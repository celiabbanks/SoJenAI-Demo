# core/models.py

from __future__ import annotations

from pathlib import Path
import os
from typing import Dict, Tuple, List, Optional, Union, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login  # NEW import


# ============================================================
# HF token + model IDs
# ============================================================
HF_TOKEN = os.getenv("HF_TOKEN")  # set on the backend server

# One-time login: this sets up auth globally in this process,
# so we don't have to pass the token into every from_pretrained call.
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        print(f"[WARN] Hugging Face login failed: {e}")

DISTIL_MODEL_ID = os.getenv(
    "DISTIL_MODEL_ID",
    "celiabbanks/sojenai-distilbert-bias",  # private HF repo
)

ROBERTA_MODEL_ID = os.getenv(
    "ROBERTA_MODEL_ID",
    "celiabbanks/sojenai-roberta-sentiment",  # your HF repo
)

USE_GPU = os.getenv("SOJEN_USE_GPU", "false").lower() in {"1", "true", "yes"}
DEVICE = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
MAX_LEN = int(os.getenv("SOJEN_MAX_LEN", "256"))


# Canonical categories and sentiment labels
CATEGORIES: List[str] = [
    "political",
    "racial",
    "sexist",
    "classist",
    "ageism",
    "antisemitic",
    "bullying",
    "brand",
]
SENT_LABELS = ["negative", "neutral", "positive"]

_bias_tok = None
_bias_model = None
_sent_tok = None
_sent_model = None


# ============================================================
# Model loader (local Path OR HF repo ID)
# ============================================================
def _load(source: Union[Path, str]):
    """
    Load a model + tokenizer either from a local Path or from a
    Hugging Face repo ID (string).

    HF auth is handled globally via huggingface_hub.login(HF_TOKEN),
    so we do NOT need to pass the token on each call.
    """
    # Local folder case
    if isinstance(source, Path):
        if not source.exists():
            raise FileNotFoundError(f"Model path does not exist: {source}")
        tok = AutoTokenizer.from_pretrained(source, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(source)

    # Hugging Face repo ID case
    else:
        repo_id = str(source)
        tok = AutoTokenizer.from_pretrained(
            repo_id,
            use_fast=True,
        )
        mdl = AutoModelForSequenceClassification.from_pretrained(
            repo_id,
        )

    mdl.to(DEVICE).eval()
    return tok, mdl


def load_all():
    """Idempotent: load both models if not loaded. Uses HF IDs or local paths."""
    global _bias_tok, _bias_model, _sent_tok, _sent_model

    if _bias_tok is None or _bias_model is None:
        _bias_tok, _bias_model = _load(DISTIL_MODEL_ID)

    if _sent_tok is None or _sent_model is None:
        _sent_tok, _sent_model = _load(ROBERTA_MODEL_ID)


def _bias_head_size() -> int:
    if _bias_model is None:
        raise RuntimeError("Bias model not loaded; call load_all() first.")
    return int(_bias_model.config.num_labels)


# ============================================================
# Seed lexicon for heuristic category guessing (binary head)
# ============================================================

LEX: Dict[str, List[str]] = {
    "racial": [
        "race",
        "racist",
        "ethnic",
        "black",
        "white",
        "asian",
        "african american",
        "hispanic",
        "latino",
        "black women are too angry",
        "asian women are submissive",
        "all white women exhibit the 'Karen behavior'",
        "latina chicks are easy",
    ],
    "sexist": [
        "women are",
        "men are",
        "boys will be boys",
        "girls can't",
        "women should",
        "feminazi",
        "men should",
        "too emotional",
    ],
    "classist": ["poor", "rich", "lower class", "upper class", "welfare", "lazy"],
    "ageism": [
        "old people",
        "should retire",
        "let younger one have the job",
        "boomers",
        "over-qualified",
        "not as agile",
        "too old",
        "too young",
        "kids these days",
    ],
    "antisemitic": ["jew", "jews", "zionist", "antisemitic", "rothschild"],
    "bullying": [
        "idiot",
        "moron",
        "retard",
        "useless",
        "pathetic",
        "trash",
        "die",
        "dick head",
        "jerk",
        "numb nuts",
        "pussy",
    ],
    "brand": [
        # core brand words
        "brand",
        "our brand",
        "my brand",
        "this brand",
        "their brand",
        "the brand",
        # brand reputation
        "brand image",
        "brand reputation",
        "brand perception",
        "brand safety",
        # social media dragging / pile-on
        "being dragged online",
        "being dragged on twitter",
        "dragged on twitter",
        "dragged on social media",
        "dragged on tiktok",
        "dragged on instagram",
        # boycott / cancel language
        "boycott",
        "boycotting",
        "cancel this brand",
        "cancel the brand",
        # product-level harsh sentiment
        "product is trash",
        "this product sucks",
        "this company sucks",
        "hate this company",
    ],
    "political": [
        "democrat",
        "republican",
        "liberal",
        "conservative",
        "leftist",
        "maga",
        "socialist",
        "commie",
        "communist",
    ],
}

# Words that often signal negative group stereotypes in simple templates
NEGATIVE_STEREOTYPE_WORDS = [
    "bad",
    "lazy",
    "weak",
    "hostile",
    "submissive",
    "stupid",
    "dumb",
    "crazy",
    "inferior",
    "violent",
    "dangerous",
    "aggressive",
    "ugly",
    "irresponsible",
    "emotional",
    "overly emotional",
    "annoying",
    "untrustworthy",
    "criminals",
    "rapists",
    "dirty",
    "illegal",
    "not good at",
    "can't drive",
    "bad drivers",
]


def _guess_category_probs(text: str) -> Dict[str, float]:
    lo = (text or "").lower()
    scores = {c: 0.0 for c in CATEGORIES}
    for cat, keys in LEX.items():
        for k in keys:
            if k in lo:
                scores[cat] += 1.0
    total = sum(scores.values())
    if total <= 0:
        return scores
    return {k: float(v / total) for k, v in scores.items()}


# ============================================================
# Your bias terms & safe phrases (for mitigation context)
# ============================================================

BIAS_TERMS: Dict[str, List[str]] = {
    "racial": [
        "nigger",
        "thug",
        "hoodlum",
        "spic",
        "chinc",
        "coon",
        "wetback",
        "black women are hostile",
        "they take our jobs",
        "haitian immigrants eat pets",
        "affirmative action is reverse discrimination",
        "darkie",
        "migrant",
        "you’re here because of dei",
        "i won't get in an elevator alone with a black person",
        "angry black woman",
        "blacks have to be twice as good",
        "illegal alien",
        "terrorist",
        "towelhead",
        "they don't belong here",
        "they are too lazy to work",
        "they are good at sports, bad at school",
        "immigrants are rapists and criminals",
    ],
    "antisemitic": [
        "jews own the media",
        "jews run america",
        "jew bitch",
        "jew bastard",
        "stingy as a jew",
        "holocaust denier",
        "nazism",
        "skinhead",
        "semite",
    ],
    "sexist": [
        "female dike",
        "female butch",
        "bitch",
        "bitch slut",
        "whore",
        "bitches and hoes",
        "women do not belong in the workplace",
        "women should not be sports casters",
        "they are only good for bearing children",
        "feminazi",
        "slut",
        "gold digger",
        "women are weak",
        "a woman's role is in the home",
        "patriarchy",
        "women are too emotional",
        "men are stronger than women",
        "women should not have careers",
        "i will not vote for a woman for us president",
        "men don't cry",
        "only men should become a us president",
        "she is a weak executive and grins alot",
    ],
    "classist": [
        "lazy",
        "freeloader",
        "criminal",
        "hoe",
        "ghetto",
        "white trash",
        "you did not go to an ivy league school",
        "welfare queen",
        "elitist",
        "blue-collar",
        "low-life",
        "privileged",
        "self-made vs. born with a silver spoon",
        "poor people don't work hard enough",
        "rich people are greedy",
    ],
    "ageism": [
        "old fart",
        "overqualified",
        "past her prime",
        "past your prime",
        "go play with your grandchildren",
        "grandma",
        "grandpa",
        "sleepy joe",
        "cannot keep up",
        "old people are a drain on society",
        "you are no longer relevant",
        "she's ancient",
        "you're ancient",
        "over the hill",
    ],
    "political": [
        "republican",
        "conservative party",
        "democrat",
        "liberal party",
        "green party",
        "the squad",
        "far right",
        "far left",
        "project 2025",
        "president trump",
        "president biden",
        "vice president harris",
        "vice president jd vance",
        "obama",
        "jill biden",
        "melania trump",
        "antifa",
        "klu klux klan",
        "nra",
        "guns and abortion",
    ],
    "brand": [
        "love this brand",
        "my brand",
        "prefer this brand",
        "birkin bag",
        "expensive taste",
        "favorite team",
        "favorite movie",
        "favorite apparel",
        "favorite food",
    ],
    "bullying": [
        "scaredy cat",
        "can't cut it",
        "dare you",
        "spit on you",
        "you should kill yourself",
        "go hide in a corner",
        "not popular enough",
        "you don't fit in",
        "shame on you",
        "you're ugly",
        "too fat",
        "too skinny",
        "put a bag over your head",
        "hide your face",
        "cover your titties",
        "hide yourself",
        "body not meant for those clothes",
        "you can't do anything right",
        "bonehead",
        "you can't handle it",
        "you're stupid",
        "you're a moron",
        "she's big as an apartment complex",
        "she can afford to miss a meal",
    ],
}

SAFE_PHRASES: List[str] = [
    "love my grandparents",
    "my adorable grandparents",
    "grandma's cooking is delicious",
    "my grandma makes the best apple pie",
    "grandma shared stories from her youth",
    "grandma's advice is always so helpful",
    "my grandma loves gardening",
    "my grandpa used to be a teacher",
    "spending time with my grandparents is the best",
    "i am a foreigner to learning technology",
    "let's pig out and enjoy this ice cream",
    "you're so skinny oh how i want to be too",
    "we are better than that is not a classist statement",
    "having wealth can make you more wealth",
    "your behavior is ugly and needs improvement",
    "the older generation has a lot to teach us",
    "young people today have so many new opportunities",
    "he is a successful foreigner who has made america his home",
    "i am too lazy to run for the u.s. congress",
    "i feel lazy today",
    "lazy saturday mornings are the best",
    "george washington carver was a prominent black scientist of the early 20th century with his sustainable farming techniques",
    "african american hip hop artists are some of the wealthiest people in the u.s.",
]

# ============================================================
# Core predictors
# ============================================================


@torch.inference_mode()
def predict_bias_type(text: str) -> Dict[str, float]:
    """
    If the bias model has 8 labels, return per-category probabilities.
    If it's binary (non_biased/biased), combine model 'presence' with
    a lexicon-based category guess.
    """
    load_all()
    enc = _bias_tok(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)
    logits = _bias_model(**enc).logits.squeeze(0)

    head = _bias_head_size()

    if head == len(CATEGORIES):
        probs = F.softmax(logits, dim=-1)
        return {CATEGORIES[i]: float(probs[i].item()) for i in range(len(CATEGORIES))}
    elif head == 2:
        # binary: [non_biased, biased]
        probs = F.softmax(logits, dim=-1)
        presence = float(probs[1].item())
        cat_guess = _guess_category_probs(text)
        return {k: float(v * presence) for k, v in cat_guess.items()}
    else:
        # unexpected head size; fallback to heuristic
        return _guess_category_probs(text)


@torch.inference_mode()
def predict_implicit_explicit(text: str) -> int:
    """
    Returns:
        0 = neutral / none
        1 = explicit (strong slurs or direct attacks)
        2 = implicit (stereotypes / generalizations about a group)
    """
    lo = (text or "").lower()

    # --- Explicit bias markers: strong slurs or direct "you should X" attacks ---
    explicit_terms = [
        # racial slurs
        "nigger",
        "spic",
        "chinc",
        "wetback",
        "darkie",
        "towelhead",
        "illegal alien",
        # sexist / misogynistic
        "bitch",
        "whore",
        "bitches and hoes",
        "slut",
        "bitch slut",
        # antisemitic
        "jew bitch",
        "stingy as a jew",
        # bullying / self-harm
        "you should kill yourself",
        "go kill yourself",
        "you should die",
        "you're stupid",
        "you're a moron",
        "idiot",
        "pathetic",
        "trash",
    ]
    if any(term in lo for term in explicit_terms):
        return 1  # explicit

    # --- Implicit stereotypes: group + negative attribute ---
    group_terms = [
        "women",
        "men",
        "black women",
        "black people",
        "asian women",
        "asians",
        "jews",
        "immigrants",
        "old people",
        "young people",
        "poor people",
        "rich people",
        "those people",
        "they all",
        "they are",
    ]

    has_group = any(g in lo for g in group_terms)
    has_negative_stereo = any(neg in lo for neg in NEGATIVE_STEREOTYPE_WORDS)

    if has_group and has_negative_stereo:
        return 2  # implicit stereotype, e.g. "women are bad drivers"

    # existing light implicit cues
    implicit_hits = ["always", "never", "those people", "they all", "lazy"]
    if any(t in lo for t in implicit_hits):
        return 2

    return 0


@torch.inference_mode()
def predict_sentiment(text: str) -> str:
    load_all()
    enc = _sent_tok(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)
    logits = _sent_model(**enc).logits.squeeze(0)
    probs = F.softmax(logits, dim=-1)
    idx = int(torch.argmax(probs).item())
    return SENT_LABELS[max(0, min(idx, len(SENT_LABELS) - 1))]


# ============================================================
# Severity assessment
# ============================================================

SEVERE_BLOCK_CATEGORIES = {"racial", "antisemitic", "sexist", "bullying"}


def assess_bias_severity(
    text: str, scores: Dict[str, float]
) -> Tuple[str, Optional[str], Dict[str, float]]:
    """
    Returns:
        severity_label: "none", "low", "medium", "high"
        reason: human-readable explanation
        meta: dict with max_prob, top_label, implicit_flag, lexicon info, etc.
    """
    lo = (text or "").lower()
    if not scores:
        scores = {k: 0.0 for k in CATEGORIES}

    # --------------------------------------------------------
    # Build lexicon_hits from BOTH BIAS_TERMS and LEX,
    # with a negativity gate for stereotype-like patterns.
    # --------------------------------------------------------
    lexicon_hits: Dict[str, List[str]] = {}

    # 1) Exact phrases from your curated BIAS_TERMS
    for cat, terms in BIAS_TERMS.items():
        for term in terms:
            if term.lower() in lo:
                lexicon_hits.setdefault(cat, []).append(term)

    # 2) Pattern-based hits from LEX (e.g., "women are", "asian")
    #    For higher-risk categories, require a negative stereotype word.
    for cat, keys in LEX.items():
        for key in keys:
            key_lo = key.lower()
            if key_lo in lo:
                if cat in {"racial", "sexist", "classist", "ageism", "antisemitic", "bullying"}:
                    # Only treat as a hit if there's a negative stereotype term too
                    if not any(neg in lo for neg in NEGATIVE_STEREOTYPE_WORDS):
                        continue
                lexicon_hits.setdefault(cat, []).append(key)

    all_zero = all((v is None) or (float(v) <= 0.0) for v in scores.values())
    implicit_explicit = predict_implicit_explicit(text)

    # Model max score (if any)
    if scores:
        top_label_model, max_prob_model = max(scores.items(), key=lambda kv: kv[1])
    else:
        top_label_model, max_prob_model = (None, 0.0)

    # --------------------------------------------------------
    # Case A: Model is timid (<0.20) BUT lexicon says "this is a stereotype"
    # --------------------------------------------------------
    if max_prob_model < 0.2 and lexicon_hits:
        priority = [
            "racial",
            "antisemitic",
            "sexist",
            "bullying",
            "classist",
            "ageism",
            "brand",
            "political",
        ]

        def _cat_rank(c: str) -> tuple[int, int]:
            count = len(lexicon_hits.get(c, []))
            idx = priority.index(c) if c in priority else 999
            return (-count, idx)

        sorted_cats = sorted(lexicon_hits.keys(), key=_cat_rank)
        primary = sorted_cats[0]

        if primary in SEVERE_BLOCK_CATEGORIES:
            severity = "high"
            max_prob = 0.8
            reason = (
                "Lexicon indicates a strong identity-based stereotype toward a protected group "
                "even though the model's numeric signal is low. Treated as high severity."
            )
        else:
            severity = "medium"
            max_prob = 0.5
            reason = (
                "Lexicon indicates biased phrasing even though the model's numeric signal is weak. "
                "Treated as medium severity."
            )

        meta = {
            "max_prob": float(max_prob),
            "top_label": primary,
            "implicit_explicit": implicit_explicit,
            "lexicon_override": True,
            "lexicon_hits": lexicon_hits,
        }
        return severity, reason, meta

    # --------------------------------------------------------
    # Case B: Model ~0 and NO lexicon hits → genuinely clean
    # --------------------------------------------------------
    if all_zero and not lexicon_hits:
        severity = "none"
        reason = (
            "Model scores are near zero across all categories and no lexicon hits detected; "
            "treated as no detectable bias."
        )
        meta = {
            "max_prob": 0.0,
            "top_label": None,
            "implicit_explicit": implicit_explicit,
            "lexicon_override": False,
            "lexicon_hits": lexicon_hits,
        }
        return severity, reason, meta

    # --------------------------------------------------------
    # Case C: Normal model-driven severity (model has some signal)
    # --------------------------------------------------------
    top_label = top_label_model
    max_prob = max_prob_model

    if max_prob < 0.2:
        severity = "none"
        reason = "Model confidence is low (below 0.20); treated as no actionable bias."
    elif max_prob < 0.4:
        severity = "low"
        reason = "Model detects weak bias signal; suitable for coaching or monitoring."
    elif max_prob < 0.7:
        severity = "medium"
        reason = "Model detects moderate bias signal; candidate for mitigation/rewrite."
    else:
        severity = "high"
        reason = "Model detects strong bias signal toward a protected group."

    if top_label in SEVERE_BLOCK_CATEGORIES and severity == "medium":
        severity = "high"
        reason += " Escalated due to category being high-risk."
    if implicit_explicit == 1 and severity != "none":
        if severity == "low":
            severity = "medium"
        elif severity == "medium":
            severity = "high"
        reason += " Marked explicit based on phrasing."

    meta = {
        "max_prob": float(max_prob),
        "top_label": top_label,
        "implicit_explicit": implicit_explicit,
        "lexicon_override": False,
        "lexicon_hits": lexicon_hits,
    }
    return severity, reason, meta


# ============================================================
# Mitigation
# ============================================================

def mitigate_text(text: str) -> Dict[str, Any]:
    """
    High-level mitigation helper used by the /v1/mitigate endpoint.

    Strategy:
      - severity = "none": no rewrite, light advisory.
      - severity = "low": gentle advisory, no strong rewrite.
      - severity = "medium" or "high": ALWAYS provide a category-specific suggested rewrite,
        plus an advisory explaining why the original is harmful.

    Returned keys:
      - original: the raw input text
      - severity: none/low/medium/high
      - mode: "none" | "advisory" | "rewrite"
      - rewritten: suggested safer wording (may be None for none/low)
      - advisory: explanation for the user
      - scores, meta: debugging + UI
    """
    scores = predict_bias_type(text)
    severity, reason, meta = assess_bias_severity(text, scores)
    top_label = meta.get("top_label", "unknown")
    max_prob = meta.get("max_prob", 0.0)
    lex_hits = meta.get("lexicon_hits", {}) or {}

    # ---------------------------
    # NONE severity: no-op
    # ---------------------------
    if severity == "none":
        return {
            "original": text,
            "severity": severity,
            "mode": "none",
            "rewritten": None,
            "advisory": (
                "SoJenAI-Moderator did not detect sufficient risk in this message "
                "to justify a rewrite. No mitigation is applied."
            ),
            "scores": scores,
            "meta": meta,
        }

    # ---------------------------
    # LOW severity: coaching only
    # ---------------------------
    if severity == "low":
        return {
            "original": text,
            "severity": severity,
            "mode": "advisory",
            "rewritten": None,
            "advisory": (
                "SoJenAI-Moderator sees a weak bias or tone risk. You may want to soften "
                "the language or add context, but an automatic rewrite is not required "
                "at this level."
            ),
            "scores": scores,
            "meta": meta,
        }

    # ---------------------------
    # MEDIUM / HIGH severity:
    # always propose a safer rewrite
    # ---------------------------
    cats_present = [c for c, terms in (lex_hits or {}).items() if terms]
    is_intersectional = ("racial" in cats_present and "sexist" in cats_present)

    # Category-specific generic suggested rewrites
    if is_intersectional:
        suggested = (
            "I want to understand people's individual experiences without making "
            "assumptions about anyone based on both their race and gender."
        )
    elif top_label == "racial":
        suggested = (
            "I want to focus on individual behavior and experiences without making "
            "assumptions about any racial or ethnic group."
        )
    elif top_label == "sexist":
        suggested = (
            "I want to talk about specific situations without making assumptions "
            "about any gender as a whole."
        )
    elif top_label == "classist":
        suggested = (
            "I want to discuss challenges and opportunities without judging people "
            "based on income or social class."
        )
    elif top_label == "ageism":
        suggested = (
            "I want to talk about the situation without assuming that age alone "
            "determines someone's abilities or value."
        )
    elif top_label == "antisemitic":
        suggested = (
            "I want to express my concerns without making harmful generalizations "
            "about Jewish people or any religious group."
        )
    elif top_label == "bullying":
        suggested = (
            "I want to address the issue directly but in a way that is respectful "
            "and does not attack someone's worth or appearance."
        )
    elif top_label == "brand":
        suggested = (
            "I'm frustrated with this experience, but I want to describe what went wrong "
            "without attacking people or using hostile language."
        )
    else:
        # Fallback when label is missing or unknown
        suggested = (
            "I want to share my concerns in a way that focuses on specific behavior "
            "or impact without targeting any group based on identity."
        )

    # Build a more detailed advisory that teaches the user
    base_msg = (
        "SoJenAI-Moderator has detected a strong bias or stereotype in this message. "
        "Statements that generalize an entire group (for example, by gender, race, "
        "age, or other identity) can reinforce harmful stereotypes and create an unsafe environment."
    )

    if is_intersectional:
        base_msg += (
            " In this case, the wording combines stereotypes about both race and gender "
            "which amplifies the harm by targeting people with that combined identity."
        )
    elif top_label == "racial":
        base_msg += (
            " In this case, the language assigns a negative trait to a racialized group "
            "as a whole, rather than focusing on specific behavior in context."
        )
    elif top_label == "sexist":
        base_msg += (
            " In this case, the language assigns a negative trait to a gender group "
            "as a whole (for example, saying that all women or all men share a flaw), "
            "which is a form of gender stereotyping."
        )
    elif top_label == "bullying":
        base_msg += (
            " In this case, the language attacks a person's worth or appearance, "
            "which can be experienced as harassment or bullying."
        )

    return {
        "original": text,
        "severity": severity,
        "mode": "rewrite",
        "rewritten": suggested,
        "advisory": base_msg,
        "scores": scores,
        "meta": meta,
    }


# ============================================================
# Debug helper
# ============================================================

def model_debug_summary() -> Dict[str, Any]:
    try:
        return {
            "bias": {
                "num_labels": int(_bias_model.config.num_labels) if _bias_model else None,
                "id2label": getattr(_bias_model.config, "id2label", None) if _bias_model else None,
                "path": str(DISTIL_MODEL_ID),
            },
            "sentiment": {
                "num_labels": int(_sent_model.config.num_labels) if _sent_model else None,
                "id2label": getattr(_sent_model.config, "id2label", None) if _sent_model else None,
                "path": str(ROBERTA_MODEL_ID),
            },
            "device": DEVICE,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
