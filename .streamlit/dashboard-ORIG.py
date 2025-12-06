# SoJenAI-Demo/dashboard.py

import os
from typing import List, Dict, Any

import requests
import streamlit as st
import pandas as pd
from PIL import Image

# -----------------------------
# Config
# -----------------------------
API_BASE = os.getenv("SOJEN_API_BASE", "http://127.0.0.1:8010")
HEALTH_ENDPOINT = f"{API_BASE}/health"
INFER_ENDPOINT = f"{API_BASE}/v1/infer"
MITIGATE_ENDPOINT = f"{API_BASE}/v1/mitigate"

# If the logo is in the same folder as this file, keep as-is.
# If you moved it into assets/images, you can change this to "../assets/images/..."
LOGO_FILENAME = "JenAI-Moderator_CommIntell.png"


# -----------------------------
# Helpers
# -----------------------------
def load_logo():
    # First try local path
    if os.path.exists(LOGO_FILENAME):
        try:
            return Image.open(LOGO_FILENAME)
        except Exception:
            return None

    # Optionally, also try assets/images relative to project root
    alt_path = os.path.join("assets", "images", LOGO_FILENAME)
    if os.path.exists(alt_path):
        try:
            return Image.open(alt_path)
        except Exception:
            return None

    return None


def call_health() -> Dict[str, Any]:
    resp = requests.get(HEALTH_ENDPOINT, timeout=5)
    resp.raise_for_status()
    return resp.json()


def call_infer(texts: List[str]) -> Dict[str, Any]:
    payload = {"texts": texts}
    resp = requests.post(INFER_ENDPOINT, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def call_mitigate(text: str) -> Dict[str, Any]:
    """
    NEW: align with current /v1/mitigate API

    Backend expects: { "text": "<comment>" }
    and returns keys like:
      - severity
      - mode
      - rewritten
      - advisory
      - scores
      - meta
      - lexicon_hits
    """
    payload = {"text": text}
    resp = requests.post(MITIGATE_ENDPOINT, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# -----------------------------
# Streamlit UI setup
# -----------------------------
st.set_page_config(
    page_title="SoJen.AI — Communication Intelligence",
    layout="wide",
)

logo = load_logo()

# Initialize session state for inference results
if "infer_results" not in st.session_state:
    st.session_state.infer_results = None
    st.session_state.type_order = []
    st.session_state.device = None

# Sidebar
with st.sidebar:
    # Brand / logo
    if logo is not None:
        st.image(
            logo,
            caption="JenAI-Moderator • Communication Intelligence",
            use_container_width=True,
        )

    st.markdown("### About this API")
    st.markdown(
        """
**Creator:** Celia Banks  
**Project:** SoJen.AI — JenAI-Moderator  

JenAI-Moderator provides *Communication Intelligence* for:
- Bias detection across multiple categories
- Severity assessment (none/low/medium/high)
- Advisory or rewrite responses
"""
    )

    st.markdown("### Models")
    st.markdown(
        """
- **Bias model:** DistilBERT-based classifier  
- **Sentiment model:** RoBERTa-based classifier  
- **Categories:** political, racial, sexist, classist, ageism, antisemitic, bullying, brand
"""
    )

    st.markdown("### Backend status")
    try:
        health = call_health()
        device = health.get("device", "n/a")
        st.success(f"API OK — device: `{device}`")
    except Exception as e:
        st.error(f"Health check failed: {e}")

# ------------

# -----------------------------
# Device Badge (GPU / CPU)
# -----------------------------
try:
    health = call_health()
    device = health.get("device", "n/a")

    badge_color = "#52c41a" if device == "cuda" else "#d9d9d9"
    badge_text = "GPU Acceleration" if device == "cuda" else "CPU Mode"

    st.markdown(
        f"""
        <div style="
            margin-top:10px;
            padding:6px 12px;
            border-radius:999px;
            background-color:{badge_color};
            color:white;
            font-size:13px;
            font-weight:600;
            text-align:center;
        ">
            {badge_text}
        </div>
        """,
        unsafe_allow_html=True,
    )
except Exception:
    st.warning("Could not determine accelerator status.")


# ------------




    st.markdown("### API Details")
    st.markdown("**Base URL:**")
    st.code(API_BASE, language="bash")
    st.markdown("**Endpoints:**")
    st.write("- `GET /health` — health & device")
    st.write("- `POST /v1/infer` — bias scores")
    st.write("- `POST /v1/mitigate` — advisory/rewrite")


# Main title
st.title("SoJen.AI — JenAI-Moderator")
st.subheader("Communication Intelligence for Bias Detection & Rewrite")

st.markdown(
    """
JenAI-Moderator analyzes text for bias categories and offers a more constructive rewrite
while preserving the speaker’s intent — or, for high-severity cases, advisory feedback
on why the content is harmful.
"""
)

# ------------

# Performance Indicator (GPU/CPU)
if "device" in st.session_state and st.session_state.device:
    perf_device = st.session_state.device

    perf_color = "#52c41a" if perf_device == "cuda" else "#faad14"
    perf_label = "Ultra-fast GPU Mode" if perf_device == "cuda" else "Standard CPU Mode"

    st.markdown(
        f"""
        <div style="
            margin-top:4px;
            padding:8px 14px;
            border-radius:8px;
            background-color:{perf_color};
            color:white;
            font-size:14px;
            font-weight:600;
            display:inline-block;
        ">
            {perf_label}
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div style="
            margin-top:4px;
            padding:8px 14px;
            border-radius:8px;
            background-color:#d9d9d9;
            color:#444;
            font-size:14px;
            font-weight:600;
            display:inline-block;
        ">
            Performance Mode: Unknown
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------


# -----------------------------
# Input section (Single text only)
# -----------------------------
st.markdown("### Input")

default_text = "Women are bad drivers."

text_single = st.text_area(
    "Enter text for analysis:",
    value=default_text,
    height=140,
)

texts: List[str] = [text_single] if text_single.strip() else []

run_button = st.button(
    "Analyze with JenAI-Moderator",
    type="primary",
    use_container_width=True,
)

# If Analyze clicked, call /v1/infer and store in session_state
if run_button:
    if not texts:
        st.warning("Please enter at least one text.")
    else:
        with st.spinner("Calling /v1/infer..."):
            try:
                infer_res = call_infer(texts)
            except Exception as e:
                st.error(f"Error calling /v1/infer: {e}")
            else:
                st.session_state.device = infer_res.get("device", "unknown")
                st.session_state.type_order = infer_res.get("type_order", [])
                st.session_state.infer_results = infer_res.get("results", [])


# -----------------------------
# Results section (always use session_state)
# -----------------------------

# -----------------------------
# Results section (always use session_state)
# -----------------------------
device = st.session_state.device
type_order = st.session_state.type_order
results = st.session_state.infer_results

if results:
    st.markdown("---")
    st.markdown(f"**Model device:** `{device}`")
    st.markdown("---")

    for idx, item in enumerate(results):
        text = item.get("text", "")
        # Prefer scores_ordered, but fall back to raw scores or zeros
        scores_ordered = item.get("scores_ordered") or {}
        raw_scores = item.get("scores") or {}
        meta = item.get("meta", {}) or {}
        sev_meta = meta.get("severity_meta", {}) or {}

        # Use severity/top_label from severity_meta so it reflects lexicon overrides
        top_label = sev_meta.get("top_label", item.get("top_label"))
        severity = item.get("severity", "none")

        # Ensure we always have a category list and score vector
        if not type_order:
            type_order = list(raw_scores.keys())

        if not scores_ordered and type_order:
            scores_ordered = {cat: float(raw_scores.get(cat, 0.0)) for cat in type_order}

        exp_label = f"Text #{idx+1} — Top category: **{top_label or 'none'}**"
        with st.expander(exp_label, expanded=(idx == 0)):
            # Original text
            st.markdown("**Original text**")
            st.write(text)

            # Implicit / explicit / neutral indicator
            implicit_flag = sev_meta.get("implicit_explicit", 0)
            implicit_map = {
                0: "neutral / none",
                1: "explicit",
                2: "implicit",
            }
            implicit_label = implicit_map.get(implicit_flag, "unknown")

            sev_display = severity.capitalize()
            st.markdown(
                f"**Severity:** `{sev_display}`  •  **Bias style:** `{implicit_label}`"
            )

            # Scores as table + bar chart (even if all near-zero)
            if type_order:
                data = {
                    "category": type_order,
                    "score": [
                        float(scores_ordered.get(cat, 0.0)) for cat in type_order
                    ],
                }
                df = pd.DataFrame(data)

                st.markdown("**Bias category scores (model probabilities)**")
                st.dataframe(
                    df.style.highlight_max(subset=["score"], color="#ffe6e6"),
                    use_container_width=True,
                )

                st.markdown("**Visualization**")
                st.bar_chart(
                    df.set_index("category")["score"],
                    use_container_width=True,
                )
            else:
                # This should be rare now, but keep a graceful fallback
                st.info(
                    "Model returned no category scores for this text; this usually means "
                    "it detected no discernible bias signal."
                )

            # Meta info
            with st.expander("Model metadata", expanded=False):
                st.json(meta)

            # -----------------------------
            # JenAI-Moderator Rewrite section
            # -----------------------------
            st.markdown("---")
            st.markdown("#### JenAI-Moderator Rewrite")

            rewrite_col1, rewrite_col2 = st.columns([1, 3])

            with rewrite_col1:
                if logo is not None:
                    st.image(logo, width=120)
                else:
                    st.markdown("**JenAI-Moderator**")

            with rewrite_col2:
                st.write(
                    "Click **Run Rewrite** to get a more constructive or advisory "
                    "response for this message, depending on its severity. "
                    "You can then edit the suggested wording directly."
                )

                if st.button("Run Rewrite", key=f"rewrite_{idx}"):
                    with st.spinner("Calling /v1/mitigate..."):
                        try:
                            mit = call_mitigate(text)
                        except Exception as e:
                            st.error(f"Error calling /v1/mitigate: {e}")
                        else:
                            mode = mit.get("mode", "rewrite")
                            m_severity = mit.get("severity", severity)
                            advisory = mit.get("advisory", "")
                            rewritten = mit.get("rewritten", None)
                            mit_meta = mit.get("meta", {}) or {}
                            primary_cat = mit_meta.get("top_label", top_label)

                            # Severity badge
                            sev_label = m_severity.capitalize()
                            badge_color = {
                                "high": "#ff4d4f",
                                "medium": "#faad14",
                                "low": "#52c41a",
                                "none": "#d9d9d9",
                            }.get(m_severity, "#d9d9d9")

                            st.markdown(
                                f"""
                                <div style="
                                    display:inline-block;
                                    padding:4px 10px;
                                    border-radius:999px;
                                    background-color:{badge_color};
                                    color:white;
                                    font-size:12px;
                                    margin-bottom:6px;
                                ">
                                    Severity: {sev_label}{(" • " + str(primary_cat)) if primary_cat else ""}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # -----------
                            # Main response text
                            st.markdown("**JenAI-Moderator explanation:**")
                            if advisory:
                                st.markdown(advisory)
                            else:
                                st.markdown(
                                    "_No detailed advisory was provided for this message._"
                                )

                            # Show suggested rewrite from the model, if present
                            if rewritten:
                                st.markdown("**JenAI-Moderator suggested rewrite:**")
                                st.code(rewritten)


                            # -----------

                            # Persona header
                            st.markdown(
                                """
                                <div style="
                                    margin-top:6px;
                                    padding:12px;
                                    border-radius:8px;
                                    background-color:#eef3ff;
                                    border-left:4px solid #4a6cff;
                                ">
                                    <strong>JenAI-Moderator</strong><br>
                                    <em>Communication Intelligence Response</em>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Main response text
                            st.markdown("**JenAI-Moderator explanation:**")
                            if advisory:
                                st.markdown(advisory)
                            else:
                                st.markdown(
                                    "_No detailed advisory was provided for this message._"
                                )

                            # Show suggested rewrite from the model, if present
                            if rewritten:
                                st.markdown("**JenAI-Moderator suggested rewrite:**")
                                st.code(rewritten)
                                st.caption(
                                    "You can copy this suggested rewrite into your product, social platform, "
                                    "HR system, or internal tooling. It is designed to preserve intent while "
                                    "removing harmful stereotypes or attacks."
                                )
                            else:
                                st.caption(
                                    "For this message, JenAI-Moderator is providing advisory feedback only. "
                                    "You can draft your own alternative phrasing based on the explanation above."
                                )


                            # Mode explanation
                            if mode == "advisory":
                                st.markdown(
                                    "_JenAI-Moderator is in **advisory mode** for "
                                    "this message due to its assessed severity toward "
                                    "a protected group. The response explains why the "
                                    "content may be harmful and suggests a different way to "
                                    "express underlying concerns._"
                                )
                            elif mode == "rewrite":
                                st.markdown(
                                    "_JenAI-Moderator is in **rewrite mode**, providing "
                                    "a clearer and less harmful version of the message "
                                    "while preserving intent._"
                                )
                            else:  # mode == "none"
                                st.markdown(
                                    f"_JenAI-Moderator has **not proposed a rewrite** at this severity level "
                                    f"(`severity = {m_severity}`). The model signal is too low to justify "
                                    "an automatic bias mitigation rewrite._"
                                )

