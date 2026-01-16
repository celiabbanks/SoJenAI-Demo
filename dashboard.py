# SoJenAI-Demo/.streamlit/dashboard.py

import torch
import os
from typing import List, Dict, Any
from pathlib import Path

import requests
import streamlit as st
import pandas as pd
from PIL import Image

from gtts import gTTS
import io



# -----------------------------
# Config
# -----------------------------
API_BASE = os.getenv("SOJEN_API_BASE", "http://127.0.0.1:8010")
HEALTH_ENDPOINT = f"{API_BASE}/health"
INFER_ENDPOINT = f"{API_BASE}/v1/infer"
MITIGATE_ENDPOINT = f"{API_BASE}/v1/mitigate"

LOGO_FILENAME = "SoJenAI-Moderator_CommIntell.png"


# -----------------------------
# Helpers
# -----------------------------

def load_logo():
    """
    Load the SoJenAI-Moderator logo from assets/images/
    relative to the project root.
    """
    dashboard_dir = Path(__file__).resolve().parent
    project_root = dashboard_dir  # now dashboard.py is at project root
    img_path = project_root / "assets" / "images" / LOGO_FILENAME

    if img_path.exists():
        try:
            return Image.open(img_path)
        except Exception:
            return None
    return None


def call_health() -> Dict[str, Any]:
    resp = requests.get(HEALTH_ENDPOINT, timeout=5)
    resp.raise_for_status()
    return resp.json()


def call_infer(texts: List[str]) -> Dict[str, Any]:
    payload = {"texts": texts}
    resp = requests.post(INFER_ENDPOINT, json=payload, timeout=120) # increased timeout from 30 sec to 120 for HF connection
    resp.raise_for_status()
    return resp.json()


def call_mitigate(text: str) -> Dict[str, Any]:
    """
    /v1/mitigate expects a JSON body like:
        {"text": "..."}
    and returns:
        {
          "mode": "rewrite" | "advisory" | "none",
          "severity": "...",
          "advisory": "...",
          "rewritten": "...",
          "meta": {...}
        }
    """
    payload = {"text": text}
    resp = requests.post(MITIGATE_ENDPOINT, json=payload, timeout=120) # increased timeout from 30 sec to 120
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


# Initialize session state
if "infer_results" not in st.session_state:
    st.session_state.infer_results = None
    st.session_state.type_order = []
    st.session_state.device = None
    st.session_state.backend_device = None  # from /health


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    # Brand / logo
    if logo is not None:
        st.image(
            logo,
            caption="SoJenAI-Moderator • Communication Intelligence",
            use_container_width=True,
        )

    st.markdown("### About this API")
    st.markdown(
        """
**Creator:** Celia Banks  
**Project:** SoJen.AI — SoJenAI-Moderator  

SoJenAI-Moderator provides *Communication Intelligence* for:
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
    health = None
    device = "n/a"
    try:
        health = call_health()
        device = health.get("device", "n/a")
        st.success(f"API OK — device: `{device}`")
        # Save for main performance indicator
        st.session_state.backend_device = device
    except Exception as e:
        st.error(f"Health check failed: {e}")

    # Device badge (GPU / CPU) if health worked
    if health is not None:
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

    st.markdown("### API Details")
    st.markdown("**Base URL:**")
    st.code(API_BASE, language="bash")
    st.markdown("**Endpoints:**")
    st.write("- `GET /health` — health & device")
    st.write("- `POST /v1/infer` — bias scores + severity")
    st.write("- `POST /v1/mitigate` — advisory/rewrite")

    # -----------------------------
    # Disclaimer 
    # -----------------------------
    st.markdown("### Legal / IP Notice")
    st.caption(
        "This demo is part of the patent-pending SoJen.AI system and is "
        "provided for evaluation only. Please do not share this link externally."
    )




# -----------------------------
# Main title + Performance mode
# -----------------------------
# st.title("SoJen.AI — SoJenAI-Moderator")
st.markdown("<h1>SoJen.AI™ — SoJenAI-Moderator</h1>", unsafe_allow_html=True)
st.subheader("Communication Intelligence for Bias Detection & Rewrite")

# Voice-enabled badge
st.markdown(
    """
    <div style="
        margin-top:6px;
        padding:6px 12px;
        border-radius:999px;
        background-color:#4a6cff;
        color:#ffffff;
        display:inline-block;
        font-size:12px;
        font-weight:600;
    ">
        SoJenAI-Moderator voice enabled
    </div>
    """,
    unsafe_allow_html=True,
)


# Performance indicator: prefer device from last /v1/infer, else from /health
perf_device = st.session_state.device or st.session_state.backend_device
if perf_device:
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

# -----------------------------
# Landing page intro 
# -----------------------------
st.markdown("""
# **Welcome to the SoJen.AI Communication Intelligence Demo**

This dashboard previews the engine that powers **SoJen.AI** — a system designed to interpret **tone**, **intent**, 
and **emotional context** in digital communication.  
Through our AI persona, **SoJenAI-Moderator**, the engine provides real-time, emotionally intelligent guidance that helps 
users communicate more clearly and calmly.

This demo represents **Layer 1** of the SoJen.AI offering: a commercial communication intelligence API.

**Layer 2 — the SoJen.AI social platform — is protected under a U.S. patent application and is not shown here.**  
The full platform extends this engine into a wellness-centered digital environment built for calmer, safer communication.

Use this demo to explore how the engine understands messages, detects escalation, and generates constructive alternatives in real time.
""")

# -----------------------------
# How to Use This Demo 
# -----------------------------
st.markdown("""
## **How to Use This Demo**

Welcome to the **SoJen.AI demonstration dashboard**.  
This demo highlights how our communication intelligence engine interprets **tone**, **intent**, and **emotional context** 
in real time and provides supportive guidance through our AI persona, **SoJenAI-Moderator**.

1. **Enter any message** into the input text box below — workplace note, student message, customer interaction, or social post. Just replace the default comment text.
2. Comment examples to test the different bias categories:
   - **Brand:  I hate Kellogg brand cereals.**
   - **Bullying: You are such a moron.**
   - **Sexism: Women are bad drivers.**
   - **Racism: Black women are hostile.**
   - **Classism: Welfare queen should get a job.**
   - **Antisemitism: Jews own the media.**
   - **Ageism: Grandma should retire and let someone younger have her job.**
   - **Political: MAGA Republicans are foolish people.**             
3. The engine will analyze your message and return:
   - **Tone and intent classification**
   - **Emotional context inference**
   - **Bias or escalation indicators** (when present)
4. Inside the results expander, click **Run Rewrite** to get a **SoJenAI-Moderator response** — a clearer, calmer, or more constructive version of your message.
5. When a rewrite or advisory appears, click **▶️ Play SoJenAI-Moderator voice** to **hear** the response spoken in a professional tone.
6. Try different tones — **neutral, frustrated, confused, overwhelmed, dismissive, emotional** — and see how the system responds.
7. All processing is done via the SoJen.AI **ML + Generative AI engine**, which powers both our commercial API and our 
   **patent-pending social platform**.

This demo is provided for evaluation as part of having met the **University of Michigan School of Information MADS capstone project** and SoJen.AI’s 
**patent-pending research and development. Please do not share this link externally.**
""")

# -----------------------------
# Input section (Single text)
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
    "Analyze with SoJenAI-Moderator",
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
# Results section (simplified with audio)
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

        # Implicit / explicit / neutral indicator
        implicit_flag = sev_meta.get("implicit_explicit", 0)
        implicit_map = {
            0: "neutral / none",
            1: "explicit",
            2: "implicit",
        }
        implicit_label = implicit_map.get(implicit_flag, "unknown")

        exp_label = f"Text #{idx+1} — Top category: **{top_label or 'none'}**"
        with st.expander(exp_label, expanded=(idx == 0)):
            # DEBUG so we know the expander content is rendering
            st.markdown("**DEBUG: Inside expander content.**")

            # Original text
            st.markdown("**Original text**")
            st.write(text)

            sev_display = severity.capitalize()
            st.markdown(
                f"**Severity:** `{sev_display}`  •  **Bias Style:** `{implicit_label}`"
            )

            # Scores as table + bar chart
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
                st.info(
                    "Model returned no category scores for this text; "
                    "this usually means it detected no discernible bias signal."
                )

            # Meta info
            with st.expander("Model metadata", expanded=False):
                st.json(meta)

            # -----------------------------
            # JenAI-Moderator Rewrite + Audio
            # -----------------------------
            st.markdown("---")
            st.markdown("#### SoJenAI-Moderator Rewrite")

            rewrite_col1, rewrite_col2 = st.columns([1, 3])

            with rewrite_col1:
                if logo is not None:
                    st.image(logo, width=120)
                else:
                    st.markdown("**SoJenAI-Moderator**")

            with rewrite_col2:
                st.write(
                    "Click **Run Rewrite** to get a more constructive or advisory "
                    "response for this message, depending on its severity."
                )

                if st.button("Run Rewrite", key=f"rewrite_{idx}"):
                    st.markdown("**DEBUG: Run Rewrite handler entered.**")

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
                                    <strong>SoJenAI-Moderator</strong><br>
                                    <em>Communication Intelligence Response</em>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Explanation
                            st.markdown("**SoJenAI-Moderator explanation:**")
                            if advisory:
                                st.markdown(advisory)
                            else:
                                st.markdown(
                                    "_No detailed advisory was provided for this message._"
                                )

                            # Suggested rewrite
                            if rewritten:
                                st.markdown("**SoJenAI-Moderator suggested rewrite:**")
                                st.code(rewritten)
                                st.caption(
                                    "You can copy this suggested rewrite into your product, "
                                    "social platform, HR system, or internal tooling. "
                                    "It is designed to preserve intent while removing "
                                    "harmful stereotypes or attacks."
                                )
                            else:
                                st.caption(
                                    "For this message, SoJenAI-Moderator is providing advisory "
                                    "feedback only. You can draft your own alternative phrasing "
                                    "based on the explanation above."
                                )

                            # --------------------------------
                            # SoJenAI-Moderator voice: inline audio
                            # --------------------------------
                            spoken_text = rewritten or advisory  # Prefer rewrite; fallback to advisory

                            if spoken_text:
                                st.markdown("##### Hear SoJenAI-Moderator")
                                try:
                                    st.caption("Generating SoJenAI-Moderator audio…")

                                    tts = gTTS(text=spoken_text, lang="en", slow=False)
                                    audio_bytes = io.BytesIO()
                                    tts.write_to_fp(audio_bytes)
                                    audio_bytes.seek(0)
                                    st.audio(audio_bytes.read(), format="audio/mp3")
                                except Exception as e:
                                    st.error(f"Unable to generate voice right now: {e}")

                            # Mode explanation
                            if mode == "advisory":
                                st.markdown(
                                    "_SoJenAI-Moderator is in **advisory mode** for "
                                    "this message due to its assessed severity toward "
                                    "a protected group. The response explains why the "
                                    "content may be harmful and suggests a different way to "
                                    "express underlying concerns._"
                                )
                            elif mode == "rewrite":
                                st.markdown(
                                    "_SoJenAI-Moderator is in **rewrite mode**, providing "
                                    "a clearer and less harmful version of the message "
                                    "while preserving intent._"
                                )
                            else:  # mode == "none"
                                st.markdown(
                                    f"_SoJenAI-Moderator has **not proposed a rewrite** at this severity level "
                                    f"(`severity = {m_severity}`). The model signal is too low to justify "
                                    "an automatic bias mitigation rewrite._"
                                )


# -----------------------------
# What This Demo Does NOT Show (NEW)
# -----------------------------
st.markdown("""
## **What This Demo Does *Not* Show**

This demonstration focuses solely on the SoJen.AI communication intelligence engine.  
It does **not** represent the full design or functionality of the patent-pending SoJen.AI social platform.

Specifically, this demo does **not** include:

- The SoJen.AI platform UI, community structure, or interaction flows  
- Age segmentation, onboarding, or wellness-based design logic  
- Internal ML training data, feature extraction, or model architecture  
- Multi-user safety layers or moderation pathways  
- Creator ecosystems, brand-safe environments, or sponsored rooms  
- The full social platform prototype or product experience  

This demo is intended for evaluation of the **engine capabilities only**, not the total patented architecture.
""")


# -----------------------------
# Copyright Footer
# -----------------------------
st.markdown("""
<hr style="margin-top:40px;">

<div style="text-align:center; font-size:13px; color: #666;">
© 2025 SoJen.AI — All Rights Reserved.<br>
This demonstration is part of a patent-pending system. Unauthorized sharing or distribution is prohibited.
</div>
""", unsafe_allow_html=True)

