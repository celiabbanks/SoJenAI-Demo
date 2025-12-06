========================================================
SOJENAI-DEMO — JenAI-Moderator
Portable API + Dashboard Investor Demo
========================================================

This folder contains a self-contained demonstration of the
JenAI-Moderator Communication Intelligence API and dashboard.

Contents:
• FastAPI backend (app/)
• Streamlit dashboard (.streamlit/)
• Inference engine (core/)
• Preloaded HuggingFace models (assets/models/)
• Branding assets (assets/images/)
• Launcher scripts and environment setup

========================================================
1. SETUP (FIRST TIME ONLY)
========================================================

STEP 1 — Open a terminal inside the SoJenAI-Demo folder

Windows:
    Shift + Right-click on folder → “Open PowerShell window here”

Mac/Linux:
    Right-click folder → “Open in Terminal”

STEP 2 — Create a virtual environment

Windows:
    python -m venv venv

Mac/Linux:
    python3 -m venv venv

STEP 3 — Activate the environment

Windows:
    venv\Scripts\activate

Mac/Linux:
    source venv/bin/activate

STEP 4 — Install demo dependencies

    pip install --upgrade pip
    pip install -r requirements.txt


========================================================
2. HOW TO RUN THE API (FASTAPI)
========================================================

Once the venv is active:

Windows:
    Double-click:  run_api.bat
    (or run in terminal:  run_api.bat)

Mac/Linux:
    chmod +x run_api.sh       (first time)
    ./run_api.sh

The API will start on:
    http://127.0.0.1:8010

Health check:
    http://127.0.0.1:8010/health

(Optionally) Smoke test:
    http://127.0.0.1:8010/smoke


========================================================
3. HOW TO RUN THE DASHBOARD (STREAMLIT UI)
========================================================

With the same venv active:

Windows:
    Double-click:  run_dashboard.bat

Mac/Linux:
    chmod +x run_dashboard.sh   (first time)
    ./run_dashboard.sh

By default, the dashboard opens at:
    http://localhost:8501/


========================================================
4. DEMO FLOW (FOR INVESTOR PRESENTATION)
========================================================

1. Start the API (run_api script).
2. Start the dashboard (run_dashboard script).
3. In the dashboard, paste example comments such as:

   • “I won’t get in an elevator alone with a black person.”
   • “Women should not be sports casters.”
   • “Women are bad drivers.”
   • “People from New York City are rude.”

4. Explain what the system shows:
   • Detected category (racial, sexist, bullying, etc.)
   • Severity (none / low / medium / high)
   • Whether mitigation is advisory, rewrite, or no-op
   • That JenAI-Moderator does NOT over-police low-signal messages

5. Show the mitigation panel:
   • Advisory explanations for high-severity harmful content
   • Constructive rewrites for moderate-risk content
   • “No rewrite at this severity level” messaging for weak/ambiguous cases

6. Open the “Why this was handled this way” expander (if implemented):
   • Show top model scores
   • Show lexicon influence
   • Emphasize explainability and transparency


========================================================
5. TROUBLESHOOTING
========================================================

• If models cannot be loaded:
      Ensure these folders exist and contain HuggingFace files:
          assets/models/distilbert_bias_finetuned/
          assets/models/roberta_sentiment/

• If the logo does not appear:
      Ensure:
          assets/images/JenAI-Moderator_CommIntell.png
      exists in the demo folder.

• If port 8010 is in use:
      Edit run_api.bat or run_api.sh and change:
          --port 8010
      to another free port, e.g. 8011

• If Streamlit fails to start:
      Confirm the venv is activated and requirements are installed.


========================================================
6. EXPECTED FOLDER STRUCTURE
========================================================

SoJenAI-Demo/
    app/
        main.py
        predictor_smoke.py
        inference.py
        config.py
        __init__.py
    core/
        models.py
        __init__.py
    .streamlit/
        dashboard.py
        config.toml       (optional)
        __init__.py
    assets/
        images/
            JenAI-Moderator_CommIntell.png
        models/
            distilbert_bias_finetuned/
                config.json
                pytorch_model.bin
                tokenizer.json
                tokenizer_config.json
                ...
            roberta_sentiment/
                config.json
                pytorch_model.bin
                merges.txt
                vocab.json
                tokenizer.json
                tokenizer_config.json
                ...
    requirements.txt
    run_api.bat
    run_dashboard.bat
    run_api.sh
    run_dashboard.sh
    install_demo.bat      (optional helper)
    install_demo.sh       (optional helper)
    README_RUN.txt
    .gitignore


========================================================
Demo Ready.
========================================================
