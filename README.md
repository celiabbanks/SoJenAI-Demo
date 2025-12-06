## README.md for SoJen.AI Demo
SoJen.AI™ — Communication Intelligence Demo
Bias Detection • Tone & Intent Understanding • Generative Mitigation

Patent-Pending System • Proprietary Technology

## Overview

SoJen.AI™ is a communication intelligence system designed to understand tone, intent, and emotional context in real time.
It detects bias, identifies escalation risk, and delivers supportive rewrites through its AI persona, JenAI-Moderator™.

This repository contains a demo environment showcasing:

A FastAPI backend for inference & mitigation

A Streamlit dashboard for interactive testing

Integration of machine learning & generative AI

The JenAI-Moderator™ Communication Intelligence response layer

Important:
This demo represents Layer 1 of the SoJen.AI offering (the commercial API).
The SoJen Social Platform, which is protected under a U.S. patent application, is not included in this repository.

## What SoJen.AI Does

SoJen.AI targets one of the most widespread problems today:
miscommunication that escalates conflict across workplaces, classrooms, social networks, and customer interactions.

The engine analyzes:

Bias categories (racial, sexist, political, bullying, etc.)

Severity (none → high)

Implicit vs explicit bias

Emotional cues (frustration, hostility, confusion, overwhelm)

Tone & intent mismatches

Then it generates:

Supportive advisory guidance

Clear, calm rewrites

Conflict-reducing alternatives

Context-aware suggestions

All delivered through the JenAI-Moderator™ persona, a core part of SoJen.AI’s brand identity and user experience.

## Architecture Overview
Backend (FastAPI)

/v1/infer → bias, intent & severity inference

/v1/mitigate → JenAI-Moderator rewrite generation

/health → model and device status

Dashboard (Streamlit)

Interactive web UI used for:

Entering sample text

Visualizing category scores

Examining model metadata

Triggering mitigation rewrites

Understanding system behavior

Machine Learning Engine

Uses a multi-model architecture:

DistilBERT classifier

RoBERTa sentiment & tone-layer

Custom severity + implicit/explicit modules

Lexicon enhancement layer

Generative mitigation (persona-based rewrite engine)

## Intellectual Property Notice

The SoJen.AI system, its algorithms, architecture, AI persona, and social platform design are:

Proprietary

Confidential

Protected under U.S. patent application

No portion of this work may be copied, redistributed, or used
without explicit written authorization from the creator, Celia Banks.

This repository contains only the demo interface, not the full patent-protected platform logic.

## How to Use the Demo

Enter any message into the text box.

Click Analyze with JenAI-Moderator to view:

Bias category probabilities

Severity

Implicit/explicit designation

Metadata

Click Run Rewrite to generate:

Advisory explanation

Constructive rewrite

Severity badge

Try various tones:

frustrated

accusatory

confused

overwhelmed

dismissive

biased

This helps illustrate the engine behind the future SoJen Social Platform™.

## Demo Deployment

This repository is intended for deployment on:

Render / Railway (FastAPI backend)

Streamlit Cloud (dashboard interface)

The dashboard uses:

SOJEN_API_BASE="https://your-fastapi-service-url"

to connect to the live backend.

## Repository Structure
SoJenAI-Demo/

| Path                                  | Description                                   |
|---------------------------------------|-----------------------------------------------|
| `app/`                                | FastAPI backend service                       |
| `app/main.py`                         | API entrypoint (inference + mitigation)       |
| `app/inference.py`                    | ML inference workflow                         |
| `app/predictor_smoke.py`              | Smoke-test routes                             |
| `app/config.py`                       | Settings and configuration                    |
| `assets/images/JenAI-Moderator_CommIntell.png` | Brand persona asset                   |
| `dashboard.py`                        | Streamlit demo interface                      |
| `requirements.txt`                    | Python dependencies                           |
| `.gitignore`                          | Large file & env exclusions                   |
| `README.md`                           | Project documentation                         |

Note: Large model files (*.safetensors, *.bin, *.pt) are intentionally excluded from version control.

## Use Cases
Enterprise

HR compliance

Internal communications

Customer support tone intelligence

Education

Classroom safety

Student well-being dashboards

Tone-aware conflict prevention

Social Platforms

Community moderation

De-escalation

Bias mitigation

## Supportive guidance
(Full platform covered by patent, not included here)

## Technology Stack

Python 3.12+

FastAPI

Streamlit

Transformers (Hugging Face)

PyTorch

Pydantic

Uvicorn

Pillow

Requests

Pandas


## License
© 2025 SoJen.AI — All Rights Reserved.
This repository contains proprietary and confidential materials.
Unauthorized copying, distribution, or use is prohibited.

Celia Banks
Founder & Developer — SoJen.AI™
University of Michigan School of Information MADS Program Alumni

For collaboration or evaluation inquiries, contact privately.
