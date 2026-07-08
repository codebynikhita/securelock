# 🛡️ SecureLock — AI-Powered Fake & Clone Account Detection

<div align="center">

![SecureLock Banner](https://img.shields.io/badge/SecureLock-Account%20Intelligence-6366f1?style=for-the-badge&logo=shield&logoColor=white)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189ABE?style=flat-square)](https://xgboost.ai)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=flat-square&logo=render&logoColor=white)](https://render.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Live Demo → [securelock-1.onrender.com](https://securelock-1.onrender.com)**

</div>

---

## The Story Behind This Project

Honestly? This started as a university major project that kind of just... happened to me.

You know how group projects go — everyone's got ideas, there's a shared doc somewhere, a few meetings, some half-built code. By the end we had something that technically ran. 

But I kept thinking about it after submission. The *idea* was genuinely good — fake account detection is a real problem and the prototype we had felt like it was sitting at like 30% of what it could be. So I just kept working on it in my own time.

One thing led to another and I ended up:
- Swapping the static data with a **live scraper** that pulls real Instagram stats in real-time
- Retraining the models on **50,000 accounts** instead of the original 5,000
- Rebuilding the UI from scratch so it actually looks like something worth showing
- Adding an **admin dashboard** with real charts and detection logs
- Actually deploying it — not just `localhost:5000`

It sort of became my thing. Not complaining about it at all — just how it went.

---

## What It Does

SecureLock analyzes any social media account and tells you whether it's **Genuine**, **Fake**, or a **Clone** — using three independent AI classifiers that vote on the result.

**Enter a username → Live data scraped → AI verdict in seconds.**

### How the AI Works

| Classifier | Type | Role |
|---|---|---|
| **Random Forest** | Ensemble (Bagging) | High-recall fake detection |
| **XGBoost** | Ensemble (Boosting) | Precision clone detection |
| **K-Nearest Neighbors** | Distance-based | Anomaly spotting via neighbor similarity |

All three vote independently. The **combined risk score** is the maximum fake/clone probability across all classifiers — not just an average — which means if even one model is confident, you get warned.

---

## Features

- 🔴 **Live Data Scraping** — Googlebot User-Agent bypass fetches real-time followers, following, posts & account age from Instagram without any API key or login
- 🤖 **3-Classifier Ensemble** — RF + XGBoost + KNN trained on 50,000 labelled accounts, 99.8% test accuracy
- 👥 **Clone Detection** — Levenshtein distance + cosine similarity + Euclidean metrics to find impersonation accounts
- 📊 **Admin Dashboard** — Live charts, detection logs, platform breakdowns, flagged account management
- ✏️ **Manual Override** — Edit any metric and instantly re-analyze with your corrected values
- 📄 **PDF Reports** — One-click print/download of the full analysis report
- 🌐 **Multi-Platform** — Twitter, Instagram, Facebook, LinkedIn (live scraping works best on Instagram)
- ⚠️ **Smart Error Handling** — Shows clear "Account Not Found" instead of fake estimated data for non-existent accounts

---

## Tech Stack

```
Backend       Flask (Python)
ML Models     scikit-learn, XGBoost, NumPy, Pandas
Database      SQLite (via database.py)
Frontend      Vanilla HTML/CSS/JS — no frameworks, no bloat
Scraping      urllib + Googlebot User-Agent bypass (no Selenium)
Fonts         Inter, Space Grotesk, JetBrains Mono (Google Fonts)
Charts        Chart.js (admin dashboard)
Deployment    Render (Web Service), Gunicorn
```

---

## Project Structure

```
securelock/
├── app.py                  # Flask routes & scraping logic
├── model.py                # SecureLockModel — loads & runs all 3 classifiers
├── model_pipeline.py       # Training pipeline (50k dataset generation + training)
├── database.py             # SQLite detection logs & admin queries
├── scrape_live_instagram.py # Instagram Googlebot scraper
├── requirements.txt        # Python dependencies
├── Procfile                # Render/Gunicorn start command
├── render.yaml             # Render deployment config
├── static/
│   ├── css/style.css       # Full premium design system
│   └── js/main.js          # Platform tabs, suggestion chips, modal logic
└── templates/
    ├── base.html           # Sticky navbar, footer
    ├── index.html          # Landing page + results page
    ├── admin_dashboard.html # Analytics dashboard
    └── admin_login.html    # Admin auth
```

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/codebynikhita/securelock.git
cd securelock

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the models (first time only — generates models/ folder)
python model_pipeline.py

# 5. Run the app
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

> **Note on model files:** The trained `.pkl` model files are excluded from Git (too large). Run `python model_pipeline.py` once to generate them locally. On Render, the build command handles this automatically.

---

## Deploying to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and configures everything
5. Hit **Deploy** — live in ~3 minutes

The `render.yaml` in this repo already sets:
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app --workers 1 --timeout 120`

---

## Admin Dashboard

Go to `/admin/login` on the live site.

Default credentials (change in `app.py` before deploying):
```
Username: admin
Password: securelock2026
```

The dashboard shows:
- Total scans, fake accounts caught, clones detected
- Detection history chart (line graph)
- Classification breakdown (pie chart)
- Flagged accounts table with dismiss/verify actions

---

## Accuracy & Model Notes

| Metric | Value |
|---|---|
| Training set size | 50,000 synthetic accounts |
| Test accuracy (RF) | 99.8% |
| Test accuracy (XGBoost) | 99.9% |
| Test accuracy (KNN) | 98.7% |
| Features used | followers, following, posts, account_age, profile_picture, follower_ratio |

> **On live Instagram data:** Accuracy depends on what Instagram's public meta tags return. Some numbers may be rounded (e.g. "104M" → 104,000,000). The manual override lets you correct any metric before re-analyzing.

---

## Known Limitations

- Live scraping works best on **Instagram** — Twitter/Facebook block or redirect more aggressively
- Account age is **estimated** from Instagram's sequential user ID mapping — verify via "About this account" in the Instagram app for exact year
- Instagram may occasionally return cached/CDN numbers that differ slightly from the app display

---

## What I Learned

Building this alone taught me more than the group project ever did:
- How real scraping works (and why it's hard)
- Why ensemble models outperform single classifiers
- How to actually deploy something — not just run `python app.py` locally
- That finishing something imperfect is infinitely better than leaving something perfect unfinished

---

## License

MIT — use it, fork it, improve it. Just don't submit it as your own university project 😄

---

<div align="center">
Built solo by <strong>Nikhita G P</strong> &nbsp;|&nbsp; 2026 &nbsp;|&nbsp;
<a href="https://securelock-1.onrender.com">Live Demo</a>
</div>
