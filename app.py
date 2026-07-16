"""
Flask Web Application — AI-Powered Phishing Website Detection and Prevention Framework
========================================================================================
Integrates all three detection modules:
  Module A: URL heuristic-based risk scorer (rule-based, no ML)
  Module B: Content/NLP-based classifier
  Module C: Real-time reputation API check
"""

import os
import time
import warnings
from urllib.parse import urlparse

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*urllib3.*")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from scipy.sparse import csr_matrix, hstack

from ContentExtraction import fetch_html, extract_structural_features, extract_visible_text
from moduleA import module_a_predict

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# Load trained models at startup (Module B only)
# --------------------------------------------------------------------------- #

MODELS_DIR = "models"

# Module B
b_vectorizer = joblib.load(os.path.join(MODELS_DIR, "module_b_tfidf_vectorizer.pkl"))
b_scaler = joblib.load(os.path.join(MODELS_DIR, "module_b_scaler.pkl"))
b_logreg = joblib.load(os.path.join(MODELS_DIR, "module_b_logistic_regression.pkl"))
b_svm = joblib.load(os.path.join(MODELS_DIR, "module_b_linear_svm.pkl"))

STRUCTURAL_FEATURE_COLS = [
    "num_forms", "has_password_field", "num_iframes", "num_scripts",
    "num_links", "external_form_action", "title_brand_mismatch",
    "favicon_mismatch", "has_meta_refresh", "right_click_disabled",
]


# --------------------------------------------------------------------------- #
# Module B — Content/NLP Prediction
# --------------------------------------------------------------------------- #

def module_b_predict(url: str) -> dict:
    html = fetch_html(url, cache_dir="data/processed")
    if html is None or len(html.strip()) == 0:
        return {"available": False, "score": 0.0, "flags": [], "note": "Page unreachable"}

    struct_feats = extract_structural_features(html, url)
    text = extract_visible_text(html)

    df = pd.DataFrame([{**struct_feats, "text": text}])
    df["text"] = df["text"].fillna("")

    text_features = b_vectorizer.transform(df["text"])
    struct = df[STRUCTURAL_FEATURE_COLS].fillna(0).values
    struct_scaled = b_scaler.transform(struct)
    X = hstack([text_features, csr_matrix(struct_scaled)])

    prob_logreg = float(b_logreg.predict_proba(X)[:, 1][0])
    prob_svm = float(b_svm.predict_proba(X)[:, 1][0])
    avg = (prob_logreg + prob_svm) / 2.0

    phishing_score = 1.0 - avg
    flags = []
    if struct_feats["has_password_field"]:
        flags.append("Password field detected")
    if struct_feats["num_iframes"] > 0:
        flags.append(f"{struct_feats['num_iframes']} iframe(s)")
    if struct_feats["external_form_action"]:
        flags.append("External form action")
    if struct_feats["title_brand_mismatch"]:
        flags.append("Brand mismatch")
    if struct_feats["favicon_mismatch"]:
        flags.append("Favicon mismatch")
    if struct_feats["has_meta_refresh"]:
        flags.append("Meta refresh redirect")

    return {
        "available": True,
        "score": round(phishing_score, 4),
        "flags": flags,
    }


# --------------------------------------------------------------------------- #
# Module C — Reputation Check
# --------------------------------------------------------------------------- #

def module_c_predict(url: str) -> dict:
    from moduleC import get_reputation_verdict
    uh_key = os.environ.get("URLHAUS_API_KEY")
    vt_key = os.environ.get("VT_API_KEY")
    verdict = get_reputation_verdict(url, uh_key=uh_key, vt_key=vt_key)
    flagged = verdict.get("flagged")
    sources = verdict.get("sources", [])

    if flagged is None:
        return {"available": False, "score": 0.0, "flags": [], "note": "No reputation source reachable"}

    phishing_score = 1.0 if flagged == 1 else 0.0
    flags = []
    if flagged == 1:
        flags.extend([f"Flagged by {s}" for s in sources])
    else:
        flags.append("Clean across all sources")

    return {
        "available": True,
        "score": round(phishing_score, 4),
        "flags": flags,
    }


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #

KNOWN_LEGITIMATE_DOMAINS = {
    "google.com", "facebook.com", "amazon.com", "microsoft.com", "apple.com",
    "netflix.com", "paypal.com", "github.com", "linkedin.com", "twitter.com",
    "instagram.com", "youtube.com", "wikipedia.org", "yahoo.com", "reddit.com",
    "tiktok.com", "whatsapp.com", "zoom.us", "dropbox.com", "slack.com",
    "openai.com", "anthropic.com", "bing.com", "cloudflare.com", "adobe.com",
    "spotify.com", "tesla.com", "nvidia.com", "intel.com", "ibm.com",
    "oracle.com", "salesforce.com", "uber.com", "airbnb.com", "stripe.com",
    "shopify.com", "wordpress.com", "medium.com", "quora.com", "ebay.com",
    "walmart.com", "target.com", "bestbuy.com", "costco.com", "homedepot.com",
    "chase.com", "wellsfargo.com", "bankofamerica.com", "citibank.com",
    "hsbc.com", "barclays.com", "bbc.com", "cnn.com", "nytimes.com",
    "reuters.com", "bloomberg.com", "forbes.com", "wired.com", "arstechnica.com",
    "stackoverflow.com", "npmjs.com", "pypi.org", "docker.com", "aws.amazon.com",
    "cloud.google.com", "azure.microsoft.com", "digitalocean.com", "heroku.com",
    "vercel.com", "netlify.com", "fastly.com", "akamai.com",
}


def _is_known_legitimate(domain: str) -> bool:
    dl = domain.lower().replace("www.", "")
    return dl in KNOWN_LEGITIMATE_DOMAINS or any(dl.endswith("." + d) for d in KNOWN_LEGITIMATE_DOMAINS)


def fuse_predictions(mod_a: dict, mod_b: dict, mod_c: dict, url: str = "") -> dict:
    w_a, w_b, w_c = 0.05, 0.60, 0.35

    s_a = mod_a["score"] if mod_a["available"] else 0.5
    s_b = mod_b["score"] if mod_b["available"] else 0.5
    s_c = mod_c["score"] if mod_c["available"] else 0.5

    # Known legitimate domain override: if domain is well-known and reputation is clean,
    # force score low regardless of Module B false positives or C availability
    if url:
        domain = (urlparse(url).hostname or "").replace("www.", "")
        if _is_known_legitimate(domain) and s_a < 0.15:
            return {"verdict": "LEGITIMATE", "confidence": 0.99}

    if mod_b.get("available", True):
        # All modules available: standard weighted average
        final = s_a * w_a + s_b * w_b + s_c * w_c
    elif s_a >= 0.35:
        # Page unreachable AND Module A says phishing → trust URL heuristics
        final = s_a * 0.85 + s_c * 0.15
    elif s_c >= 0.5:
        # Page unreachable AND Module C says flagged → trust reputation
        final = s_c * 0.70 + s_a * 0.30
    else:
        # Page unreachable but no strong signal → lean legitimate
        final = s_a * 0.20 + s_c * 0.80

    confidence = abs(final - 0.5) * 2

    if final >= 0.6:
        verdict = "PHISHING"
    elif final >= 0.35:
        verdict = "SUSPICIOUS"
    else:
        verdict = "LEGITIMATE"

    return {"verdict": verdict, "confidence": round(confidence, 4)}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "URL must include a protocol (https:// or http://)"}), 400

    t0 = time.time()

    mod_a, mod_b, mod_c = {}, {}, {}
    try:
        mod_a = module_a_predict(url)
    except Exception as e:
        mod_a = {"available": False, "score": 0.0, "flags": [], "note": str(e)}

    try:
        mod_b = module_b_predict(url)
    except Exception as e:
        mod_b = {"available": False, "score": 0.0, "flags": [], "note": str(e)}

    try:
        mod_c = module_c_predict(url)
    except Exception as e:
        mod_c = {"available": False, "score": 0.0, "flags": [], "note": str(e)}

    fusion = fuse_predictions(mod_a, mod_b, mod_c, url=url)

    return jsonify({
        "module_a": mod_a,
        "module_b": mod_b,
        "module_c": mod_c,
        "fusion": fusion,
        "elapsed_seconds": round(time.time() - t0, 2),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
