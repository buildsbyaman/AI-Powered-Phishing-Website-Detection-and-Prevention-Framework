"""
Module C — Real-Time Reputation Check
=======================================
Queries URLhaus and VirusTotal for a given URL and returns a
reputation-based phishing signal. Unlike Modules A and B, this module is not
a trained classifier — it's an API-lookup component, so there is nothing to
"train." What it does provide is an optional evaluation mode:
given a labeled CSV of URLs, it reports how accurate the reputation signal
alone is — useful for the ablation study.

Setup:
    - URLhaus API key: sign up at https://urlhaus.abuse.ch/api/
    - VirusTotal API key: get one at https://www.virustotal.com/
    - Store keys as environment variables (never hardcode secrets):
        export URLHAUS_API_KEY="..."
        export VT_API_KEY="..."

Usage as a library:
    from moduleC import get_reputation_verdict
    verdict = get_reputation_verdict("http://example-suspicious-site.com")
    # -> {"flagged": 1, "sources": ["urlhaus"]}

Usage for standalone evaluation against a labeled test set:
    python moduleC.py --evaluate Datasets/urls_labeled.csv
    (CSV requires columns: url, label   where label: 1 = legitimate, 0 = phishing)
"""

import argparse
import base64
import json
import os
import time
import warnings
from typing import Optional
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*urllib3.*")

import pandas as pd
import requests

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("[!] matplotlib not installed — skipping graphs. "
          "Install with: pip install matplotlib")

URLHAUS_ENDPOINT = "https://urlhaus-api.abuse.ch/v1/url/"
VT_ENDPOINT = "https://www.virustotal.com/api/v3/urls"
TIMEOUT_SECONDS = 4
CACHE_FILE = "data/processed/module_c_cache.json"


# --------------------------------------------------------------------------- #
# 1. Caching (avoids duplicate API calls, which is important given rate limits)
# --------------------------------------------------------------------------- #

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


_CACHE = _load_cache()


# --------------------------------------------------------------------------- #
# 2. Individual reputation sources
# --------------------------------------------------------------------------- #

def check_urlhaus(url: str, api_key: Optional[str] = None) -> Optional[int]:
    """
    Returns 1 if URL is found in URLhaus (abuse.ch), 0 if not found,
    None if the check could not be completed.
    """
    api_key = api_key or os.environ.get("URLHAUS_API_KEY", "")
    if not api_key:
        return None
    try:
        resp = requests.post(URLHAUS_ENDPOINT, data={"url": url},
                              headers={"Auth-Key": api_key}, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        status = resp.json().get("query_status", "")
        if status in ("online", "offline"):
            return 1
        return 0
    except Exception:
        return None


def check_virustotal(url: str, api_key: Optional[str] = None) -> Optional[int]:
    """
    Returns 1 if VirusTotal flags the URL as malicious, 0 if clean,
    None if the check could not be completed.
    """
    api_key = api_key or os.environ.get("VT_API_KEY", "")
    if not api_key:
        return None
    try:
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        resp = requests.get(f"{VT_ENDPOINT}/{url_id}",
                            headers={"x-apikey": api_key}, timeout=TIMEOUT_SECONDS)
        if resp.status_code == 404:
            return 0
        resp.raise_for_status()
        stats = resp.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
        malicious = stats.get("malicious", 0) + stats.get("suspicious", 0)
        return 1 if malicious > 0 else 0
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 3. Combined verdict (with graceful degradation)
# --------------------------------------------------------------------------- #

def get_reputation_verdict(url: str, uh_key: Optional[str] = None,
                            vt_key: Optional[str] = None,
                            use_cache: bool = True) -> dict:
    """
    Combines URLhaus and VirusTotal into a single reputation signal.
    If both sources are unreachable, returns flagged=None so the fusion layer
    knows to fall back to Modules A and B alone.
    """
    if use_cache and url in _CACHE:
        return _CACHE[url]

    uh_result = check_urlhaus(url, uh_key)
    vt_result = check_virustotal(url, vt_key)

    sources = [("urlhaus", uh_result), ("virustotal", vt_result)]

    sources_flagged = [name for name, r in sources if r == 1]
    checked_sources = [name for name, r in sources if r is not None]

    if not checked_sources:
        verdict = {"flagged": None, "sources": [], "note": "no reputation source reachable"}
    else:
        verdict = {"flagged": 1 if sources_flagged else 0, "sources": sources_flagged}

    if use_cache:
        _CACHE[url] = verdict
        _save_cache(_CACHE)

    return verdict


# --------------------------------------------------------------------------- #
# 4. Standalone evaluation mode (optional — for the Module C ablation results)
# --------------------------------------------------------------------------- #

def evaluate_reputation_module(csv_path: str, out_dir: str = "outputs", delay: float = 0.3):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )

    df = pd.read_csv(csv_path)
    y_true, y_pred = [], []
    unreachable = 0

    for i, row in df.iterrows():
        verdict = get_reputation_verdict(row["url"])
        if verdict["flagged"] is None:
            unreachable += 1
            continue
        predicted_label = 0 if verdict["flagged"] == 1 else 1
        y_true.append(row["label"])
        y_pred.append(predicted_label)

        if (i + 1) % 50 == 0:
            print(f"[+] Checked {i + 1}/{len(df)} URLs ({unreachable} unreachable so far)")
        time.sleep(delay)

    if not y_true:
        print("[!] No URLs could be checked — reputation APIs were unreachable "
              "or no API keys are configured. Skipping metrics.")
        return

    print(f"\n[+] Reputation module checked {len(y_true)}/{len(df)} URLs "
          f"({unreachable} unreachable/uncovered)")

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "Coverage": len(y_true) / len(df),
    }

    print("\n" + "=" * 60 + "\nMODULE C — REPUTATION-ONLY METRICS\n" + "=" * 60)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "module_c_metrics.csv"), index=False)
    print(f"\n[+] Saved metrics -> {os.path.join(out_dir, 'module_c_metrics.csv')}")

    if PLOTTING_AVAILABLE and len(set(y_pred)) > 1:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.title("Module C — Reputation-Only Confusion Matrix")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.xticks([0, 1], ["Phishing", "Legitimate"])
        plt.yticks([0, 1], ["Phishing", "Legitimate"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout()
        path = os.path.join(out_dir, "module_c_confusion_matrix.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[+] Saved confusion matrix -> {path}")
    elif PLOTTING_AVAILABLE:
        print("[!] Skipping confusion matrix plot — reputation module only ever "
              "predicted a single class on this sample (not informative as a graph).")


# --------------------------------------------------------------------------- #
# 5. CLI entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Module C reputation checker / evaluator")
    parser.add_argument("--evaluate", type=str, default=None,
                         help="Path to a labeled CSV (url, label) to evaluate reputation-only accuracy")
    parser.add_argument("--url", type=str, default=None,
                         help="Check a single URL and print its reputation verdict")
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    if args.url:
        print(json.dumps(get_reputation_verdict(args.url), indent=2))
    elif args.evaluate:
        evaluate_reputation_module(args.evaluate, out_dir=args.out_dir)
    else:
        print("Provide --url <url> for a single check or --evaluate <csv> for batch evaluation.")


if __name__ == "__main__":
    main()
