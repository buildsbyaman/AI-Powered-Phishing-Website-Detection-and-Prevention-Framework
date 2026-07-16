"""
Module B — Content Extraction Utilities
=========================================
Fetches live webpage HTML for a given list of URLs and extracts:
  1. Structural/DOM features (forms, iframes, password fields, brand mismatch, etc.)
  2. Visible page text (for TF-IDF vectorization downstream in train_module_b.py)

Results are cached to disk (data/processed/) so repeated experimentation doesn't
re-scrape the same URLs — scraping is the slowest and least reliable part of this
pipeline, so caching matters more here than anywhere else in the project.

Usage (as a library, called from train_module_b.py):
    from content_extraction import build_content_dataset
    df = build_content_dataset("data/raw/urls_labeled.csv", cache_dir="data/processed")

Expected input CSV: two columns -> url, label   (label: 1 = legitimate, 0 = phishing)
"""

import os
import re
import time
import hashlib
import warnings
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*urllib3.*")

import pandas as pd
import requests
import tldextract
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TIMEOUT_SECONDS = 6

# A small reference list of well-known brand names used for the
# brand-impersonation heuristic below. Expand this list for your project.
KNOWN_BRANDS = {
    "paypal": "paypal.com", "microsoft": "microsoft.com", "apple": "apple.com",
    "amazon": "amazon.com", "google": "google.com", "facebook": "facebook.com",
    "netflix": "netflix.com", "instagram": "instagram.com", "bank of america": "bankofamerica.com",
    "chase": "chase.com", "wells fargo": "wellsfargo.com", "linkedin": "linkedin.com",
}


def _cache_path(url: str, cache_dir: str) -> str:
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, "html_cache", f"{url_hash}.html")


def fetch_html(url: str, cache_dir: str = "data/processed") -> Optional[str]:
    """Fetch raw HTML for a URL, using a disk cache to avoid re-fetching."""
    cache_file = _cache_path(url, cache_dir)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        html = resp.text
        with open(cache_file, "w", encoding="utf-8", errors="ignore") as f:
            f.write(html)
        return html
    except Exception:
        return None  # unreachable page — handled by caller (treated as a signal, not a crash)


def _external_form_action(soup: BeautifulSoup, url: str) -> int:
    """Flag if any <form> submits to a different domain than the page itself."""
    page_domain = tldextract.extract(url).registered_domain
    for form in soup.find_all("form"):
        action = form.get("action", "")
        if action.startswith("http"):
            action_domain = tldextract.extract(action).registered_domain
            if action_domain and action_domain != page_domain:
                return 1
    return 0


def _brand_title_mismatch(soup: BeautifulSoup, url: str) -> int:
    """Flag if the page title/text claims a known brand but the domain doesn't match it."""
    page_domain = tldextract.extract(url).registered_domain
    title = soup.title.get_text().lower() if soup.title else ""
    for brand, real_domain in KNOWN_BRANDS.items():
        if brand in title and page_domain != real_domain:
            return 1
    return 0


def _favicon_mismatch(soup: BeautifulSoup, url: str) -> int:
    """Flag if the favicon is hosted on a different domain than the page."""
    page_domain = tldextract.extract(url).registered_domain
    icon = soup.find("link", rel=lambda v: v and "icon" in v.lower())
    if icon and icon.get("href", "").startswith("http"):
        icon_domain = tldextract.extract(icon["href"]).registered_domain
        if icon_domain and icon_domain != page_domain:
            return 1
    return 0


def extract_structural_features(html: str, url: str) -> dict:
    """Extract DOM/structural features from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")

    features = {
        "num_forms": len(soup.find_all("form")),
        "has_password_field": int(bool(soup.find("input", {"type": "password"}))),
        "num_iframes": len(soup.find_all("iframe")),
        "num_scripts": len(soup.find_all("script")),
        "num_links": len(soup.find_all("a")),
        "external_form_action": _external_form_action(soup, url),
        "title_brand_mismatch": _brand_title_mismatch(soup, url),
        "favicon_mismatch": _favicon_mismatch(soup, url),
        "has_meta_refresh": int(bool(soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)}))),
        "right_click_disabled": int("event.button==2" in html or "contextmenu" in html.lower()),
    }
    return features


def extract_visible_text(html: str) -> str:
    """Extract visible page text for TF-IDF vectorization."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _process_url(url, label, cache_dir):
    html = fetch_html(url, cache_dir)
    if html is None or len(html.strip()) == 0:
        return None
    try:
        struct_feats = extract_structural_features(html, url)
        text = extract_visible_text(html)
    except Exception:
        return None
    struct_feats["text"] = text
    struct_feats["url"] = url
    struct_feats["label"] = label
    return struct_feats


def build_content_dataset(url_label_csv: str, cache_dir: str = "data/processed",
                           delay: float = 0.5, workers: int = 10) -> pd.DataFrame:
    """
    Scrape (or load from cache) each URL in url_label_csv and return a DataFrame
    with structural features + visible text + label, ready for train_module_b.py.
    Unreachable URLs are dropped and reported, not silently ignored.
    """
    processed_path = os.path.join(cache_dir, "module_b_dataset.csv")
    if os.path.exists(processed_path):
        print(f"[+] Found existing processed dataset -> {processed_path} (skipping re-scrape)")
        return pd.read_csv(processed_path)

    df = pd.read_csv(url_label_csv)
    rows = []
    unreachable = 0
    total = len(df)

    print(f"[+] Scraping {total} URLs with {workers} workers...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_url, row["url"], row["label"], cache_dir): i
            for i, row in df.iterrows()
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is None:
                unreachable += 1
            else:
                rows.append(result)
            if i % 50 == 0:
                print(f"[+] Processed {i}/{total} URLs ({unreachable} unreachable so far)")

    result_df = pd.DataFrame(rows)
    os.makedirs(cache_dir, exist_ok=True)
    result_df.to_csv(processed_path, index=False)

    print(f"\n[+] Finished scraping: {len(result_df)} usable pages, "
          f"{unreachable} unreachable/skipped")
    print(f"[+] Saved processed dataset -> {processed_path}")

    return result_df