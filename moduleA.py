"""
Module A — URL Heuristic Risk Scorer (Rule-Based)
===================================================
Scores a URL based on 12 risk checks — no ML model, no training required.

Checks:
  1. HTTPS vs HTTP
  2. IP address as domain (private vs public)
  3. Subdomain count
  4. Suspicious TLD
  5. URL length
  6. Special character ratio
  7. @ symbol (URL confusion)
  8. Non-standard port
  9. URL shortener domain
  10. Login/brand keywords in domain
  11. Phishing bait keywords in URL
  12. Excessive % encoding

Usage:
    from moduleA import module_a_predict
    result = module_a_predict("http://192.168.1.100/paypal/login")
"""

import re
from urllib.parse import urlparse
from typing import Dict

SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "work", "buzz", "online", "icu", "monster",
    "cfd", "surf", "tk", "ml", "ga", "cf", "gq", "pw", "cc", "ws",
    "biz", "info", "site", "tech", "support",
}

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "buff.ly",
    "ow.ly", "rb.gy", "cutt.ly", "shorturl.at", "lnkd.in",
}

BANKING_KEYWORDS = [
    "bank", "banking", "chase", "wellsfargo", "bankofamerica", "citi",
    "discover", "capitalone", "tdbank", "pnc", "usbank", "hsbc",
    "barclays", "lloyds", "natwest", "santander", "revolut",
]

PAYMENT_KEYWORDS = [
    "paypal", "venmo", "cashapp", "stripe", "square", "wise",
    "westernunion", "moneygram", "zelle", "applepay", "googlepay",
]

CRYPTO_KEYWORDS = [
    "crypto", "bitcoin", "wallet", "metamask", "coinbase", "binance",
    "kraken", "ledger", "trezor", "defi", "uniswap", "opensea",
]

LOGINTARGET_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update",
    "confirm", "password", "credential", "auth", "sso", "oauth",
]

PHISHING_KEYWORDS = [
    "free", "winner", "claim", "prize", "lottery", "urgent",
    "suspended", "limited", "alert", "unusual", "activity",
    "restore", "unlock", "verify-now", "click-here",
]


def _domain_keyword_score(domain: str, keywords: list) -> float:
    dl = domain.lower().replace("www.", "")
    matched = [kw for kw in keywords if kw in dl]
    if not matched:
        return 0.0
    return min(len(matched) * 0.2, 1.0)


def _check_ip_address(domain: str) -> float:
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain):
        octets = domain.split(".")
        if octets[0] == "10" or (octets[0] == "172" and 16 <= int(octets[1]) <= 31) or (octets[0] == "192" and octets[1] == "168"):
            return 1.0
        return 0.8
    if re.match(r"^\[([0-9a-f:]+)\]$", domain):
        return 1.0
    return 0.0


def _tld_risk_score(tld: str) -> float:
    t = tld.lower().split(".")[-1]
    if t in SUSPICIOUS_TLDS:
        return 0.8
    if t in {"gov", "edu", "mil"}:
        return 0.0
    if t in {"com", "org", "net"}:
        return 0.1
    return 0.3


def module_a_predict(url: str) -> Dict:
    parsed = urlparse(url)
    domain = (parsed.hostname or "").lower().replace("www.", "")
    full = url
    ext = __import__("tldextract").extract(url)
    tld = ext.suffix
    subdomain = ext.subdomain
    score = 0.0
    flags = []

    # 1. HTTPS
    if parsed.scheme == "http":
        score += 0.10
        flags.append("No HTTPS")

    # 2. IP address as domain
    ip_risk = _check_ip_address(domain)
    score += ip_risk * 0.35
    if ip_risk > 0:
        flags.append("IP address as domain")

    # 3. Subdomains (more = riskier)
    sub_count = subdomain.count(".") + (1 if subdomain else 0)
    if sub_count >= 3:
        score += 0.15
        flags.append(f"Excessive subdomains ({sub_count})")
    elif sub_count >= 2:
        score += 0.08

    # 4. TLD risk
    tld_risk = _tld_risk_score(tld)
    score += tld_risk * 0.10
    if tld_risk >= 0.6:
        flags.append(f"Suspicious TLD (.{tld})")

    # 5. URL length
    url_len = len(full)
    if url_len > 100:
        score += 0.10
        flags.append("Very long URL")
    elif url_len > 75:
        score += 0.05

    # 6. Special character ratio
    letters = sum(c.isalpha() for c in full)
    digits = sum(c.isdigit() for c in full)
    special = len(full) - letters - digits
    sp_ratio = special / max(len(full), 1)
    if sp_ratio > 0.35:
        score += 0.12
        flags.append("High special character ratio")

    # 7. @ symbol (URL confusion)
    if "@" in full:
        score += 0.20
        flags.append("Contains @ symbol (URL confusion)")

    # 8. Port in URL
    if parsed.port and parsed.port not in (80, 443):
        score += 0.10
        flags.append(f"Non-standard port ({parsed.port})")

    # 9. Shortened URL
    if domain in SHORTENER_DOMAINS:
        score += 0.15
        flags.append(f"URL shortener ({domain})")

    # 10. Domain-only risk keywords
    score += _domain_keyword_score(domain, LOGINTARGET_KEYWORDS) * 0.15
    login_matched = [kw for kw in LOGINTARGET_KEYWORDS if kw in domain]
    if login_matched:
        flags.append(f"Login keyword in domain ({','.join(login_matched[:2])})")

    score += _domain_keyword_score(domain, BANKING_KEYWORDS) * 0.12
    score += _domain_keyword_score(domain, PAYMENT_KEYWORDS) * 0.12
    score += _domain_keyword_score(domain, CRYPTO_KEYWORDS) * 0.10

    # 11. Phishing bait keywords in full URL
    phish_matched = [kw for kw in PHISHING_KEYWORDS if kw in full.lower()]
    if phish_matched:
        score += min(len(phish_matched) * 0.12, 0.30)
        flags.append(f"Phishing bait keywords ({','.join(phish_matched[:3])})")

    # 12. % encoding (obfuscation)
    pct_count = full.count("%")
    if pct_count > 5:
        score += 0.12
        flags.append("Excessive % encoding")

    # Clamp
    score = min(score, 1.0)

    return {
        "available": True,
        "score": round(score, 4),
        "flags": flags,
    }
