

"""
matching.py
-----------
Fuzzy address matching between Aadhaar QR-extracted address and a database.

Handles edge cases like:
- Wing A vs Wing B (same building, different flat)
- Same pincode/landmark but different house number
"""

import re
import sqlite3
import pandas as pd
from difflib import SequenceMatcher


# ==============================================================================
# TEXT NORMALISATION
# ==============================================================================

def normalise(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[,./\\|;:\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_query_string(address_parts: dict) -> str:
    ORDER = [
        "House", "Street", "Landmark", "Locality",
        "Village/Town/City", "Post Office", "Sub District",
        "District", "State", "Pincode",
    ]
    parts = [str(address_parts.get(f, "")).strip() for f in ORDER]
    parts = [p for p in parts if p]
    return normalise(" ".join(parts))


# ==============================================================================
# SIMILARITY HELPERS
# ==============================================================================

def token_overlap_score(a: str, b: str) -> float:
    """Jaccard token overlap — good for partial/reordered addresses."""
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def sequence_ratio(a: str, b: str) -> float:
    """SequenceMatcher ratio — good for near-identical strings."""
    return SequenceMatcher(None, a, b).ratio()


def combined_score(a: str, b: str) -> float:
    """Weighted combination of token overlap and sequence ratio."""
    return 0.6 * token_overlap_score(a, b) + 0.4 * sequence_ratio(a, b)


# ==============================================================================
# KEY FIELD BONUS
# Key fields that must match for a valid address — includes House now
# to handle Wing A vs Wing B / Flat 101 vs Flat 202 cases
# ==============================================================================

KEY_FIELDS = [
    "House",              # ← CRITICAL: Wing/Flat number
    "State",
    "District",
    "Pincode",
    "Village/Town/City",
    "Post Office",
]

def key_field_bonus(address_parts: dict, db_address_raw: str) -> float:
    """
    +5% bonus for each key field that appears verbatim in DB address.
    House field gets double bonus since it's the most distinguishing field.
    """
    db_norm = normalise(db_address_raw)
    bonus   = 0.0
    for field in KEY_FIELDS:
        val = normalise(str(address_parts.get(field, "")))
        if val and val in db_norm:
            if field == "House":
                bonus += 0.10   # double bonus for house match
            else:
                bonus += 0.05
    return min(bonus, 0.30)     # cap at 30%


# ==============================================================================
# HOUSE NUMBER PENALTY
# Handles Wing A vs Wing B — same building, different unit
# ==============================================================================

def house_penalty(address_parts: dict, db_address_raw: str) -> float:
    """
    If extracted house number exists but does NOT appear in DB address,
    apply a 60% penalty. This prevents Wing B matching Wing A records.
    """
    house_val = normalise(str(address_parts.get("House", "")))
    if not house_val:
        return 1.0   # no house info → no penalty, return multiplier of 1
    db_norm = normalise(db_address_raw)
    if house_val in db_norm:
        return 1.0   # house matches → no penalty
    return 0.4       # house doesn't match → 60% penalty


# ==============================================================================
# DATABASE LOADER
# ==============================================================================

def load_database(db_path: str = None,
                  csv_path: str = None,
                  df: pd.DataFrame = None) -> pd.DataFrame:
    if df is not None:
        return df.copy()

    if csv_path:
        return pd.read_csv(csv_path)

    if db_path:
        conn = sqlite3.connect(db_path)
        # Try all known table names
        for tbl in ("residents", "aadhaar", "users", "records", "data"):
            try:
                frame = pd.read_sql(f"SELECT * FROM {tbl}", conn)
                conn.close()
                return frame
            except Exception:
                pass
        # Last resort: auto-detect table
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            if tables:
                frame = pd.read_sql(f"SELECT * FROM {tables[0]}", conn)
                conn.close()
                return frame
        except Exception:
            pass
        conn.close()
        raise ValueError(f"Could not find any table in database: {db_path}")

    raise ValueError("Supply one of: db_path, csv_path, or df.")


# ==============================================================================
# MAIN MATCHING FUNCTION
# ==============================================================================

def match_address(
    address_parts: dict,
    db_path:        str             = None,
    csv_path:       str             = None,
    df:             pd.DataFrame    = None,
    threshold:      float           = 0.40,
    address_col:    str             = "Address",
) -> dict:
    """
    Match extracted Aadhaar address against every row in the database.

    Handles:
    - Fuzzy full-address matching
    - Key field bonuses (State, District, Pincode, House)
    - House/Wing number penalty to avoid Wing A matching Wing B

    Returns dict with: status, score, matched_row, all_scores, reason
    """
    database = load_database(db_path=db_path, csv_path=csv_path, df=df)
    query    = build_query_string(address_parts)

    if not query:
        return {
            "status":      "Invalid Address",
            "score":       0.0,
            "matched_row": None,
            "all_scores":  [],
            "reason":      "Extracted address is empty.",
        }

    results = []
    for _, row in database.iterrows():
        db_addr_raw = str(row.get(address_col, ""))
        if db_addr_raw.lower() in ("empty", "nan", "", "none"):
            results.append((row, 0.0))
            continue

        db_addr_norm = normalise(db_addr_raw)

        # Base fuzzy score
        score = combined_score(query, db_addr_norm)

        # Key field bonus
        score += key_field_bonus(address_parts, db_addr_raw)

        # House number penalty (Wing A vs Wing B fix)
        score *= house_penalty(address_parts, db_addr_raw)

        score = min(score, 1.0)
        results.append((row, score))

    results.sort(key=lambda x: x[1], reverse=True)

    best_row, best_score = results[0] if results else (None, 0.0)

    all_scores = [
        {
            "id":    int(r.get("id", i)),
            "name":  str(r.get("full_name", "")),
            "score": round(s, 4),
        }
        for i, (r, s) in enumerate(results)
    ]

    return {
        "status":      "Valid Address" if best_score >= threshold else "Invalid Address",
        "score":       round(best_score, 4),
        "matched_row": best_row.to_dict() if best_row is not None else None,
        "all_scores":  all_scores,
        "reason":      None,
    }


