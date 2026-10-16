"""Plate text normalisation for comparison (Arabic + Latin, Eastern/Western Arabic numerals).

Only used for scoring: raw model output is stored unchanged.
"""

from __future__ import annotations

import re
import unicodedata

_DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}        # Arabic-Indic U+0660..
_DIGITS.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})  # Extended (Persian/Urdu) U+06F0..
_REMOVE_RX = re.compile(r"[\s\-_.,:;/\\|·•ـً-ٰٟ‌-‏‪-‮⁦-⁩]")

DEFAULT_FOLD = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ي": "ى", "ة": "ه"}

# Official Saudi plate letter pairs (Arabic -> Latin).
KSA_AR_TO_LATIN = {
    "ا": "A", "ب": "B", "ح": "J", "د": "D", "ر": "R", "س": "S", "ص": "X", "ط": "T", "ع": "E",
    "ق": "G", "ك": "K", "ل": "L", "م": "Z", "ن": "N", "ه": "H", "و": "U", "ى": "V",
}


def normalize_plate(
    text: str | None,
    *,
    country: str | None = None,
    fold: dict[str, str] | None = None,
    ksa_arabic_to_latin: bool = True,
    canonical_order: bool = True,
) -> str:
    """Canonical form used for exact match and CER.

    - Eastern Arabic / Persian digits -> 0-9; NFKC (Arabic presentation forms -> base letters)
    - spaces, separators, tatweel, diacritics and bidi marks removed; Latin upper-cased
    - letter variants folded (alef forms, ya/alef maqsura, ta marbuta)
    - KSA: Arabic letters mapped to their official Latin pair, so either plate row matches
    - canonical_order: digits first, then letters (each group keeps its order), so
      "ABJ 1234" == "1234 ABJ" == "١٢٣٤ ا ب ح" (for KSA)
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).translate(_DIGITS)
    s = _REMOVE_RX.sub("", s).upper()
    fold = DEFAULT_FOLD if fold is None else fold
    if fold:
        s = "".join(fold.get(c, c) for c in s)
    if ksa_arabic_to_latin and country == "ksa":
        s = "".join(KSA_AR_TO_LATIN.get(c, c) for c in s)
    if canonical_order:
        s = "".join(c for c in s if c.isdigit()) + "".join(c for c in s if not c.isdigit())
    return s


def digits_only(normalized: str) -> str:
    return "".join(c for c in normalized if c.isdigit())


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate of hyp against ref (can exceed 1.0 with many insertions)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)
