"""
P1 - Preprocessing for Amazon ML Challenge 2026

Responsibilities
----------------
1. Preserve raw source fields.
2. Normalize business names and addresses.
3. Generate multiple deterministic name representations.
4. Generate multiple deterministic address representations.
5. Canonicalize country values.
6. Preserve missing-value information.
7. Provide character n-gram utilities for P2.
8. Support chunked TSV processing for large datasets.

P1 does NOT:
- perform fuzzy matching
- generate candidate pairs
- calculate pairwise similarity
- make entity-match decisions
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


# ============================================================
# OUTPUT ENCODING
# ============================================================

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

if (BASE_DIR / "dataset" / "train").exists():
    TRAIN_DIR = BASE_DIR / "dataset" / "train"
    TEST_DIR = BASE_DIR / "dataset" / "test"

elif (BASE_DIR.parent / "ml_dataset" / "data" / "train").exists():
    TRAIN_DIR = BASE_DIR.parent / "ml_dataset" / "data" / "train"
    TEST_DIR = BASE_DIR.parent / "ml_dataset" / "data" / "test"

elif (BASE_DIR / "ml_dataset" / "data" / "train").exists():
    TRAIN_DIR = BASE_DIR / "ml_dataset" / "data" / "train"
    TEST_DIR = BASE_DIR / "ml_dataset" / "data" / "test"

else:
    TRAIN_DIR = BASE_DIR / "dataset" / "train"
    TEST_DIR = BASE_DIR / "dataset" / "test"


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_CHUNKSIZE = 100_000

NAME_NGRAM_SIZES = (3, 4)
ADDRESS_NGRAM_SIZES = (3, 4)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value) -> str:
    """
    Normalize arbitrary text deterministically.

    Steps:
        1. Missing value -> ""
        2. Convert to string
        3. Unicode NFC normalization
        4. Unicode case folding
        5. Keep letters, numbers, combining marks and whitespace
        6. Replace punctuation/symbols with spaces
        7. Collapse repeated whitespace

    This function intentionally does NOT:
        - transliterate scripts
        - translate languages
        - perform fuzzy correction
        - remove arbitrary words
    """
    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize("NFC", value)
    value = value.casefold()

    result = []

    for char in value:
        category = unicodedata.category(char)

        if category.startswith(("L", "N", "M")) or char.isspace():
            result.append(char)
        else:
            result.append(" ")

    value = "".join(result)

    value = re.sub(r"\s+", " ", value).strip()

    return value


# ============================================================
# BUSINESS NAME
# ============================================================

# Legal suffixes are represented as tuples so that multi-token
# Indic word replacements for multilingual business names
INDIC_WORD_REPLACEMENTS = {
    "एसएस": "ss", "प्राइवेट": "private", "लिमिटेड": "limited", "कंपनी": "company",
    "एंटरप्राइजेज": "enterprises", "वेंचर्स": "ventures", "सर्विसेज": "services",
    "इंडस्ट्रीज": "industries", "फूड": "food", "होटल": "hotel",
    "टेक्नोलॉजी": "technology", "सॉल्यूशंस": "solutions", "एसोसिएशन": "association",
    "बैंक": "bank", "ट्रस्ट": "trust", "कॉर्पोरेशन": "corporation",
    "इन्वेस्टमेंट्स": "investments", "ट्रेडर्स": "traders", "इंटरनेशनल": "international",
    "ग्रुप": "group", "हरियाणा": "haryana", "ಕರ್ನಾಟಕ": "karnataka", "ತಮಿಳುನಾಡು": "tamil nadu",
    "ಮಹಾರಾಷ್ಟ್ರ": "maharashtra", "ದೆಹಲಿ": "delhi", "ರಾಜಸ್ಥಾನ": "rajasthan",
    "తెలంగాణ": "telangana", "ఆంధ్రప్రదేశ్": "andhra pradesh", "ఒడిశా": "odisha",
    "ଶକ୍ତି": "shakti", "ଆଗ୍ରୋ": "agro", "ଲିମିଟେଡ୍": "limited",
    "గ్రేట్": "great", "ఇంపెక్స్": "impex", "ప్రైవేట్": "private", "లిమిటెడ్": "limited",
    "రెడ్": "red", "వెంచర్స్": "ventures", "हॉस्पिटल": "hospital", "मल्टीस्पेशलिटी": "multispeciality",
    "ऑटोमोबाइल्स": "automobiles", "टेक्सटाइल्स": "textiles", "फार्मा": "pharma",
    "हेल्थकेयर": "healthcare", "एजुकेशन": "education", "सिक्योरिटीज": "securities",
}

HONORIFIC_PREFIXES = {"the", "smt", "m/s", "mr", "mrs", "ms", "shri", "dr", "pllc", "llc", "inc", "ltd", "partners"}

LEGAL_SUFFIX_TUPLES = [
    # 3-token suffixes
    ("limited", "liability", "company"),
    ("l", "l", "c"),
    ("l", "l", "p"),
    ("private", "limited", "company"),
    ("pvt", "ltd", "co"),

    # 2-token suffixes
    ("private", "limited"),
    ("pvt", "limited"),
    ("pvt", "ltd"),
    ("pty", "ltd"),
    ("co", "ltd"),
    ("gmbh", "co", "kg"),
    ("sp", "z", "o", "o"),

    # 1-token suffixes
    ("incorporated",),
    ("corporation",),
    ("limited",),
    ("company",),
    ("gmbh",),
    ("sarl",),
    ("corp",),
    ("llc",),
    ("pllc",),
    ("inc",),
    ("lnc",),
    ("ltd",),
    ("plc",),
    ("llp",),
    ("sas",),
    ("sasu",),
    ("sa",),
    ("lp",),
    ("co",),
    ("pvt",),
    ("pc",),
    ("pa",),
    ("federation",),
    ("foundation",),
    ("center",),
    ("centre",),
    ("services",),
]

ALL_LEGAL_SUFFIX_TOKENS = {
    token
    for suffix in LEGAL_SUFFIX_TUPLES
    for token in suffix
}


def normalize_business_name(value) -> str:
    """Normalize a business name using the generic text normalizer with Indic transliteration."""
    if pd.isna(value) or not value:
        return ""
    val_str = str(value)
    for k, v in INDIC_WORD_REPLACEMENTS.items():
        if k in val_str:
            val_str = val_str.replace(k, f" {v} ")
    return normalize_text(val_str)


def extract_name_core(name_clean: str) -> str:
    """
    Remove trailing legal-entity suffixes, strip web domain extensions and honorific prefixes.

    Important:
        - Trailing suffixes (and common stacked combinations) are removed.
        - Domain suffixes (.com, .org, etc.) are stripped.
        - Leading noise/honorific prefixes (e.g. Smt, The, M/s) are stripped.
        - Suffix-only names are preserved.
    """
    if not name_clean:
        return ""

    # Strip domain extensions like .com, .org, .net, .in, .co
    name_clean = re.sub(r"\.(com|org|net|in|co|io|biz|info|gov)$", "", name_clean)
    name_clean = re.sub(r"\s+(com|org|net)$", "", name_clean)

    tokens = name_clean.split()
    if not tokens:
        return ""

    # Never reduce a name that consists entirely of suffix tokens.
    if all(token in ALL_LEGAL_SUFFIX_TOKENS for token in tokens):
        return name_clean

    # Strip leading honorific / noise prefixes if more tokens follow
    if tokens[0] in HONORIFIC_PREFIXES and len(tokens) > 1:
        tokens = tokens[1:]

    current = list(tokens)

    while True:
        matched = False
        for suffix in LEGAL_SUFFIX_TUPLES:
            suffix_len = len(suffix)
            if len(current) <= suffix_len:
                continue

            if tuple(current[-suffix_len:]) == suffix:
                current = current[:-suffix_len]
                matched = True
                break

        if not matched:
            break

    if not current:
        return name_clean

    return " ".join(current)


def extract_first_two_tokens(core_name: str) -> str:
    """Extract first two significant non-stopword tokens from core name."""
    if not core_name:
        return ""
    tokens = [t for t in core_name.split() if t not in GENERIC_BUSINESS_STOPWORDS and len(t) > 1]
    if len(tokens) >= 2:
        return " ".join(tokens[:2])
    return ""


# ============================================================
# PHONETIC REPRESENTATION
# ============================================================

def soundex(token: str) -> str:
    """
    Standard deterministic Soundex implementation.

    Soundex is primarily useful for Latin/English-like names.
    It should therefore be treated as one blocking signal rather
    than a universal multilingual representation.
    """
    token = str(token).upper()

    if not token or not token[0].isalpha():
        return ""

    mapping = {
        "B": "1",
        "F": "1",
        "P": "1",
        "V": "1",

        "C": "2",
        "G": "2",
        "J": "2",
        "K": "2",
        "Q": "2",
        "S": "2",
        "X": "2",
        "Z": "2",

        "D": "3",
        "T": "3",

        "L": "4",

        "M": "5",
        "N": "5",

        "R": "6",
    }

    first = token[0]
    encoded = [first]
    previous_code = mapping.get(first, "")

    for char in token[1:]:
        code = mapping.get(char, "")

        if code:
            if code != previous_code:
                encoded.append(code)

            previous_code = code

        elif char in "AEIOUY":
            previous_code = ""

        else:
            # H and W do not reset repeated consonant groups.
            pass

    code_string = "".join(encoded)

    digits = code_string[1:].replace("0", "")

    return (code_string[0] + digits + "000")[:4]


def compute_phonetic_key(value: str) -> str:
    """
    Compute a token-sorted phonetic representation.

    Example:

        'delta telecommunication'

    becomes something conceptually like:

        'D430 T425'
    """
    if not value:
        return ""

    tokens = [token for token in value.split() if token]

    codes = [
        soundex(token)
        for token in tokens
    ]

    codes = [code for code in codes if code]

    if not codes:
        return ""

    return " ".join(sorted(codes))


# ============================================================
# TOKEN REPRESENTATIONS
# ============================================================

def sorted_token_key(value: str) -> str:
    """
    Return a deterministic token-order-independent representation.

    Example:
        'metro coalition llc'
            -> 'coalition llc metro'
    """
    if not value:
        return ""

    tokens = value.split()

    return " ".join(sorted(tokens))


def compact_text(value: str) -> str:
    """Remove whitespace from a normalized representation."""
    if not value:
        return ""

    return re.sub(r"\s+", "", value)


# ============================================================
# CHARACTER N-GRAMS
# ============================================================

def char_ngrams(
    value: str,
    n: int = 3,
) -> tuple[str, ...]:
    """
    Generate character n-grams from normalized text.

    Spaces are retained as boundaries after normalization.

    Example:
        'delta'
        n=3

        -> ('del', 'elt', 'lta')

    This function is intentionally not called automatically for
    every dataframe row because materializing millions of n-gram
    tuples can consume substantial memory.

    P2 can call this while constructing an inverted index.
    """
    if not value:
        return ()

    if n <= 0:
        raise ValueError("n must be greater than zero")

    if len(value) < n:
        return ()

    return tuple(
        value[i:i + n]
        for i in range(len(value) - n + 1)
    )


def unique_char_ngrams(
    value: str,
    n: int = 3,
) -> tuple[str, ...]:
    """Generate unique character n-grams while preserving order."""
    grams = char_ngrams(value, n)

    if not grams:
        return ()

    return tuple(dict.fromkeys(grams))


def multi_char_ngrams(
    value: str,
    sizes: Iterable[int] = NAME_NGRAM_SIZES,
) -> tuple[str, ...]:
    """
    Generate unique n-grams across multiple n values.

    Intended for P2 indexing rather than storing as a dataframe
    column by default.
    """
    result = []

    for n in sizes:
        result.extend(unique_char_ngrams(value, n))

    return tuple(dict.fromkeys(result))


# ============================================================
# DIACRITIC / ACCENT NORMALIZATION & SIGNIFICANT TOKENS
# ============================================================

GENERIC_BUSINESS_STOPWORDS = {
    "the", "and", "of", "for", "in", "at", "by", "with", "&",
    "group", "services", "service", "international", "global",
    "solutions", "solution", "holdings", "holding", "enterprises",
    "enterprise", "industries", "industry", "management", "consulting",
    "consultants", "tech", "technology", "technologies", "trading",
    "commercial", "logistics", "associates", "partners", "products",
    "systems", "worldwide", "corporation", "company", "limited"
}


def strip_accents_ascii(value: str) -> str:
    """Normalize text to ASCII by removing accents and combining diacritical marks.

    Example:
        'café müller société' -> 'cafe muller societe'
    """
    if not value:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(value))
    ascii_chars = [c for c in nfkd if not unicodedata.combining(c)]
    return "".join(ascii_chars).encode("ascii", "ignore").decode("ascii")


def significant_token_key(value: str) -> str:
    """Extract sorted non-stopword tokens for robust semantic token blocking."""
    if not value:
        return ""
    tokens = [t for t in value.split() if t not in GENERIC_BUSINESS_STOPWORDS and len(t) > 1]
    if not tokens:
        tokens = value.split()
    return " ".join(sorted(tokens))


# ============================================================
# ADDRESS
# ============================================================

def normalize_address(value) -> str:
    """Normalize a business address using the generic text normalizer."""
    return normalize_text(value)


STREET_TYPES = {
    "street": "st", "st": "st", "saint": "st",
    "avenue": "ave", "ave": "ave",
    "road": "rd", "rd": "rd",
    "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct",
    "boulevard": "blvd", "blvd": "blvd",
    "way": "way", "place": "pl", "pl": "pl",
    "circle": "cir", "cir": "cir",
    "highway": "hwy", "hwy": "hwy",
    "expressway": "expy", "parkway": "pkwy",
    "marg": "rd", "rasta": "rd", "salai": "rd", "path": "rd",
}

US_STATE_MAP = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}

INDIA_STATE_MAP = {
    "uttar pradesh": "up", "up": "up", "delhi": "dl", "maharashtra": "mh", "mh": "mh",
    "karnataka": "ka", "ka": "ka", "tamil nadu": "tn", "tn": "tn", "rajasthan": "rj", "rj": "rj",
    "gujarat": "gj", "gj": "gj", "west bengal": "wb", "wb": "wb", "haryana": "hr", "hr": "hr",
    "punjab": "pb", "pb": "pb", "andhra pradesh": "ap", "ap": "ap", "telangana": "tg", "tg": "tg",
    "ts": "tg", "kerala": "kl", "kl": "kl", "madhya pradesh": "mp", "mp": "mp", "bihar": "br", "br": "br",
    "odisha": "od", "orissa": "od", "od": "od", "jharkhand": "jh", "jh": "jh", "uttarakhand": "uk",
    "assam": "as", "as": "as", "goa": "ga", "ga": "ga", "himachal pradesh": "hp", "hp": "hp",
}


def extract_address_components(address_clean: str, country_clean: str = "") -> dict[str, str]:
    """
    Extract lightweight deterministic address components.

    Returned fields:
        house_number
        street_root
        postal_code
        geo_token
        address_street_key
        address_house_geo_key
        address_house_postcode_key
        address_component_key (composite key of house# + postal + country)
    """
    if not address_clean:
        return {
            "house_number": "",
            "street_root": "",
            "postal_code": "",
            "geo_token": "",
            "address_street_key": "",
            "address_house_geo_key": "",
            "address_house_postcode_key": "",
            "address_component_key": "",
        }

    # Normalize address_clean to lowercase tokens
    norm_addr = normalize_text(address_clean)
    tokens = norm_addr.split()
    if not tokens:
        return {
            "house_number": "",
            "street_root": "",
            "postal_code": "",
            "geo_token": "",
            "address_street_key": "",
            "address_house_geo_key": "",
            "address_house_postcode_key": "",
            "address_component_key": "",
        }

    house_number = ""
    house_idx = -1

    # First check raw address tokens for unit prefixes like "af-0684", "004303"
    raw_tokens = str(address_clean).lower().replace(",", " ").split()
    for i, token in enumerate(raw_tokens):
        m = re.match(r"^(?:no|door|hno|plot|shop|flat|[a-z]{1,3})?[\.\#\-]?0*(\d+[a-z]?)$", token)
        if m:
            house_number = m.group(1)
            house_idx = i
            break
        m2 = re.match(r"^([a-z]{0,3}\-?0*\d+[\-\/]\w+.*)$", token)
        if m2:
            num_part = re.sub(r"^[a-z]{0,3}\-?0*", "", m2.group(1))
            house_number = num_part if num_part else m2.group(1)
            house_idx = i
            break

    # If not found from raw tokens, check normalized tokens
    if not house_number:
        for i, token in enumerate(tokens):
            m = re.match(r"^(?:no|door|hno|plot|shop|flat|[a-z]{1,3})?0*(\d+[a-z]?)$", token)
            if m:
                house_number = m.group(1)
                house_idx = i
                break

    # Extract primary street root (e.g. "home", "elkins", "swift", "fordsbush", "laytonia")
    street_root = ""
    for i, token in enumerate(tokens):
        if token in STREET_TYPES:
            if i > 0 and tokens[i - 1] != house_number and len(tokens[i - 1]) > 2:
                street_root = tokens[i - 1]
                break
    if not street_root and house_idx != -1 and house_idx + 1 < len(tokens):
        nxt = tokens[house_idx + 1]
        if len(nxt) > 2 and nxt not in STREET_TYPES:
            street_root = nxt

    # Generic postal-code extraction (5-6 digits or alphanumeric postal patterns)
    postal_code = ""
    for token in tokens:
        if re.fullmatch(r"\d{5,6}", token):
            postal_code = token
            break

    # State or locality geo token (check multi-token states first, then single token)
    geo_token = ""
    padded_addr = f" {norm_addr} "
    for state_name, code in INDIA_STATE_MAP.items():
        if f" {state_name} " in padded_addr:
            geo_token = code
            break
    if not geo_token:
        for state_name, code in US_STATE_MAP.items():
            if f" {state_name} " in padded_addr:
                geo_token = code
                break
    if not geo_token:
        for token in tokens:
            if token in US_STATE_MAP.values():
                geo_token = token
                break
            if token in INDIA_STATE_MAP.values():
                geo_token = token
                break

    component_key = ""
    if house_number and postal_code:
        component_key = f"{house_number}_{postal_code}_{country_clean}".strip("_")
    elif postal_code and country_clean:
        component_key = f"post_{postal_code}_{country_clean}"

    street_key = f"{house_number}_{street_root}_{country_clean}".strip("_") if (house_number and street_root) else ""
    house_geo_key = f"{house_number}_{geo_token}_{country_clean}".strip("_") if (house_number and geo_token) else ""
    house_postcode_key = f"{house_number}_{postal_code}_{country_clean}".strip("_") if (house_number and postal_code) else ""

    return {
        "house_number": house_number,
        "street_root": street_root,
        "postal_code": postal_code,
        "geo_token": geo_token,
        "address_street_key": street_key,
        "address_house_geo_key": house_geo_key,
        "address_house_postcode_key": house_postcode_key,
        "address_component_key": component_key,
    }


# ============================================================
# COUNTRY
# ============================================================

COUNTRY_ALIASES = {
    # United States
    "us": "us", "usa": "us", "united states": "us", "united states of america": "us",
    # India
    "in": "india", "ind": "india", "india": "india", "republic of india": "india",
    # France
    "fr": "france", "fra": "france", "france": "france", "french republic": "france",
    # United Kingdom
    "uk": "uk", "united kingdom": "uk", "gb": "uk", "gbr": "uk", "great britain": "uk",
    # Germany
    "de": "germany", "deu": "germany", "germany": "germany", "deutschland": "germany",
    # Canada
    "ca": "canada", "can": "canada", "canada": "canada",
    # Australia
    "au": "australia", "aus": "australia", "australia": "australia",
    # China
    "cn": "china", "chn": "china", "china": "china", "prc": "china", "peoples republic of china": "china",
    # Japan
    "jp": "japan", "jpn": "japan", "japan": "japan",
    # Brazil
    "br": "brazil", "bra": "brazil", "brazil": "brazil", "brasil": "brazil",
    # Italy
    "it": "italy", "ita": "italy", "italy": "italy", "italia": "italy",
    # Spain
    "es": "spain", "esp": "spain", "spain": "spain", "espana": "spain",
    # Mexico
    "mx": "mexico", "mex": "mexico", "mexico": "mexico",
    # Netherlands
    "nl": "netherlands", "nld": "netherlands", "netherlands": "netherlands", "holland": "netherlands",
    # Singapore
    "sg": "singapore", "sgp": "singapore", "singapore": "singapore",
    # Switzerland
    "ch": "switzerland", "che": "switzerland", "switzerland": "switzerland", "suisse": "switzerland", "schweiz": "switzerland",
    # Sweden
    "se": "sweden", "swe": "sweden", "sweden": "sweden", "sverige": "sweden",
    # Russia
    "ru": "russia", "rus": "russia", "russia": "russia", "russian federation": "russia",
    # South Africa
    "za": "south africa", "zaf": "south africa", "south africa": "south africa",
    # UAE
    "ae": "uae", "are": "uae", "uae": "uae", "united arab emirates": "uae",
    # Poland
    "pl": "poland", "pol": "poland", "poland": "poland", "polska": "poland",
    # Belgium
    "be": "belgium", "bel": "belgium", "belgium": "belgium", "belgique": "belgium",
    # Austria
    "at": "austria", "aut": "austria", "austria": "austria", "osterreich": "austria",
    # Ireland
    "ie": "ireland", "irl": "ireland", "ireland": "ireland",
    # New Zealand
    "nz": "new zealand", "nzl": "new zealand", "new zealand": "new zealand",
    # South Korea
    "kr": "south korea", "kor": "south korea", "south korea": "south korea", "korea": "south korea",
    # Turkey
    "tr": "turkey", "tur": "turkey", "turkey": "turkey", "turkiye": "turkey",
    # Norway
    "no": "norway", "nor": "norway", "norge": "norway",
    # Denmark
    "dk": "denmark", "dnk": "denmark", "danmark": "denmark",
    # Finland
    "fi": "finland", "fin": "finland", "suomi": "finland",
    # Portugal
    "pt": "portugal", "prt": "portugal",
    # Greece
    "gr": "greece", "grc": "greece", "hellas": "greece",
    # Czechia
    "cz": "czechia", "cze": "czechia", "czech republic": "czechia",
    # Romania
    "ro": "romania", "rou": "romania",
    # Hungary
    "hu": "hungary", "hun": "hungary",
    # Argentina
    "ar": "argentina", "arg": "argentina",
    # Chile
    "cl": "chile", "chl": "chile",
    # Colombia
    "co": "colombia", "col": "colombia",
    # Malaysia
    "my": "malaysia", "mys": "malaysia",
    # Indonesia
    "id": "indonesia", "idn": "indonesia",
    # Thailand
    "th": "thailand", "tha": "thailand",
    # Vietnam
    "vn": "vietnam", "vnm": "vietnam",
    # Philippines
    "ph": "philippines", "phl": "philippines",
}


def normalize_country(value) -> str:
    """Normalize country values to deterministic canonical names."""
    if pd.isna(value):
        return ""

    value = str(value).strip().casefold()

    if not value:
        return ""

    value = re.sub(r"[^\w\s]", "", value)
    value = re.sub(r"\s+", " ", value).strip()

    if not value:
        return ""

    return COUNTRY_ALIASES.get(value, value)


# ============================================================
# DATAFRAME PREPROCESSING
# ============================================================

def _safe_column(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Return a column if available, otherwise an empty string series.

    This keeps preprocessing robust to schema validation and makes
    missing-column behavior explicit.
    """
    if column in df.columns:
        return df[column]

    return pd.Series(
        [""] * len(df),
        index=df.index,
        dtype="string",
    )


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess one dataframe chunk.

    The function preserves the original source columns and adds
    deterministic derived representations.
    """
    df = df.copy()

    # --------------------------------------------------------
    # Validate expected source schema
    # --------------------------------------------------------

    expected_columns = {
        "entity_id",
        "business_name",
        "business_address",
        "country",
    }

    missing_columns = expected_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # --------------------------------------------------------
    # Original missing-value flags
    # --------------------------------------------------------

    df["name_was_missing"] = df["business_name"].isna()
    df["address_was_missing"] = df["business_address"].isna()
    df["country_was_missing"] = df["country"].isna()

    # --------------------------------------------------------
    # Business name
    # --------------------------------------------------------

    df["name_clean"] = (
        df["business_name"]
        .map(normalize_business_name)
    )

    df["name_compact"] = (
        df["name_clean"]
        .map(compact_text)
    )

    df["name_token_key"] = (
        df["name_clean"]
        .map(sorted_token_key)
    )

    df["name_core"] = (
        df["name_clean"]
        .map(extract_name_core)
    )

    df["name_core_compact"] = (
        df["name_core"]
        .map(compact_text)
    )

    df["name_core_token_key"] = (
        df["name_core"]
        .map(sorted_token_key)
    )

    df["name_phonetic_key"] = (
        df["name_core"]
        .map(compute_phonetic_key)
    )

    df["name_ascii"] = (
        df["name_clean"]
        .map(strip_accents_ascii)
    )

    df["name_ascii_compact"] = (
        df["name_ascii"]
        .map(compact_text)
    )

    df["name_significant_token_key"] = (
        df["name_core"]
        .map(significant_token_key)
    )

    df["name_first_two_tokens"] = (
        df["name_core"]
        .map(extract_first_two_tokens)
    )

    df["name_has_digits"] = df["name_clean"].str.contains(r"\d", regex=True, na=False)

    # --------------------------------------------------------
    # Business-name statistics useful for P2/P3
    # --------------------------------------------------------

    df["name_token_count"] = (
        df["name_clean"]
        .map(lambda value: len(value.split()) if value else 0)
    )

    df["name_core_token_count"] = (
        df["name_core"]
        .map(lambda value: len(value.split()) if value else 0)
    )

    # --------------------------------------------------------
    # Country (normalized early for address components)
    # --------------------------------------------------------

    df["country_clean"] = (
        df["country"]
        .map(normalize_country)
    )

    # --------------------------------------------------------
    # Address
    # --------------------------------------------------------

    df["address_clean"] = (
        df["business_address"]
        .map(normalize_address)
    )

    df["address_token_key"] = (
        df["address_clean"]
        .map(sorted_token_key)
    )

    df["address_token_count"] = (
        df["address_clean"]
        .map(lambda value: len(value.split()) if value else 0)
    )

    df["address_has_digits"] = df["address_clean"].str.contains(r"\d", regex=True, na=False)

    # Lightweight structured address features.
    address_components = [
        extract_address_components(addr, ctry)
        for addr, ctry in zip(df["address_clean"], df["country_clean"])
    ]

    df["address_house_number"] = [item["house_number"] for item in address_components]
    df["address_street_root"] = [item["street_root"] for item in address_components]
    df["address_postal_code"] = [item["postal_code"] for item in address_components]
    df["address_component_key"] = [item["address_component_key"] for item in address_components]
    df["address_street_key"] = [item["address_street_key"] for item in address_components]
    df["address_house_geo_key"] = [item["address_house_geo_key"] for item in address_components]
    df["address_house_postcode_key"] = [item["address_house_postcode_key"] for item in address_components]

    df["dedup_group_key"] = df["name_compact"] + "_" + df["address_clean"] + "_" + df["country_clean"]

    # --------------------------------------------------------
    # Derived missing flags
    # --------------------------------------------------------

    df["name_is_empty"] = df["name_clean"].eq("")
    df["name_core_is_empty"] = df["name_core"].eq("")
    df["address_is_empty"] = df["address_clean"].eq("")
    df["country_is_empty"] = df["country_clean"].eq("")

    return df


# ============================================================
# CHUNKED FILE READER
# ============================================================

def iter_preprocessed_file(
    input_path,
    chunksize: int = DEFAULT_CHUNKSIZE,
):
    """
    Read and preprocess a TSV file incrementally.

    Important for this challenge because the source files are
    hundreds of MB and should not be loaded completely into RAM.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero")

    for chunk in pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=True,
        chunksize=chunksize,
    ):
        yield preprocess_dataframe(chunk)


# ============================================================
# VALIDATION
# ============================================================

def validate_file(
    input_path,
    rows: int = 1000,
) -> None:
    """Run a lightweight preprocessing validation on one file."""
    input_path = Path(input_path)

    print("\n" + "=" * 80)
    print(f"VALIDATING: {input_path.name}")
    print("=" * 80)

    chunk = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=True,
        nrows=rows,
    )

    processed = preprocess_dataframe(chunk)

    print(f"Rows tested: {len(processed):,}")

    print("\nDerived columns:")
    print(
        [
            column
            for column in processed.columns
            if column not in chunk.columns
        ]
    )

    sample_columns = [
        "entity_id",
        "business_name",
        "name_clean",
        "name_compact",
        "name_token_key",
        "name_core",
        "name_core_compact",
        "name_phonetic_key",
        "business_address",
        "address_clean",
        "address_token_key",
        "address_house_number",
        "address_postal_code",
        "country_clean",
        "name_was_missing",
        "address_was_missing",
        "country_was_missing",
    ]

    print("\nSample:")
    print(
        processed[sample_columns]
        .head(5)
        .to_string(index=False)
    )

    print("\nUnexpected NaN values in derived columns:")

    derived_columns = [
        "name_clean",
        "name_compact",
        "name_token_key",
        "name_core",
        "name_core_compact",
        "name_core_token_key",
        "name_phonetic_key",
        "address_clean",
        "address_token_key",
        "country_clean",
    ]

    print(
        processed[derived_columns]
        .isna()
        .sum()
    )

    print("\nCharacter n-gram utility test:")

    if len(processed) > 0:
        example = processed.iloc[0]["name_core"]

        print(f"Input: {example!r}")
        print(f"3-grams: {char_ngrams(example, 3)[:10]}")
        print(f"4-grams: {char_ngrams(example, 4)[:10]}")

    print("\nValidation complete.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    files = [
        TRAIN_DIR / "train_source1.tsv",
        TRAIN_DIR / "train_source2.tsv",
        TRAIN_DIR / "train_source3.tsv",

        TEST_DIR / "test_source1.tsv",
        TEST_DIR / "test_source2.tsv",
        TEST_DIR / "test_source3.tsv",
    ]

    print()
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("P1 PREPROCESSING VALIDATION")
    print("=" * 80)

    for file_path in files:

        if not file_path.exists():
            print(f"\nWARNING: File not found: {file_path}")
            continue

        validate_file(file_path)

    print()
    print("=" * 80)
    print("P1 PREPROCESSING VALIDATION COMPLETE")
    print("=" * 80)