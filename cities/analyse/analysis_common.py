from pathlib import Path
import re
import unicodedata

import pandas as pd
import numpy as np


# ============================================================
# ORTAK ANALIZ AYARLARI
# ------------------------------------------------------------
# Bu dosya cities/analyse klasöründe durmalı.
# Tüm analiz dosyaları aynı 64 şehirlik ana listeyi buradan kullanır.
# Böylece total_score.csv 55/56 şehre düşmez.
# ============================================================

BASE_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_APP_DIR / "data"
TOTAL_DATA_DIR = BASE_APP_DIR / "total_data"
ANALYSE_DATA_DIR = BASE_APP_DIR / "analyse_data"

TOTAL_SCORE_FILE = TOTAL_DATA_DIR / "total_score.csv"

CITY_META = [{'city': 'Washington D.C.', 'region': 'North America', 'timezone': 'America/New_York', 'lat': 38.9072, 'lon': -77.0369}, {'city': 'New York', 'region': 'North America', 'timezone': 'America/New_York', 'lat': 40.7128, 'lon': -74.006}, {'city': 'Los Angeles', 'region': 'North America', 'timezone': 'America/Los_Angeles', 'lat': 34.0522, 'lon': -118.2437}, {'city': 'Ottawa', 'region': 'North America', 'timezone': 'America/Toronto', 'lat': 45.4215, 'lon': -75.6972}, {'city': 'Mexico City', 'region': 'Latin America', 'timezone': 'America/Mexico_City', 'lat': 19.4326, 'lon': -99.1332}, {'city': 'Monterrey', 'region': 'Latin America', 'timezone': 'America/Monterrey', 'lat': 25.6866, 'lon': -100.3161}, {'city': 'Brasília', 'region': 'Latin America', 'timezone': 'America/Sao_Paulo', 'lat': -15.7939, 'lon': -47.8828}, {'city': 'São Paulo', 'region': 'Latin America', 'timezone': 'America/Sao_Paulo', 'lat': -23.5558, 'lon': -46.6396}, {'city': 'Buenos Aires', 'region': 'Latin America', 'timezone': 'America/Argentina/Buenos_Aires', 'lat': -34.6037, 'lon': -58.3816}, {'city': 'Santiago', 'region': 'Latin America', 'timezone': 'America/Santiago', 'lat': -33.4489, 'lon': -70.6693}, {'city': 'Bogotá', 'region': 'Latin America', 'timezone': 'America/Bogota', 'lat': 4.711, 'lon': -74.0721}, {'city': 'Lima', 'region': 'Latin America', 'timezone': 'America/Lima', 'lat': -12.0464, 'lon': -77.0428}, {'city': 'San José', 'region': 'Latin America', 'timezone': 'America/Costa_Rica', 'lat': 9.9281, 'lon': -84.0907}, {'city': 'London', 'region': 'Europe', 'timezone': 'Europe/London', 'lat': 51.5072, 'lon': -0.1276}, {'city': 'Paris', 'region': 'Europe', 'timezone': 'Europe/Paris', 'lat': 48.8566, 'lon': 2.3522}, {'city': 'Berlin', 'region': 'Europe', 'timezone': 'Europe/Berlin', 'lat': 52.52, 'lon': 13.405}, {'city': 'Rome', 'region': 'Europe', 'timezone': 'Europe/Rome', 'lat': 41.9028, 'lon': 12.4964}, {'city': 'Madrid', 'region': 'Europe', 'timezone': 'Europe/Madrid', 'lat': 40.4168, 'lon': -3.7038}, {'city': 'Amsterdam', 'region': 'Europe', 'timezone': 'Europe/Amsterdam', 'lat': 52.3676, 'lon': 4.9041}, {'city': 'Brussels', 'region': 'Europe', 'timezone': 'Europe/Brussels', 'lat': 50.8503, 'lon': 4.3517}, {'city': 'Vienna', 'region': 'Europe', 'timezone': 'Europe/Vienna', 'lat': 48.2082, 'lon': 16.3738}, {'city': 'Bern', 'region': 'Europe', 'timezone': 'Europe/Zurich', 'lat': 46.948, 'lon': 7.4474}, {'city': 'Stockholm', 'region': 'Europe', 'timezone': 'Europe/Stockholm', 'lat': 59.3293, 'lon': 18.0686}, {'city': 'Oslo', 'region': 'Europe', 'timezone': 'Europe/Oslo', 'lat': 59.9139, 'lon': 10.7522}, {'city': 'Copenhagen', 'region': 'Europe', 'timezone': 'Europe/Copenhagen', 'lat': 55.6761, 'lon': 12.5683}, {'city': 'Helsinki', 'region': 'Europe', 'timezone': 'Europe/Helsinki', 'lat': 60.1699, 'lon': 24.9384}, {'city': 'Dublin', 'region': 'Europe', 'timezone': 'Europe/Dublin', 'lat': 53.3498, 'lon': -6.2603}, {'city': 'Lisbon', 'region': 'Europe', 'timezone': 'Europe/Lisbon', 'lat': 38.7223, 'lon': -9.1393}, {'city': 'Athens', 'region': 'Europe', 'timezone': 'Europe/Athens', 'lat': 37.9838, 'lon': 23.7275}, {'city': 'Warsaw', 'region': 'Europe', 'timezone': 'Europe/Warsaw', 'lat': 52.2297, 'lon': 21.0122}, {'city': 'Prague', 'region': 'Europe', 'timezone': 'Europe/Prague', 'lat': 50.0755, 'lon': 14.4378}, {'city': 'Budapest', 'region': 'Europe', 'timezone': 'Europe/Budapest', 'lat': 47.4979, 'lon': 19.0402}, {'city': 'Ankara', 'region': 'Europe', 'timezone': 'Europe/Istanbul', 'lat': 39.9334, 'lon': 32.8597}, {'city': 'Istanbul', 'region': 'Europe', 'timezone': 'Europe/Istanbul', 'lat': 41.0082, 'lon': 28.9784}, {'city': 'Moscow', 'region': 'Europe', 'timezone': 'Europe/Moscow', 'lat': 55.7558, 'lon': 37.6173}, {'city': 'Saint Petersburg', 'region': 'Europe', 'timezone': 'Europe/Moscow', 'lat': 59.9311, 'lon': 30.3609}, {'city': 'Reykjavík', 'region': 'Europe', 'timezone': 'Atlantic/Reykjavik', 'lat': 64.1466, 'lon': -21.9426}, {'city': 'Tallinn', 'region': 'Europe', 'timezone': 'Europe/Tallinn', 'lat': 59.437, 'lon': 24.7536}, {'city': 'Riga', 'region': 'Europe', 'timezone': 'Europe/Riga', 'lat': 56.9496, 'lon': 24.1052}, {'city': 'Vilnius', 'region': 'Europe', 'timezone': 'Europe/Vilnius', 'lat': 54.6872, 'lon': 25.2797}, {'city': 'Luxembourg City', 'region': 'Europe', 'timezone': 'Europe/Luxembourg', 'lat': 49.6116, 'lon': 6.1319}, {'city': 'Bratislava', 'region': 'Europe', 'timezone': 'Europe/Bratislava', 'lat': 48.1486, 'lon': 17.1077}, {'city': 'Ljubljana', 'region': 'Europe', 'timezone': 'Europe/Ljubljana', 'lat': 46.0569, 'lon': 14.5058}, {'city': 'Tokyo', 'region': 'East & Southeast Asia', 'timezone': 'Asia/Tokyo', 'lat': 35.6762, 'lon': 139.6503}, {'city': 'Osaka', 'region': 'East & Southeast Asia', 'timezone': 'Asia/Tokyo', 'lat': 34.6937, 'lon': 135.5023}, {'city': 'Seoul', 'region': 'East & Southeast Asia', 'timezone': 'Asia/Seoul', 'lat': 37.5665, 'lon': 126.978}, {'city': 'Busan', 'region': 'East & Southeast Asia', 'timezone': 'Asia/Seoul', 'lat': 35.1796, 'lon': 129.0756}, {'city': 'Beijing', 'region': 'East & Southeast Asia', 'timezone': 'Asia/Shanghai', 'lat': 39.9042, 'lon': 116.4074}, {'city': 'Shanghai', 'region': 'East & Southeast Asia', 'timezone': 'Asia/Shanghai', 'lat': 31.2304, 'lon': 121.4737}, {'city': 'Jakarta', 'region': 'East & Southeast Asia', 'timezone': 'Asia/Jakarta', 'lat': -6.2088, 'lon': 106.8456}, {'city': 'Singapore', 'region': 'East & Southeast Asia', 'timezone': 'Asia/Singapore', 'lat': 1.3521, 'lon': 103.8198}, {'city': 'New Delhi', 'region': 'South Asia, Middle East & Africa', 'timezone': 'Asia/Kolkata', 'lat': 28.6139, 'lon': 77.209}, {'city': 'Mumbai', 'region': 'South Asia, Middle East & Africa', 'timezone': 'Asia/Kolkata', 'lat': 19.076, 'lon': 72.8777}, {'city': 'Riyadh', 'region': 'South Asia, Middle East & Africa', 'timezone': 'Asia/Riyadh', 'lat': 24.7136, 'lon': 46.6753}, {'city': 'Johannesburg', 'region': 'South Asia, Middle East & Africa', 'timezone': 'Africa/Johannesburg', 'lat': -26.2041, 'lon': 28.0473}, {'city': 'Cape Town', 'region': 'South Asia, Middle East & Africa', 'timezone': 'Africa/Johannesburg', 'lat': -33.9249, 'lon': 18.4241}, {'city': 'Nairobi', 'region': 'South Asia, Middle East & Africa', 'timezone': 'Africa/Nairobi', 'lat': -1.2921, 'lon': 36.8219}, {'city': 'Cairo', 'region': 'South Asia, Middle East & Africa', 'timezone': 'Africa/Cairo', 'lat': 30.0444, 'lon': 31.2357}, {'city': 'Tel Aviv', 'region': 'South Asia, Middle East & Africa', 'timezone': 'Asia/Jerusalem', 'lat': 32.0853, 'lon': 34.7818}, {'city': 'Baku', 'region': 'South Asia, Middle East & Africa', 'timezone': 'Asia/Baku', 'lat': 40.4093, 'lon': 49.8671}, {'city': 'Canberra', 'region': 'Oceania', 'timezone': 'Australia/Sydney', 'lat': -35.2809, 'lon': 149.13}, {'city': 'Sydney', 'region': 'Oceania', 'timezone': 'Australia/Sydney', 'lat': -33.8688, 'lon': 151.2093}, {'city': 'Melbourne', 'region': 'Oceania', 'timezone': 'Australia/Melbourne', 'lat': -37.8136, 'lon': 144.9631}, {'city': 'Auckland', 'region': 'Oceania', 'timezone': 'Pacific/Auckland', 'lat': -36.8509, 'lon': 174.7645}]

GREEN_DATA = [['Washington D.C.', 0.022, 0.235, 0.744], ['New York', 0.201, 0.1, 0.509], ['Los Angeles', 0.226, 0.176, 0.759], ['London', 0.266, 0.317, 0.719], ['Paris', 0.009, 0.18, 0.868], ['Berlin', 0.144, 0.459, 0.577], ['Tokyo', 0.268, 0.043, 0.806], ['Osaka', 0.041, 0.0, 0.677], ['Canberra', 0.163, 0.861, 0.204], ['Sydney', 0.783, 0.697, 0.255], ['Melbourne', 1.0, 0.589, 0.245], ['Ottawa', 0.649, 0.632, 0.059], ['Seoul', 0.095, 0.35, 0.438], ['Busan', 0.326, 0.402, 0.122], ['Rome', 0.212, 0.543, 0.323], ['Madrid', 0.109, 0.464, 0.472], ['Amsterdam', 0.029, 0.304, 0.405], ['Brussels', 0.049, 0.347, 0.626], ['Vienna', 0.063, 0.404, 0.553], ['Bern', 0.0, 0.626, 0.367], ['Stockholm', 0.028, 0.359, 0.568], ['Oslo', 0.098, 0.676, 0.174], ['Copenhagen', 0.036, 0.171, 0.506], ['Helsinki', 0.11, 0.248, 0.146], ['Dublin', 0.012, 0.211, 0.853], ['Lisbon', 0.006, 0.21, 0.816], ['Athens', 0.138, 0.304, 0.458], ['Warsaw', 0.121, 0.489, 0.47], ['Prague', 0.077, 0.397, 0.469], ['Budapest', 0.082, 0.369, 0.641], ['Mexico City', 0.328, 0.367, 0.671], ['Monterrey', 0.375, 0.657, 0.266], ['Brasília', 0.452, 1.0, 0.096], ['São Paulo', 0.253, 0.522, 0.588], ['Buenos Aires', 0.027, 0.065, 1.0], ['Santiago', 0.24, 0.354, 0.522], ['Bogotá', 0.059, 0.168, 0.915], ['Lima', 0.407, 0.03, 0.136], ['Ankara', 0.314, 0.43, 0.282], ['Istanbul', 0.742, 0.388, 0.169], ['Moscow', 0.432, 0.31, 0.119], ['Saint Petersburg', 0.196, 0.427, 0.32], ['Beijing', 0.383, 0.255, 0.503], ['Shanghai', 0.719, 0.172, 0.464], ['New Delhi', 0.458, 0.19, 0.496], ['Mumbai', 0.343, 0.354, 0.223], ['Jakarta', 0.324, 0.132, 0.763], ['Riyadh', 0.687, 0.065, 0.964], ['Johannesburg', 0.411, 0.476, 0.556], ['Cape Town', 0.269, 0.307, 0.333], ['Nairobi', 0.111, 0.634, 0.455], ['Cairo', 0.396, 0.117, 0.2], ['Tel Aviv', 0.001, 0.049, 0.812], ['Singapore', 0.332, 0.235, 0.151], ['Baku', 0.21, 0.134, 0.372], ['Auckland', 0.332, 0.414, 0.13], ['Reykjavík', 0.033, 0.297, 0.0], ['San José', 0.117, 0.544, 0.554], ['Tallinn', 0.019, 0.458, 0.504], ['Riga', 0.043, 0.512, 0.365], ['Vilnius', 0.059, 0.721, 0.274], ['Luxembourg City', 0.023, 0.772, 0.193], ['Bratislava', 0.054, 0.567, 0.225], ['Ljubljana', 0.038, 0.856, 0.192]]


CONTAINS_CITY_MAP = {
    "washington": "Washington D.C.",
    "washington d c": "Washington D.C.",
    "new york": "New York",
    "los angeles": "Los Angeles",
    "london": "London",
    "paris": "Paris",
    "berlin": "Berlin",
    "tokyo": "Tokyo",
    "osaka": "Osaka",
    "canberra": "Canberra",
    "sydney": "Sydney",
    "melbourne": "Melbourne",
    "ottawa": "Ottawa",
    "seoul": "Seoul",
    "busan": "Busan",
    "rome": "Rome",
    "roma": "Rome",
    "madrid": "Madrid",
    "amsterdam": "Amsterdam",
    "brussels": "Brussels",
    "bruxelles": "Brussels",
    "vienna": "Vienna",
    "wien": "Vienna",
    "bern": "Bern",
    "stockholm": "Stockholm",
    "oslo": "Oslo",
    "copenhagen": "Copenhagen",
    "kobenhavn": "Copenhagen",
    "helsinki": "Helsinki",
    "dublin": "Dublin",
    "lisbon": "Lisbon",
    "lisboa": "Lisbon",
    "athens": "Athens",
    "warsaw": "Warsaw",
    "warszawa": "Warsaw",
    "prague": "Prague",
    "praha": "Prague",
    "budapest": "Budapest",
    "mexico city": "Mexico City",
    "ciudad de mexico": "Mexico City",
    "monterrey": "Monterrey",
    "brasilia": "Brasília",
    "brasília": "Brasília",
    "sao paulo": "São Paulo",
    "são paulo": "São Paulo",
    "buenos aires": "Buenos Aires",
    "santiago": "Santiago",
    "bogota": "Bogotá",
    "bogotá": "Bogotá",
    "lima": "Lima",
    "ankara": "Ankara",
    "istanbul": "Istanbul",
    "moscow": "Moscow",
    "moskva": "Moscow",
    "saint petersburg": "Saint Petersburg",
    "st petersburg": "Saint Petersburg",
    "beijing": "Beijing",
    "pekin": "Beijing",
    "shanghai": "Shanghai",
    "new delhi": "New Delhi",
    "delhi": "New Delhi",
    "mumbai": "Mumbai",
    "jakarta": "Jakarta",
    "riyadh": "Riyadh",
    "johannesburg": "Johannesburg",
    "cape town": "Cape Town",
    "nairobi": "Nairobi",
    "cairo": "Cairo",
    "tel aviv": "Tel Aviv",
    "singapore": "Singapore",
    "baku": "Baku",
    "auckland": "Auckland",
    "reykjavik": "Reykjavík",
    "reykjavík": "Reykjavík",
    "san jose": "San José",
    "san josé": "San José",
    "tallinn": "Tallinn",
    "riga": "Riga",
    "vilnius": "Vilnius",
    "luxembourg city": "Luxembourg City",
    "luxembourg": "Luxembourg City",
    "bratislava": "Bratislava",
    "ljubljana": "Ljubljana",
}


def read_csv_safely(path):
    encodings = ["utf-8-sig", "utf-8", "cp1254", "latin1"]

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            return clean_columns(df)
        except Exception:
            pass

    raise ValueError(f"{path} okunamadı.")


def clean_columns(df):
    df = df.copy()
    df.columns = [
        str(col)
        .replace("\ufeff", "")
        .replace("ï»¿", "")
        .strip()
        for col in df.columns
    ]
    return df


def clean_numeric(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()
    x = x.replace("%", "")
    x = x.replace(",", ".")

    if x.lower() in ["", "nan", "none", "veri yok", "-", "null"]:
        return np.nan

    try:
        return float(x)
    except Exception:
        return np.nan


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def city_key_from_any_name(name):
    norm = normalize_text(name)

    for pattern, target_city in CONTAINS_CITY_MAP.items():
        pattern_norm = normalize_text(pattern)
        if pattern_norm and pattern_norm in norm:
            return normalize_text(target_city)

    return norm


def detect_column(df, candidates):
    existing = list(df.columns)
    lowered = {str(col).lower(): col for col in existing}

    for candidate in candidates:
        if candidate in existing:
            return candidate

    for candidate in candidates:
        key = candidate.lower()
        if key in lowered:
            return lowered[key]

    return None


def safe_median(series, fallback=0.5, lower=None, upper=None):
    s = pd.to_numeric(series, errors="coerce")
    if lower is not None or upper is not None:
        s = s.clip(lower=lower, upper=upper)

    if not s.notna().any():
        return float(fallback)

    value = s.median(skipna=True)
    if pd.isna(value):
        value = fallback
    return float(value)


def fill_numeric_with_median(series, fallback=0.5, lower=None, upper=None):
    s = pd.to_numeric(series, errors="coerce")
    if lower is not None or upper is not None:
        s = s.clip(lower=lower, upper=upper)

    was_missing = s.isna()
    fill_value = safe_median(s, fallback=fallback, lower=lower, upper=upper)
    s = s.fillna(fill_value)

    if lower is not None or upper is not None:
        s = s.clip(lower=lower, upper=upper)

    return s, was_missing, fill_value


def minmax_01(series):
    s = pd.to_numeric(series, errors="coerce")
    min_val = s.min(skipna=True)
    max_val = s.max(skipna=True)

    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(0.5, index=s.index, dtype=float)

    return ((s - min_val) / (max_val - min_val)).clip(0, 1)


def to_0_1_score(series):
    s = series.apply(clean_numeric)
    max_val = s.max(skipna=True)

    if pd.isna(max_val):
        return s

    if max_val <= 1:
        return s.clip(0, 1)

    return (s / 100).clip(0, 1)


def get_master_city_dataframe():
    df = pd.DataFrame(CITY_META)
    df["city_key"] = df["city"].apply(city_key_from_any_name)
    df = df.drop_duplicates(subset=["city_key"], keep="first")
    df = df[["city", "city_key", "region", "timezone", "lat", "lon"]].copy()
    return df.reset_index(drop=True)


def get_master_city_score_base():
    return get_master_city_dataframe()[["city", "city_key"]].copy()


def ensure_score_has_64_cities(score_df, score_col, fallback=0.5):
    master = get_master_city_score_base()

    df = score_df.copy()
    if "city_key" not in df.columns:
        if "city" not in df.columns:
            raise ValueError("score_df içinde city veya city_key kolonu olmalı.")
        df["city_key"] = df["city"].apply(city_key_from_any_name)

    if score_col not in df.columns:
        raise ValueError(f"score_df içinde {score_col} kolonu yok.")

    small = (
        df[["city_key", score_col]]
        .copy()
        .groupby("city_key", as_index=False)
        .agg({score_col: "mean"})
    )

    out = master.merge(small, on="city_key", how="left")

    fill_value = safe_median(out[score_col], fallback=fallback, lower=0, upper=1)
    out[score_col + "_was_missing_in_score_file"] = out[score_col].isna()
    out[score_col] = (
        pd.to_numeric(out[score_col], errors="coerce")
        .fillna(fill_value)
        .clip(0, 1)
        .round(4)
    )

    return out, fill_value


def update_total_score_from_score_df(score_df, score_col, fallback=0.5):
    """
    total_score.csv dosyasını her zaman 64 şehirlik ana liste üzerinden günceller.
    Eski total_score.csv 55/56 şehir olsa bile ana listeye çekilir.
    """
    TOTAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    base = get_master_city_score_base()

    if TOTAL_SCORE_FILE.exists():
        total_df = read_csv_safely(TOTAL_SCORE_FILE)
        if "city" not in total_df.columns:
            total_df = base.copy()
        else:
            total_df["city_key"] = total_df["city"].apply(city_key_from_any_name)
            value_cols = [col for col in total_df.columns if col not in ["city", "city_key"]]
            total_df = (
                total_df[["city_key"] + value_cols]
                .drop_duplicates(subset=["city_key"], keep="first")
            )
            total_df = base.merge(total_df, on="city_key", how="left")
    else:
        total_df = base.copy()

    if score_col in total_df.columns:
        total_df = total_df.drop(columns=[score_col])

    score64, fill_value = ensure_score_has_64_cities(
        score_df=score_df,
        score_col=score_col,
        fallback=fallback,
    )

    score_small = score64[["city_key", score_col]].copy()

    merged = total_df.merge(score_small, on="city_key", how="left")
    merged[score_col] = (
        pd.to_numeric(merged[score_col], errors="coerce")
        .fillna(fill_value)
        .fillna(fallback)
        .clip(0, 1)
        .round(4)
    )

    merged = merged.drop_duplicates(subset=["city_key"], keep="first")
    merged = merged.drop(columns=["city_key"])

    # city ilk kolon kalsın
    cols = ["city"] + [col for col in merged.columns if col != "city"]
    merged = merged[cols]

    merged.to_csv(TOTAL_SCORE_FILE, index=False, encoding="utf-8-sig")

    return merged


def save_analyse_csv(df, output_file):
    ANALYSE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    return output_file
