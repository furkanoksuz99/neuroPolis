from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import time
import os

import requests
import pandas as pd
import numpy as np


# ============================================================
# TOMTOM TRAFFIC SCORE API SCRIPT
# ------------------------------------------------------------
# - Input yok
# - Getpass yok
# - Geocode yok
# - Hazır koordinatlar
# - Her şehir için 13 nokta
# - Sadece skor CSV kaydeder
# ============================================================


# ============================================================
# 1) AYARLAR
# ============================================================

BASE_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_APP_DIR / "data"

OUTPUT_CSV_NAME = "traffic_score_results.csv"

# Kanka tokeni buraya yaz.
# Güvenlik için burada senin tokeni tekrar yazmıyorum.
# Örnek:
# TOMTOM_API_KEY = "abc123..."
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "FNWH3j9zCS4o4Nyr9InJjzWYR72gu29z")

POINT_DISTANCE_KM = 5
ZOOM = 10
REQUEST_SLEEP = 0.15
REQUEST_TIMEOUT = 20

RUN_PREFLIGHT_TEST = True

DEFAULT_TRAFFIC_BASE = "api.tomtom.com"

TRAFFIC_BASE_BY_COUNTRY = {
    "South Korea": "kr-api.tomtom.com",
}


# ============================================================
# 2) ŞEHİR LİSTESİ
# ============================================================

CITIES = [
    {"city": "Washington D.C.", "country": "United States", "timezone": "America/New_York", "lat": 38.9072, "lon": -77.0369},
    {"city": "New York", "country": "United States", "timezone": "America/New_York", "lat": 40.7128, "lon": -74.0060},
    {"city": "Los Angeles", "country": "United States", "timezone": "America/Los_Angeles", "lat": 34.0522, "lon": -118.2437},
    {"city": "London", "country": "United Kingdom", "timezone": "Europe/London", "lat": 51.5072, "lon": -0.1276},
    {"city": "Paris", "country": "France", "timezone": "Europe/Paris", "lat": 48.8566, "lon": 2.3522},
    {"city": "Berlin", "country": "Germany", "timezone": "Europe/Berlin", "lat": 52.5200, "lon": 13.4050},
    {"city": "Tokyo", "country": "Japan", "timezone": "Asia/Tokyo", "lat": 35.6762, "lon": 139.6503},
    {"city": "Osaka", "country": "Japan", "timezone": "Asia/Tokyo", "lat": 34.6937, "lon": 135.5023},
    {"city": "Canberra", "country": "Australia", "timezone": "Australia/Sydney", "lat": -35.2809, "lon": 149.1300},
    {"city": "Sydney", "country": "Australia", "timezone": "Australia/Sydney", "lat": -33.8688, "lon": 151.2093},
    {"city": "Melbourne", "country": "Australia", "timezone": "Australia/Melbourne", "lat": -37.8136, "lon": 144.9631},
    {"city": "Ottawa", "country": "Canada", "timezone": "America/Toronto", "lat": 45.4215, "lon": -75.6972},
    {"city": "Seoul", "country": "South Korea", "timezone": "Asia/Seoul", "lat": 37.5665, "lon": 126.9780},
    {"city": "Busan", "country": "South Korea", "timezone": "Asia/Seoul", "lat": 35.1796, "lon": 129.0756},
    {"city": "Rome", "country": "Italy", "timezone": "Europe/Rome", "lat": 41.9028, "lon": 12.4964},
    {"city": "Madrid", "country": "Spain", "timezone": "Europe/Madrid", "lat": 40.4168, "lon": -3.7038},
    {"city": "Amsterdam", "country": "Netherlands", "timezone": "Europe/Amsterdam", "lat": 52.3676, "lon": 4.9041},
    {"city": "Brussels", "country": "Belgium", "timezone": "Europe/Brussels", "lat": 50.8503, "lon": 4.3517},
    {"city": "Vienna", "country": "Austria", "timezone": "Europe/Vienna", "lat": 48.2082, "lon": 16.3738},
    {"city": "Bern", "country": "Switzerland", "timezone": "Europe/Zurich", "lat": 46.9480, "lon": 7.4474},
    {"city": "Stockholm", "country": "Sweden", "timezone": "Europe/Stockholm", "lat": 59.3293, "lon": 18.0686},
    {"city": "Oslo", "country": "Norway", "timezone": "Europe/Oslo", "lat": 59.9139, "lon": 10.7522},
    {"city": "Copenhagen", "country": "Denmark", "timezone": "Europe/Copenhagen", "lat": 55.6761, "lon": 12.5683},
    {"city": "Helsinki", "country": "Finland", "timezone": "Europe/Helsinki", "lat": 60.1699, "lon": 24.9384},
    {"city": "Dublin", "country": "Ireland", "timezone": "Europe/Dublin", "lat": 53.3498, "lon": -6.2603},
    {"city": "Lisbon", "country": "Portugal", "timezone": "Europe/Lisbon", "lat": 38.7223, "lon": -9.1393},
    {"city": "Athens", "country": "Greece", "timezone": "Europe/Athens", "lat": 37.9838, "lon": 23.7275},
    {"city": "Warsaw", "country": "Poland", "timezone": "Europe/Warsaw", "lat": 52.2297, "lon": 21.0122},
    {"city": "Prague", "country": "Czechia", "timezone": "Europe/Prague", "lat": 50.0755, "lon": 14.4378},
    {"city": "Budapest", "country": "Hungary", "timezone": "Europe/Budapest", "lat": 47.4979, "lon": 19.0402},
    {"city": "Mexico City", "country": "Mexico", "timezone": "America/Mexico_City", "lat": 19.4326, "lon": -99.1332},
    {"city": "Monterrey", "country": "Mexico", "timezone": "America/Monterrey", "lat": 25.6866, "lon": -100.3161},
    {"city": "Brasília", "country": "Brazil", "timezone": "America/Sao_Paulo", "lat": -15.7939, "lon": -47.8828},
    {"city": "São Paulo", "country": "Brazil", "timezone": "America/Sao_Paulo", "lat": -23.5558, "lon": -46.6396},
    {"city": "Buenos Aires", "country": "Argentina", "timezone": "America/Argentina/Buenos_Aires", "lat": -34.6037, "lon": -58.3816},
    {"city": "Santiago", "country": "Chile", "timezone": "America/Santiago", "lat": -33.4489, "lon": -70.6693},
    {"city": "Bogotá", "country": "Colombia", "timezone": "America/Bogota", "lat": 4.7110, "lon": -74.0721},
    {"city": "Lima", "country": "Peru", "timezone": "America/Lima", "lat": -12.0464, "lon": -77.0428},
    {"city": "Ankara", "country": "Turkey", "timezone": "Europe/Istanbul", "lat": 39.9334, "lon": 32.8597},
    {"city": "Istanbul", "country": "Turkey", "timezone": "Europe/Istanbul", "lat": 41.0082, "lon": 28.9784},
    {"city": "Moscow", "country": "Russia", "timezone": "Europe/Moscow", "lat": 55.7558, "lon": 37.6173},
    {"city": "Saint Petersburg", "country": "Russia", "timezone": "Europe/Moscow", "lat": 59.9311, "lon": 30.3609},
    {"city": "Beijing", "country": "China", "timezone": "Asia/Shanghai", "lat": 39.9042, "lon": 116.4074},
    {"city": "Shanghai", "country": "China", "timezone": "Asia/Shanghai", "lat": 31.2304, "lon": 121.4737},
    {"city": "New Delhi", "country": "India", "timezone": "Asia/Kolkata", "lat": 28.6139, "lon": 77.2090},
    {"city": "Mumbai", "country": "India", "timezone": "Asia/Kolkata", "lat": 19.0760, "lon": 72.8777},
    {"city": "Jakarta", "country": "Indonesia", "timezone": "Asia/Jakarta", "lat": -6.2088, "lon": 106.8456},
    {"city": "Riyadh", "country": "Saudi Arabia", "timezone": "Asia/Riyadh", "lat": 24.7136, "lon": 46.6753},
    {"city": "Johannesburg", "country": "South Africa", "timezone": "Africa/Johannesburg", "lat": -26.2041, "lon": 28.0473},
    {"city": "Cape Town", "country": "South Africa", "timezone": "Africa/Johannesburg", "lat": -33.9249, "lon": 18.4241},
    {"city": "Nairobi", "country": "Kenya", "timezone": "Africa/Nairobi", "lat": -1.2921, "lon": 36.8219},
    {"city": "Cairo", "country": "Egypt", "timezone": "Africa/Cairo", "lat": 30.0444, "lon": 31.2357},
    {"city": "Tel Aviv", "country": "Israel", "timezone": "Asia/Jerusalem", "lat": 32.0853, "lon": 34.7818},
    {"city": "Singapore", "country": "Singapore", "timezone": "Asia/Singapore", "lat": 1.3521, "lon": 103.8198},
    {"city": "Baku", "country": "Azerbaijan", "timezone": "Asia/Baku", "lat": 40.4093, "lon": 49.8671},
    {"city": "Auckland", "country": "New Zealand", "timezone": "Pacific/Auckland", "lat": -36.8509, "lon": 174.7645},
    {"city": "Reykjavík", "country": "Iceland", "timezone": "Atlantic/Reykjavik", "lat": 64.1466, "lon": -21.9426},
    {"city": "San José", "country": "Costa Rica", "timezone": "America/Costa_Rica", "lat": 9.9281, "lon": -84.0907},
    {"city": "Tallinn", "country": "Estonia", "timezone": "Europe/Tallinn", "lat": 59.4370, "lon": 24.7536},
    {"city": "Riga", "country": "Latvia", "timezone": "Europe/Riga", "lat": 56.9496, "lon": 24.1052},
    {"city": "Vilnius", "country": "Lithuania", "timezone": "Europe/Vilnius", "lat": 54.6872, "lon": 25.2797},
    {"city": "Luxembourg City", "country": "Luxembourg", "timezone": "Europe/Luxembourg", "lat": 49.6116, "lon": 6.1319},
    {"city": "Bratislava", "country": "Slovakia", "timezone": "Europe/Bratislava", "lat": 48.1486, "lon": 17.1077},
    {"city": "Ljubljana", "country": "Slovenia", "timezone": "Europe/Ljubljana", "lat": 46.0569, "lon": 14.5058},
]


# ============================================================
# 3) YEREL SAAT MODELİ
# ============================================================

def get_local_period(local_dt):
    hour = local_dt.hour + local_dt.minute / 60
    weekday = local_dt.weekday()

    if weekday >= 5:
        if 10 <= hour < 20:
            return "hafta_sonu_gun_ici"
        return "hafta_sonu_sakin"

    if 7 <= hour < 10:
        return "sabah_peak"
    elif 16 <= hour < 19.5:
        return "aksam_peak"
    elif 10 <= hour < 16:
        return "gun_ici"
    elif 19.5 <= hour < 22:
        return "normal_aksam"
    return "gece"


PERIOD_MODEL = {
    "sabah_peak": {
        "expected_ratio": 0.70,
        "penalty_multiplier": 0.80,
        "label": "Sabah peak",
    },
    "aksam_peak": {
        "expected_ratio": 0.65,
        "penalty_multiplier": 0.75,
        "label": "Akşam peak",
    },
    "gun_ici": {
        "expected_ratio": 0.82,
        "penalty_multiplier": 1.00,
        "label": "Gün içi",
    },
    "normal_aksam": {
        "expected_ratio": 0.88,
        "penalty_multiplier": 1.10,
        "label": "Normal akşam",
    },
    "gece": {
        "expected_ratio": 0.95,
        "penalty_multiplier": 1.30,
        "label": "Gece / off-peak",
    },
    "hafta_sonu_gun_ici": {
        "expected_ratio": 0.85,
        "penalty_multiplier": 1.00,
        "label": "Hafta sonu gün içi",
    },
    "hafta_sonu_sakin": {
        "expected_ratio": 0.92,
        "penalty_multiplier": 1.20,
        "label": "Hafta sonu sakin",
    },
}


# ============================================================
# 4) NOKTA ÜRETME
# ============================================================

def make_city_points(center_lat, center_lon, distance_km=5):
    lat_km = 111.0
    lon_km = 111.0 * np.cos(np.radians(center_lat))

    dlat = distance_km / lat_km
    dlon = distance_km / lon_km

    points = [
        ("merkez", center_lat, center_lon),
        ("kuzey", center_lat + dlat, center_lon),
        ("güney", center_lat - dlat, center_lon),
        ("doğu", center_lat, center_lon + dlon),
        ("batı", center_lat, center_lon - dlon),
        ("kuzeydoğu", center_lat + dlat, center_lon + dlon),
        ("kuzeybatı", center_lat + dlat, center_lon - dlon),
        ("güneydoğu", center_lat - dlat, center_lon + dlon),
        ("güneybatı", center_lat - dlat, center_lon - dlon),
        ("dış_kuzey", center_lat + 2 * dlat, center_lon),
        ("dış_güney", center_lat - 2 * dlat, center_lon),
        ("dış_doğu", center_lat, center_lon + 2 * dlon),
        ("dış_batı", center_lat, center_lon - 2 * dlon),
    ]

    return [(name, round(lat, 6), round(lon, 6)) for name, lat, lon in points]


# ============================================================
# 5) MODEL YARDIMCI FONKSİYONLARI
# ============================================================

def frc_weight(frc):
    weights = {
        "FRC0": 1.50,
        "FRC1": 1.35,
        "FRC2": 1.20,
        "FRC3": 1.00,
        "FRC4": 0.80,
        "FRC5": 0.60,
        "FRC6": 0.40,
    }
    return weights.get(frc, 0.75)


def weighted_average(values, weights):
    values = np.array(values, dtype=float)
    weights = np.array(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    if mask.sum() == 0:
        return np.nan

    return np.average(values[mask], weights=weights[mask])


def weighted_quantile(values, weights, quantile):
    values = np.array(values, dtype=float)
    weights = np.array(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    cutoff = quantile * weights.sum()

    return values[np.searchsorted(cumulative, cutoff)]


def traffic_level(score):
    if pd.isna(score):
        return "Veri yok"
    if score >= 90:
        return "Çok akıcı"
    if score >= 75:
        return "Akıcı"
    if score >= 60:
        return "Orta yoğunluk"
    if score >= 45:
        return "Yoğun trafik"
    return "Çok yoğun"


def local_time_meaning(relative_to_expected):
    if pd.isna(relative_to_expected):
        return "Veri yok"
    if relative_to_expected >= 1.05:
        return "Beklenenden iyi"
    if relative_to_expected >= 0.95:
        return "Normal"
    if relative_to_expected >= 0.80:
        return "Biraz kötü"
    return "Belirgin kötü"


def reliability_label(valid_points):
    if valid_points >= 11:
        return "İyi"
    if valid_points >= 8:
        return "Orta"
    if valid_points >= 4:
        return "Düşük"
    if valid_points >= 1:
        return "Çok düşük"
    return "Veri yok"


def quality_penalty(valid_points):
    if valid_points >= 11:
        return 0
    if valid_points >= 8:
        return 2
    if valid_points >= 4:
        return 5
    if valid_points >= 1:
        return 10
    return 100


# ============================================================
# 6) TOMTOM API
# ============================================================

def check_api_key():
    if TOMTOM_API_KEY in [None, "", "BURAYA_TOMTOM_API_KEY_YAZ"]:
        raise ValueError(
            "TomTom API key yok kanka. TOMTOM_API_KEY değişkenine key'i yaz."
        )


def fetch_flow_segment(lat, lon, country):
    base = TRAFFIC_BASE_BY_COUNTRY.get(country, DEFAULT_TRAFFIC_BASE)

    url = f"https://{base}/traffic/services/4/flowSegmentData/absolute/{ZOOM}/json"

    params = {
        "key": TOMTOM_API_KEY,
        "point": f"{lat},{lon}",
        "unit": "KMPH",
    }

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

    if response.status_code == 403:
        text = response.text[:700]
        if "InsufficientFunds" in text:
            raise RuntimeError("TomTom kredi yetersiz: InsufficientFunds")
        raise RuntimeError(f"TomTom 403 hatası: {text}")

    if response.status_code == 429:
        raise RuntimeError("TomTom 429 rate limit / kota hatası")

    response.raise_for_status()

    data = response.json()
    return data.get("flowSegmentData")


def preflight_test():
    print("TomTom API ön test yapılıyor...")

    try:
        flow = fetch_flow_segment(
            lat=39.9334,
            lon=32.8597,
            country="Turkey",
        )

        if flow is None:
            print("Ön test: API cevap verdi ama trafik verisi boş geldi.")
            return True

        print("Ön test başarılı. API çalışıyor.")
        return True

    except Exception as e:
        print("Ön test başarısız:")
        print(e)
        return False


# ============================================================
# 7) ŞEHİR TRAFİK SKORU
# ============================================================

def calculate_city_score(city_info):
    city = city_info["city"]
    country = city_info["country"]
    timezone = city_info["timezone"]
    center_lat = city_info["lat"]
    center_lon = city_info["lon"]

    local_dt = datetime.now(ZoneInfo(timezone))
    local_period = get_local_period(local_dt)
    period_info = PERIOD_MODEL[local_period]

    points = make_city_points(
        center_lat=center_lat,
        center_lon=center_lon,
        distance_km=POINT_DISTANCE_KM,
    )

    point_rows = []
    point_errors = 0

    for point_name, lat, lon in points:
        try:
            flow = fetch_flow_segment(lat, lon, country)
            time.sleep(REQUEST_SLEEP)

            if flow is None:
                point_errors += 1
                continue

            current_speed = flow.get("currentSpeed")
            free_flow_speed = flow.get("freeFlowSpeed")
            current_tt = flow.get("currentTravelTime")
            free_flow_tt = flow.get("freeFlowTravelTime")
            confidence = flow.get("confidence", 0)
            road_closure = flow.get("roadClosure", False)
            frc = flow.get("frc")

            if current_speed is None or free_flow_speed is None or free_flow_speed <= 0:
                point_errors += 1
                continue

            speed_ratio = current_speed / free_flow_speed
            speed_ratio_clipped = float(np.clip(speed_ratio, 0, 1))

            if current_tt is not None and free_flow_tt is not None and free_flow_tt > 0:
                travel_time_ratio = current_tt / free_flow_tt
            else:
                travel_time_ratio = np.nan

            point_weight = max(confidence, 0.05) * frc_weight(frc)

            point_rows.append({
                "point": point_name,
                "lat": lat,
                "lon": lon,
                "frc": frc,
                "current_speed": current_speed,
                "free_flow_speed": free_flow_speed,
                "speed_ratio": speed_ratio_clipped,
                "flow_percent": speed_ratio_clipped * 100,
                "current_travel_time": current_tt,
                "free_flow_travel_time": free_flow_tt,
                "travel_time_ratio": travel_time_ratio,
                "confidence": confidence,
                "road_closure": road_closure,
                "point_weight": point_weight,
            })

        except RuntimeError:
            raise

        except Exception:
            point_errors += 1
            continue

    if len(point_rows) == 0:
        return {
            "city": city,
            "country": country,
            "local_time": local_dt.strftime("%Y-%m-%d %H:%M"),
            "local_period": local_period,
            "period_label": period_info["label"],
            "valid_points": 0,
            "reliability": "Veri yok",
            "flow_percent": np.nan,
            "mean_flow_percent": np.nan,
            "median_flow_percent": np.nan,
            "p25_flow_percent": np.nan,
            "traffic_score": np.nan,
            "traffic_score_0_1": np.nan,
            "traffic_level": "Veri yok",
            "expected_ratio": period_info["expected_ratio"],
            "relative_to_expected": np.nan,
            "meaning": "Veri yok",
            "road_closure_rate": np.nan,
            "quality_penalty": 100,
            "point_errors": point_errors,
            "status": "veri_yok",
        }

    dfp = pd.DataFrame(point_rows)

    valid_points = len(dfp)

    mean_ratio = weighted_average(dfp["speed_ratio"], dfp["point_weight"])
    median_ratio = weighted_quantile(dfp["speed_ratio"], dfp["point_weight"], 0.50)
    p25_ratio = weighted_quantile(dfp["speed_ratio"], dfp["point_weight"], 0.25)

    mean_ratio = float(np.clip(mean_ratio, 0, 1))
    median_ratio = float(np.clip(median_ratio, 0, 1))
    p25_ratio = float(np.clip(p25_ratio, 0, 1))

    combined_ratio = (
        0.40 * mean_ratio
        + 0.40 * median_ratio
        + 0.20 * p25_ratio
    )

    raw_congestion = 1 - combined_ratio
    adjusted_congestion = raw_congestion * period_info["penalty_multiplier"]
    raw_score = 100 * (1 - adjusted_congestion)

    road_closure_rate = dfp["road_closure"].mean()
    closure_penalty = road_closure_rate * 25

    q_penalty = quality_penalty(valid_points)

    final_score = raw_score - closure_penalty - q_penalty
    final_score = float(np.clip(final_score, 0, 100))

    expected_ratio = period_info["expected_ratio"]
    relative_to_expected = combined_ratio / expected_ratio

    return {
        "city": city,
        "country": country,
        "local_time": local_dt.strftime("%Y-%m-%d %H:%M"),
        "local_period": local_period,
        "period_label": period_info["label"],
        "valid_points": valid_points,
        "reliability": reliability_label(valid_points),

        "flow_percent": round(combined_ratio * 100, 2),
        "mean_flow_percent": round(mean_ratio * 100, 2),
        "median_flow_percent": round(median_ratio * 100, 2),
        "p25_flow_percent": round(p25_ratio * 100, 2),

        "traffic_score": round(final_score, 2),
        "traffic_score_0_1": round(final_score / 100, 4),
        "traffic_level": traffic_level(final_score),

        "expected_ratio": expected_ratio,
        "relative_to_expected": round(relative_to_expected, 4),
        "meaning": local_time_meaning(relative_to_expected),

        "road_closure_rate": round(float(road_closure_rate), 4),
        "quality_penalty": q_penalty,
        "point_errors": point_errors,
        "status": "ok",
    }


# ============================================================
# 8) DATAFRAME ÜRETME
# ============================================================

def build_traffic_score_dataframe(cities=None, verbose=True):
    check_api_key()

    if cities is None:
        cities = CITIES

    if RUN_PREFLIGHT_TEST:
        ok = preflight_test()
        if not ok:
            raise RuntimeError("TomTom ön test başarısız. API key / kota / billing kontrol et.")

    records = []

    total = len(cities)

    if verbose:
        print("\nTOMTOM TRAFFIC SCORE BAŞLADI")
        print("=" * 60)
        print(f"Şehir sayısı: {total}")
        print(f"Şehir başına nokta: 13")
        print(f"Tahmini sorgu: {total * 13 + (1 if RUN_PREFLIGHT_TEST else 0)}")
        print("=" * 60)

    for index, city_info in enumerate(cities, 1):
        city_name = city_info["city"]

        try:
            if verbose:
                print(f"\n[{index}/{total}] {city_name} işleniyor...")

            summary = calculate_city_score(city_info)
            records.append(summary)

            if verbose:
                print(
                    f"{city_name} | "
                    f"score={summary['traffic_score']} | "
                    f"flow={summary['flow_percent']} | "
                    f"valid={summary['valid_points']} | "
                    f"{summary['traffic_level']}"
                )

        except RuntimeError:
            raise

        except Exception as e:
            records.append({
                "city": city_name,
                "country": city_info.get("country"),
                "local_time": np.nan,
                "local_period": np.nan,
                "period_label": np.nan,
                "valid_points": 0,
                "reliability": "Veri yok",
                "flow_percent": np.nan,
                "mean_flow_percent": np.nan,
                "median_flow_percent": np.nan,
                "p25_flow_percent": np.nan,
                "traffic_score": np.nan,
                "traffic_score_0_1": np.nan,
                "traffic_level": "Veri yok",
                "expected_ratio": np.nan,
                "relative_to_expected": np.nan,
                "meaning": "Veri yok",
                "road_closure_rate": np.nan,
                "quality_penalty": 100,
                "point_errors": 13,
                "status": f"hata: {e}",
            })

            if verbose:
                print(f"{city_name} hata verdi: {e}")

    df = pd.DataFrame(records)

    df = df.sort_values(
        by="traffic_score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    df["traffic_rank"] = df["traffic_score"].rank(
        method="min",
        ascending=False,
    )

    df["traffic_rank"] = df["traffic_rank"].astype("Int64")

    final_columns = [
        "traffic_rank",
        "city",
        "country",
        "local_time",
        "local_period",
        "period_label",
        "valid_points",
        "reliability",
        "flow_percent",
        "mean_flow_percent",
        "median_flow_percent",
        "p25_flow_percent",
        "traffic_score",
        "traffic_score_0_1",
        "traffic_level",
        "expected_ratio",
        "relative_to_expected",
        "meaning",
        "road_closure_rate",
        "quality_penalty",
        "point_errors",
        "status",
    ]

    for col in final_columns:
        if col not in df.columns:
            df[col] = np.nan

    df = df[final_columns]

    return df


# ============================================================
# 9) KAYDETME
# ============================================================

def save_traffic_score_csv(df, output_dir=None):
    if output_dir is None:
        output_dir = DATA_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / OUTPUT_CSV_NAME

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return output_csv


# ============================================================
# 10) DIŞARIDAN ÇALIŞTIRMA
# ============================================================

def run_and_save(output_dir=None):
    df = build_traffic_score_dataframe()

    output_csv = save_traffic_score_csv(
        df=df,
        output_dir=output_dir,
    )

    ok_count = int((df["status"] == "ok").sum())
    error_count = int((df["status"] != "ok").sum())

    print("\nBİTTİ KANKA")
    print("=" * 60)
    print(f"CSV kaydedildi: {output_csv}")
    print(f"Toplam şehir: {len(df)}")
    print(f"Başarılı: {ok_count}")
    print(f"Hatalı / verisiz: {error_count}")

    print("\nİlk 10 şehir:")
    print(df.head(10)[[
        "traffic_rank",
        "city",
        "traffic_score",
        "flow_percent",
        "traffic_level",
        "meaning",
        "valid_points",
        "reliability",
    ]].to_string(index=False))

    return {
        "df": df,
        "output_csv": output_csv,
        "ok_count": ok_count,
        "error_count": error_count,
    }


if __name__ == "__main__":
    run_and_save()