from pathlib import Path
import time
import re
import unicodedata
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


# ============================================================
# AYARLAR
# ============================================================

YEAR = 2025
START_DATE = f"{YEAR}-01-01"
END_DATE = f"{YEAR}-12-31"

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

PRECIP_DAY_THRESHOLD_INCH = 0.04

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ============================================================
# MANUEL 2025 NÜFUS VERİSİ
# ============================================================

MANUAL_POPULATION_2025 = {
    "Washington D.C.": 6500000,
    "New York": 13920148,
    "Los Angeles": 12740420,
    "London": 10416420,
    "Paris": 11200000,
    "Berlin": 4600000,
    "Tokyo": 33412512,
    "Osaka": 12964145,
    "Canberra": 500000,
    "Sydney": 5400000,
    "Melbourne": 5300000,
    "Ottawa": 1500000,
    "Seoul": 22500000,
    "Busan": 3400000,
    "Rome": 4300000,
    "Madrid": 6800000,
    "Amsterdam": 1200000,
    "Brussels": 2100000,
    "Vienna": 2000000,
    "Bern": 440000,
    "Stockholm": 2400000,
    "Oslo": 1100000,
    "Copenhagen": 1400000,
    "Helsinki": 1300000,
    "Dublin": 1600000,
    "Lisbon": 3000000,
    "Athens": 3200000,
    "Warsaw": 3100000,
    "Prague": 1400000,
    "Budapest": 3000000,
    "Mexico City": 22900000,
    "Monterrey": 5300000,
    "Brasília": 4900000,
    "São Paulo": 22990000,
    "Buenos Aires": 14017736,
    "Santiago": 7100000,
    "Bogotá": 10624315,
    "Lima": 10580241,
    "Ankara": 5910306,
    "Istanbul": 15800000,
    "Moscow": 13400000,
    "Saint Petersburg": 5652922,
    "Beijing": 21600000,
    "Shanghai": 29558908,
    "New Delhi": 30222405,
    "Mumbai": 21800000,
    "Jakarta": 41913860,
    "Riyadh": 7442772,
    "Johannesburg": 6200000,
    "Cape Town": 4852662,
    "Nairobi": 5412741,
    "Cairo": 25600000,
    "Tel Aviv": 4000000,
    "Singapore": 5638700,
    "Baku": 2500000,
    "Auckland": 1700000,
    "Reykjavík": 250000,
    "San José": 1800000,
    "Tallinn": 620000,
    "Riga": 1000000,
    "Vilnius": 720000,
    "Luxembourg City": 210000,
    "Bratislava": 700000,
    "Ljubljana": 550000,
}


# ============================================================
# ŞEHİR LİSTESİ
# ============================================================

CITY_LIST = [
    {"city": "Washington D.C.", "country_code": "US"},
    {"city": "New York", "country_code": "US"},
    {"city": "Los Angeles", "country_code": "US"},
    {"city": "London", "country_code": "GB"},
    {"city": "Paris", "country_code": "FR"},
    {"city": "Berlin", "country_code": "DE"},
    {"city": "Tokyo", "country_code": "JP"},
    {"city": "Osaka", "country_code": "JP"},
    {"city": "Canberra", "country_code": "AU"},
    {"city": "Sydney", "country_code": "AU"},
    {"city": "Melbourne", "country_code": "AU"},
    {"city": "Ottawa", "country_code": "CA"},
    {"city": "Seoul", "country_code": "KR"},
    {"city": "Busan", "country_code": "KR"},
    {"city": "Rome", "country_code": "IT"},
    {"city": "Madrid", "country_code": "ES"},
    {"city": "Amsterdam", "country_code": "NL"},
    {"city": "Brussels", "country_code": "BE"},
    {"city": "Vienna", "country_code": "AT"},
    {"city": "Bern", "country_code": "CH"},
    {"city": "Stockholm", "country_code": "SE"},
    {"city": "Oslo", "country_code": "NO"},
    {"city": "Copenhagen", "country_code": "DK"},
    {"city": "Helsinki", "country_code": "FI"},
    {"city": "Dublin", "country_code": "IE"},
    {"city": "Lisbon", "country_code": "PT"},
    {"city": "Athens", "country_code": "GR"},
    {"city": "Warsaw", "country_code": "PL"},
    {"city": "Prague", "country_code": "CZ"},
    {"city": "Budapest", "country_code": "HU"},
    {"city": "Mexico City", "country_code": "MX"},
    {"city": "Monterrey", "country_code": "MX"},
    {"city": "Brasília", "country_code": "BR"},
    {"city": "São Paulo", "country_code": "BR"},
    {"city": "Buenos Aires", "country_code": "AR"},
    {"city": "Santiago", "country_code": "CL"},
    {"city": "Bogotá", "country_code": "CO"},
    {"city": "Lima", "country_code": "PE"},
    {"city": "Ankara", "country_code": "TR"},
    {"city": "Istanbul", "country_code": "TR"},
    {"city": "Moscow", "country_code": "RU"},
    {"city": "Saint Petersburg", "country_code": "RU"},
    {"city": "Beijing", "country_code": "CN"},
    {"city": "Shanghai", "country_code": "CN"},
    {"city": "New Delhi", "country_code": "IN"},
    {"city": "Mumbai", "country_code": "IN"},
    {"city": "Jakarta", "country_code": "ID"},
    {"city": "Riyadh", "country_code": "SA"},
    {"city": "Johannesburg", "country_code": "ZA"},
    {"city": "Cape Town", "country_code": "ZA"},
    {"city": "Nairobi", "country_code": "KE"},
    {"city": "Cairo", "country_code": "EG"},
    {"city": "Tel Aviv", "country_code": "IL"},
    {"city": "Singapore", "country_code": "SG"},
    {"city": "Baku", "country_code": "AZ"},
    {"city": "Auckland", "country_code": "NZ"},
    {"city": "Reykjavík", "country_code": "IS"},
    {"city": "San José", "country_code": "CR"},
    {"city": "Tallinn", "country_code": "EE"},
    {"city": "Riga", "country_code": "LV"},
    {"city": "Vilnius", "country_code": "LT"},
    {"city": "Luxembourg City", "country_code": "LU"},
    {"city": "Bratislava", "country_code": "SK"},
    {"city": "Ljubljana", "country_code": "SI"},
]


# ============================================================
# MANUEL KOORDİNAT YEDEKLERİ
# ============================================================

MANUAL_CITY_COORDS = {
    "San José": {
        "City Name": "San José",
        "Country": "Costa Rica",
        "ISO2": "CR",
        "Latitude": 9.9281,
        "Longitude": -84.0907,
        "Elevation (m)": np.nan,
        "Timezone": "America/Costa_Rica",
    },
    "Luxembourg City": {
        "City Name": "Luxembourg City",
        "Country": "Luxembourg",
        "ISO2": "LU",
        "Latitude": 49.6116,
        "Longitude": 6.1319,
        "Elevation (m)": np.nan,
        "Timezone": "Europe/Luxembourg",
    },
}


# ============================================================
# WIKIPEDIA AREA SCRAPER
# ============================================================

CITY_URL_OVERRIDES = {
    "New York": "https://en.wikipedia.org/wiki/New_York_City",
    "Washington D.C.": "https://en.wikipedia.org/wiki/Washington,_D.C.",
    "San José": "https://en.wikipedia.org/wiki/San_Jos%C3%A9,_Costa_Rica",
    "Luxembourg City": "https://en.wikipedia.org/wiki/Luxembourg_City",
    "São Paulo": "https://en.wikipedia.org/wiki/S%C3%A3o_Paulo",
    "Brasília": "https://en.wikipedia.org/wiki/Bras%C3%ADlia",
    "Bogotá": "https://en.wikipedia.org/wiki/Bogot%C3%A1",
    "Reykjavík": "https://en.wikipedia.org/wiki/Reykjav%C3%ADk",
    "Mexico City": "https://en.wikipedia.org/wiki/Mexico_City",
    "Saint Petersburg": "https://en.wikipedia.org/wiki/Saint_Petersburg",
    "New Delhi": "https://en.wikipedia.org/wiki/New_Delhi",
    "Tel Aviv": "https://en.wikipedia.org/wiki/Tel_Aviv",
}


def city_name_to_wikipedia_url(city_name):
    if city_name in CITY_URL_OVERRIDES:
        return CITY_URL_OVERRIDES[city_name]

    safe_name = quote(city_name.replace(" ", "_"))
    return f"https://en.wikipedia.org/wiki/{safe_name}"


def web_page_to_soup(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)

        if response.status_code != 200:
            print(f"Wikipedia HTTP {response.status_code}: {url}")
            return None

        return BeautifulSoup(response.text, "html.parser")

    except Exception as e:
        print(f"Wikipedia fetch error: {url} - {e}")
        return None


def find_infobox(soup):
    if soup is None:
        return None

    for table in soup.find_all("table"):
        classes = table.get("class", [])
        if "infobox" in classes and "vcard" in classes:
            return table

    for table in soup.find_all("table"):
        classes = table.get("class", [])
        if "infobox" in classes:
            return table

    return None


def _extract_sq_mi(text):
    text = unicodedata.normalize("NFKD", text)

    if "/sq" in text or "per sq" in text.lower() or "density" in text.lower():
        return np.nan

    if "sq" not in text:
        return np.nan

    match = re.search(r"([\d,]+\.?\d*)\s*sq", text)

    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except Exception:
            return np.nan

    return np.nan


def area_from_soup(soup):
    infobox = find_infobox(soup)

    if infobox is None:
        return np.nan

    for row in infobox.find_all("tr"):
        row_text = unicodedata.normalize("NFKD", row.get_text(" ", strip=True))

        if "sq" not in row_text:
            continue

        if "/sq" in row_text or "per sq" in row_text.lower() or "density" in row_text.lower():
            continue

        if "sq mi" in row_text or "sq\u202fmi" in row_text:
            val = _extract_sq_mi(row_text)

            if not pd.isna(val):
                return val

    return np.nan


def get_area_from_wikipedia(city_name):
    url = city_name_to_wikipedia_url(city_name)
    soup = web_page_to_soup(url)
    area = area_from_soup(soup)

    return area, url


# ============================================================
# OPEN-METEO GEOCODING
# ============================================================

def get_city_location(city, country_code):
    params = {
        "name": city,
        "count": 10,
        "language": "en",
        "format": "json",
        "countryCode": country_code,
    }

    try:
        response = requests.get(GEOCODING_URL, params=params, timeout=20)
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if len(results) == 0:
            print(f"Open-Meteo konum bulunamadı: {city}")

            if city in MANUAL_CITY_COORDS:
                print(f"Manuel koordinat kullanılıyor: {city}")
                return MANUAL_CITY_COORDS[city]

            return None

        city_lower = city.lower().replace(".", "").strip()
        selected = None

        for result in results:
            result_name = str(result.get("name", "")).lower().replace(".", "").strip()
            result_country = result.get("country_code")

            if result_name == city_lower and result_country == country_code:
                selected = result
                break

        if selected is None:
            selected = results[0]

        return {
            "City Name": city,
            "Country": selected.get("country"),
            "ISO2": selected.get("country_code"),
            "Latitude": selected.get("latitude"),
            "Longitude": selected.get("longitude"),
            "Elevation (m)": selected.get("elevation"),
            "Timezone": selected.get("timezone"),
        }

    except Exception as e:
        print(f"Geocoding hatası: {city} - {e}")

        if city in MANUAL_CITY_COORDS:
            print(f"Manuel koordinat kullanılıyor: {city}")
            return MANUAL_CITY_COORDS[city]

        return None


# ============================================================
# OPEN-METEO 2025 WEATHER
# ============================================================

def get_2025_weather_summary(latitude, longitude, timezone):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "sunshine_duration",
        ]),
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": timezone if timezone else "auto",
    }

    try:
        response = requests.get(ARCHIVE_URL, params=params, timeout=40)
        response.raise_for_status()

        data = response.json()

        if "daily" not in data:
            print("Daily weather verisi bulunamadı.")
            return None

        daily = pd.DataFrame(data["daily"])

        numeric_cols = [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "sunshine_duration",
        ]

        for col in numeric_cols:
            if col not in daily.columns:
                daily[col] = np.nan

            daily[col] = pd.to_numeric(daily[col], errors="coerce")

        avg_high = daily["temperature_2m_max"].mean()
        avg_low = daily["temperature_2m_min"].mean()
        total_precip = daily["precipitation_sum"].sum(skipna=True)

        precip_days = (
            daily["precipitation_sum"] >= PRECIP_DAY_THRESHOLD_INCH
        ).sum()

        sunshine_hours = daily["sunshine_duration"].sum(skipna=True) / 3600

        return {
            "Avg High Temp 2025 (F)": round(avg_high, 2) if not pd.isna(avg_high) else np.nan,
            "Avg Low Temp 2025 (F)": round(avg_low, 2) if not pd.isna(avg_low) else np.nan,
            "Annual Precipitation Days 2025": int(precip_days),
            "Annual Precipitation 2025 (in)": round(total_precip, 2),
            "Annual Sunshine Hours 2025": round(sunshine_hours, 2),
        }

    except Exception as e:
        print(f"Weather API hatası: {latitude}, {longitude} - {e}")
        return None



# ============================================================
# CLIMATE & DEMOGRAPHIC COMFORT SCORE (0-1)
# ============================================================

SCORE_COLUMN = "Climate Demographic Comfort Score"

# Ağırlıklar toplamı 1.00 olacak şekilde kuruldu.
# Not: Her alt skor önce 0-1 arasına çevrilir, sonra bu ağırlıklarla birleşir.
SCORE_WEIGHTS = {
    "density_score": 0.25,        # Population / Area -> yoğunluk arttıkça risk artar
    "population_score": 0.05,     # Büyük nüfus hafif baskı kabul edilir
    "high_temp_score": 0.21,      # Yıllık ort. yüksek sıcaklık konforu
    "low_temp_score": 0.14,       # Yıllık ort. düşük sıcaklık konforu
    "sunshine_score": 0.20,       # Güneşlenme süresi pozitif
    "rain_days_score": 0.07,      # Yağışlı gün sayısı negatif
    "precip_score": 0.03,         # Toplam yağış miktarı negatif
    "elevation_score": 0.05,      # Çok yüksek rakım hafif negatif
}


def _to_numeric(series):
    """Kolonu güvenli şekilde numeric seriye çevirir."""
    return pd.to_numeric(series, errors="coerce")


def _winsor_bounds(series, lower_q=0.05, upper_q=0.95):
    """
    Uç değerlerin skoru aşırı bozmasını engellemek için 5%-95% sınırları kullanır.
    Veri sayısı azsa min-max'a döner.
    """
    s = _to_numeric(series).replace([np.inf, -np.inf], np.nan).dropna()

    if len(s) == 0:
        return np.nan, np.nan

    if len(s) < 10:
        lower = s.min()
        upper = s.max()
    else:
        lower = s.quantile(lower_q)
        upper = s.quantile(upper_q)

    if pd.isna(lower) or pd.isna(upper) or lower == upper:
        lower = s.min()
        upper = s.max()

    return lower, upper


def benefit_score_0_1(series, lower=None, upper=None):
    """
    Yüksek değer iyiyse 0-1 skoru üretir.
    Örn: Sunshine Hours yüksekse iyi.
    """
    s = _to_numeric(series).replace([np.inf, -np.inf], np.nan)

    if lower is None or upper is None:
        lower, upper = _winsor_bounds(s)

    if pd.isna(lower) or pd.isna(upper) or lower == upper:
        return pd.Series(np.nan, index=s.index, dtype=float)

    return ((s - lower) / (upper - lower)).clip(0, 1)


def cost_score_0_1(series, lower=None, upper=None):
    """
    Yüksek değer kötüyse 0-1 skoru üretir.
    Örn: Nüfus yoğunluğu, yağışlı gün, toplam yağış yüksekse kötü.
    """
    return 1 - benefit_score_0_1(series, lower=lower, upper=upper)


def range_score_0_1(series, ideal_low, ideal_high, hard_low, hard_high):
    """
    İdeal aralıkta 1, aşırı düşük/yüksek değerlerde 0'a yaklaşan skor üretir.
    Sıcaklık gibi 'ne çok düşük ne çok yüksek' olması gereken değişkenlerde kullanılır.
    """
    s = _to_numeric(series).replace([np.inf, -np.inf], np.nan)
    score = pd.Series(np.nan, index=s.index, dtype=float)

    # İdeal aralık -> tam puan
    ideal_mask = (s >= ideal_low) & (s <= ideal_high)
    score.loc[ideal_mask] = 1.0

    # İdealden düşük ama tolerans içinde -> doğrusal artış
    low_mask = (s >= hard_low) & (s < ideal_low)
    score.loc[low_mask] = (s.loc[low_mask] - hard_low) / (ideal_low - hard_low)

    # İdealden yüksek ama tolerans içinde -> doğrusal düşüş
    high_mask = (s > ideal_high) & (s <= hard_high)
    score.loc[high_mask] = (hard_high - s.loc[high_mask]) / (hard_high - ideal_high)

    # Hard sınırların dışı -> 0 puan
    outside_mask = (s < hard_low) | (s > hard_high)
    score.loc[outside_mask] = 0.0

    return score.clip(0, 1)


def elevation_comfort_score_0_1(elevation_m):
    """
    0-1000 m arası tam puan.
    1000-2500 m arasında puan kademeli düşer.
    2500 m ve üzeri 0'a yaklaşır.
    """
    e = _to_numeric(elevation_m).replace([np.inf, -np.inf], np.nan)
    score = pd.Series(np.nan, index=e.index, dtype=float)

    score.loc[e <= 1000] = 1.0

    mid_mask = (e > 1000) & (e < 2500)
    score.loc[mid_mask] = 1 - ((e.loc[mid_mask] - 1000) / 1500)

    score.loc[e >= 2500] = 0.0

    return score.clip(0, 1)


def weighted_score_0_1(component_scores, weights):
    """
    Eksik alt skor varsa final skoru tamamen NaN yapmaz.
    O şehir için mevcut alt skorların ağırlıklarını yeniden 1'e ölçekler.
    """
    score_df = pd.DataFrame(component_scores)
    weight_series = pd.Series(weights, dtype=float)

    weighted_values = score_df.mul(weight_series, axis=1)
    available_weight = score_df.notna().mul(weight_series, axis=1).sum(axis=1)

    final_score = weighted_values.sum(axis=1, skipna=True) / available_weight
    final_score = final_score.where(available_weight > 0, np.nan)

    return final_score.clip(0, 1)


def add_climate_demographic_score(df):
    """
    Mevcut veri setine 0-1 arası Climate Demographic Comfort Score kolonu ekler.

    Model:
    score =
        0.25 * population_density_score
      + 0.05 * population_scale_score
      + 0.21 * avg_high_temp_comfort_score
      + 0.14 * avg_low_temp_comfort_score
      + 0.20 * sunshine_score
      + 0.07 * rainy_days_score
      + 0.03 * precipitation_amount_score
      + 0.05 * elevation_comfort_score
    """
    df = df.copy()

    population = _to_numeric(df.get("Population", pd.Series(index=df.index, dtype=float)))
    area = _to_numeric(df.get("Area (sq mi)", pd.Series(index=df.index, dtype=float)))

    # Nüfus yoğunluğu: kapsamlar uyumlu olmalı.
    # Not: Metro nüfusu / city-proper alanı kullanılırsa yoğunluk şişer.
    population_density = population / area.replace(0, np.nan)

    density_score = cost_score_0_1(np.log1p(population_density))
    population_score = cost_score_0_1(np.log1p(population))

    high_temp_score = range_score_0_1(
        df.get("Avg High Temp 2025 (F)", pd.Series(index=df.index, dtype=float)),
        ideal_low=64,
        ideal_high=77,
        hard_low=45,
        hard_high=95,
    )

    low_temp_score = range_score_0_1(
        df.get("Avg Low Temp 2025 (F)", pd.Series(index=df.index, dtype=float)),
        ideal_low=45,
        ideal_high=60,
        hard_low=20,
        hard_high=80,
    )

    sunshine_score = benefit_score_0_1(
        df.get("Annual Sunshine Hours 2025", pd.Series(index=df.index, dtype=float))
    )

    rain_days_score = cost_score_0_1(
        df.get("Annual Precipitation Days 2025", pd.Series(index=df.index, dtype=float))
    )

    precip_score = cost_score_0_1(
        df.get("Annual Precipitation 2025 (in)", pd.Series(index=df.index, dtype=float))
    )

    elevation_score = elevation_comfort_score_0_1(
        df.get("Elevation (m)", pd.Series(index=df.index, dtype=float))
    )

    component_scores = {
        "density_score": density_score,
        "population_score": population_score,
        "high_temp_score": high_temp_score,
        "low_temp_score": low_temp_score,
        "sunshine_score": sunshine_score,
        "rain_days_score": rain_days_score,
        "precip_score": precip_score,
        "elevation_score": elevation_score,
    }

    df[SCORE_COLUMN] = weighted_score_0_1(component_scores, SCORE_WEIGHTS).round(4)

    return df


# ============================================================
# ANA DATAFRAME ÜRETİCİ
# ============================================================

def build_city_weather_dataframe():
    results = []
    total = len(CITY_LIST)
    missing_population = []

    for idx, item in enumerate(CITY_LIST, 1):
        city = item["city"]
        country_code = item["country_code"]

        print(f"\n[{idx}/{total}] {city} - {country_code}")

        location = get_city_location(city, country_code)

        if location is None:
            print(f"SKIP: Konum alınamadı: {city}")
            continue

        weather = get_2025_weather_summary(
            latitude=location["Latitude"],
            longitude=location["Longitude"],
            timezone=location["Timezone"],
        )

        if weather is None:
            weather = {
                "Avg High Temp 2025 (F)": np.nan,
                "Avg Low Temp 2025 (F)": np.nan,
                "Annual Precipitation Days 2025": np.nan,
                "Annual Precipitation 2025 (in)": np.nan,
                "Annual Sunshine Hours 2025": np.nan,
            }

        area_sq_mi, wiki_url = get_area_from_wikipedia(city)

        population_2025 = MANUAL_POPULATION_2025.get(city, np.nan)

        if pd.isna(population_2025):
            missing_population.append(city)

        row = {
            "City Name": location["City Name"],
            "Country": location["Country"],
            "ISO2": location["ISO2"],
            "Analysis Year": YEAR,

            "Latitude": location["Latitude"],
            "Longitude": location["Longitude"],
            "Elevation (m)": location["Elevation (m)"],

            "Population": population_2025,
            "Population Year": YEAR,
            "Population Source": "Manual 2025 estimate",
            "Population Scope": "Urban agglomeration / metro-style estimate",

            "Area (sq mi)": area_sq_mi,
            "Area Source": "Wikipedia infobox",
            "Area Note": (
                "Area is usually city proper / administrative area; "
                "do not calculate density with metro population unless scope is aligned."
            ),

            **weather,

            "Data Frequency": "Annual",
            "Weather Source": "Open-Meteo Historical Weather API",
            "Weather Date Range": f"{START_DATE} to {END_DATE}",
            "Timezone": location["Timezone"],
            "Wikipedia URL": wiki_url,
        }

        results.append(row)

        time.sleep(1)

    df = pd.DataFrame(results)

    df = add_climate_demographic_score(df)

    column_order = [
        "City Name",
        "Country",
        "ISO2",
        "Analysis Year",

        "Latitude",
        "Longitude",
        "Elevation (m)",

        "Population",
        "Population Year",
        "Population Source",
        "Population Scope",

        "Area (sq mi)",
        "Area Source",
        "Area Note",

        "Avg High Temp 2025 (F)",
        "Avg Low Temp 2025 (F)",
        "Annual Precipitation Days 2025",
        "Annual Precipitation 2025 (in)",
        "Annual Sunshine Hours 2025",
        SCORE_COLUMN,

        "Data Frequency",
        "Weather Source",
        "Weather Date Range",
        "Timezone",
        "Wikipedia URL",
    ]

    df = df[column_order]

    quality_report = {
        "row_count": len(df),
        "missing_values": df.isna().sum().to_dict(),
        "missing_population": missing_population,
        "missing_area": df[df["Area (sq mi)"].isna()]["City Name"].tolist(),
        "duplicate_city_count": int(df["City Name"].duplicated().sum()),
        "temperature_problem_count": int(
            (df["Avg High Temp 2025 (F)"] < df["Avg Low Temp 2025 (F)"]).sum()
        ),
        "precipitation_problem_count": int(
            (
                (df["Annual Precipitation Days 2025"] < 0)
                | (df["Annual Precipitation Days 2025"] > 366)
            ).sum()
        ),
        "sunshine_problem_count": int(
            (
                (df["Annual Sunshine Hours 2025"] < 0)
                | (df["Annual Sunshine Hours 2025"] > 8760)
            ).sum()
        ),
    }

    return df, quality_report


# ============================================================
# CSV / EXCEL KAYDETME
# ============================================================

def save_city_weather_outputs(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "cities_weather_area_population_2025.csv"
    output_excel = output_dir / "cities_weather_area_population_2025.xlsx"

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    df.to_excel(output_excel, index=False, engine="openpyxl")

    return output_csv, output_excel


# ============================================================
# TEST AMAÇLI ÇALIŞTIRMA
# ============================================================

def run_and_save(output_dir=None):
    if output_dir is None:
        output_dir = Path.cwd() / "data"

    df, quality_report = build_city_weather_dataframe()
    output_csv, output_excel = save_city_weather_outputs(df, output_dir)

    return {
        "df": df,
        "quality_report": quality_report,
        "output_csv": output_csv,
        "output_excel": output_excel,
    }