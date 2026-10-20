from pathlib import Path
import time

import requests
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ==============================
# AYARLAR
# ==============================

ANALYSIS_YEAR = 2025

OUTPUT_CSV_NAME = "economic_data_imf_2025_fixed.csv"
OUTPUT_EXCEL_NAME = "economic_data_imf_2025_fixed.xlsx"


# Bu dosya cities/services içinde olduğu için:
# parent.parent = cities klasörü
BASE_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_APP_DIR / "data"


# ==============================
# ŞEHİR - ÜLKE EŞLEŞTİRMESİ
# ==============================

city_to_country_iso2 = {
    "Washington D.C.": "US",
    "New York": "US",
    "Los Angeles": "US",

    "London": "GB",
    "Paris": "FR",
    "Berlin": "DE",

    "Tokyo": "JP",
    "Osaka": "JP",

    "Canberra": "AU",
    "Sydney": "AU",
    "Melbourne": "AU",
    "Ottawa": "CA",

    "Seoul": "KR",
    "Busan": "KR",

    "Rome": "IT",
    "Madrid": "ES",
    "Amsterdam": "NL",
    "Brussels": "BE",
    "Vienna": "AT",
    "Bern": "CH",
    "Stockholm": "SE",
    "Oslo": "NO",
    "Copenhagen": "DK",
    "Helsinki": "FI",
    "Dublin": "IE",
    "Lisbon": "PT",
    "Athens": "GR",
    "Warsaw": "PL",
    "Prague": "CZ",
    "Budapest": "HU",

    "Mexico City": "MX",
    "Monterrey": "MX",

    "Brasília": "BR",
    "São Paulo": "BR",

    "Buenos Aires": "AR",
    "Santiago": "CL",
    "Bogotá": "CO",
    "Lima": "PE",

    "Ankara": "TR",
    "Istanbul": "TR",

    "Moscow": "RU",
    "Saint Petersburg": "RU",

    "Beijing": "CN",
    "Shanghai": "CN",

    "New Delhi": "IN",
    "Mumbai": "IN",

    "Jakarta": "ID",
    "Riyadh": "SA",

    "Johannesburg": "ZA",
    "Cape Town": "ZA",

    "Nairobi": "KE",
    "Cairo": "EG",
    "Tel Aviv": "IL",

    "Singapore": "SG",
    "Baku": "AZ",
    "Auckland": "NZ",

    "Reykjavík": "IS",
    "San José": "CR",

    "Tallinn": "EE",
    "Riga": "LV",
    "Vilnius": "LT",

    "Luxembourg City": "LU",
    "Bratislava": "SK",
    "Ljubljana": "SI",
}


# ==============================
# ISO2 -> IMF ISO3
# ==============================

iso2_to_iso3 = {
    "US": "USA",
    "GB": "GBR",
    "FR": "FRA",
    "DE": "DEU",
    "JP": "JPN",
    "AU": "AUS",
    "CA": "CAN",
    "KR": "KOR",
    "IT": "ITA",
    "ES": "ESP",
    "NL": "NLD",
    "BE": "BEL",
    "AT": "AUT",
    "CH": "CHE",
    "SE": "SWE",
    "NO": "NOR",
    "DK": "DNK",
    "FI": "FIN",
    "IE": "IRL",
    "PT": "PRT",
    "GR": "GRC",
    "PL": "POL",
    "CZ": "CZE",
    "HU": "HUN",
    "MX": "MEX",
    "BR": "BRA",
    "AR": "ARG",
    "CL": "CHL",
    "CO": "COL",
    "PE": "PER",
    "TR": "TUR",
    "RU": "RUS",
    "CN": "CHN",
    "IN": "IND",
    "ID": "IDN",
    "SA": "SAU",
    "ZA": "ZAF",
    "KE": "KEN",
    "EG": "EGY",
    "IL": "ISR",
    "SG": "SGP",
    "AZ": "AZE",
    "NZ": "NZL",
    "IS": "ISL",
    "CR": "CRI",
    "EE": "EST",
    "LV": "LVA",
    "LT": "LTU",
    "LU": "LUX",
    "SK": "SVK",
    "SI": "SVN",
}


# ==============================
# IMF GÖSTERGELERİ
# ==============================

indicators = {
    "GDP_per_capita": "NGDPDPC",
    "Inflation": "PCPIPCH",
    "Unemployment": "LUR",
}


# ==============================
# REQUEST SESSION + RETRY
# ==============================

def create_session():
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


# ==============================
# IMF VERİ ÇEKME
# ==============================

def get_imf_indicator(session, country_iso3, indicator_code, year):
    url = (
        "https://www.imf.org/external/datamapper/api/v1/"
        f"{indicator_code}/{country_iso3}?periods={year}"
    )

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()

        value = (
            data.get("values", {})
            .get(indicator_code, {})
            .get(country_iso3, {})
            .get(str(year))
        )

        if value is None:
            return None

        return float(value)

    except Exception as e:
        print(f"Hata: {country_iso3} - {indicator_code} - {year} -> {e}")
        return None


# ==============================
# NORMALİZASYON
# ==============================

def normalize_positive(series):
    """
    Değer yükseldikçe skor yükselir.
    Örnek: GDP per capita.
    """
    series = pd.to_numeric(series, errors="coerce")

    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series(np.nan, index=series.index)

    if max_val == min_val:
        return pd.Series(0.5, index=series.index)

    return (series - min_val) / (max_val - min_val)


def normalize_negative(series):
    """
    Değer yükseldikçe skor düşer.
    Örnek: enflasyon, işsizlik.
    """
    series = pd.to_numeric(series, errors="coerce")

    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series(np.nan, index=series.index)

    if max_val == min_val:
        return pd.Series(0.5, index=series.index)

    return 1 - ((series - min_val) / (max_val - min_val))


# ==============================
# DATAFRAME ÜRETİCİ
# ==============================

def build_economic_dataframe():
    session = create_session()

    results = []
    country_cache = {}

    for city, iso2 in city_to_country_iso2.items():
        print(f"Çekiliyor: {city}")

        country_iso3 = iso2_to_iso3.get(iso2)

        if country_iso3 is None:
            print(f"Ülke kodu bulunamadı: {city} - {iso2}")
            continue

        # Aynı ülke için tekrar tekrar API'ye gitmeyelim.
        if country_iso3 not in country_cache:
            gdp_per_capita = get_imf_indicator(
                session=session,
                country_iso3=country_iso3,
                indicator_code=indicators["GDP_per_capita"],
                year=ANALYSIS_YEAR,
            )

            inflation = get_imf_indicator(
                session=session,
                country_iso3=country_iso3,
                indicator_code=indicators["Inflation"],
                year=ANALYSIS_YEAR,
            )

            unemployment = get_imf_indicator(
                session=session,
                country_iso3=country_iso3,
                indicator_code=indicators["Unemployment"],
                year=ANALYSIS_YEAR,
            )

            country_cache[country_iso3] = {
                "GDP per capita USD": gdp_per_capita,
                "Inflation (%)": inflation,
                "Unemployment (%)": unemployment,
            }

            time.sleep(0.5)

        country_data = country_cache[country_iso3]

        data_complete = (
            country_data["GDP per capita USD"] is not None
            and country_data["Inflation (%)"] is not None
            and country_data["Unemployment (%)"] is not None
        )

        results.append({
            "City": city,
            "Country ISO2": iso2,
            "Country ISO3": country_iso3,
            "Analysis Year": ANALYSIS_YEAR,

            "GDP per capita USD": country_data["GDP per capita USD"],
            "Inflation (%)": country_data["Inflation (%)"],
            "Unemployment (%)": country_data["Unemployment (%)"],

            "Data Frequency": "Annual",
            "Data Source": "IMF World Economic Outlook",
            "Data Status": "2025 IMF WEO estimate/projection",
            "Data Complete": data_complete,
        })

    df = pd.DataFrame(results)

    numeric_columns = [
        "GDP per capita USD",
        "Inflation (%)",
        "Unemployment (%)",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ==============================
    # SKOR HESAPLAMA
    # ==============================

    df["GDP Score"] = normalize_positive(df["GDP per capita USD"])
    df["Inflation Score"] = normalize_negative(df["Inflation (%)"])
    df["Unemployment Score"] = normalize_negative(df["Unemployment (%)"])

    GDP_WEIGHT = 40
    INFLATION_WEIGHT = 30
    UNEMPLOYMENT_WEIGHT = 30

    df["Economic Score"] = (
        df["GDP Score"] * GDP_WEIGHT
        + df["Inflation Score"] * INFLATION_WEIGHT
        + df["Unemployment Score"] * UNEMPLOYMENT_WEIGHT
    ).round(2)

    # Ara skorları 0-100 yapıyoruz.
    df["GDP Score"] = (df["GDP Score"] * 100).round(2)
    df["Inflation Score"] = (df["Inflation Score"] * 100).round(2)
    df["Unemployment Score"] = (df["Unemployment Score"] * 100).round(2)

    df = df.sort_values(by="Economic Score", ascending=False)

    quality_report = {
        "row_count": len(df),
        "missing_values": df.isna().sum().to_dict(),
        "missing_rows": df[df["Data Complete"] == False][
            [
                "City",
                "Country ISO3",
                "GDP per capita USD",
                "Inflation (%)",
                "Unemployment (%)",
            ]
        ].to_dict(orient="records"),
        "duplicate_city_count": int(df["City"].duplicated().sum()),
        "min_economic_score": df["Economic Score"].min(),
        "max_economic_score": df["Economic Score"].max(),
    }

    return df, quality_report


# ==============================
# CSV / EXCEL KAYDETME
# ==============================

def save_economic_outputs(df, output_dir=None):
    if output_dir is None:
        output_dir = DATA_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / OUTPUT_CSV_NAME
    output_excel = output_dir / OUTPUT_EXCEL_NAME

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    df.to_excel(output_excel, index=False, engine="openpyxl")

    return output_csv, output_excel


# ==============================
# DIŞARIDAN ÇALIŞTIRILACAK ANA FONKSİYON
# ==============================

def run_and_save(output_dir=None):
    df, quality_report = build_economic_dataframe()
    output_csv, output_excel = save_economic_outputs(df, output_dir)

    print("\nBitti kanka! Ekonomik veri hazır.")
    print(f"CSV: {output_csv}")
    print(f"Excel: {output_excel}")

    print("\nİlk 10 satır:")
    print(df.head(10))

    print("\nEksik veri olan satırlar:")
    if len(quality_report["missing_rows"]) == 0:
        print("Eksik veri yok kral, her şey tamam.")
    else:
        print(quality_report["missing_rows"])

    return {
        "df": df,
        "quality_report": quality_report,
        "output_csv": output_csv,
        "output_excel": output_excel,
    }


# Bu dosyayı direkt çalıştırırsan da çalışsın diye:
if __name__ == "__main__":
    run_and_save()