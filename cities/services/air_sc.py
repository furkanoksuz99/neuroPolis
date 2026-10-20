from pathlib import Path
import os
import time

import requests
import pandas as pd
import numpy as np


# ============================================================
# AYARLAR
# ============================================================

BASE_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_APP_DIR / "data"

OUTPUT_CSV_NAME = "global_aqi_raporu.csv"

# Güvenli kullanım:
# Terminalde token vermek istersen:
# set WAQI_TOKEN=senin_tokenin
#
# Direkt buraya da yazabilirsin:
TOKEN = os.getenv("WAQI_TOKEN", "16afffdcdea31743024ad7e390f9691ef959c1d6")

REQUEST_TIMEOUT = 10
SLEEP_SECONDS = 1


# ============================================================
# ŞEHİR LİSTESİ
# ============================================================

CITIES = [
    "washington", "new york", "los angeles",
    "london", "paris", "berlin",
    "tokyo", "osaka",
    "canberra", "sydney", "melbourne",
    "ottawa",
    "seoul", "busan",
    "rome", "madrid",
    "amsterdam", "brussels",
    "vienna", "bern",
    "stockholm", "oslo",
    "copenhagen", "helsinki",
    "dublin", "lisbon",
    "athens", "warsaw",
    "prague", "budapest",
    "mexico city", "monterrey",
    "brasilia", "sao paulo",
    "buenos aires", "santiago",
    "bogota", "lima",
    "ankara", "istanbul",
    "moscow", "saint petersburg",
    "beijing", "shanghai",
    "new delhi", "mumbai",
    "jakarta",
    "riyadh",
    "johannesburg", "cape town",
    "nairobi",
    "cairo",
    "tel aviv",
    "singapore",
    "baku",
    "auckland",
    "reykjavik",
    "san jose",
    "tallinn",
    "riga",
    "vilnius",
    "luxembourg",
    "bratislava",
    "ljubljana",
]


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def temiz_sayi(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def iaqi_al(data, degisken):
    """
    WAQI iaqi içinden pm25, pm10, sıcaklık, nem vb. değerleri güvenli çeker.
    """
    try:
        return temiz_sayi(data.get("iaqi", {}).get(degisken, {}).get("v", np.nan))
    except Exception:
        return np.nan


def hesapla_hava_kalitesi_skoru(aqi):
    """
    AQI düşükse iyi, yüksekse kötü.

    0 AQI   -> 1.00
    300 AQI -> 0.00
    300 üstü de 0.00 kabul edilir.
    """
    if pd.isna(aqi):
        return np.nan

    score = 1 - min(aqi, 300) / 300
    return round(score, 4)


def fetch_city_aqi(session, city, token):
    """
    Tek şehir için WAQI verisi çeker.
    """
    url = f"https://api.waqi.info/feed/{city}/?token={token}"

    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    veri = response.json()

    if veri.get("status") != "ok":
        return {
            "query_city": city,
            "station_city": np.nan,
            "aqi": np.nan,
            "pm25": np.nan,
            "pm10": np.nan,
            "temperature": np.nan,
            "humidity": np.nan,
            "pressure": np.nan,
            "wind": np.nan,
            "air_quality_score_0_1": np.nan,
            "air_quality_score_0_100": np.nan,
            "status": "veri_yok",
        }

    data = veri.get("data", {})

    station_city = data.get("city", {}).get("name", city)

    aqi = temiz_sayi(data.get("aqi", np.nan))
    pm25 = iaqi_al(data, "pm25")
    pm10 = iaqi_al(data, "pm10")
    temperature = iaqi_al(data, "t")
    humidity = iaqi_al(data, "h")
    pressure = iaqi_al(data, "p")
    wind = iaqi_al(data, "w")

    air_quality_score_0_1 = hesapla_hava_kalitesi_skoru(aqi)

    if pd.notna(air_quality_score_0_1):
        air_quality_score_0_100 = round(air_quality_score_0_1 * 100, 2)
    else:
        air_quality_score_0_100 = np.nan

    return {
        "query_city": city,
        "station_city": station_city,
        "aqi": aqi,
        "pm25": pm25,
        "pm10": pm10,
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "wind": wind,
        "air_quality_score_0_1": air_quality_score_0_1,
        "air_quality_score_0_100": air_quality_score_0_100,
        "status": "ok",
    }


# ============================================================
# ANA VERİ ÇEKME
# ============================================================

def build_aqi_dataframe(cities=None, token=None, verbose=True):
    if cities is None:
        cities = CITIES

    if token is None:
        token = TOKEN

    if token in [None, "", "BURAYA_WAQI_TOKEN_YAZ"]:
        raise ValueError(
            "WAQI token yok kanka. TOKEN değişkenine token yaz veya terminalde WAQI_TOKEN olarak tanımla."
        )

    session = requests.Session()

    records = []

    if verbose:
        print("GLOBAL AQI RAPORU")
        print("=" * 60)

    for index, city in enumerate(cities, 1):
        try:
            record = fetch_city_aqi(
                session=session,
                city=city,
                token=token,
            )

            records.append(record)

            if verbose:
                print(f"\n[{index}/{len(cities)}] {record['query_city']}")
                print(f"Station: {record['station_city']}")
                print(f"AQI: {record['aqi'] if pd.notna(record['aqi']) else 'Veri yok'}")
                print(f"PM2.5: {record['pm25'] if pd.notna(record['pm25']) else 'Veri yok'}")
                print(f"PM10: {record['pm10'] if pd.notna(record['pm10']) else 'Veri yok'}")
                print(f"Sıcaklık: {record['temperature'] if pd.notna(record['temperature']) else 'Veri yok'}")
                print(f"Nem: {record['humidity'] if pd.notna(record['humidity']) else 'Veri yok'}")
                print(f"Hava Kalitesi Skoru 0-1: {record['air_quality_score_0_1'] if pd.notna(record['air_quality_score_0_1']) else 'Veri yok'}")
                print(f"Hava Kalitesi Skoru 0-100: {record['air_quality_score_0_100'] if pd.notna(record['air_quality_score_0_100']) else 'Veri yok'}")
                print(f"Status: {record['status']}")

        except Exception as e:
            if verbose:
                print(f"\n[{index}/{len(cities)}] Hata: {city} -> {e}")

            records.append({
                "query_city": city,
                "station_city": np.nan,
                "aqi": np.nan,
                "pm25": np.nan,
                "pm10": np.nan,
                "temperature": np.nan,
                "humidity": np.nan,
                "pressure": np.nan,
                "wind": np.nan,
                "air_quality_score_0_1": np.nan,
                "air_quality_score_0_100": np.nan,
                "status": f"hata: {e}",
            })

        time.sleep(SLEEP_SECONDS)

    df = pd.DataFrame(records)

    return df


# ============================================================
# KAYDETME
# ============================================================

def save_aqi_outputs(df, output_dir=None):
    if output_dir is None:
        output_dir = DATA_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / OUTPUT_CSV_NAME

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return output_csv


# ============================================================
# DIŞARIDAN ÇALIŞTIRMA
# ============================================================

def run_and_save(output_dir=None):
    df = build_aqi_dataframe()

    output_csv = save_aqi_outputs(
        df=df,
        output_dir=output_dir,
    )

    ok_count = int((df["status"] == "ok").sum())
    missing_count = int((df["status"] != "ok").sum())

    print("\nBitti kanka! AQI verileri hazır.")
    print(f"CSV: {output_csv}")
    print(f"Toplam şehir: {len(df)}")
    print(f"Başarılı: {ok_count}")
    print(f"Veri yok / hata: {missing_count}")

    print("\nEn kirli 10 şehir:")
    en_kirli = (
        df.dropna(subset=["aqi"])
          .sort_values("aqi", ascending=False)
          .head(10)
    )

    if en_kirli.empty:
        print("AQI verisi yok.")
    else:
        for _, row in en_kirli.iterrows():
            print(f"{row['station_city']} → AQI: {row['aqi']}")

    return {
        "df": df,
        "output_csv": output_csv,
        "ok_count": ok_count,
        "missing_count": missing_count,
    }


if __name__ == "__main__":
    run_and_save()