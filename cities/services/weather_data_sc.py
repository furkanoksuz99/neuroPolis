from pathlib import Path
from datetime import datetime
import csv
import json
import urllib.parse
import urllib.request

import pandas as pd


# ============================================================
# AYARLAR
# ============================================================

OUTPUT_CSV_NAME = "current_weather_results.csv"
OUTPUT_EXCEL_NAME = "current_weather_results.xlsx"

BASE_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_APP_DIR / "data"


# ============================================================
# ŞEHİR LİSTESİ
# ============================================================

CITIES = [
    ("Washington D.C.", 38.9072, -77.0369),
    ("New York", 40.7128, -74.0060),
    ("Los Angeles", 34.0522, -118.2437),
    ("London", 51.5074, -0.1278),
    ("Paris", 48.8566, 2.3522),
    ("Berlin", 52.5200, 13.4050),
    ("Tokyo", 35.6762, 139.6503),
    ("Osaka", 34.6937, 135.5023),
    ("Canberra", -35.2809, 149.1300),
    ("Sydney", -33.8688, 151.2093),
    ("Melbourne", -37.8136, 144.9631),
    ("Ottawa", 45.4215, -75.6972),
    ("Seoul", 37.5665, 126.9780),
    ("Busan", 35.1796, 129.0756),
    ("Rome", 41.9028, 12.4964),
    ("Madrid", 40.4168, -3.7038),
    ("Amsterdam", 52.3676, 4.9041),
    ("Brussels", 50.8503, 4.3517),
    ("Vienna", 48.2082, 16.3738),
    ("Bern", 46.9480, 7.4474),
    ("Stockholm", 59.3293, 18.0686),
    ("Oslo", 59.9139, 10.7522),
    ("Copenhagen", 55.6761, 12.5683),
    ("Helsinki", 60.1699, 24.9384),
    ("Dublin", 53.3498, -6.2603),
    ("Lisbon", 38.7169, -9.1395),
    ("Athens", 37.9838, 23.7275),
    ("Warsaw", 52.2297, 21.0122),
    ("Prague", 50.0755, 14.4378),
    ("Budapest", 47.4979, 19.0402),
    ("Mexico City", 19.4326, -99.1332),
    ("Monterrey", 25.6866, -100.3161),
    ("Brasília", -15.7942, -47.8822),
    ("São Paulo", -23.5505, -46.6333),
    ("Buenos Aires", -34.6037, -58.3816),
    ("Santiago", -33.4489, -70.6693),
    ("Bogotá", 4.7110, -74.0721),
    ("Lima", -12.0464, -77.0428),
    ("Ankara", 39.9334, 32.8597),
    ("Istanbul", 41.0082, 28.9784),
    ("Moscow", 55.7558, 37.6173),
    ("Saint Petersburg", 59.9311, 30.3609),
    ("Beijing", 39.9042, 116.4074),
    ("Shanghai", 31.2304, 121.4737),
    ("New Delhi", 28.6139, 77.2090),
    ("Mumbai", 19.0760, 72.8777),
    ("Jakarta", -6.2088, 106.8456),
    ("Riyadh", 24.7136, 46.6753),
    ("Johannesburg", -26.2041, 28.0473),
    ("Cape Town", -33.9249, 18.4241),
    ("Nairobi", -1.2921, 36.8219),
    ("Cairo", 30.0444, 31.2357),
    ("Tel Aviv", 32.0853, 34.7818),
    ("Singapore", 1.3521, 103.8198),
    ("Baku", 40.4093, 49.8671),
    ("Auckland", -36.8509, 174.7645),
    ("Reykjavík", 64.1355, -21.8954),
    ("San José", 9.9281, -84.0907),
    ("Tallinn", 59.4370, 24.7536),
    ("Riga", 56.9460, 24.1059),
    ("Vilnius", 54.6872, 25.2797),
    ("Luxembourg City", 49.6116, 6.1319),
    ("Bratislava", 48.1486, 17.1077),
    ("Ljubljana", 46.0569, 14.5058),
]


# ============================================================
# WMO HAVA KODLARI
# ============================================================

WMO_CODES = {
    0: ("Açık / Güneşli", "☀️"),
    1: ("Büyük Ölçüde Açık", "🌤️"),
    2: ("Parçalı Bulutlu", "⛅"),
    3: ("Kapalı", "☁️"),
    45: ("Sisli", "🌫️"),
    48: ("Don Sisi", "🌫️"),
    51: ("Hafif Çiseleme", "🌦️"),
    53: ("Orta Çiseleme", "🌦️"),
    55: ("Yoğun Çiseleme", "🌧️"),
    61: ("Hafif Yağmur", "🌧️"),
    63: ("Orta Yağmur", "🌧️"),
    65: ("Şiddetli Yağmur", "🌧️"),
    71: ("Hafif Kar", "🌨️"),
    73: ("Orta Kar", "❄️"),
    75: ("Yoğun Kar", "❄️"),
    77: ("Kar Taneleri", "🌨️"),
    80: ("Hafif Sağanak", "🌦️"),
    81: ("Orta Sağanak", "🌧️"),
    82: ("Şiddetli Sağanak", "⛈️"),
    85: ("Hafif Kar Sağanağı", "🌨️"),
    86: ("Yoğun Kar Sağanağı", "❄️"),
    95: ("Gök Gürültülü Fırtına", "⛈️"),
    96: ("Dolu ile Fırtına", "⛈️"),
    99: ("Şiddetli Dolu Fırtınası", "⛈️"),
}


def wmo_to_text(code):
    return WMO_CODES.get(code, (f"Bilinmiyor ({code})", "❓"))


def is_sunny(code):
    return code in (0, 1)


# ============================================================
# TEK ŞEHİR İÇİN ANLIK HAVA VERİSİ ÇEKME
# ============================================================

def fetch_weather(city_name, latitude, longitude):
    params = urllib.parse.urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "weather_code",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover",
            "is_day",
        ]),
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    })

    url = f"https://api.open-meteo.com/v1/forecast?{params}"

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))

        current = data["current"]

        weather_code = current.get("weather_code")
        weather_desc, weather_emoji = wmo_to_text(weather_code)

        return {
            "City": city_name,
            "Latitude": latitude,
            "Longitude": longitude,

            "Temperature (C)": current.get("temperature_2m"),
            "Feels Like (C)": current.get("apparent_temperature"),
            "Humidity (%)": current.get("relative_humidity_2m"),
            "Wind Speed (km/h)": current.get("wind_speed_10m"),
            "Wind Direction": current.get("wind_direction_10m"),
            "Cloud Cover (%)": current.get("cloud_cover"),

            "Weather Code": weather_code,
            "Weather Description": weather_desc,
            "Weather Emoji": weather_emoji,

            "Is Sunny": is_sunny(weather_code),
            "Is Day": bool(current.get("is_day")),
            "Timezone": data.get("timezone", ""),

            "Fetched At": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "Data Source": "Open-Meteo Current Weather API",
            "Error": None,
        }

    except Exception as e:
        return {
            "City": city_name,
            "Latitude": latitude,
            "Longitude": longitude,

            "Temperature (C)": None,
            "Feels Like (C)": None,
            "Humidity (%)": None,
            "Wind Speed (km/h)": None,
            "Wind Direction": None,
            "Cloud Cover (%)": None,

            "Weather Code": None,
            "Weather Description": None,
            "Weather Emoji": None,

            "Is Sunny": None,
            "Is Day": None,
            "Timezone": None,

            "Fetched At": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "Data Source": "Open-Meteo Current Weather API",
            "Error": str(e),
        }


# ============================================================
# TÜM ŞEHİRLER İÇİN VERİ ÇEKME
# ============================================================

def fetch_all_current_weather(cities=CITIES, verbose=True):
    results = []

    for city_name, latitude, longitude in cities:
        row = fetch_weather(city_name, latitude, longitude)
        results.append(row)

        if verbose:
            if row["Error"]:
                print(f"✗ {city_name:<22} HATA: {row['Error']}")
            else:
                sunny_text = "✅ Güneşli" if row["Is Sunny"] else "❌"
                print(
                    f"{row['Weather Emoji']} {city_name:<22} "
                    f"{row['Temperature (C)']:+.1f}°C "
                    f"Hissedilen {row['Feels Like (C)']:+.1f}°C "
                    f"Nem %{row['Humidity (%)']} "
                    f"Rüzgar {row['Wind Speed (km/h)']} km/h "
                    f"{row['Weather Description']} {sunny_text}"
                )

    return results


# ============================================================
# DATAFRAME ÜRETME
# ============================================================

def build_current_weather_dataframe():
    results = fetch_all_current_weather()

    df = pd.DataFrame(results)

    column_order = [
        "City",
        "Latitude",
        "Longitude",
        "Temperature (C)",
        "Feels Like (C)",
        "Humidity (%)",
        "Wind Speed (km/h)",
        "Wind Direction",
        "Cloud Cover (%)",
        "Weather Code",
        "Weather Description",
        "Weather Emoji",
        "Is Sunny",
        "Is Day",
        "Timezone",
        "Fetched At",
        "Data Source",
        "Error",
    ]

    df = df[column_order]

    successful_df = df[df["Error"].isna()]
    error_df = df[df["Error"].notna()]

    sunny_df = successful_df[successful_df["Is Sunny"] == True]
    rainy_df = successful_df[
        successful_df["Weather Code"].isin([51, 53, 55, 61, 63, 65, 80, 81, 82])
    ]
    snowy_df = successful_df[
        successful_df["Weather Code"].isin([71, 73, 75, 77, 85, 86])
    ]

    if successful_df.empty:
        hottest_city = None
        coldest_city = None
        avg_temperature = None
    else:
        hottest_row = successful_df.loc[successful_df["Temperature (C)"].idxmax()]
        coldest_row = successful_df.loc[successful_df["Temperature (C)"].idxmin()]

        hottest_city = {
            "City": hottest_row["City"],
            "Temperature (C)": hottest_row["Temperature (C)"],
        }

        coldest_city = {
            "City": coldest_row["City"],
            "Temperature (C)": coldest_row["Temperature (C)"],
        }

        avg_temperature = round(successful_df["Temperature (C)"].mean(), 2)

    quality_report = {
        "row_count": len(df),
        "successful_count": len(successful_df),
        "error_count": len(error_df),
        "sunny_count": len(sunny_df),
        "rainy_count": len(rainy_df),
        "snowy_count": len(snowy_df),
        "hottest_city": hottest_city,
        "coldest_city": coldest_city,
        "avg_temperature": avg_temperature,
        "missing_values": df.isna().sum().to_dict(),
        "error_rows": error_df[["City", "Error"]].to_dict(orient="records"),
    }

    return df, quality_report


# ============================================================
# CSV / EXCEL KAYDETME
# ============================================================

def save_current_weather_outputs(df, output_dir=None):
    if output_dir is None:
        output_dir = DATA_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / OUTPUT_CSV_NAME
    output_excel = output_dir / OUTPUT_EXCEL_NAME

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    df.to_excel(output_excel, index=False, engine="openpyxl")

    return output_csv, output_excel


# ============================================================
# ANA ÇALIŞTIRMA FONKSİYONU
# ============================================================

def run_and_save(output_dir=None):
    df, quality_report = build_current_weather_dataframe()
    output_csv, output_excel = save_current_weather_outputs(df, output_dir)

    print("\nBitti kanka! Anlık hava durumu verisi hazır.")
    print(f"CSV: {output_csv}")
    print(f"Excel: {output_excel}")

    print("\nVeri kalite özeti:")
    print(f"Toplam şehir: {quality_report['row_count']}")
    print(f"Başarılı: {quality_report['successful_count']}")
    print(f"Hatalı: {quality_report['error_count']}")
    print(f"Güneşli şehir sayısı: {quality_report['sunny_count']}")
    print(f"Yağmurlu şehir sayısı: {quality_report['rainy_count']}")
    print(f"Karlı şehir sayısı: {quality_report['snowy_count']}")
    print(f"Ortalama sıcaklık: {quality_report['avg_temperature']}")

    print("\nEn sıcak şehir:")
    print(quality_report["hottest_city"])

    print("\nEn soğuk şehir:")
    print(quality_report["coldest_city"])

    if quality_report["error_rows"]:
        print("\nHatalı şehirler:")
        print(quality_report["error_rows"])

    return {
        "df": df,
        "quality_report": quality_report,
        "output_csv": output_csv,
        "output_excel": output_excel,
    }


if __name__ == "__main__":
    run_and_save()