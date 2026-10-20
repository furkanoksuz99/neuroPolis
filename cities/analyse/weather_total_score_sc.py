import pandas as pd
import numpy as np

from cities.analyse.analysis_common import (
    DATA_DIR,
    ANALYSE_DATA_DIR,
    TOTAL_SCORE_FILE,
    get_master_city_dataframe,
    read_csv_safely,
    clean_columns,
    clean_numeric,
    city_key_from_any_name,
    detect_column,
    fill_numeric_with_median,
    update_total_score_from_score_df,
    save_analyse_csv,
)


# ============================================================
# WEATHER SCORE TO TOTAL DATA
# ------------------------------------------------------------
# Okuduğu dosya:
# - cities/data/current_weather_results.csv
#
# Yazdığı dosyalar:
# - cities/analyse_data/weather_score.csv
# - cities/total_data/total_score.csv
#
# Eklediği kolon:
# - weather
# ============================================================

WEATHER_FILE = DATA_DIR / "current_weather_results.csv"
ANALYSE_OUTPUT_FILE = ANALYSE_DATA_DIR / "weather_score.csv"

OUTPUT_COLUMN = "weather"


def to_bool(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    x = str(x).strip().lower()
    return x in ["true", "1", "yes", "evet"]


def north_season(month):
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "autumn"


def get_city_season(lat, month):
    lat = float(lat)
    if abs(lat) < 23.5:
        return "tropical"
    if lat >= 0:
        return north_season(month)
    shifted_month = ((month + 5) % 12) + 1
    return north_season(shifted_month)


def get_local_month(row):
    try:
        dt_utc = pd.to_datetime(row["fetched_at"], utc=True)
        dt_local = dt_utc.tz_convert(row["timezone"])
        return int(dt_local.month)
    except Exception:
        return int(pd.Timestamp.utcnow().month)


def gaussian_score(value, optimal, sigma):
    return float(np.exp(-0.5 * ((value - optimal) / sigma) ** 2))


def piecewise_score(value, ideal_min, ideal_max, hard_min, hard_max):
    if ideal_min <= value <= ideal_max:
        return 1.0
    if value < ideal_min:
        if value <= hard_min:
            return 0.0
        return float((value - hard_min) / (ideal_min - hard_min))
    if value >= hard_max:
        return 0.0
    return float((hard_max - value) / (hard_max - ideal_max))


def absolute_temperature_score(temp_c):
    return gaussian_score(temp_c, 21.0, 7.0)


def humidity_score(rh):
    return piecewise_score(rh, 40, 60, 0, 100)


def wind_score(wind_kmh):
    return piecewise_score(wind_kmh, 0, 15, 0, 55)


def sunshine_score(cloud_cover_pct, is_sunny_bool):
    base = 1.0 - (cloud_cover_pct / 100.0) * 0.9
    bonus = 0.1 if is_sunny_bool else 0.0
    return float(min(1.0, max(0.0, base + bonus)))


def precipitation_penalty(weather_code):
    try:
        code = int(weather_code)
    except Exception:
        return 0.15

    if code == 0:
        return 0.00
    if code in [1, 2, 3]:
        return 0.05
    if 45 <= code <= 48:
        return 0.20
    if 51 <= code <= 57:
        return 0.30
    if 61 <= code <= 67:
        return 0.40
    if 71 <= code <= 77:
        return 0.50
    if 80 <= code <= 82:
        return 0.40
    if 85 <= code <= 86:
        return 0.45
    if 95 <= code <= 99:
        return 0.60
    return 0.15


def precipitation_score(weather_code):
    return float(max(0.0, 1.0 - precipitation_penalty(weather_code)))


def safe_sigma(std_value):
    if pd.isna(std_value) or std_value <= 0:
        return 4.0
    return max(4.0, float(std_value) * 1.25)


def prepare_weather_source():
    if not WEATHER_FILE.exists():
        print(f"\nHava durumu dosyası yok: {WEATHER_FILE}")
        return pd.DataFrame(columns=["city_key"])

    df = read_csv_safely(WEATHER_FILE)
    df = clean_columns(df)

    city_col = detect_column(df, ["city", "City"])
    temp_col = detect_column(df, ["temperature", "Temperature (C)"])
    feels_col = detect_column(df, ["feels_like", "Feels Like (C)"])
    humidity_col = detect_column(df, ["humidity", "Humidity (%)"])
    wind_col = detect_column(df, ["wind_speed", "Wind Speed (km/h)"])
    cloud_col = detect_column(df, ["cloud_cover", "Cloud Cover (%)"])
    code_col = detect_column(df, ["weather_code", "Weather Code"])
    sunny_col = detect_column(df, ["is_sunny", "Is Sunny"])
    fetched_col = detect_column(df, ["fetched_at", "Fetched At"])

    required = {
        "city": city_col,
        "temperature": temp_col,
        "feels_like": feels_col,
        "humidity": humidity_col,
        "wind_speed": wind_col,
        "cloud_cover": cloud_col,
        "weather_code": code_col,
    }

    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(f"current_weather_results.csv içinde eksik kolonlar var: {missing}. Mevcut kolonlar: {list(df.columns)}")

    out = pd.DataFrame()
    out["source_city"] = df[city_col].astype(str)
    out["city_key"] = out["source_city"].apply(city_key_from_any_name)
    out["temperature"] = df[temp_col].apply(clean_numeric)
    out["feels_like"] = df[feels_col].apply(clean_numeric)
    out["humidity"] = df[humidity_col].apply(clean_numeric)
    out["wind_speed"] = df[wind_col].apply(clean_numeric)
    out["cloud_cover"] = df[cloud_col].apply(clean_numeric)
    out["weather_code"] = df[code_col].apply(clean_numeric)
    out["is_sunny"] = df[sunny_col].apply(to_bool) if sunny_col is not None else False
    out["fetched_at"] = df[fetched_col] if fetched_col is not None else pd.Timestamp.utcnow().isoformat()

    out = out[out["city_key"] != ""].copy()

    grouped = (
        out.groupby("city_key", as_index=False)
        .agg({
            "temperature": "mean",
            "feels_like": "mean",
            "humidity": "mean",
            "wind_speed": "mean",
            "cloud_cover": "mean",
            "weather_code": "mean",
            "is_sunny": "max",
            "fetched_at": "first",
        })
    )

    return grouped


def prepare_weather_data():
    meta = get_master_city_dataframe()
    source = prepare_weather_source()
    df = meta.merge(source, on="city_key", how="left")

    default_values = {
        "temperature": 21.0,
        "feels_like": 21.0,
        "humidity": 50.0,
        "wind_speed": 10.0,
        "cloud_cover": 50.0,
        "weather_code": 0.0,
    }

    for col, fallback in default_values.items():
        df[col], missing_mask, fill_value = fill_numeric_with_median(
            df[col],
            fallback=fallback,
            lower=0 if col in ["humidity", "wind_speed", "cloud_cover", "weather_code"] else None,
            upper=100 if col in ["humidity", "cloud_cover"] else None,
        )
        df[col + "_was_missing"] = missing_mask

    if "is_sunny" not in df.columns:
        df["is_sunny"] = False
    df["is_sunny"] = df["is_sunny"].fillna(False).apply(to_bool)

    if "fetched_at" not in df.columns:
        df["fetched_at"] = pd.Timestamp.utcnow().isoformat()
    df["fetched_at"] = df["fetched_at"].fillna(pd.Timestamp.utcnow().isoformat())

    return df


def build_weather_score_dataframe():
    df = prepare_weather_data()

    df["local_month"] = df.apply(get_local_month, axis=1)
    df["season"] = df.apply(lambda row: get_city_season(row["lat"], row["local_month"]), axis=1)

    df["temp_combined"] = 0.60 * df["temperature"] + 0.40 * df["feels_like"]

    region_season_stats = (
        df.groupby(["region", "season"])["temp_combined"]
        .agg(region_season_avg_temp="mean", region_season_std_temp="std", region_season_city_count="count")
        .reset_index()
    )
    df = df.merge(region_season_stats, on=["region", "season"], how="left")

    region_stats = (
        df.groupby("region")["temp_combined"]
        .agg(region_avg_temp="mean", region_std_temp="std", region_city_count="count")
        .reset_index()
    )
    df = df.merge(region_stats, on="region", how="left")

    season_stats = (
        df.groupby("season")["temp_combined"]
        .agg(season_avg_temp="mean", season_std_temp="std", season_city_count="count")
        .reset_index()
    )
    df = df.merge(season_stats, on="season", how="left")

    global_avg_temp = df["temp_combined"].mean()
    global_std_temp = df["temp_combined"].std()
    global_city_count = len(df)

    def choose_temperature_reference(row):
        if row["region_season_city_count"] >= 2:
            return pd.Series({
                "reference_temp": row["region_season_avg_temp"],
                "reference_std": row["region_season_std_temp"],
                "reference_count": row["region_season_city_count"],
                "reference_type": "region_season",
            })
        if row["region_city_count"] >= 2:
            return pd.Series({
                "reference_temp": row["region_avg_temp"],
                "reference_std": row["region_std_temp"],
                "reference_count": row["region_city_count"],
                "reference_type": "region_fallback",
            })
        if row["season_city_count"] >= 2:
            return pd.Series({
                "reference_temp": row["season_avg_temp"],
                "reference_std": row["season_std_temp"],
                "reference_count": row["season_city_count"],
                "reference_type": "season_fallback",
            })
        return pd.Series({
            "reference_temp": global_avg_temp,
            "reference_std": global_std_temp,
            "reference_count": global_city_count,
            "reference_type": "global_fallback",
        })

    df = pd.concat([df, df.apply(choose_temperature_reference, axis=1)], axis=1)
    df["reference_sigma"] = df["reference_std"].apply(safe_sigma)

    df["s_thermal_absolute"] = df["temp_combined"].apply(absolute_temperature_score)
    df["s_thermal_regional"] = df.apply(lambda row: gaussian_score(row["temp_combined"], row["reference_temp"], row["reference_sigma"]), axis=1)
    df["s_thermal"] = 0.45 * df["s_thermal_absolute"] + 0.55 * df["s_thermal_regional"]
    df["s_humidity"] = df["humidity"].apply(humidity_score)
    df["s_sunshine"] = df.apply(lambda row: sunshine_score(row["cloud_cover"], row["is_sunny"]), axis=1)
    df["s_wind"] = df["wind_speed"].apply(wind_score)
    df["s_rain"] = df["weather_code"].apply(precipitation_score)

    df[OUTPUT_COLUMN] = (
        0.30 * df["s_thermal"]
        + 0.20 * df["s_humidity"]
        + 0.30 * df["s_sunshine"]
        + 0.15 * df["s_wind"]
        + 0.05 * df["s_rain"]
    )
    df[OUTPUT_COLUMN] = df[OUTPUT_COLUMN].clip(0, 1).round(4)

    detail_cols = [
        "city", "city_key", "region", "timezone", "lat", "lon",
        "local_month", "season", "temperature", "feels_like", "temp_combined",
        "humidity", "wind_speed", "cloud_cover", "weather_code", "is_sunny", "fetched_at",
        "temperature_was_missing", "feels_like_was_missing", "humidity_was_missing",
        "wind_speed_was_missing", "cloud_cover_was_missing", "weather_code_was_missing",
        "reference_temp", "reference_std", "reference_sigma", "reference_count", "reference_type",
        "s_thermal_absolute", "s_thermal_regional", "s_thermal", "s_humidity", "s_sunshine", "s_wind", "s_rain",
        OUTPUT_COLUMN,
    ]

    result = df[detail_cols].copy()

    for col in result.columns:
        if col not in ["city", "city_key", "region", "timezone", "season", "is_sunny", "fetched_at", "reference_type"]:
            result[col] = pd.to_numeric(result[col], errors="coerce").round(4)

    return result


def run_and_save(output_dir=None):
    score_df = build_weather_score_dataframe()

    analyse_df = score_df.drop(columns=["city_key"], errors="ignore")
    analyse_output = save_analyse_csv(analyse_df, ANALYSE_OUTPUT_FILE)

    total_df = update_total_score_from_score_df(
        score_df=score_df,
        score_col=OUTPUT_COLUMN,
        fallback=0.5,
    )

    print("\nBitti kanka! Weather skor total_score.csv içine eklendi.")
    print(f"Total CSV: {TOTAL_SCORE_FILE}")
    print(f"Analyse CSV: {analyse_output}")
    print(f"Analyse şehir sayısı: {len(score_df)}")
    print(f"Total şehir sayısı: {len(total_df)}")

    return {
        "df": total_df,
        "score_df": score_df,
        "output_csv": TOTAL_SCORE_FILE,
        "analyse_csv": analyse_output,
    }


if __name__ == "__main__":
    run_and_save()
