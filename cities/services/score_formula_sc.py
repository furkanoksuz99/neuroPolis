from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# AYARLAR
# ============================================================

BASE_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_APP_DIR / "data"

ANNUAL_WEATHER_FILE = "cities_weather_area_population_2025.csv"
ECONOMIC_FILE = "economic_data_imf_2025_fixed.csv"
NEWS_SENTIMENT_FILE = "city_sentiment_scores.csv"
CURRENT_WEATHER_FILE = "current_weather_results.csv"

OUTPUT_CSV_NAME = "city_scores.csv"
OUTPUT_EXCEL_NAME = "city_scores.xlsx"

OUTPUT_TREND_CSV_NAME = "city_score_trends_7d.csv"
OUTPUT_TREND_EXCEL_NAME = "city_score_trends_7d.xlsx"


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def read_csv_safely(file_name):
    path = DATA_DIR / file_name

    if not path.exists():
        print(f"Dosya bulunamadı: {path}")
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"CSV okuma hatası: {path} -> {e}")
        return pd.DataFrame()


def read_required_csv(file_name):
    path = DATA_DIR / file_name

    if not path.exists():
        raise FileNotFoundError(
            f"{file_name} bulunamadı. Beklenen yol: {path}"
        )

    return pd.read_csv(path)


def ensure_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan

    return df


def detect_column(df, candidates):
    existing = list(df.columns)
    lowered = {str(col).lower(): col for col in existing}

    for candidate in candidates:
        if candidate in existing:
            return candidate

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def clamp_series(series, min_value=0, max_value=100):
    return to_numeric(series).clip(min_value, max_value)


def normalize_positive(series, default_value=50):
    """
    Değer yükseldikçe skor yükselir.
    """
    s = to_numeric(series)

    if s.notna().sum() == 0:
        return pd.Series(default_value, index=series.index, dtype=float)

    min_val = s.min()
    max_val = s.max()

    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series(default_value, index=series.index, dtype=float)

    if max_val == min_val:
        return pd.Series(default_value, index=series.index, dtype=float)

    result = ((s - min_val) / (max_val - min_val)) * 100
    return result.fillna(default_value).clip(0, 100)


def normalize_negative(series, default_value=50):
    """
    Değer yükseldikçe skor düşer.
    """
    s = to_numeric(series)

    if s.notna().sum() == 0:
        return pd.Series(default_value, index=series.index, dtype=float)

    min_val = s.min()
    max_val = s.max()

    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series(default_value, index=series.index, dtype=float)

    if max_val == min_val:
        return pd.Series(default_value, index=series.index, dtype=float)

    result = 100 - (((s - min_val) / (max_val - min_val)) * 100)
    return result.fillna(default_value).clip(0, 100)


def comfort_score(series, target, tolerance, default_value=50):
    """
    Hedef değere yakınsa yüksek skor verir.
    """
    s = to_numeric(series)

    result = 100 - ((s - target).abs() / tolerance * 100)
    result = result.clip(0, 100)

    return result.fillna(default_value)


def weighted_score(score_series_list, weight_list):
    total_weight = sum(weight_list)

    if total_weight == 0:
        return score_series_list[0] * 0 + 50

    result = 0

    for score, weight in zip(score_series_list, weight_list):
        result += clamp_series(score).fillna(50) * weight

    return (result / total_weight).clip(0, 100)


def fahrenheit_to_celsius(series):
    return (to_numeric(series) - 32) * 5 / 9


def safe_log1p(series):
    s = to_numeric(series)
    s = s.where(s >= 0, np.nan)
    return np.log1p(s)


def stable_city_seed(city):
    h = hashlib.md5(str(city).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def normalize_sentiment_label(label):
    """
    Sentiment dosyasından gelen label değerini temizler.
    Beklenenler:
    Pozitif / Nötr / Negatif
    positive / neutral / negative
    """
    if pd.isna(label):
        return "Nötr"

    text = str(label).strip().lower()

    if text in ["pozitif", "positive", "pos"]:
        return "Pozitif"

    if text in ["negatif", "negative", "neg"]:
        return "Negatif"

    return "Nötr"


def sentiment_label_to_zone(label):
    label = normalize_sentiment_label(label)

    if label == "Pozitif":
        return "Positive Zone"

    if label == "Negatif":
        return "Risk Zone"

    return "Neutral Zone"


# ============================================================
# VERİLERİ HAZIRLAMA
# ============================================================

def prepare_sentiment_data():
    """
    Ana kaynak burası.

    city_sentiment_scores.csv içindeki:
    - overall
    - sentiment_label

    direkt kullanılacak.

    Emotion Score = overall
    Social Label = sentiment_label
    """

    df = read_required_csv(NEWS_SENTIMENT_FILE)

    city_col = detect_column(df, ["City", "city"])
    country_col = detect_column(df, ["Country", "country"])
    overall_col = detect_column(df, ["overall", "Social News Score", "social_news_score"])
    label_col = detect_column(df, ["sentiment_label", "Social Label", "social_label"])

    positive_col = detect_column(df, ["positive_pct", "Positive News %", "positive_news_pct"])
    negative_col = detect_column(df, ["negative_pct", "Negative News %", "negative_news_pct"])
    neutral_col = detect_column(df, ["neutral_pct", "Neutral News %", "neutral_news_pct"])
    headline_count_col = detect_column(df, ["headline_count", "Headline Count", "news_count"])

    volatility_col = detect_column(df, ["volatility", "Sentiment Volatility"])
    subjectivity_col = detect_column(df, ["avg_subjectivity", "Avg Subjectivity"])
    theme_col = detect_column(df, ["dominant_theme", "Dominant Theme"])
    top_pos_col = detect_column(df, ["top_positive_headline", "Top Positive Headline"])
    top_neg_col = detect_column(df, ["top_negative_headline", "Top Negative Headline"])

    if city_col is None:
        raise ValueError(
            f"{NEWS_SENTIMENT_FILE} içinde city / City kolonu yok. "
            f"Mevcut kolonlar: {list(df.columns)}"
        )

    if overall_col is None:
        raise ValueError(
            f"{NEWS_SENTIMENT_FILE} içinde overall veya Social News Score kolonu yok. "
            f"Mevcut kolonlar: {list(df.columns)}"
        )

    if label_col is None:
        raise ValueError(
            f"{NEWS_SENTIMENT_FILE} içinde sentiment_label veya Social Label kolonu yok. "
            f"Mevcut kolonlar: {list(df.columns)}"
        )

    out = pd.DataFrame()

    out["City"] = df[city_col].astype(str)

    if country_col is not None:
        out["News Country"] = df[country_col].astype(str)
    else:
        out["News Country"] = ""

    out["Sentiment Overall"] = to_numeric(df[overall_col])
    out["Emotion Score"] = out["Sentiment Overall"].round(1)

    # Kartta büyük sayı olarak tam sayı istersen bunu kullanırsın.
    out["Emotion Score Display"] = out["Sentiment Overall"].round(0).astype("Int64")

    out["Social News Score"] = out["Sentiment Overall"].round(1)
    out["Social Score"] = out["Sentiment Overall"].round(1)

    out["Social Label"] = df[label_col].apply(normalize_sentiment_label)
    out["Zone Label"] = out["Social Label"].apply(sentiment_label_to_zone)

    out["Positive News %"] = to_numeric(df[positive_col]) if positive_col else np.nan
    out["Negative News %"] = to_numeric(df[negative_col]) if negative_col else np.nan
    out["Neutral News %"] = to_numeric(df[neutral_col]) if neutral_col else np.nan
    out["Headline Count"] = to_numeric(df[headline_count_col]) if headline_count_col else np.nan

    out["Sentiment Volatility"] = to_numeric(df[volatility_col]) if volatility_col else np.nan
    out["Avg Subjectivity"] = to_numeric(df[subjectivity_col]) if subjectivity_col else np.nan

    out["Dominant Theme"] = df[theme_col] if theme_col else np.nan
    out["Top Positive Headline"] = df[top_pos_col] if top_pos_col else np.nan
    out["Top Negative Headline"] = df[top_neg_col] if top_neg_col else np.nan

    # overall boşsa sıkıntı; direkt hata verelim.
    if out["Emotion Score"].isna().any():
        missing_cities = out.loc[out["Emotion Score"].isna(), "City"].tolist()

        raise ValueError(
            f"{NEWS_SENTIMENT_FILE} içinde bazı şehirlerde overall boş: {missing_cities[:10]}"
        )

    return out


def prepare_annual_weather_data():
    df = read_csv_safely(ANNUAL_WEATHER_FILE)

    if df.empty:
        return df

    if "City Name" in df.columns:
        df = df.rename(columns={"City Name": "City"})

    needed_columns = [
        "City",
        "Country",
        "ISO2",
        "Population",
        "Area (sq mi)",
        "Avg High Temp 2025 (F)",
        "Avg Low Temp 2025 (F)",
        "Annual Precipitation Days 2025",
        "Annual Precipitation 2025 (in)",
        "Annual Sunshine Hours 2025",
    ]

    df = ensure_columns(df, needed_columns)

    df["Annual Avg Temp 2025 (F)"] = (
        to_numeric(df["Avg High Temp 2025 (F)"])
        + to_numeric(df["Avg Low Temp 2025 (F)"])
    ) / 2

    df["Annual Avg Temp 2025 (C)"] = fahrenheit_to_celsius(
        df["Annual Avg Temp 2025 (F)"]
    )

    df["Population Density"] = (
        to_numeric(df["Population"])
        / to_numeric(df["Area (sq mi)"])
    )

    keep_columns = [
        "City",
        "Country",
        "ISO2",
        "Population",
        "Area (sq mi)",
        "Population Density",
        "Annual Avg Temp 2025 (C)",
        "Annual Precipitation Days 2025",
        "Annual Precipitation 2025 (in)",
        "Annual Sunshine Hours 2025",
    ]

    return df[keep_columns]


def prepare_economic_data():
    df = read_csv_safely(ECONOMIC_FILE)

    if df.empty:
        return df

    needed_columns = [
        "City",
        "Country ISO2",
        "Country ISO3",
        "GDP per capita USD",
        "Inflation (%)",
        "Unemployment (%)",
        "Economic Score",
    ]

    df = ensure_columns(df, needed_columns)

    if df["Economic Score"].isna().all():
        gdp_score = normalize_positive(df["GDP per capita USD"])
        inflation_score = normalize_negative(df["Inflation (%)"])
        unemployment_score = normalize_negative(df["Unemployment (%)"])

        df["Economic Score"] = weighted_score(
            [gdp_score, inflation_score, unemployment_score],
            [40, 30, 30],
        )

    keep_columns = [
        "City",
        "Country ISO2",
        "Country ISO3",
        "GDP per capita USD",
        "Inflation (%)",
        "Unemployment (%)",
        "Economic Score",
    ]

    return df[keep_columns]


def prepare_current_weather_data():
    df = read_csv_safely(CURRENT_WEATHER_FILE)

    if df.empty:
        return df

    needed_columns = [
        "City",
        "Temperature (C)",
        "Feels Like (C)",
        "Humidity (%)",
        "Wind Speed (km/h)",
        "Wind Direction",
        "Cloud Cover (%)",
        "Weather Code",
        "Weather Description",
        "Is Sunny",
        "Is Day",
        "Fetched At",
    ]

    df = ensure_columns(df, needed_columns)

    return df[needed_columns]


# ============================================================
# ANA SKOR DOSYASI
# ============================================================

def build_score_dataframe():
    sentiment_df = prepare_sentiment_data()
    annual_df = prepare_annual_weather_data()
    economic_df = prepare_economic_data()
    current_weather_df = prepare_current_weather_data()

    # Ana base artık sentiment dosyasıdır.
    # Çünkü Emotion Score sentiment overall'dan gelecek.
    base_df = sentiment_df.copy()

    if not annual_df.empty:
        base_df = base_df.merge(annual_df, on="City", how="left")

    if not economic_df.empty:
        base_df = base_df.merge(economic_df, on="City", how="left")

    if not current_weather_df.empty:
        base_df = base_df.merge(current_weather_df, on="City", how="left")

    df = base_df.copy()

    # Country seçimi:
    # Önce yıllık veri Country, yoksa sentiment country.
    if "Country" not in df.columns:
        df["Country"] = np.nan

    df["Country"] = df["Country"].fillna(df.get("News Country", ""))

    # ------------------------------------------------------------
    # ARA DEĞİŞKENLER
    # Bunlar sadece alt kartlar için.
    # Emotion Score'a karışmayacak.
    # ------------------------------------------------------------

    df["Population Density Log"] = safe_log1p(df.get("Population Density", np.nan))
    df["Population Log"] = safe_log1p(df.get("Population", np.nan))

    density_good_score = normalize_negative(df["Population Density Log"])
    population_good_score = normalize_negative(df["Population Log"])

    current_temp_good = comfort_score(
        df.get("Temperature (C)", pd.Series(np.nan, index=df.index)),
        target=22,
        tolerance=25,
    )

    annual_temp_good = comfort_score(
        df.get("Annual Avg Temp 2025 (C)", pd.Series(np.nan, index=df.index)),
        target=18,
        tolerance=25,
    )

    humidity_good = comfort_score(
        df.get("Humidity (%)", pd.Series(np.nan, index=df.index)),
        target=50,
        tolerance=50,
    )

    cloud_good = normalize_negative(
        df.get("Cloud Cover (%)", pd.Series(np.nan, index=df.index))
    )

    wind_positive_score = normalize_positive(
        df.get("Wind Speed (km/h)", pd.Series(np.nan, index=df.index))
    )

    wind_negative_score = normalize_negative(
        df.get("Wind Speed (km/h)", pd.Series(np.nan, index=df.index))
    )

    sunshine_score = normalize_positive(
        df.get("Annual Sunshine Hours 2025", pd.Series(np.nan, index=df.index))
    )

    precipitation_comfort_score = comfort_score(
        df.get("Annual Precipitation Days 2025", pd.Series(np.nan, index=df.index)),
        target=90,
        tolerance=160,
    )

    economy_score = clamp_series(
        df.get("Economic Score", pd.Series(50, index=df.index))
    ).fillna(50)

    # ------------------------------------------------------------
    # ALT KART SKORLARI
    # Bunlar sadece gösterim için.
    # Emotion Score bunlardan hesaplanmıyor.
    # ------------------------------------------------------------

    df["Air Quality Score"] = weighted_score(
        [
            density_good_score,
            wind_positive_score,
            humidity_good,
            precipitation_comfort_score,
        ],
        [45, 20, 20, 15],
    ).round(2)

    df["AQI Proxy"] = (180 - df["Air Quality Score"] * 1.4).clip(5, 180).round(0)

    df["Traffic Score"] = weighted_score(
        [
            density_good_score,
            population_good_score,
            economy_score,
        ],
        [50, 30, 20],
    ).round(2)

    df["Traffic Load Proxy"] = (100 - df["Traffic Score"]).round(2)

    # Social Score artık sentiment overall ile birebir aynı.
    df["Social Score"] = df["Sentiment Overall"].round(1)
    df["Social News Score"] = df["Sentiment Overall"].round(1)

    df["Weather Score"] = weighted_score(
        [
            current_temp_good,
            annual_temp_good,
            sunshine_score,
            precipitation_comfort_score,
            humidity_good,
            cloud_good,
            wind_negative_score,
        ],
        [20, 20, 20, 15, 10, 10, 5],
    ).round(2)

    df["Economy Score"] = economy_score.round(2)

    df["Noise Score"] = weighted_score(
        [
            df["Traffic Score"],
            density_good_score,
            wind_negative_score,
        ],
        [55, 35, 10],
    ).round(2)

    df["Estimated Noise (dB)"] = (75 - df["Noise Score"] * 0.35).clip(35, 80).round(1)

    # ------------------------------------------------------------
    # KRİTİK KISIM
    # Emotion Score artık sadece sentiment overall.
    # Burada air/weather/economy/traffic/noise ASLA karışmıyor.
    # ------------------------------------------------------------

    df["Emotion Score"] = df["Sentiment Overall"].round(1)
    df["Emotion Score Display"] = df["Sentiment Overall"].round(0).astype("Int64")

    df["Social Label"] = df["Social Label"].apply(normalize_sentiment_label)
    df["Zone Label"] = df["Social Label"].apply(sentiment_label_to_zone)

    # ------------------------------------------------------------
    # RANK
    # Rank da sentiment overall'a göre.
    # ------------------------------------------------------------

    df = df.sort_values(by="Emotion Score", ascending=False).reset_index(drop=True)

    df["Rank"] = df["Emotion Score"].rank(
        method="min",
        ascending=False,
    ).astype(int)

    total_city_count = len(df)
    df["Rank Text"] = df["Rank"].apply(lambda x: f"#{x} of {total_city_count}")

    # ------------------------------------------------------------
    # 7 GÜNLÜK TREND
    # Trend de emotion score yani sentiment overall bazlı.
    # ------------------------------------------------------------

    df["Trend 7D JSON"] = df.apply(
        lambda row: json.dumps(
            generate_7_day_trend(row["City"], row["Emotion Score"]),
            ensure_ascii=False,
        ),
        axis=1,
    )

    df["Trend Change"] = df["Trend 7D JSON"].apply(
        lambda x: round(json.loads(x)[-1] - json.loads(x)[0], 2)
    )

    df["Updated At"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ------------------------------------------------------------
    # FINAL KOLONLAR
    # ------------------------------------------------------------

    final_columns = [
        "City",
        "Country",
        "ISO2",

        # Ana skor
        "Emotion Score",
        "Emotion Score Display",
        "Social Label",
        "Zone Label",
        "Rank",
        "Rank Text",

        # Sentiment detayları
        "Sentiment Overall",
        "Social Score",
        "Social News Score",
        "Positive News %",
        "Negative News %",
        "Neutral News %",
        "Headline Count",
        "Sentiment Volatility",
        "Avg Subjectivity",
        "Dominant Theme",
        "Top Positive Headline",
        "Top Negative Headline",

        # Alt kart skorları
        "Air Quality Score",
        "AQI Proxy",

        "Traffic Score",
        "Traffic Load Proxy",

        "Weather Score",
        "Temperature (C)",
        "Feels Like (C)",
        "Humidity (%)",
        "Wind Speed (km/h)",
        "Cloud Cover (%)",
        "Weather Description",

        "Economy Score",
        "GDP per capita USD",
        "Inflation (%)",
        "Unemployment (%)",

        "Noise Score",
        "Estimated Noise (dB)",

        # Ek veriler
        "Population",
        "Area (sq mi)",
        "Population Density",

        "Annual Avg Temp 2025 (C)",
        "Annual Precipitation Days 2025",
        "Annual Sunshine Hours 2025",

        "Trend 7D JSON",
        "Trend Change",
        "Updated At",
    ]

    for col in final_columns:
        if col not in df.columns:
            df[col] = np.nan

    df = df[final_columns]

    quality_report = {
        "row_count": len(df),
        "missing_values": df.isna().sum().to_dict(),
        "top_city": df.iloc[0]["City"] if len(df) > 0 else None,
        "top_score": float(df.iloc[0]["Emotion Score"]) if len(df) > 0 else None,
        "bottom_city": df.iloc[-1]["City"] if len(df) > 0 else None,
        "bottom_score": float(df.iloc[-1]["Emotion Score"]) if len(df) > 0 else None,
    }

    return df, quality_report


# ============================================================
# TREND DATA
# ============================================================

def generate_7_day_trend(city, base_score):
    """
    Şimdilik stabil pseudo trend.
    Base score artık sentiment overall.
    """
    seed = stable_city_seed(city)

    values = []

    for i in range(7):
        wave = math.sin((seed % 10) + i) * 2.5
        drift = (i - 3) * 0.35
        value = base_score + wave + drift

        values.append(round(max(0, min(100, value)), 2))

    return values


def build_trend_dataframe(score_df):
    rows = []
    today = datetime.utcnow().date()

    for _, row in score_df.iterrows():
        city = row["City"]
        trend_values = json.loads(row["Trend 7D JSON"])

        for i, score in enumerate(trend_values):
            trend_date = today - timedelta(days=6 - i)

            rows.append({
                "City": city,
                "Date": trend_date,
                "Emotion Score": score,
            })

    return pd.DataFrame(rows)


# ============================================================
# KAYDETME
# ============================================================

def save_score_outputs(score_df, trend_df, output_dir=None):
    if output_dir is None:
        output_dir = DATA_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / OUTPUT_CSV_NAME
    output_excel = output_dir / OUTPUT_EXCEL_NAME

    trend_csv = output_dir / OUTPUT_TREND_CSV_NAME
    trend_excel = output_dir / OUTPUT_TREND_EXCEL_NAME

    score_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    score_df.to_excel(output_excel, index=False, engine="openpyxl")

    trend_df.to_csv(trend_csv, index=False, encoding="utf-8-sig")
    trend_df.to_excel(trend_excel, index=False, engine="openpyxl")

    return output_csv, output_excel, trend_csv, trend_excel


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def run_and_save(output_dir=None):
    score_df, quality_report = build_score_dataframe()

    if score_df.empty:
        print("Skor dosyası üretilemedi çünkü veri yok.")
        return {
            "df": score_df,
            "quality_report": quality_report,
            "output_csv": None,
            "output_excel": None,
        }

    trend_df = build_trend_dataframe(score_df)

    output_csv, output_excel, trend_csv, trend_excel = save_score_outputs(
        score_df=score_df,
        trend_df=trend_df,
        output_dir=output_dir,
    )

    print("\nBitti kanka! Skor verileri hazır.")
    print("Emotion Score artık sadece city_sentiment_scores.csv içindeki overall değerinden geliyor.")
    print("Social Label artık sadece sentiment_label değerinden geliyor.")
    print(f"Skor CSV: {output_csv}")
    print(f"Skor Excel: {output_excel}")
    print(f"Trend CSV: {trend_csv}")
    print(f"Trend Excel: {trend_excel}")

    print("\nÖzet:")
    print(f"Şehir sayısı: {quality_report['row_count']}")
    print(f"En yüksek şehir: {quality_report['top_city']} - {quality_report['top_score']}")
    print(f"En düşük şehir: {quality_report['bottom_city']} - {quality_report['bottom_score']}")

    print("\nİlk 10 şehir:")
    print(score_df.head(10)[[
        "City",
        "Emotion Score",
        "Emotion Score Display",
        "Social Label",
        "Zone Label",
        "Rank Text",
    ]])

    return {
        "df": score_df,
        "trend_df": trend_df,
        "quality_report": quality_report,
        "output_csv": output_csv,
        "output_excel": output_excel,
        "trend_csv": trend_csv,
        "trend_excel": trend_excel,
    }


if __name__ == "__main__":
    run_and_save()