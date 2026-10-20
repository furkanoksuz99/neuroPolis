import pandas as pd
import numpy as np

from cities.analyse.analysis_common import (
    DATA_DIR,
    ANALYSE_DATA_DIR,
    TOTAL_SCORE_FILE,
    get_master_city_score_base,
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
# EMOTIONAL SCORE TO TOTAL DATA
# ------------------------------------------------------------
# Okuduğu dosya:
# - cities/data/city_sentiment_scores.csv
#
# Yazdığı dosyalar:
# - cities/analyse_data/emotional_score.csv
# - cities/total_data/total_score.csv
#
# Eklediği kolon:
# - emotional
# ============================================================

SENTIMENT_FILE = DATA_DIR / "city_sentiment_scores.csv"
ANALYSE_OUTPUT_FILE = ANALYSE_DATA_DIR / "emotional_score.csv"

OUTPUT_COLUMN = "emotional"


def prepare_emotional_source():
    if not SENTIMENT_FILE.exists():
        print(f"\nSentiment dosyası yok: {SENTIMENT_FILE}")
        return pd.DataFrame(columns=["city", "city_key", "overall_raw", OUTPUT_COLUMN])

    df = read_csv_safely(SENTIMENT_FILE)
    df = clean_columns(df)

    city_col = detect_column(df, ["city", "City"])
    country_col = detect_column(df, ["country", "Country"])
    score_col = detect_column(df, ["overall", "Sentiment Overall", "Emotion Score", "Social News Score"])
    label_col = detect_column(df, ["sentiment_label", "Sentiment Label", "label"])

    if city_col is None:
        raise ValueError(f"city_sentiment_scores.csv içinde city kolonu yok. Mevcut kolonlar: {list(df.columns)}")

    if score_col is None:
        raise ValueError(f"city_sentiment_scores.csv içinde overall skoru yok. Mevcut kolonlar: {list(df.columns)}")

    out = pd.DataFrame()
    out["city"] = df[city_col].astype(str)
    out["city_key"] = out["city"].apply(city_key_from_any_name)
    out["country"] = df[country_col].astype(str) if country_col is not None else ""
    out["overall_raw"] = df[score_col].apply(clean_numeric)
    out[OUTPUT_COLUMN] = (out["overall_raw"] / 100).clip(0, 1)
    out["sentiment_label"] = df[label_col].astype(str) if label_col is not None else ""

    optional_cols = [
        "positive_pct",
        "negative_pct",
        "neutral_pct",
        "headline_count",
        "volatility",
        "avg_subjectivity",
        "dominant_theme",
        "top_positive_headline",
        "top_negative_headline",
        "_weighted_mean",
        "_median_raw",
    ]

    for col in optional_cols:
        detected = detect_column(df, [col])
        if detected is not None:
            out[col] = df[detected]

    numeric_optional = [
        "positive_pct",
        "negative_pct",
        "neutral_pct",
        "headline_count",
        "volatility",
        "avg_subjectivity",
        "_weighted_mean",
        "_median_raw",
    ]

    agg_dict = {
        "city": "first",
        "country": "first",
        "overall_raw": "mean",
        OUTPUT_COLUMN: "mean",
        "sentiment_label": "first",
    }

    for col in optional_cols:
        if col in out.columns:
            if col in numeric_optional:
                out[col] = out[col].apply(clean_numeric)
                agg_dict[col] = "mean"
            else:
                agg_dict[col] = "first"

    out = out[out["city_key"] != ""].copy()
    out = out.groupby("city_key", as_index=False).agg(agg_dict)

    return out


def build_emotional_score_dataframe():
    master = get_master_city_score_base()
    source = prepare_emotional_source()

    if source.empty:
        df = master.copy()
        df["country"] = ""
        df["overall_raw"] = np.nan
        df[OUTPUT_COLUMN] = np.nan
        df["sentiment_label"] = ""
    else:
        source = source.rename(columns={"city": "source_city"})
        df = master.merge(source, on="city_key", how="left")
        df["country"] = df.get("country", "")
        df["sentiment_label"] = df.get("sentiment_label", "")

    # Eksikleri doldur: overall_raw 0-100, emotional 0-1.
    df["overall_raw"], raw_missing, raw_fill = fill_numeric_with_median(
        df["overall_raw"],
        fallback=50,
        lower=0,
        upper=100,
    )

    df[OUTPUT_COLUMN] = (df["overall_raw"] / 100).clip(0, 1)
    df[OUTPUT_COLUMN], score_missing, score_fill = fill_numeric_with_median(
        df[OUTPUT_COLUMN],
        fallback=0.5,
        lower=0,
        upper=1,
    )

    df["emotional_was_missing"] = raw_missing | score_missing
    df["overall_raw"] = df["overall_raw"].round(2)
    df[OUTPUT_COLUMN] = df[OUTPUT_COLUMN].round(4)

    for col in ["country", "sentiment_label"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    optional_cols = [
        "positive_pct",
        "negative_pct",
        "neutral_pct",
        "headline_count",
        "volatility",
        "avg_subjectivity",
        "dominant_theme",
        "top_positive_headline",
        "top_negative_headline",
        "_weighted_mean",
        "_median_raw",
    ]

    for col in optional_cols:
        if col not in df.columns:
            df[col] = np.nan

    result_cols = [
        "city",
        "city_key",
        "country",
        "overall_raw",
        OUTPUT_COLUMN,
        "sentiment_label",
        "emotional_was_missing",
    ] + optional_cols

    result = df[result_cols].copy()
    return result


def run_and_save(output_dir=None):
    score_df = build_emotional_score_dataframe()

    analyse_df = score_df.drop(columns=["city_key"], errors="ignore")
    analyse_output = save_analyse_csv(analyse_df, ANALYSE_OUTPUT_FILE)

    total_df = update_total_score_from_score_df(
        score_df=score_df,
        score_col=OUTPUT_COLUMN,
        fallback=0.5,
    )

    print("\nBitti kanka! Emotional skor total_score.csv içine eklendi.")
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
