from pathlib import Path

import pandas as pd
import numpy as np

from cities.analyse.analysis_common import (
    DATA_DIR,
    TOTAL_SCORE_FILE,
    ANALYSE_DATA_DIR,
    GREEN_DATA,
    read_csv_safely,
    clean_columns,
    clean_numeric,
    city_key_from_any_name,
    to_0_1_score,
    fill_numeric_with_median,
    update_total_score_from_score_df,
    save_analyse_csv,
)


# ============================================================
# ENVIRONMENTAL SCORE
# ------------------------------------------------------------
# Okuduğu dosyalar:
# - cities/data/traffic_score_results.csv
# - cities/data/global_aqi_raporu.csv
#
# Yazdığı dosyalar:
# - cities/analyse_data/environmental_score.csv
# - cities/analyse_data/environmental_fill_report.csv
# - cities/total_data/total_score.csv
#
# Çıktı kolonu:
# - envoirmental
# ============================================================

TRAFFIC_FILE = DATA_DIR / "traffic_score_results.csv"
AQI_FILE = DATA_DIR / "global_aqi_raporu.csv"

ANALYSE_OUTPUT_FILE = ANALYSE_DATA_DIR / "environmental_score.csv"
FILL_REPORT_FILE = ANALYSE_DATA_DIR / "environmental_fill_report.csv"

OUTPUT_COLUMN = "envoirmental"

WEIGHTS = {
    "traffic_good": 0.25,
    "air_quality_good": 0.32,
    "green_good": 0.30,
    "builtup_good": 0.13,
}


def prepare_green_data():
    df_green = pd.DataFrame(
        GREEN_DATA,
        columns=["city", "area_norm", "green_good", "builtup_norm"],
    )

    df_green["builtup_good"] = 1 - df_green["builtup_norm"]
    df_green["city_key"] = df_green["city"].apply(city_key_from_any_name)

    return df_green[
        [
            "city",
            "city_key",
            "area_norm",
            "green_good",
            "builtup_norm",
            "builtup_good",
        ]
    ].copy()


def prepare_traffic_data():
    if not TRAFFIC_FILE.exists():
        print(f"\nTrafik dosyası yok: {TRAFFIC_FILE}")
        print("traffic_good medyan/fallback ile doldurulacak.")
        return pd.DataFrame(columns=["city_key", "traffic_good"])

    df = read_csv_safely(TRAFFIC_FILE)
    df = clean_columns(df)

    if "city" not in df.columns:
        raise ValueError(
            f"traffic_score_results.csv içinde city kolonu yok. Mevcut kolonlar: {list(df.columns)}"
        )

    score_col = None
    for candidate in ["traffic_score_0_1", "traffic_score", "flow_percent"]:
        if candidate in df.columns:
            score_col = candidate
            break

    if score_col is None:
        raise ValueError(
            f"traffic_score_results.csv içinde trafik skoru yok. Mevcut kolonlar: {list(df.columns)}"
        )

    df["city_key"] = df["city"].apply(city_key_from_any_name)
    df["traffic_good"] = to_0_1_score(df[score_col])

    # flow_percent geldiyse yüksek akış iyi; trafik skoru zaten iyi kabul edilir.
    out = (
        df.groupby("city_key", as_index=False)
        .agg({"traffic_good": "mean"})
    )
    out["traffic_good"] = out["traffic_good"].clip(0, 1).round(4)
    return out


def prepare_aqi_data():
    if not AQI_FILE.exists():
        print(f"\nAQI dosyası yok: {AQI_FILE}")
        print("air_quality_good medyan/fallback ile doldurulacak.")
        return pd.DataFrame(columns=["city_key", "air_quality_good"])

    df = read_csv_safely(AQI_FILE)
    df = clean_columns(df)

    city_col = "query_city" if "query_city" in df.columns else "city" if "city" in df.columns else None

    if city_col is None:
        raise ValueError(
            f"global_aqi_raporu.csv içinde query_city/city kolonu yok. Mevcut kolonlar: {list(df.columns)}"
        )

    score_col = None
    for candidate in ["air_quality_score_0_1", "air_quality_score", "aqi_score"]:
        if candidate in df.columns:
            score_col = candidate
            break

    if score_col is None:
        raise ValueError(
            f"global_aqi_raporu.csv içinde air_quality_score_0_1 kolonu yok. Mevcut kolonlar: {list(df.columns)}"
        )

    df["city_key"] = df[city_col].apply(city_key_from_any_name)
    df["air_quality_good"] = to_0_1_score(df[score_col])

    out = (
        df.groupby("city_key", as_index=False)
        .agg({"air_quality_good": "mean"})
    )
    out["air_quality_good"] = out["air_quality_good"].clip(0, 1).round(4)
    return out


def build_environmental_score_dataframe():
    df_green = prepare_green_data()
    df_traffic = prepare_traffic_data()
    df_aqi = prepare_aqi_data()

    df = df_green.merge(df_traffic, on="city_key", how="left")
    df = df.merge(df_aqi, on="city_key", how="left")

    model_cols = [
        "traffic_good",
        "air_quality_good",
        "green_good",
        "builtup_good",
    ]

    fill_report = []

    for col in model_cols:
        df[col], was_missing, fill_value = fill_numeric_with_median(
            df[col],
            fallback=0.5,
            lower=0,
            upper=1,
        )
        df[col + "_was_missing"] = was_missing
        fill_report.append({
            "column": col,
            "missing_count": int(was_missing.sum()),
            "filled_with": round(float(fill_value), 4),
        })

    fill_report_df = pd.DataFrame(fill_report)
    save_analyse_csv(fill_report_df, FILL_REPORT_FILE)

    print("\nEnvironmental eksik doldurma raporu:")
    print(fill_report_df.to_string(index=False))

    df[OUTPUT_COLUMN] = (
        WEIGHTS["traffic_good"] * df["traffic_good"]
        + WEIGHTS["air_quality_good"] * df["air_quality_good"]
        + WEIGHTS["green_good"] * df["green_good"]
        + WEIGHTS["builtup_good"] * df["builtup_good"]
    )

    df[OUTPUT_COLUMN] = df[OUTPUT_COLUMN].clip(0, 1).round(4)

    result = df[
        [
            "city",
            "city_key",
            "traffic_good",
            "air_quality_good",
            "area_norm",
            "green_good",
            "builtup_norm",
            "builtup_good",
            "traffic_good_was_missing",
            "air_quality_good_was_missing",
            "green_good_was_missing",
            "builtup_good_was_missing",
            OUTPUT_COLUMN,
        ]
    ].copy()

    result = result.sort_values(by=OUTPUT_COLUMN, ascending=False).reset_index(drop=True)

    return result, fill_report_df


def run_and_save(output_dir=None):
    score_df, fill_report_df = build_environmental_score_dataframe()

    analyse_df = score_df.drop(columns=["city_key"], errors="ignore")
    analyse_output = save_analyse_csv(analyse_df, ANALYSE_OUTPUT_FILE)

    total_df = update_total_score_from_score_df(
        score_df=score_df,
        score_col=OUTPUT_COLUMN,
        fallback=0.5,
    )

    print("\nBitti kanka! Environmental skor üretildi.")
    print(f"Total CSV: {TOTAL_SCORE_FILE}")
    print(f"Analyse CSV: {analyse_output}")
    print(f"Fill Report CSV: {FILL_REPORT_FILE}")
    print(f"Analyse şehir sayısı: {len(score_df)}")
    print(f"Total şehir sayısı: {len(total_df)}")

    return {
        "df": total_df,
        "score_df": score_df,
        "fill_report": fill_report_df,
        "output_csv": TOTAL_SCORE_FILE,
        "analyse_csv": analyse_output,
        "fill_report_csv": FILL_REPORT_FILE,
    }


if __name__ == "__main__":
    run_and_save()
