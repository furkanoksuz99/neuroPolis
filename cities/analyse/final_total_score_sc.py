import pandas as pd
import numpy as np

from cities.analyse.analysis_common import (
    ANALYSE_DATA_DIR,
    TOTAL_DATA_DIR,
    TOTAL_SCORE_FILE,
    get_master_city_score_base,
    read_csv_safely,
    clean_columns,
    clean_numeric,
    city_key_from_any_name,
    fill_numeric_with_median,
    save_analyse_csv,
)


# ============================================================
# FINAL TOTAL SCORE BUILDER
# ------------------------------------------------------------
# analyse_data içindeki 5 skor dosyasını okur ve total_score.csv'yi
# SIFIRDAN 64 şehir üzerinden kurar.
#
# Bu dosya total_score.csv'nin 55/56 şehre düşmesini kesin çözer.
# ============================================================

FINAL_FILL_REPORT_FILE = ANALYSE_DATA_DIR / "final_total_fill_report.csv"

FILES = {
    "envoirmental": ANALYSE_DATA_DIR / "environmental_score.csv",
    "weather": ANALYSE_DATA_DIR / "weather_score.csv",
    "climate_demographic": ANALYSE_DATA_DIR / "climate_demographic_score.csv",
    "emotional": ANALYSE_DATA_DIR / "emotional_score.csv",
    "economic": ANALYSE_DATA_DIR / "economic_score.csv",
}


def load_score_file(score_name, file_path):
    master = get_master_city_score_base()

    if not file_path.exists():
        print(f"\n{score_name} dosyası yok: {file_path}")
        print(f"{score_name} kolonu 0.5 fallback ile doldurulacak.")
        out = master.copy()
        out[score_name] = np.nan
        return out

    df = read_csv_safely(file_path)
    df = clean_columns(df)

    if "city" not in df.columns:
        raise ValueError(f"{file_path.name} içinde city kolonu yok. Mevcut kolonlar: {list(df.columns)}")

    if score_name not in df.columns:
        raise ValueError(f"{file_path.name} içinde {score_name} kolonu yok. Mevcut kolonlar: {list(df.columns)}")

    small = pd.DataFrame()
    small["city"] = df["city"].astype(str)
    small["city_key"] = small["city"].apply(city_key_from_any_name)
    small[score_name] = df[score_name].apply(clean_numeric)

    small = small[small["city_key"] != ""].copy()
    small = small.groupby("city_key", as_index=False).agg({score_name: "mean"})

    out = master.merge(small, on="city_key", how="left")
    return out


def build_final_total_score():
    final_df = get_master_city_score_base()
    fill_reports = []

    for score_name, file_path in FILES.items():
        score_df = load_score_file(score_name, file_path)
        score_df = score_df[["city_key", score_name]].copy()

        final_df = final_df.merge(score_df, on="city_key", how="left")

        final_df[score_name], missing_mask, fill_value = fill_numeric_with_median(
            final_df[score_name],
            fallback=0.5,
            lower=0,
            upper=1,
        )

        final_df[score_name] = final_df[score_name].round(4)

        fill_reports.append({
            "column": score_name,
            "missing_count": int(missing_mask.sum()),
            "filled_with": round(float(fill_value), 4),
            "source_file": str(file_path),
        })

    final_df = final_df.drop_duplicates(subset=["city_key"], keep="first")
    final_df = final_df[
        [
            "city",
            "envoirmental",
            "weather",
            "climate_demographic",
            "emotional",
            "economic",
        ]
    ].copy()

    # Şehir sırası ana liste sırası kalsın.
    fill_report_df = pd.DataFrame(fill_reports)

    return final_df, fill_report_df


def run_and_save(output_dir=None):
    TOTAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    final_df, fill_report_df = build_final_total_score()

    final_df.to_csv(
        TOTAL_SCORE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    save_analyse_csv(fill_report_df, FINAL_FILL_REPORT_FILE)

    print("\nBitti kanka! total_score.csv 64 şehir üzerinden sıfırdan kuruldu.")
    print(f"Total CSV: {TOTAL_SCORE_FILE}")
    print(f"Fill Report CSV: {FINAL_FILL_REPORT_FILE}")
    print(f"Şehir sayısı: {len(final_df)}")

    if len(final_df) != 64:
        print("\nUYARI: Final şehir sayısı 64 değil. Master city list kontrol edilmeli.")

    print("\nİlk 10 satır:")
    print(final_df.head(10).to_string(index=False))

    print("\nFinal eksik doldurma raporu:")
    print(fill_report_df.to_string(index=False))

    return {
        "df": final_df,
        "missing_report": fill_report_df,
        "output_csv": TOTAL_SCORE_FILE,
        "fill_report_csv": FINAL_FILL_REPORT_FILE,
    }


if __name__ == "__main__":
    run_and_save()
