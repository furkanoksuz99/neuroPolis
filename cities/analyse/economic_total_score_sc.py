import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from cities.analyse.analysis_common import (
    DATA_DIR,
    ANALYSE_DATA_DIR,
    TOTAL_SCORE_FILE,
    get_master_city_score_base,
    clean_columns,
    read_csv_safely,
    clean_numeric,
    city_key_from_any_name,
    detect_column,
    fill_numeric_with_median,
    minmax_01,
    update_total_score_from_score_df,
    save_analyse_csv,
)


# ============================================================
# ECONOMIC SCORE TO TOTAL DATA
# ------------------------------------------------------------
# Okuduğu dosya:
# - cities/data/economi.xlsx
# Sheet:
# - economic_data
#
# Yazdığı dosyalar:
# - cities/analyse_data/economic_score.csv
# - cities/total_data/total_score.csv
#
# Eklediği kolon:
# - economic
# ============================================================

ECONOMIC_FILE = DATA_DIR / "economi.xlsx"
ECONOMIC_SHEET_NAME = "economic_data"
ANALYSE_OUTPUT_FILE = ANALYSE_DATA_DIR / "economic_score.csv"

OUTPUT_COLUMN = "economic"


def read_excel_safely(path, sheet_name):
    if not path.exists():
        print(f"\nEkonomi Excel dosyası yok: {path}")
        return pd.DataFrame()

    try:
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        return clean_columns(df)
    except Exception as e:
        raise ValueError(f"{path} okunamadı. Hata: {e}")


def prepare_economic_source():
    df = read_excel_safely(ECONOMIC_FILE, ECONOMIC_SHEET_NAME)

    if df.empty:
        return pd.DataFrame(columns=["city", "city_key", "country", "GDP", "Inflation (%)", "Unemployment (%)"])

    city_col = detect_column(df, ["City", "city"])
    country_col = detect_column(df, ["Country", "country"])
    gdp_col = detect_column(df, ["GDP", "GDP per capita USD", "GDP Per Capita USD", "GDP_per_capita_USD", "gdp"])
    inflation_col = detect_column(df, ["Inflation (%)", "Inflation", "inflation"])
    unemployment_col = detect_column(df, ["Unemployment (%)", "Unemployment", "unemployment"])

    if city_col is None:
        raise ValueError(f"economi.xlsx içinde City kolonu yok. Mevcut kolonlar: {list(df.columns)}")
    if gdp_col is None:
        raise ValueError(f"economi.xlsx içinde GDP kolonu yok. Mevcut kolonlar: {list(df.columns)}")
    if inflation_col is None:
        raise ValueError(f"economi.xlsx içinde Inflation kolonu yok. Mevcut kolonlar: {list(df.columns)}")
    if unemployment_col is None:
        raise ValueError(f"economi.xlsx içinde Unemployment kolonu yok. Mevcut kolonlar: {list(df.columns)}")

    out = pd.DataFrame()
    out["city"] = df[city_col].astype(str)
    out["city_key"] = out["city"].apply(city_key_from_any_name)
    out["country"] = df[country_col].astype(str) if country_col is not None else ""
    out["GDP"] = df[gdp_col].apply(clean_numeric)
    out["Inflation (%)"] = df[inflation_col].apply(clean_numeric)
    out["Unemployment (%)"] = df[unemployment_col].apply(clean_numeric)

    out = out[out["city_key"] != ""].copy()

    out = (
        out.groupby("city_key", as_index=False)
        .agg({
            "city": "first",
            "country": "first",
            "GDP": "mean",
            "Inflation (%)": "mean",
            "Unemployment (%)": "mean",
        })
    )

    return out


def build_economic_score_dataframe():
    master = get_master_city_score_base()
    source = prepare_economic_source().rename(columns={"city": "source_city"})

    df = master.merge(source, on="city_key", how="left")

    if "country" not in df.columns:
        df["country"] = ""
    df["country"] = df["country"].fillna("")

    # GDP pozitif olmalı. Eksik/0/negatif değerler medyanla doldurulur.
    df["GDP"] = pd.to_numeric(df["GDP"], errors="coerce")
    df.loc[df["GDP"] <= 0, "GDP"] = np.nan

    df["GDP"], gdp_missing, gdp_fill = fill_numeric_with_median(
        df["GDP"],
        fallback=1.0,
        lower=0.000001,
        upper=None,
    )

    df["Inflation (%)"], infl_missing, infl_fill = fill_numeric_with_median(
        df["Inflation (%)"],
        fallback=3.0,
        lower=None,
        upper=None,
    )

    df["Unemployment (%)"], unemp_missing, unemp_fill = fill_numeric_with_median(
        df["Unemployment (%)"],
        fallback=5.0,
        lower=None,
        upper=None,
    )

    df["economic_was_missing"] = gdp_missing | infl_missing | unemp_missing

    df["Log_GDP"] = np.log(df["GDP"])
    df["Inv_Inflation"] = df["Inflation (%)"] * -1
    df["Inv_Unemployment"] = df["Unemployment (%)"] * -1

    features = ["Log_GDP", "Inv_Inflation", "Inv_Unemployment"]

    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[features])

        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(scaled).flatten()
        raw_weights = pca.components_[0]

        proxy_good_economy = scaled[:, 0] + scaled[:, 1] + scaled[:, 2]
        corr = np.corrcoef(pc1, proxy_good_economy)[0, 1]

        if pd.notna(corr) and corr < 0:
            pc1 = -pc1
            raw_weights = -raw_weights

        explained_variance = round(float(pca.explained_variance_ratio_[0]), 4)

        weights_df = pd.DataFrame({
            "variable": features,
            "raw_pca_weight": raw_weights,
            "normalized_weight": np.abs(raw_weights) / np.sum(np.abs(raw_weights)),
        })

        print("\nEkonomik PCA ağırlıkları:")
        print(weights_df.round(4).to_string(index=False))
        print(f"\nPC1 açıklanan varyans oranı: {explained_variance}")

        df["Economic_PC1"] = pc1
        df[OUTPUT_COLUMN] = minmax_01(df["Economic_PC1"]).round(4)

    except Exception as e:
        print(f"\nPCA hesaplanamadı, ekonomik skor basit ortalama ile üretilecek. Hata: {e}")
        z = pd.DataFrame(StandardScaler().fit_transform(df[features]), columns=features)
        composite = z["Log_GDP"] + z["Inv_Inflation"] + z["Inv_Unemployment"]
        df["Economic_PC1"] = composite
        df[OUTPUT_COLUMN] = minmax_01(composite).round(4)

    result = df[
        [
            "city",
            "city_key",
            "country",
            "GDP",
            "Inflation (%)",
            "Unemployment (%)",
            "Log_GDP",
            "Inv_Inflation",
            "Inv_Unemployment",
            "Economic_PC1",
            "economic_was_missing",
            OUTPUT_COLUMN,
        ]
    ].copy()

    for col in ["GDP", "Inflation (%)", "Unemployment (%)", "Log_GDP", "Inv_Inflation", "Inv_Unemployment", "Economic_PC1", OUTPUT_COLUMN]:
        result[col] = pd.to_numeric(result[col], errors="coerce").round(4)

    return result


def run_and_save(output_dir=None):
    score_df = build_economic_score_dataframe()

    analyse_df = score_df.drop(columns=["city_key"], errors="ignore")
    analyse_output = save_analyse_csv(analyse_df, ANALYSE_OUTPUT_FILE)

    total_df = update_total_score_from_score_df(
        score_df=score_df,
        score_col=OUTPUT_COLUMN,
        fallback=0.5,
    )

    print("\nBitti kanka! Economic skor total_score.csv içine eklendi.")
    print(f"Okunan dosya: {ECONOMIC_FILE}")
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
