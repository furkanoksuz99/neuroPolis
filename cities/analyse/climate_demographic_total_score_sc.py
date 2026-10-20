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
    minmax_01,
    update_total_score_from_score_df,
    save_analyse_csv,
)


# ============================================================
# CLIMATE + DEMOGRAPHIC SCORE TO TOTAL DATA
# ------------------------------------------------------------
# Okuduğu dosya:
# - cities/data/cities_weather_area_population_2025.csv
#
# Yazdığı dosyalar:
# - cities/analyse_data/climate_demographic_score.csv
# - cities/total_data/total_score.csv
#
# Eklediği kolon:
# - climate_demographic
# ============================================================

INPUT_FILE = DATA_DIR / "cities_weather_area_population_2025.csv"
ANALYSE_OUTPUT_FILE = ANALYSE_DATA_DIR / "climate_demographic_score.csv"

OUTPUT_COLUMN = "climate_demographic"

FIXED_DENSITY_WEIGHT = 0.20
REMAINING_WEIGHT = 1 - FIXED_DENSITY_WEIGHT


def temperature_comfort(series, ideal_low, ideal_high, hard_low, hard_high):
    s = pd.to_numeric(series, errors="coerce")
    score = pd.Series(np.nan, index=s.index, dtype=float)

    ideal_mask = (s >= ideal_low) & (s <= ideal_high)
    score[ideal_mask] = 1.0

    low_mask = (s >= hard_low) & (s < ideal_low)
    score[low_mask] = (s[low_mask] - hard_low) / (ideal_low - hard_low)

    high_mask = (s > ideal_high) & (s <= hard_high)
    score[high_mask] = (hard_high - s[high_mask]) / (hard_high - ideal_high)

    outside_mask = (s < hard_low) | (s > hard_high)
    score[outside_mask] = 0.0

    return score.clip(0, 1)


def elevation_comfort(series):
    s = pd.to_numeric(series, errors="coerce")
    score = pd.Series(np.nan, index=s.index, dtype=float)

    score[s <= 1000] = 1.0

    mid_mask = (s > 1000) & (s < 3000)
    score[mid_mask] = (3000 - s[mid_mask]) / 2000

    score[s >= 3000] = 0.0

    return score.clip(0, 1)


def z_standardize(series):
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=1)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def run_pca_weights(df, z_cols):
    data = df[z_cols].copy()
    data = data.fillna(0)

    try:
        corr_matrix = data.corr().fillna(0)
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        loadings = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
        loading_df = pd.DataFrame(
            loadings,
            index=z_cols,
            columns=[f"PC{i + 1}" for i in range(len(z_cols))]
        )
        pc1_loadings = loading_df["PC1"].copy()
        if pc1_loadings.sum() < 0:
            pc1_loadings = -pc1_loadings
        squared_loadings = pc1_loadings ** 2
        if squared_loadings.sum() == 0:
            raise ValueError("PCA loadings sıfır geldi.")
        return squared_loadings / squared_loadings.sum()
    except Exception:
        return pd.Series(1 / len(z_cols), index=z_cols)


def prepare_climate_source():
    if not INPUT_FILE.exists():
        print(f"\nClimate demographic input dosyası yok: {INPUT_FILE}")
        return pd.DataFrame(columns=["city", "city_key"])

    df = read_csv_safely(INPUT_FILE)
    df = clean_columns(df)

    city_col = detect_column(df, ["City Name", "City", "city"])
    country_col = detect_column(df, ["Country", "country"])

    if city_col is None:
        raise ValueError(f"{INPUT_FILE.name} içinde şehir kolonu yok. Mevcut kolonlar: {list(df.columns)}")

    required_map = {
        "Latitude": detect_column(df, ["Latitude", "lat"]),
        "Longitude": detect_column(df, ["Longitude", "lon"]),
        "Elevation (m)": detect_column(df, ["Elevation (m)", "Elevation", "elevation"]),
        "Population": detect_column(df, ["Population", "population"]),
        "Area (sq mi)": detect_column(df, ["Area (sq mi)", "Area", "area"]),
        "Avg High Temp 2025 (F)": detect_column(df, ["Avg High Temp 2025 (F)", "Avg High Temp"]),
        "Avg Low Temp 2025 (F)": detect_column(df, ["Avg Low Temp 2025 (F)", "Avg Low Temp"]),
        "Annual Precipitation Days 2025": detect_column(df, ["Annual Precipitation Days 2025"]),
        "Annual Precipitation 2025 (in)": detect_column(df, ["Annual Precipitation 2025 (in)"]),
        "Annual Sunshine Hours 2025": detect_column(df, ["Annual Sunshine Hours 2025"]),
    }

    missing = [name for name, col in required_map.items() if col is None]
    if missing:
        raise ValueError(f"{INPUT_FILE.name} içinde eksik kolonlar var: {missing}. Mevcut kolonlar: {list(df.columns)}")

    out = pd.DataFrame()
    out["city"] = df[city_col].astype(str)
    out["city_key"] = out["city"].apply(city_key_from_any_name)
    out["country"] = df[country_col].astype(str) if country_col is not None else ""

    for out_col, source_col in required_map.items():
        out[out_col] = df[source_col].apply(clean_numeric)

    out = out[out["city_key"] != ""].copy()

    agg_dict = {"city": "first", "country": "first"}
    for col in required_map.keys():
        agg_dict[col] = "mean"

    out = out.groupby("city_key", as_index=False).agg(agg_dict)
    return out


def build_climate_demographic_score_dataframe():
    master = get_master_city_score_base()
    source = prepare_climate_source().rename(columns={"city": "source_city"})
    work = master.merge(source, on="city_key", how="left")

    if "country" not in work.columns:
        work["country"] = ""
    work["country"] = work["country"].fillna("")

    numeric_defaults = {
        "Latitude": 0,
        "Longitude": 0,
        "Elevation (m)": 500,
        "Population": 1_000_000,
        "Area (sq mi)": 100,
        "Avg High Temp 2025 (F)": 70,
        "Avg Low Temp 2025 (F)": 55,
        "Annual Precipitation Days 2025": 100,
        "Annual Precipitation 2025 (in)": 30,
        "Annual Sunshine Hours 2025": 2200,
    }

    for col, fallback in numeric_defaults.items():
        if col not in work.columns:
            work[col] = np.nan
        lower = 0 if col in ["Population", "Area (sq mi)", "Annual Precipitation Days 2025", "Annual Precipitation 2025 (in)", "Annual Sunshine Hours 2025"] else None
        work[col], missing_mask, fill_value = fill_numeric_with_median(
            work[col],
            fallback=fallback,
            lower=lower,
            upper=None,
        )
        work[col + "_was_missing"] = missing_mask

    work.loc[work["Area (sq mi)"] <= 0, "Area (sq mi)"] = work["Area (sq mi)"].median()
    work.loc[work["Population"] <= 0, "Population"] = work["Population"].median()

    work["Population Density"] = work["Population"] / work["Area (sq mi)"]
    work["Log Population Density"] = np.log1p(work["Population Density"])
    work["D_Population_Density"] = -work["Log Population Density"]

    work["D_High_Temp_Comfort"] = temperature_comfort(
        work["Avg High Temp 2025 (F)"],
        ideal_low=64,
        ideal_high=77,
        hard_low=45,
        hard_high=95,
    )

    work["D_Low_Temp_Comfort"] = temperature_comfort(
        work["Avg Low Temp 2025 (F)"],
        ideal_low=45,
        ideal_high=60,
        hard_low=25,
        hard_high=78,
    )

    work["D_Sunshine"] = work["Annual Sunshine Hours 2025"]
    work["D_Rainy_Days"] = -work["Annual Precipitation Days 2025"]
    work["D_Precipitation_Amount"] = -work["Annual Precipitation 2025 (in)"]
    work["D_Elevation_Comfort"] = elevation_comfort(work["Elevation (m)"])
    work["D_Elevation_Comfort"] = work["D_Elevation_Comfort"].fillna(work["D_Elevation_Comfort"].median()).fillna(0.5)

    pca_cols = [
        "D_High_Temp_Comfort",
        "D_Low_Temp_Comfort",
        "D_Sunshine",
        "D_Rainy_Days",
        "D_Precipitation_Amount",
        "D_Elevation_Comfort",
    ]

    work["Z_Population_Density"] = z_standardize(work["D_Population_Density"])

    z_cols = []
    for col in pca_cols:
        z_col = "Z_" + col.replace("D_", "")
        work[z_col] = z_standardize(work[col])
        z_cols.append(z_col)

    pca_weights = run_pca_weights(work, z_cols)

    final_weights = {"Z_Population_Density": FIXED_DENSITY_WEIGHT}
    for col in z_cols:
        final_weights[col] = REMAINING_WEIGHT * pca_weights[col]

    work["Climate_Demographic_Z_Composite"] = 0.0
    for col, weight in final_weights.items():
        work["Climate_Demographic_Z_Composite"] += work[col].fillna(0) * weight

    work[OUTPUT_COLUMN] = minmax_01(work["Climate_Demographic_Z_Composite"]).round(4)

    result_cols = [
        "city",
        "city_key",
        "country",
        "Latitude",
        "Longitude",
        "Elevation (m)",
        "Population",
        "Area (sq mi)",
        "Population Density",
        "Avg High Temp 2025 (F)",
        "Avg Low Temp 2025 (F)",
        "Annual Precipitation Days 2025",
        "Annual Precipitation 2025 (in)",
        "Annual Sunshine Hours 2025",
        "D_Population_Density",
        "D_High_Temp_Comfort",
        "D_Low_Temp_Comfort",
        "D_Sunshine",
        "D_Rainy_Days",
        "D_Precipitation_Amount",
        "D_Elevation_Comfort",
        "Climate_Demographic_Z_Composite",
        OUTPUT_COLUMN,
    ]

    result = work[result_cols].copy()

    for col in result.columns:
        if col not in ["city", "city_key", "country"]:
            result[col] = pd.to_numeric(result[col], errors="coerce").round(4)

    return result


def run_and_save(output_dir=None):
    score_df = build_climate_demographic_score_dataframe()

    analyse_df = score_df.drop(columns=["city_key"], errors="ignore")
    analyse_output = save_analyse_csv(analyse_df, ANALYSE_OUTPUT_FILE)

    total_df = update_total_score_from_score_df(
        score_df=score_df,
        score_col=OUTPUT_COLUMN,
        fallback=0.5,
    )

    print("\nBitti kanka! Climate demographic skor total_score.csv içine eklendi.")
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
