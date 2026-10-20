from pathlib import Path
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# AYARLAR
# ============================================================

INPUT_CSV_NAME = "global_city_news.csv"
OUTPUT_CSV_NAME = "city_sentiment_scores.csv"
OUTPUT_EXCEL_NAME = "city_sentiment_scores.xlsx"

VADER_WEIGHT = 0.65
TEXTBLOB_WEIGHT = 0.35

BASE_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_APP_DIR / "data"


# ============================================================
# PAKET KONTROLÜ
# ============================================================

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    raise ImportError(
        "vaderSentiment paketi eksik. Kurmak için: pip install vaderSentiment"
    )

try:
    from textblob import TextBlob
except ImportError:
    raise ImportError(
        "textblob paketi eksik. Kurmak için: pip install textblob"
    )


# ============================================================
# VADER ANALYZER
# ============================================================

def create_vader_analyzer():
    vader = SentimentIntensityAnalyzer()

    vader.lexicon.update({
        # Olumlu güçlendirme
        "breakthrough": 3.0,
        "milestone": 2.5,
        "landmark": 2.2,
        "surges": 2.0,
        "soars": 2.0,
        "booms": 1.8,
        "reopens": 1.5,
        "rebounds": 1.8,
        "eases": 1.5,
        "averts": 2.0,
        "prevents": 1.5,
        "rescues": 2.0,
        "acquits": 1.5,
        "cleared": 1.2,

        # Olumsuz güçlendirme
        "collapses": -3.0,
        "crumbles": -2.5,
        "spirals": -2.2,
        "massacres": -3.5,
        "slaughters": -3.5,
        "atrocity": -3.0,
        "hemorrhaging": -2.5,
        "plummets": -2.5,
        "nosedives": -2.5,
        "rampages": -2.5,
        "carnage": -3.0,
        "devastates": -2.8,
        "battered": -2.0,
        "ravaged": -2.5,
        "embattled": -2.0,
        "explodes": -2.0,
        "torched": -2.0,
        "looted": -2.0,
        "deported": -1.8,
        "evicted": -2.0,
        "crackdown": -1.8,
        "contaminated": -2.5,
        "polluted": -2.0,
        "toxic": -2.3,
        "overdoses": -2.5,
        "homicides": -2.8,
        "executes": -2.0,
        "ousted": -1.8,
        "toppled": -1.5,
        "dissolved": -1.3,
    })

    return vader


STOP = {
    "the", "a", "an", "in", "of", "to", "for", "and", "or", "at", "by", "on",
    "is", "are", "was", "were", "has", "have", "had", "be", "as", "from",
    "with", "that", "this", "it", "he", "she", "they", "we", "i", "not",
    "but", "also", "after", "over", "up", "new", "more", "about", "into",
    "its", "his", "her", "their", "than", "been", "would", "will", "said",
    "says", "say", "could", "should", "may", "might", "us", "dc", "la",
    "nyc", "before", "set", "first", "two", "three", "four", "five", "six",
    "amid", "plan", "plans", "planned", "city", "cities", "year", "years",
    "week", "weeks", "month", "months", "day", "days", "local",
    "government", "officials", "report",
}


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def tokenize(text):
    return re.findall(r"[a-z]+", str(text).lower())


def detect_column(df, candidates):
    """
    CSV kolonları bazen City/Title bazen city/title geliyor.
    İkisini de yakalıyoruz.
    """
    existing = list(df.columns)

    for candidate in candidates:
        if candidate in existing:
            return candidate

    lowered = {str(col).lower(): col for col in existing}

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


def classify(raw):
    if raw > 0.05:
        return "positive"
    if raw < -0.05:
        return "negative"
    return "neutral"


# ============================================================
# TEK HABER BAŞLIĞI SKORU
# ============================================================

def score_single(headline, vader):
    headline = str(headline)

    v_scores = vader.polarity_scores(headline)
    v_compound = v_scores["compound"]

    blob = TextBlob(headline)
    tb_polarity = blob.sentiment.polarity
    tb_subjectivity = blob.sentiment.subjectivity

    ensemble_raw = (
        VADER_WEIGHT * v_compound
        + TEXTBLOB_WEIGHT * tb_polarity
    )

    vader_content_weight = v_scores["pos"] + v_scores["neg"]
    confidence = 0.4 + 0.6 * min(vader_content_weight, 1.0)

    return {
        "ensemble_raw": ensemble_raw,
        "vader_compound": v_compound,
        "tb_polarity": tb_polarity,
        "subjectivity": tb_subjectivity,
        "confidence": confidence,
        "vader_pos": v_scores["pos"],
        "vader_neg": v_scores["neg"],
        "vader_neu": v_scores["neu"],
    }


# ============================================================
# ŞEHİR BAZINDA AGGREGATE
# ============================================================

def aggregate_city(rows):
    n = len(rows)

    if n == 0:
        return None

    raws = np.array([r["ensemble_raw"] for r in rows])
    confs = np.array([r["confidence"] for r in rows])
    headlines = [r["headline"] for r in rows]

    w_mean = float(np.average(raws, weights=confs))
    median = float(np.median(raws))

    robust_raw = 0.7 * w_mean + 0.3 * median

    score_raw = max(0.0, min(100.0, 50.0 + robust_raw * 45.0))
    overall = round(score_raw, 1)

    labels = [classify(r) for r in raws]
    label_counter = Counter(labels)

    positive_pct = round(label_counter["positive"] / n * 100, 1)
    negative_pct = round(label_counter["negative"] / n * 100, 1)
    neutral_pct = round(label_counter["neutral"] / n * 100, 1)

    volatility = round(float(np.std(raws)), 3)

    paired = sorted(zip(raws, headlines))
    top_negative = paired[0][1] if paired[0][0] < -0.05 else "—"
    top_positive = paired[-1][1] if paired[-1][0] > 0.05 else "—"

    all_tokens = [
        token
        for headline in headlines
        for token in tokenize(headline)
        if len(token) > 3 and token not in STOP
    ]

    top_words = Counter(all_tokens).most_common(4)
    dominant_theme = ", ".join(word for word, _ in top_words) if top_words else "—"

    avg_subjectivity = round(
        float(np.mean([r["subjectivity"] for r in rows])),
        3
    )

    if overall >= 58:
        sentiment_label = "Pozitif"
    elif overall >= 42:
        sentiment_label = "Nötr"
    else:
        sentiment_label = "Negatif"

    return {
        "Social News Score": overall,
        "Social Label": sentiment_label,
        "Positive News %": positive_pct,
        "Negative News %": negative_pct,
        "Neutral News %": neutral_pct,
        "Headline Count": n,
        "Sentiment Volatility": volatility,
        "Avg Subjectivity": avg_subjectivity,
        "Dominant Theme": dominant_theme,
        "Top Positive Headline": str(top_positive)[:150],
        "Top Negative Headline": str(top_negative)[:150],
        "_weighted_mean": round(w_mean, 4),
        "_median_raw": round(median, 4),
    }


# ============================================================
# CSV OKUMA
# ============================================================

def load_news_csv(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Haber CSV dosyası bulunamadı: {path}")

    df = None

    for sep in [",", "\t", ";"]:
        try:
            temp_df = pd.read_csv(path, sep=sep, encoding="utf-8")
            if len(temp_df.columns) > 1:
                df = temp_df
                break
        except Exception:
            continue

    if df is None:
        raise ValueError("Haber CSV dosyası okunamadı.")

    city_col = detect_column(df, ["City", "city"])
    title_col = detect_column(df, ["Title", "title"])
    country_col = detect_column(df, ["Country", "country"])

    if city_col is None:
        raise ValueError(f"Şehir kolonu bulunamadı. Mevcut kolonlar: {list(df.columns)}")

    if title_col is None:
        raise ValueError(f"Başlık kolonu bulunamadı. Mevcut kolonlar: {list(df.columns)}")

    rename_map = {
        city_col: "City",
        title_col: "Title",
    }

    if country_col is not None:
        rename_map[country_col] = "Country"

    df = df.rename(columns=rename_map)

    if "Country" not in df.columns:
        df["Country"] = ""

    df["City"] = df["City"].fillna("Unknown").astype(str)
    df["Country"] = df["Country"].fillna("").astype(str)
    df["Title"] = df["Title"].fillna("").astype(str)

    return df


# ============================================================
# ANA SENTIMENT DATAFRAME
# ============================================================

def build_news_sentiment_dataframe(input_dir=None):
    if input_dir is None:
        input_dir = DATA_DIR

    input_dir = Path(input_dir)
    input_path = input_dir / INPUT_CSV_NAME

    df = load_news_csv(input_path)

    print(f"Dosya yüklendi: {len(df)} satır")
    print(f"Şehir sayısı: {df['City'].nunique()}")

    vader = create_vader_analyzer()

    scored_rows = []

    for _, row in df.iterrows():
        headline = str(row["Title"]).strip()

        if not headline:
            continue

        score_data = score_single(headline, vader)

        score_data["headline"] = headline
        score_data["city"] = row["City"]
        score_data["country"] = row["Country"]

        scored_rows.append(score_data)

    city_data = {}

    for row in scored_rows:
        city_data.setdefault(row["city"], []).append(row)

    results = []

    for city, rows in city_data.items():
        aggregated = aggregate_city(rows)

        if aggregated is None:
            continue

        aggregated["City"] = city
        aggregated["Country"] = rows[0].get("country", "")

        results.append(aggregated)

    out = pd.DataFrame(results)

    if out.empty:
        return out, {
            "row_count": 0,
            "city_count": 0,
            "avg_social_score": None,
            "top_city": None,
            "bottom_city": None,
        }

    out = out.sort_values(by="Social News Score", ascending=False).reset_index(drop=True)

    column_order = [
        "City",
        "Country",
        "Social News Score",
        "Social Label",
        "Positive News %",
        "Negative News %",
        "Neutral News %",
        "Headline Count",
        "Sentiment Volatility",
        "Avg Subjectivity",
        "Dominant Theme",
        "Top Positive Headline",
        "Top Negative Headline",
        "_weighted_mean",
        "_median_raw",
    ]

    out = out[column_order]

    quality_report = {
        "row_count": len(out),
        "city_count": out["City"].nunique(),
        "avg_social_score": round(out["Social News Score"].mean(), 2),
        "top_city": out.iloc[0]["City"],
        "top_score": out.iloc[0]["Social News Score"],
        "bottom_city": out.iloc[-1]["City"],
        "bottom_score": out.iloc[-1]["Social News Score"],
        "label_counts": out["Social Label"].value_counts().to_dict(),
    }

    return out, quality_report


# ============================================================
# KAYDETME
# ============================================================

def save_news_sentiment_outputs(df, output_dir=None):
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
# DIŞARIDAN ÇALIŞTIRMA
# ============================================================

def run_and_save(output_dir=None):
    df, quality_report = build_news_sentiment_dataframe(input_dir=output_dir)
    output_csv, output_excel = save_news_sentiment_outputs(df, output_dir=output_dir)

    print("\nBitti kanka! Haber sentiment skorları hazır.")
    print(f"CSV: {output_csv}")
    print(f"Excel: {output_excel}")

    print("\nÖzet:")
    print(f"Şehir sayısı: {quality_report['city_count']}")
    print(f"Ortalama sosyal skor: {quality_report['avg_social_score']}")
    print(f"En pozitif şehir: {quality_report['top_city']} - {quality_report.get('top_score')}")
    print(f"En negatif şehir: {quality_report['bottom_city']} - {quality_report.get('bottom_score')}")
    print(f"Label dağılımı: {quality_report.get('label_counts')}")

    return {
        "df": df,
        "quality_report": quality_report,
        "output_csv": output_csv,
        "output_excel": output_excel,
    }


if __name__ == "__main__":
    run_and_save()