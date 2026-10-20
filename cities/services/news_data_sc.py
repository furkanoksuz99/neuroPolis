from pathlib import Path
from urllib.parse import quote
from datetime import datetime
import time

import feedparser
import pandas as pd


# ==============================
# AYARLAR
# ==============================

OUTPUT_CSV_NAME = "global_city_news.csv"
OUTPUT_EXCEL_NAME = "global_city_news.xlsx"

MAX_NEWS_PER_CITY = 100

BASE_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_APP_DIR / "data"


# ==============================
# CITY → COUNTRY
# ==============================

cities = [
    ("Washington D.C.", "USA"),
    ("New York", "USA"),
    ("Los Angeles", "USA"),

    ("London", "UK"),

    ("Paris", "France"),
    ("Berlin", "Germany"),

    ("Tokyo", "Japan"),
    ("Osaka", "Japan"),

    ("Canberra", "Australia"),
    ("Sydney", "Australia"),
    ("Melbourne", "Australia"),
    ("Auckland", "New Zealand"),

    ("Ottawa", "Canada"),

    ("Seoul", "South Korea"),
    ("Busan", "South Korea"),

    ("Rome", "Italy"),
    ("Madrid", "Spain"),

    ("Amsterdam", "Netherlands"),
    ("Brussels", "Belgium"),

    ("Vienna", "Austria"),
    ("Bern", "Switzerland"),
    ("Luxembourg City", "Luxembourg"),

    ("Stockholm", "Sweden"),
    ("Oslo", "Norway"),
    ("Copenhagen", "Denmark"),
    ("Helsinki", "Finland"),

    ("Dublin", "Ireland"),
    ("Lisbon", "Portugal"),
    ("Athens", "Greece"),

    ("Warsaw", "Poland"),
    ("Prague", "Czechia"),
    ("Budapest", "Hungary"),

    ("Bratislava", "Slovakia"),
    ("Ljubljana", "Slovenia"),

    ("Mexico City", "Mexico"),
    ("Monterrey", "Mexico"),

    ("Brasília", "Brazil"),
    ("São Paulo", "Brazil"),

    ("Buenos Aires", "Argentina"),
    ("Santiago", "Chile"),
    ("Bogotá", "Colombia"),
    ("Lima", "Peru"),

    ("Ankara", "Turkey"),
    ("Istanbul", "Turkey"),

    ("Moscow", "Russia"),
    ("Saint Petersburg", "Russia"),

    ("Beijing", "China"),
    ("Shanghai", "China"),

    ("New Delhi", "India"),
    ("Mumbai", "India"),

    ("Jakarta", "Indonesia"),

    ("Riyadh", "Saudi Arabia"),

    ("Johannesburg", "South Africa"),
    ("Cape Town", "South Africa"),

    ("Nairobi", "Kenya"),
    ("Cairo", "Egypt"),

    ("Tel Aviv", "Israel"),
    ("Singapore", "Singapore"),

    ("Baku", "Azerbaijan"),

    ("Reykjavík", "Iceland"),
    ("San José", "Costa Rica"),

    ("Tallinn", "Estonia"),
    ("Riga", "Latvia"),
    ("Vilnius", "Lithuania"),
]

city_to_country = dict(cities)


# ==============================
# DATE PARSE
# ==============================

def parse_entry_date(entry):
    """
    Google News RSS bazen published bilgisi verir.
    Yoksa bugünün tarihini basarız.
    """
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return datetime(*entry.published_parsed[:6]).date()

        if hasattr(entry, "updated_parsed") and entry.updated_parsed:
            return datetime(*entry.updated_parsed[:6]).date()

        return datetime.now().date()

    except Exception:
        return datetime.now().date()


# ==============================
# RSS FETCH
# ==============================

def get_news(city, country, max_news=MAX_NEWS_PER_CITY):
    """
    Tek şehir için Google News RSS üzerinden haber çeker.
    Arama sorgusunda şehir + ülke birlikte kullanıyoruz.
    Çünkü sadece city yazınca bazı şehirler karışabiliyor.
    """
    query = f"{city} {country}"

    url = (
        "https://news.google.com/rss/search?"
        f"q={quote(query)}"
        "&hl=en-US"
        "&gl=US"
        "&ceid=US:en"
    )

    feed = feedparser.parse(url)

    data = []

    for entry in feed.entries[:max_news]:
        data.append({
            "City": city,
            "Country": country,
            "Search Query": query,
            "Title": getattr(entry, "title", None),
            "News Date": parse_entry_date(entry),
            "Fetched Date": datetime.now().date(),
            "Source": getattr(entry, "source", {}).get("title") if hasattr(entry, "source") else None,
            "Link": getattr(entry, "link", None),
            "Data Source": "Google News RSS",
        })

    return data


# ==============================
# DATAFRAME ÜRETİCİ
# ==============================

def build_news_dataframe():
    all_news = []

    for city, country in cities:
        try:
            news = get_news(city, country)
            all_news.extend(news)

            print(f"{city} ✔ {len(news)}")

            time.sleep(0.5)

        except Exception as e:
            print(f"{city} ❌ {e}")

    df = pd.DataFrame(all_news)

    if df.empty:
        quality_report = {
            "row_count": 0,
            "city_count": 0,
            "duplicate_title_count": 0,
            "missing_values": {},
        }

        return df, quality_report

    # Aynı haber başlığı tekrar gelirse temizleyelim.
    df = df.drop_duplicates(subset=["City", "Title"], keep="first")

    # Tarih kolonlarını düzgün formatlayalım.
    df["News Date"] = pd.to_datetime(df["News Date"], errors="coerce")
    df["Fetched Date"] = pd.to_datetime(df["Fetched Date"], errors="coerce")

    column_order = [
        "City",
        "Country",
        "Search Query",
        "Title",
        "News Date",
        "Fetched Date",
        "Source",
        "Link",
        "Data Source",
    ]

    df = df[column_order]

    quality_report = {
        "row_count": len(df),
        "city_count": df["City"].nunique(),
        "duplicate_title_count": int(df.duplicated(subset=["City", "Title"]).sum()),
        "missing_values": df.isna().sum().to_dict(),
    }

    return df, quality_report


# ==============================
# CSV / EXCEL KAYDETME
# ==============================

def save_news_outputs(df, output_dir=None):
    if output_dir is None:
        output_dir = DATA_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / OUTPUT_CSV_NAME
    output_excel = output_dir / OUTPUT_EXCEL_NAME

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    df.to_excel(output_excel, index=False, engine="openpyxl")

    return output_csv, output_excel


# ==============================
# ANA ÇALIŞTIRMA FONKSİYONU
# ==============================

def run_and_save(output_dir=None):
    df, quality_report = build_news_dataframe()
    output_csv, output_excel = save_news_outputs(df, output_dir)

    print("\nDONE ✔")
    print(f"CSV: {output_csv}")
    print(f"Excel: {output_excel}")

    print("\nVeri kalite özeti:")
    print(f"Satır sayısı: {quality_report['row_count']}")
    print(f"Şehir sayısı: {quality_report['city_count']}")
    print(f"Tekrarlı haber başlığı sayısı: {quality_report['duplicate_title_count']}")

    print("\nEksik değerler:")
    print(quality_report["missing_values"])

    print("\nİlk 10 satır:")
    print(df.head(10))

    return {
        "df": df,
        "quality_report": quality_report,
        "output_csv": output_csv,
        "output_excel": output_excel,
    }


if __name__ == "__main__":
    run_and_save()