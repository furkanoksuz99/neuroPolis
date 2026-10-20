from pathlib import Path
import traceback

from django.conf import settings
from django.core.management.base import BaseCommand

try:
    from cities.services.news_sentiment_sc import run_and_save as run_news_sentiment
except Exception:
    run_news_sentiment = None

from cities.analyse.environmental_analyse import run_and_save as run_environmental_score
from cities.analyse.weather_total_score_sc import run_and_save as run_weather_total_score
from cities.analyse.climate_demographic_total_score_sc import run_and_save as run_climate_demographic_total_score
from cities.analyse.emotional_total_score_sc import run_and_save as run_emotional_total_score
from cities.analyse.economic_total_score_sc import run_and_save as run_economic_total_score
from cities.analyse.final_total_score_sc import run_and_save as run_final_total_score


class Command(BaseCommand):
    help = "Çekilmiş verileri analiz eder, analyse_data çıktılarını üretir ve total_score.csv dosyasını 64 şehirle kurar."

    def handle(self, *args, **options):
        data_dir = Path(settings.BASE_DIR) / "cities" / "data"
        total_data_dir = Path(settings.BASE_DIR) / "cities" / "total_data"
        analyse_data_dir = Path(settings.BASE_DIR) / "cities" / "analyse_data"

        data_dir.mkdir(parents=True, exist_ok=True)
        total_data_dir.mkdir(parents=True, exist_ok=True)
        analyse_data_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write(self.style.WARNING("\nAnaliz işlemi başladı..."))
        self.stdout.write(f"Data klasörü: {data_dir}")
        self.stdout.write(f"Analyse data klasörü: {analyse_data_dir}")
        self.stdout.write(f"Total data klasörü: {total_data_dir}\n")

        tasks = []

        if run_news_sentiment is not None:
            tasks.append({
                "name": "Google News Sentiment Analizi",
                "function": run_news_sentiment,
                "output_dir": data_dir,
            })

        tasks.extend([
            {
                "name": "Environmental Score Analizi",
                "function": run_environmental_score,
                "output_dir": None,
            },
            {
                "name": "Weather Total Score",
                "function": run_weather_total_score,
                "output_dir": None,
            },
            {
                "name": "Climate Demographic Total Score",
                "function": run_climate_demographic_total_score,
                "output_dir": None,
            },
            {
                "name": "Emotional Total Score",
                "function": run_emotional_total_score,
                "output_dir": None,
            },
            {
                "name": "Economic Total Score",
                "function": run_economic_total_score,
                "output_dir": None,
            },
            {
                "name": "Final Total Score Builder",
                "function": run_final_total_score,
                "output_dir": None,
            },
        ])

        results = {}

        for index, task in enumerate(tasks, 1):
            task_name = task["name"]
            task_function = task["function"]
            output_dir = task["output_dir"]

            self.stdout.write("\n" + "=" * 70)
            self.stdout.write(self.style.WARNING(f"[{index}/{len(tasks)}] {task_name} başlıyor..."))
            self.stdout.write("=" * 70)

            try:
                if output_dir is None:
                    result = task_function()
                else:
                    result = task_function(output_dir=output_dir)

                results[task_name] = {"status": "success", "result": result}
                self.stdout.write(self.style.SUCCESS(f"{task_name} tamamlandı."))

                if isinstance(result, dict):
                    for key in ["output_csv", "analyse_csv", "fill_report_csv"]:
                        if result.get(key):
                            self.stdout.write(f"{key}: {result[key]}")

                    df = result.get("df")
                    if df is not None and hasattr(df, "shape"):
                        self.stdout.write(f"Satır sayısı: {df.shape[0]}")

            except Exception as e:
                results[task_name] = {"status": "error", "error": str(e)}
                self.stdout.write(self.style.ERROR(f"{task_name} sırasında hata oluştu."))
                self.stdout.write(self.style.ERROR(str(e)))
                self.stdout.write(traceback.format_exc())

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write(self.style.WARNING("ANALİZ ÖZETİ"))
        self.stdout.write("=" * 70)

        success_count = 0
        error_count = 0

        for task_name, result in results.items():
            if result["status"] == "success":
                success_count += 1
                self.stdout.write(self.style.SUCCESS(f"✔ {task_name}"))
            else:
                error_count += 1
                self.stdout.write(self.style.ERROR(f"✘ {task_name}: {result['error']}"))

        self.stdout.write("\n" + "-" * 70)
        self.stdout.write(f"Başarılı analiz sayısı: {success_count}")
        self.stdout.write(f"Hatalı analiz sayısı: {error_count}")
        self.stdout.write("-" * 70)

        final_file = total_data_dir / "total_score.csv"

        if final_file.exists():
            try:
                import pandas as pd
                final_df = pd.read_csv(final_file)
                self.stdout.write(self.style.SUCCESS(f"\nFinal dosya hazır: {final_file}"))
                self.stdout.write(self.style.SUCCESS(f"Final şehir sayısı: {len(final_df)}"))
            except Exception:
                self.stdout.write(self.style.SUCCESS(f"\nFinal dosya hazır: {final_file}"))

        if error_count == 0:
            self.stdout.write(self.style.SUCCESS("\nBitti kanka, tüm analizler başarıyla tamamlandı."))
        else:
            self.stdout.write(self.style.WARNING("\nBitti ama bazı analizlerde hata var. Üstte yazıyor."))
