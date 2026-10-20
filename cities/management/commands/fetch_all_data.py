from pathlib import Path
import traceback

from django.conf import settings
from django.core.management.base import BaseCommand

from cities.services.population_data_sc import run_and_save as run_population_data
from cities.services.economic_data_sc import run_and_save as run_economic_data
from cities.services.news_data_sc import run_and_save as run_news_data
from cities.services.weather_data_sc import run_and_save as run_current_weather_data
from cities.services.news_sentiment_sc import run_and_save as run_news_sentiment
from cities.services.score_formula_sc import run_and_save as run_score_formula
from cities.services.air_sc import run_and_save as run_aqi_data
from cities.services.trafic_sc import run_and_save as run_traffic_data



class Command(BaseCommand):
    help = "Tüm şehir veri kaynaklarını tek komutla çeker ve cities/data klasörüne kaydeder."

    def handle(self, *args, **options):
        output_dir = Path(settings.BASE_DIR) / "cities" / "data"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write(self.style.WARNING("\nVeri çekme işlemi başladı..."))
        self.stdout.write(f"Çıktı klasörü: {output_dir}\n")
        
        tasks = [
         {
             "name": "Nüfus + Alan + 2025 Yıllık Hava Verisi",
             "function": run_population_data,
         },
         {
             "name": "Ekonomik Veri - IMF",
             "function": run_economic_data,
         },
         {
             "name": "Google News RSS Haber Verisi",
             "function": run_news_data,
         },
         {
             "name": "Google News Sentiment Analizi",
             "function": run_news_sentiment,
         },
         {
             "name": "Anlık Hava Durumu Verisi",
             "function": run_current_weather_data,
         },
         {
             "name": "Şehir Skor Formülleri",
             "function": run_score_formula,
         },
         {
             "name": "WAQI Hava Kalitesi Verisi",
             "function": run_aqi_data,
         },
         {
             "name": "TomTom Trafik Skoru",
             "function": run_traffic_data,
         },
        ]

        results = {}

        for index, task in enumerate(tasks, 1):
            task_name = task["name"]
            task_function = task["function"]

            self.stdout.write("\n" + "=" * 70)
            self.stdout.write(self.style.WARNING(f"[{index}/{len(tasks)}] {task_name} başlıyor..."))
            self.stdout.write("=" * 70)

            try:
                result = task_function(output_dir=output_dir)

                results[task_name] = {
                    "status": "success",
                    "result": result,
                }

                self.stdout.write(self.style.SUCCESS(f"{task_name} tamamlandı."))

                if isinstance(result, dict):
                    output_csv = result.get("output_csv")
                    output_excel = result.get("output_excel")

                    if output_csv:
                        self.stdout.write(f"CSV: {output_csv}")

                    if output_excel:
                        self.stdout.write(f"Excel: {output_excel}")

            except Exception as e:
                results[task_name] = {
                    "status": "error",
                    "error": str(e),
                }

                self.stdout.write(self.style.ERROR(f"{task_name} sırasında hata oluştu."))
                self.stdout.write(self.style.ERROR(str(e)))
                self.stdout.write(traceback.format_exc())

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write(self.style.WARNING("GENEL ÖZET"))
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
        self.stdout.write(f"Başarılı işlem sayısı: {success_count}")
        self.stdout.write(f"Hatalı işlem sayısı: {error_count}")
        self.stdout.write("-" * 70)

        if error_count == 0:
            self.stdout.write(self.style.SUCCESS("\nTüm veriler başarıyla çekildi."))
        else:
            self.stdout.write(self.style.WARNING("\nBitti ama bazı veri kaynaklarında hata var. Üstte yazıyor."))