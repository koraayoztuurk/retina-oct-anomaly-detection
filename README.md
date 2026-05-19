# Retina OCT Anomaly Detection

Bu repo, **Üretken Yapay Zekaya Giriş** dersi kapsamında geliştirilen Retina OCT anomaly detection projesinin kodlarını, deney çıktılarının özetlerini ve teslim raporlarını içerir.

Projenin ana fikri şudur: model yalnızca **normal retina OCT görüntülerinden** normal anatomiyi öğrenir; test aşamasında `CNV`, `DME` ve `DRUSEN` görüntüleri normalden sapma olarak değerlendirilir. Sapma kararı, reconstruction error ve ek anomaly score analizleriyle verilir.

## Proje Özeti

Bu çalışma klasik çok sınıflı hastalık sınıflandırması değildir. Ana problem, **normal vs patolojik** ikili anomaly detection problemidir.

Kullanılan veri kümesi Kermany OCT2017 / Mendeley OCT veri kümesidir. Veri yapısı dört sınıftan oluşur:

- `NORMAL`
- `CNV`
- `DME`
- `DRUSEN`

Eğitimde yalnızca `train/NORMAL` görüntüleri kullanılır. Patolojik sınıflar modele eğitim sırasında gösterilmez. Bu sayede proje, etiketli patolojik veri gerektirmeyen normal-only anomaly detection senaryosunu test eder.

## Final Teknik Sonuç

Final aşamasında en güçlü image-level model şu aday oldu:

```text
ae_mse_l128_e60_plateau_bn + topk_mse_5
```

Bu aday şu bileşenleri kullanır:

- Model: convolutional autoencoder
- Eğitim verisi: sadece `NORMAL` OCT görüntüleri
- Loss: `MSE`
- Latent boyut: `128`
- Görüntü boyutu: `128x128`
- Epoch: `60`
- Learning rate scheduler: `ReduceLROnPlateau`
- Mimari iyileştirme: `BatchNorm2d`
- Anomaly score: piksel bazlı squared residual değerlerinin en yüksek `%5` ortalaması, yani `topk_mse_5`

| Değerlendirme | Run / Score | AUROC | F1 | Recall | Precision | FPR |
|---|---|---:|---:|---:|---:|---:|
| Image-level | `ae_mse_l128_e60_plateau_bn + topk_mse_5` | 0.9487 | 0.8593 | 0.7613 | 0.9862 | 0.0320 |
| Patient-level | `ae_mse_l128_e60_plateau_bn + mean(topk_mse_5)` | 0.9513 | 0.9089 | 0.8541 | 0.9712 | 0.0760 |

Not: En iyi sonucu veren yapı dışında başarısız veya daha zayıf kalan deneyler de özellikle korunmuştur. Final raporda “sadece iyi sonucu seçtik” yerine, VAE, L1, MSE+SSIM, latent ablation, batch-size ablation, crop/preprocessing denemeleri, scheduler, BatchNorm, 256x256 high-resolution denemesi ve score ablation sonuçları birlikte tartışılmıştır.

## Neler Denendi?

Projede aşağıdaki ana deney aileleri çalıştırıldı:

| Deney türü | Amaç | Genel sonuç |
|---|---|---|
| AE-MSE baseline | Ara rapordaki temel sistemi kurmak | Çalışan ve güçlü başlangıç noktası oldu |
| VAE-MSE-KL | Generative latent model etkisini görmek | AE baseline kadar stabil iyileşme sağlamadı |
| L1 loss | MSE yerine mutlak fark duyarlılığını test etmek | Ana adayın gerisinde kaldı |
| MSE+SSIM loss | Yapısal benzerliği loss içine katmak | Beklenen kadar güçlü iyileşme sağlamadı |
| Latent dim `64/128/256` | Sıkıştırma kapasitesinin etkisini görmek | `128` dengeli sonuç verdi |
| Batch size ablation | Mini-batch boyutunun etkisini görmek | Final adayı kadar güçlü olmadı |
| Crop/preprocessing | Siyah/beyaz boşlukların etkisini azaltmak | Bazı örneklerde görüntü içeriğini bozma riski görüldü |
| Extended epoch | 40 epoch yerine 60 epoch denemek | İyileşme sağladı |
| LR scheduler | Plateau durumunda learning rate düşürmek | Faydalı oldu |
| BatchNorm | Eğitimi daha stabil hale getirmek | Final adayını güçlendirdi |
| 256x256 denemesi | Daha yüksek çözünürlükle küçük lezyonları yakalamak | Hesap maliyeti arttı, final adayını geçmedi |
| Score ablation | MSE yerine farklı anomaly score denemek | `topk_mse_5` en iyi image-level aday oldu |
| Patient-level analiz | Aynı hastaya ait görüntüleri birlikte yorumlamak | Image-level sonuca ek güvenilir analiz sağladı |

## Repo Yapısı

```text
.
├── configs/                         # Deney konfigürasyonları
├── data/                            # Yerel veri klasörü, gerçek OCT2017 Git'e eklenmez
├── outputs/                         # Deney çıktıları, metrikler, figürler ve karşılaştırmalar
├── report/                          # Teslim edilen ara rapor ve final rapor dosyaları
├── scripts/                         # Mock dataset ve deney ledger yardımcı scriptleri
├── tests/                           # Unit ve smoke seviyesinde kontroller
├── main.py                          # Tek deney çalıştırma girişi
├── run_experiments.py               # Konfigürasyon dosyasından çoklu deney çalıştırma
├── score_ablation.py                # Eğitilmiş checkpoint üstünden farklı anomaly score denemeleri
├── compare_experiments.py           # Deney sonuçlarını karşılaştırma
├── compare_score_ablations.py       # Score ablation sonuçlarını birleştirme
├── data_utils.py                    # Dataset, patient-level split ve preprocessing yardımcıları
├── model.py                         # AE ve VAE mimarileri
├── losses.py                        # MSE, L1, MSE+SSIM ve VAE loss fonksiyonları
├── train.py                         # Eğitim döngüsü
├── evaluate.py                      # Test, metrik ve görsel üretimi
└── utils.py                         # Output path, güvenli temizlik ve ortak yardımcılar
```

## Veri Kümesi Yerleşimi

Gerçek veri seti GitHub'a eklenmez. Büyük boyutlu olduğu için `.gitignore` içinde tutulur.

Beklenen klasör yapısı:

```text
data/
  oct2017/
    train/
      NORMAL/
      CNV/
      DME/
      DRUSEN/
    test/
      NORMAL/
      CNV/
      DME/
      DRUSEN/
```

Pipeline davranışı:

- Eğitim için yalnızca `data/oct2017/train/NORMAL` kullanılır.
- Validation seti, `train/NORMAL` içinden **patient-level split** ile ayrılır.
- Test için `test/NORMAL`, `test/CNV`, `test/DME`, `test/DRUSEN` birlikte kullanılır.
- Threshold seçimi sadece validation normal error dağılımından yapılır.
- Test seti threshold seçmek için kullanılmaz.
- Dosya adından patient ID ayrıştırılır ve aynı hastanın görüntülerinin train/validation tarafında karışmaması kontrol edilir.

## Kurulum

Bu proje yerelde PyTorch/CUDA ortamında çalıştırılmıştır. Örnek komutlarda mevcut geliştirme ortamı şu Python yolu üzerinden gösterilmiştir:

```powershell
..\odev2\.venv\Scripts\python.exe
```

Farklı bir sanal ortam kullanıyorsanız komutlardaki Python yolunu kendi ortamınıza göre değiştirebilirsiniz.

Bağımlılıkları kurmak için:

```powershell
..\odev2\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Temel bağımlılıklar:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `Pillow`

CUDA durumunu hızlı kontrol etmek için:

```powershell
..\odev2\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Tek Deney Çalıştırma

Basit AE-MSE deneyi:

```powershell
..\odev2\.venv\Scripts\python.exe main.py --run-id ae_mse_l128 --model-type ae --loss-type mse --latent-dim 128
```

Örnek override komutu:

```powershell
..\odev2\.venv\Scripts\python.exe main.py --data-root data/oct2017 --epochs 20 --batch-size 16 --num-workers 4
```

Final adayına yakın tek koşu örneği:

```powershell
..\odev2\.venv\Scripts\python.exe main.py --run-id ae_mse_l128_e60_plateau_bn --model-type ae --loss-type mse --latent-dim 128 --epochs 60 --early-stopping-patience 10 --batch-size 32 --image-size 128 --lr-scheduler plateau --lr-scheduler-factor 0.5 --lr-scheduler-patience 3 --min-learning-rate 0.00001 --use-batch-norm --num-workers 8
```

`main.py` desteklediği temel argümanlar:

- `--model-type {ae,vae}`
- `--loss-type {mse,l1,mse_ssim,vae_mse_kl}`
- `--latent-dim`
- `--image-size`
- `--batch-size`
- `--epochs`
- `--early-stopping-patience`
- `--lr-scheduler {none,plateau}`
- `--use-batch-norm`
- `--crop-mode {none,content,border,retina_margin}`
- `--default-percentile`
- `--threshold-percentiles`
- `--clean-outputs`

## Çoklu Deney Çalıştırma

Final deney matrisini çalıştırmak için:

```powershell
..\odev2\.venv\Scripts\python.exe run_experiments.py --config configs/final_experiments.json --clean-outputs --write-logs
```

Bu komut her koşu için ayrı klasör üretir:

```text
outputs/experiments/<run_id>/
```

Toplu deneylerden sonra karşılaştırma çıktıları varsayılan olarak şu klasöre yazılır:

```text
outputs/comparison/
```

Sadece belirli bir run çalıştırmak için:

```powershell
..\odev2\.venv\Scripts\python.exe run_experiments.py --config configs/final_experiments.json --only ae_mse_l128 vae_msekl_l128
```

Komutları çalıştırmadan görmek için:

```powershell
..\odev2\.venv\Scripts\python.exe run_experiments.py --config configs/final_experiments.json --dry-run
```

Mevcut çıktıları silmeden eksik koşuları tamamlamak için:

```powershell
..\odev2\.venv\Scripts\python.exe run_experiments.py --config configs/final_experiments.json --skip-existing --write-logs
```

## Kullanılan Deney Konfigürasyonları

| Dosya | İçerik |
|---|---|
| `configs/final_experiments.json` | AE, VAE, L1, MSE+SSIM ve latent-size ana deneyleri |
| `configs/extended_epoch_ablation.json` | 60 epoch denemesi |
| `configs/scheduler_ablation.json` | `ReduceLROnPlateau` denemesi |
| `configs/batchnorm_ablation.json` | BatchNorm eklenmiş final aday |
| `configs/batch_size_ablation.json` | Batch size karşılaştırmaları |
| `configs/crop_ablation.json` | İçerik kırpma denemesi |
| `configs/border_crop_ablation.json` | Border kırpma denemesi |
| `configs/retina_margin_crop_ablation.json` | Retina margin temelli kırpma denemesi |
| `configs/highres_ablation.json` | 256x256 high-resolution denemesi |

## Score Ablation

Eğitilmiş checkpoint üstünden farklı anomaly score seçeneklerini test etmek için `score_ablation.py` kullanılır. Bu script yeniden eğitim yapmaz; checkpoint ve kayıtlı config üzerinden validation/test skorlarını tekrar hesaplar.

Final aday için:

```powershell
..\odev2\.venv\Scripts\python.exe score_ablation.py --run-id ae_mse_l128_e60_plateau_bn --eval-batch-size 128 --num-workers 0
```

Tüm score ablation özetlerini birleştirmek için:

```powershell
..\odev2\.venv\Scripts\python.exe compare_score_ablations.py
```

Üretilen ana klasör:

```text
outputs/score_ablation/<run_id>/
```

Bu klasörde şunlar bulunur:

- image-level score karşılaştırmaları
- patient-level score karşılaştırmaları
- threshold karşılaştırmaları
- class-wise özetler
- bootstrap confidence interval dosyaları
- ROC overlay figürleri
- top-k residual explainability grid görselleri
- DRUSEN false negative örnekleri

## Değerlendirme Protokolü

Projede metrik üretimi şu kurallara göre yapılır:

- Ana metrik `AUROC` olarak alınır.
- Operasyon eşiği için ana tercih `p95` validation normal percentile değeridir.
- `p97` ve `p99` eşikleri karşılaştırma amacıyla ayrıca raporlanır.
- Threshold sadece validation normal reconstruction error dağılımından seçilir.
- Test seti threshold seçimi için kullanılmaz.
- Image-level metrikler ve patient-level metrikler ayrı hesaplanır.
- Patient-level analizde aynı hastaya ait görüntü skorları birlikte özetlenir.
- Class-wise analizde `NORMAL`, `CNV`, `DME`, `DRUSEN` için ayrı error ve detection count çıktıları üretilir.
- DRUSEN sınıfı ayrıca incelenir çünkü final deneylerde en zor ayrışan patoloji sınıfı olmuştur.

Üretilen temel metrikler:

- `AUROC`
- `Accuracy`
- `Precision`
- `Recall`
- `F1`
- `FPR`
- `TN / FP / FN / TP`
- class-wise mean reconstruction error
- class-wise detected count
- patient-level AUROC ve F1
- bootstrap confidence intervals

## Çıktı Yapısı

Her tek deney klasörü genel olarak şu yapıyı üretir:

```text
outputs/experiments/<run_id>/
  figures/
    training_loss.png
    roc_curve.png
    confusion_matrix.png
    test_error_distribution.png
    validation_error_histogram.png
    classwise_error_summary.png
  metrics/
    run_config.json
    dataset_checks.json
    dataset_summary.csv
    thresholds.json
    threshold_comparison.csv
    selected_threshold_metrics.json
    classwise_reconstruction_summary.csv
    test_reconstruction_errors.csv
    validation_reconstruction_errors.csv
    training_history.json
  reconstructions/
    reconstruction_examples.png
    best_worst_reconstruction_examples.png
    residual_heatmap_overlay_examples.png
    drusen_false_negative_examples.png
  summary.txt
  run.log
```

Toplu karşılaştırma klasörleri:

```text
outputs/comparison/
outputs/comparison_batch_size/
outputs/comparison_batchnorm/
outputs/comparison_crop/
outputs/comparison_highres/
outputs/comparison_preprocessing/
outputs/comparison_retina_margin/
outputs/comparison_scheduler/
```

Score ablation klasörü:

```text
outputs/score_ablation/
```

Deney defteri:

```text
outputs/experiment_ledger.csv
```

Bu CSV, final raporda hangi denemelerin yapıldığını takip etmek için üretilir.

## Mock Dataset ve Testler

Gerçek OCT2017 veri seti olmadan hızlı smoke test yapmak için küçük sentetik veri üretilebilir:

```powershell
..\odev2\.venv\Scripts\python.exe scripts/create_mock_oct_dataset.py
```

AE smoke test:

```powershell
..\odev2\.venv\Scripts\python.exe main.py --data-root data/mock_oct2017 --run-id smoke_ae --model-type ae --loss-type mse --epochs 2 --batch-size 8 --num-workers 0 --output-root tmp/smoke_ae --clean-outputs
```

VAE smoke test:

```powershell
..\odev2\.venv\Scripts\python.exe main.py --data-root data/mock_oct2017 --run-id smoke_vae --model-type vae --loss-type vae_mse_kl --epochs 2 --batch-size 8 --num-workers 0 --output-root tmp/smoke_vae --clean-outputs
```

Unit testleri çalıştırmak için:

```powershell
..\odev2\.venv\Scripts\python.exe -m unittest discover -s tests
```

Kod syntax kontrolü için:

```powershell
..\odev2\.venv\Scripts\python.exe -m py_compile main.py utils.py run_experiments.py model.py score_ablation.py compare_experiments.py scripts\build_experiment_ledger.py
```

## Rapor Dosyaları

`report/` klasörü sade tutulmuştur. GitHub üzerinde sadece teslim odaklı dosyalar bulunur:

```text
report/
  Grup12_KorayÖztürk_EmirAlpİlhan.pdf
  Grup12_KorayÖztürk_EmirAlpİlhan_Final_Rapor.docx
  Grup12_KorayÖztürk_EmirAlpİlhan_Final_Rapor.pdf
```

Bu dosyalar sırasıyla:

- ara rapor PDF
- final rapor düzenlenebilir Word dosyası
- final rapor PDF

## Git ve Büyük Dosyalar

Aşağıdaki dosyalar/klasörler Git'e alınmaz:

- `data/oct2017/`
- `tmp/`
- geçici Python/cache dosyaları
- çoğu per-run checkpoint klasörü
- eski legacy `outputs/saved_models/`
- büyük veya geçici log dosyaları

Final aday checkpoint dosyası özellikle korunmuştur:

```text
outputs/experiments/ae_mse_l128_e60_plateau_bn/saved_models/best_autoencoder.pt
```

Bunun nedeni, final score ablation sonuçlarının en iyi modeli yeniden eğitmeden tekrar üretilebilmesidir.

## Tekrarlanabilirlik Notları

Projede tekrarlanabilirlik için şu kontroller eklenmiştir:

- Eğitimde patolojik sınıf kullanılmadığı `dataset_checks.json` içinde doğrulanır.
- Train/validation patient ID kesişimi kontrol edilir.
- Eşik seçimi validation normal dağılımına dayandırılır.
- Her run kendi `run_config.json` dosyasını üretir.
- Her run ayrı klasöre yazılır, böylece deney çıktıları birbirini ezmez.
- `--clean-outputs` sadece ilgili output klasörünü temizleyecek şekilde sınırlandırılmıştır.

## Bilinen Limitler

Bu proje klinik tanı sistemi değildir. Çıktılar akademik ders projesi ve yöntem karşılaştırması amacıyla üretilmiştir.

Başlıca limitler:

- Model yalnızca reconstruction tabanlıdır; patolojileri semantik olarak sınıflandırmaz.
- DRUSEN sınıfı diğer patolojilere göre daha zor ayrışmıştır.
- Bazı crop/preprocessing yöntemleri görüntü içeriğini bozabileceği için final adayda kullanılmamıştır.
- 256x256 denemesi hesap maliyetini artırmış ancak final adayı geçmemiştir.
- Daha güçlü localization için piksel düzeyi lezyon anotasyonları gerekir.

## Kısa Komut Özeti

```powershell
# Bağımlılıkları kur
..\odev2\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Unit test
..\odev2\.venv\Scripts\python.exe -m unittest discover -s tests

# Tek deney
..\odev2\.venv\Scripts\python.exe main.py --run-id ae_mse_l128 --model-type ae --loss-type mse --latent-dim 128

# Final deney matrisi
..\odev2\.venv\Scripts\python.exe run_experiments.py --config configs/final_experiments.json --clean-outputs --write-logs

# Final aday score ablation
..\odev2\.venv\Scripts\python.exe score_ablation.py --run-id ae_mse_l128_e60_plateau_bn --eval-batch-size 128 --num-workers 0

# Score ablation özetlerini birleştir
..\odev2\.venv\Scripts\python.exe compare_score_ablations.py

# Deney ledger üret
..\odev2\.venv\Scripts\python.exe scripts/build_experiment_ledger.py
```
