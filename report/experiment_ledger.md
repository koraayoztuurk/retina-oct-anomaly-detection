# Final Teknik Deney Envanteri

Bu dosya final raporu yazılırken hiçbir denemenin unutulmaması için tutulmuştur. Başarısız veya zayıf kalan denemeler de raporda kısaca belirtilmelidir.

## Sabit Deney Kuralları

- Eğitimde yalnızca `train/NORMAL` kullanıldı; patolojik sınıflar train'e alınmadı.
- Validation ayrımı hasta ID tabanlı yapıldı; train/validation hasta kesişmesi kontrol edildi.
- Eşik seçimi test setinden değil, validation normal skor dağılımından yapıldı.
- Ana problem binary anomaly detection olarak tutuldu: `NORMAL` vs `CNV/DME/DRUSEN`.
- Ana operasyon eşiği p95 olarak kullanıldı; p97/p99 karşılaştırma tabloları da saklandı.

## En İyi Adaylar

- Image-level final adayı: `ae_mse_l128_e60` + `topk_mse_5`; AUROC=0.9457, F1=0.8464, Recall=0.7387, Precision=0.9911, FPR=0.0200.
- Patient-level final adayı: `ae_mse_l128_e60` + `mean` aggregation + `topk_mse_5`; AUROC=0.9456, F1=0.8936, Recall=0.8292, Precision=0.9688, FPR=0.0872.

## Rapor Checklist

- AE-MSE baseline anlatılacak.
- VAE + KL denemesi anlatılacak; beklenen iyileştirmeyi sağlamadığı saklanmayacak.
- L1 loss ve MSE+SSIM loss denemeleri anlatılacak; SSIM training loss'un zayıf kaldığı belirtilecek.
- Latent size ablasyonu: 64/128/256 karşılaştırılacak.
- Batch size ablasyonu: 16/32/64 karşılaştırılacak.
- Crop/preprocessing denemeleri anlatılacak: content crop ve retina margin crop full run; border crop preview.
- Score ablation anlatılacak: MSE, L1, SSIM score, retina-band, weighted retina, top-k residual ve ensemble skorlar.
- Patient-level evaluation anlatılacak.
- Bootstrap confidence interval anlatılacak.
- Top-k residual explainability ve DRUSEN false negative örnekleri anlatılacak.

## Deney Özeti

| Kategori | Run | Model | Loss | Score | Latent | Batch | Crop | AUROC | F1 | Recall | Precision | FPR | DRUSEN | Not |
|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| training_run | ae_l1_l128 | ae | l1 | l1 | 128 | 32 | none | 0.8459 | 0.7582 | 0.6293 | 0.9535 | 0.0920 | 66 | L1 loss denemesi; MSE baseline'a göre belirgin zayıf kaldı. |
| training_run | ae_mse_l128 | ae | mse | mse | 128 | 32 | none | 0.9104 | 0.8262 | 0.7160 | 0.9764 | 0.0520 | 88 | Ana 40 epoch AE-MSE baseline; score ablation ile güçlü sonuç verdi ancak 60 epoch denemesi tarafından geçildi. |
| training_run | ae_mse_l128_e60 | ae | mse | mse | 128 | 32 | none | 0.9112 | 0.8237 | 0.7133 | 0.9745 | 0.0560 | 84 | AE-MSE latent 128 için 60 epoch denemesi; validation loss ve top-k score metriklerinde 40 epoch koşusunu iyileştirdi. |
| training_run | ae_mse_l256 | ae | mse | mse | 256 | 32 | none | 0.9093 | 0.8277 | 0.7173 | 0.9782 | 0.0480 | 88 | Latent 256 ablasyonu; MSE skorunda güçlü, top-k skorla da stabil. |
| training_run | ae_mse_l256_bs16 | ae | mse | mse | 256 | 16 | none | 0.9102 | 0.8273 | 0.7187 | 0.9747 | 0.0560 | 88 | Batch size 16 denemesi; MSE'de güçlü, top-k AUROC en yükseklerden biri. |
| training_run | ae_mse_l256_bs16_crop | ae | mse | mse | 256 | 16 | content | 0.8767 | 0.7893 | 0.6693 | 0.9617 | 0.0800 | 91 | Content crop denemesi; DRUSEN biraz artsa da genel performans düşük. |
| training_run | ae_mse_l256_bs32_retina_margin | ae | mse | mse | 256 | 32 | retina_margin | 0.8862 | 0.8071 | 0.6947 | 0.9630 | 0.0800 | 96 | Retina margin crop denemesi; DRUSEN yakalama arttı ancak FPR ve genel metrikler zayıfladı. |
| training_run | ae_mse_l256_bs64 | ae | mse | mse | 256 | 64 | none | 0.9058 | 0.8155 | 0.7013 | 0.9741 | 0.0560 | 81 | Batch size 64 denemesi; batch size 16/32'ye göre daha zayıf. |
| training_run | ae_mse_l64 | ae | mse | mse | 64 | 32 | none | 0.9124 | 0.8231 | 0.7133 | 0.9727 | 0.0600 | 85 | Latent 64 ablasyonu; MSE skorunda güçlü ama top-k skorla l128/l256 gerisinde. |
| training_run | ae_mse_ssim_l128 | ae | mse_ssim | mse_ssim | 128 | 32 | none | 0.7857 | 0.6773 | 0.5360 | 0.9199 | 0.1400 | 59 | MSE+SSIM training loss denemesi; bu veri ve ayarda zayıf kaldı. |
| training_run | vae_msekl_l128 | vae | vae_mse_kl | mse | 128 | 32 | none | 0.8758 | 0.7475 | 0.6080 | 0.9702 | 0.0560 | 41 | VAE + KL denemesi; generative latent model beklenen iyileştirmeyi sağlamadı. |
| score_ablation_image_level | ae_mse_l128 |  |  | topk_mse_5 |  |  |  | 0.9437 | 0.8372 | 0.7267 | 0.9873 | 0.0280 |  | Mevcut checkpoint yeniden eğitilmeden farklı anomaly score'lar ile değerlendirildi; top-k residual skorlar en güçlü çıktı. |
| score_ablation_image_level | ae_mse_l128_e60 |  |  | topk_mse_5 |  |  |  | 0.9457 | 0.8464 | 0.7387 | 0.9911 | 0.0200 |  | Mevcut checkpoint yeniden eğitilmeden farklı anomaly score'lar ile değerlendirildi; top-k residual skorlar en güçlü çıktı. |
| score_ablation_image_level | ae_mse_l256 |  |  | topk_mse_5 |  |  |  | 0.9437 | 0.8336 | 0.7213 | 0.9872 | 0.0280 |  | Mevcut checkpoint yeniden eğitilmeden farklı anomaly score'lar ile değerlendirildi; top-k residual skorlar en güçlü çıktı. |
| score_ablation_image_level | ae_mse_l256_bs16 |  |  | topk_mse_5 |  |  |  | 0.9454 | 0.8354 | 0.7240 | 0.9873 | 0.0280 |  | Mevcut checkpoint yeniden eğitilmeden farklı anomaly score'lar ile değerlendirildi; top-k residual skorlar en güçlü çıktı. |
| score_ablation_image_level | ae_mse_l64 |  |  | topk_mse_5 |  |  |  | 0.9389 | 0.8311 | 0.7187 | 0.9854 | 0.0320 |  | Mevcut checkpoint yeniden eğitilmeden farklı anomaly score'lar ile değerlendirildi; top-k residual skorlar en güçlü çıktı. |
| score_ablation_image_level | vae_msekl_l128 |  |  | topk_mse_10 |  |  |  | 0.8840 | 0.7453 | 0.6027 | 0.9762 | 0.0440 |  | Mevcut checkpoint yeniden eğitilmeden farklı anomaly score'lar ile değerlendirildi; top-k residual skorlar en güçlü çıktı. |
| score_ablation_patient_level | ae_mse_l128 |  |  | topk_mse_5 |  |  |  | 0.9459 | 0.8921 | 0.8251 | 0.9709 | 0.0805 |  | Hasta seviyesinde mean aggregation ile hesaplandı. |
| score_ablation_patient_level | ae_mse_l128_e60 |  |  | topk_mse_5 |  |  |  | 0.9456 | 0.8936 | 0.8292 | 0.9688 | 0.0872 |  | Hasta seviyesinde mean aggregation ile hesaplandı. |
| preprocessing_preview | border_crop_preview |  |  |  |  |  | ConservativeBorderCrop |  |  |  |  |  |  | Sadece kenar boşluklarını azaltan daha konservatif kırpma önizlemesi üretildi; full training'e alınmadı. |
| preprocessing_preview | content_crop_preview |  |  |  |  |  | SafeContentCrop |  |  |  |  |  |  | İlk içerik tabanlı kırpma bazı görüntülerde fazla agresif bulundu; yine de full training denendi ve genel performansı düşürdü. |
| preprocessing_preview | retina_margin_crop_preview |  |  |  |  |  | RetinaMarginCrop |  |  |  |  |  |  | Retina sinyaline göre üst/alt boşluk bırakan kırpma denendi; DRUSEN yakalama arttı fakat genel metrikler baseline'ın gerisinde kaldı. |

## Saklanan Ana Artefaktlar

- `outputs/experiments/<run_id>/`: her training run için metrikler, figürler, reconstruction örnekleri ve config.
- `outputs/comparison*/`: model, threshold ve class-wise karşılaştırma tabloları.
- `outputs/score_ablation/`: score ablation, patient-level metrikler, bootstrap CI ve top-k explainability çıktısı.
- `outputs/preprocessing/`: crop önizleme görselleri.
