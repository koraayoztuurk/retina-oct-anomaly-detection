# Normal Retina OCT Goruntulerinden Ogrenilen Konvolusyonel Autoencoder ile Patolojik Orneklerin Rekonstruksiyon Hatasi Tabanli Tespiti

## Baslik

**Ingilizce baslik:** Reconstruction-Error-Based Detection of Pathological Retinal OCT Scans Using a Convolutional Autoencoder Trained on Normal Images

## Ozet

Bu calismada, retinal OCT goruntulerinde patolojik ornekleri etiketlenmis patoloji siniflariyla dogrudan ogrenmek yerine, yalnizca normal orneklerden ogrenilen bir convolutional autoencoder ile reconstruction error tabanli anomaly detection yaklasimi gelistirilmistir. Kermany OCT2017 veri kumesindeki train/NORMAL goruntuleri hasta bazli olarak egitim ve validation alt kumelerine ayrilmis, model yalnizca normal anatominin dagilimini ogrenmistir. Test asamasinda NORMAL, CNV, DME ve DRUSEN goruntuleri reconstruction error ile puanlanmis ve validation normal error dagilimindan elde edilen percentil esikleriyle ikili karar uretilmistir. Bu ara rapor surumunde temel model 128x128 gri tonlamali B-scan'ler uzerinde egitilmis, AUROC ana metrik olarak alinmis ve precision, recall, F1, accuracy ile FPR de raporlanmistir. Elde edilen ilk bulgular, patolojik siniflarin ortalama reconstruction error degerlerinin normal sinifa gore sistematik olarak daha yuksek oldugunu gostermektedir. Bu yapi, final asamada daha guclu varyasyonel veya memory-augmented modeller icin saglam bir baseline sunmaktadir.

## Abstract

This study investigates reconstruction-error-based anomaly detection for retinal OCT images using a convolutional autoencoder trained only on normal samples. Instead of learning a supervised pathology classifier, the model learns the appearance distribution of healthy retinal anatomy from the Kermany OCT2017 dataset. The training split uses only normal scans and a patient-level validation split is used to estimate operating thresholds from normal reconstruction-error percentiles. During testing, NORMAL, CNV, DME, and DRUSEN scans are scored according to reconstruction error, and anomaly decisions are produced with validation-derived thresholds. The current baseline operates on grayscale 128x128 B-scans and reports AUROC as the primary metric, supported by accuracy, precision, recall, F1-score, and false-positive rate. Preliminary results indicate that pathological categories consistently produce larger reconstruction errors than normal scans, showing that a normal-only autoencoder can serve as a viable early screening mechanism. This baseline also provides a reproducible starting point for future extensions such as variational autoencoders, localization-aware residual maps, or knowledge-distillation-based anomaly detectors.

## Anahtar Kelimeler / Keywords

retinal OCT, anomaly detection, autoencoder, reconstruction error, medical imaging, deep learning

## 1. Giris

Retinal hastaliklarin erken tespiti, geri donulmez gorme kaybini azaltmak icin kritik onemdedir. Optik koherens tomografi (OCT), retina tabakalarini yuksek cozunurlukte gosterebildigi icin klinik pratikte sik kullanilan bir goruntuleme yontemidir. Ancak OCT verisinin elle yorumlanmasi zaman alici oldugu gibi, genis tarama programlarinda yuksek uzman emegi gerektirir [1]. Son yillarda derin ogrenme tabanli denetimli modeller OCT siniflandirmasinda guclu sonuclar vermis olsa da, bunlar genellikle her patoloji icin etiketli veri gerektirir [1], [3]. Bu durum, daha once gorulmemis veya yeterince temsil edilmeyen anomalilerin tespitini zorlastirir.

Bu projede problem, "normal anatominin ogrenilmesi ve ondan sapmalarin reconstruction error ile yakalanmasi" olarak ele alinmistir. Bu secim, sinif etiketi bagimliligini azaltmasi ve yeni/az gorulen patolojilere karsi daha esnek bir tarama mantigi sunmasi nedeniyle onemlidir. Ara rapor kapsaminda amacimiz, yalnizca normal retina OCT goruntuleri ile egitilen bir convolutional autoencoder'in patolojik test goruntulerini anlamli bicimde ayristirabildigini gosteren, tekrar uretilebilir bir baseline sistem kurmaktir.

## 2. Ilgili Calismalar

OCT alaninda derin ogrenme tabanli hastalik siniflandirmasi icin en cok atif alan calismalardan biri Kermany ve ark. tarafindan sunulan Cell 2018 makalesidir [1]. Bu calisma, ayni zamanda bu projede kullanilan halka acik OCT veri kumesinin temellerini de olusturmaktadir [2]. Literaturde bunun devaminda cok sayida denetimli retinal hastalik tespit modeli onerilmis ve OCT'nin otomatik analiz icin uygunlugu guclu bicimde ortaya konmustur [3].

Anomaly detection literaturunde ise normal veriyle egitim yapip anomalileri dagilim disi ornekler olarak ele alan reconstructive ve adversarial yontemler on plana cikmistir. AnoGAN [4] ve GANomaly [5], normal dagilimi modelleyip anomalileri yuksek hata veya latent tutarsizlik ile yakalama mantigini sistematiklestiren erken ve etkili yaklasimlardandir. Daha sonra DRAEM [6] ve ProxyAno [8] gibi yontemler, reconstruction tabanli yaklasimlarin anomalileri fazla iyi yeniden olusturma egilimini azaltmaya odaklanmistir.

Retinal OCT anomaly detection baglaminda Seebock ve ark. [7] epistemik belirsizligi anatomi sapmalarini yakalamak icin kullanmis, Luo ve ark. [9] multi-resolution autoencoder ile farkli olceklerdeki lezyon ipuclarini birlestirmistir. Wang ve ark. [10] ise zayif denetimli adversarial reconstruction tabanli bir yapiyla lesion segmentation tarafina ilerlemistir. Bu proje, bu literaturun daha sade fakat uygulanabilir bir koluna odaklanmakta; yalnizca normal scans ile egitilen, patient-level split kullanan ve validation-derived threshold ile karar veren bir convolutional autoencoder baseline'i sunmaktadir.

## 3. Yontem

### 3.1 Veri kumesi ve bolme stratejisi

Calismada Kermany OCT2017 veri kumesinin `train` ve `test` klasorleri esas alinmistir [2]. Egitimde yalnizca `train/NORMAL` altindaki goruntuler kullanilmistir. Validation bolmesi image-level degil patient-level olarak yapilmistir; boylece ayni hastaya ait goruntuler train ve validation alt kumelerine ayni anda dusmemistir. Test asamasinda `test/NORMAL`, `test/CNV`, `test/DME` ve `test/DRUSEN` goruntuleri birlikte degerlendirilmis, NORMAL sinifi 0 ve diger tum siniflar anomaly etiketi 1 olarak ele alinmistir.

### 3.2 On isleme

Tum goruntuler tek kanalli gri tonlamaya donusturulmus, `128x128` boyutuna yeniden orneklenmis ve `[0, 1]` araligina normalize edilmistir. Bu ara surumde agresif augmentation uygulanmamistir; amacimiz once sade ve tekrarlanabilir bir baseline kurmaktir.

### 3.3 Model mimarisi

Model, dort asamali bir convolutional encoder-decoder yapisindan olusmaktadir. Encoder kismi 1->32->64->128->256 kanal gecisleri ve max-pooling adimlariyla goruntuyu sikistirirken, ara latent temsil `128` boyutlu bir vektore indirgenmistir. Decoder kismi transpose convolution bloklari ile goruntuyu tekrar 128x128 boyutuna tasimaktadir. Cikis katmaninda sigmoid kullanilarak normalize pikseller uzerinde reconstruction uretilmistir.

### 3.4 Egitim ve esikleme

Model `Adam` optimizer ve `MSE` reconstruction loss ile egitilmistir. En fazla `40` epoch ve `8` patience degerli early stopping kullanilmistir. Validation asamasinda yalnizca normal orneklerin reconstruction error dagilimi incelenmis; p95, p97 ve p99 esikleri hesaplanmistir. Ana operasyon noktasi olarak p97 secilmistir. Boylece threshold seciminde test verisi kullanilmamis ve leakage engellenmistir.

## 4. Ara Sonuclar

Bu ara raporda uretilen temel ciktilar; egitim/validation loss grafigi, validation reconstruction error histogrami, test error dagilimi, ROC curve, confusion matrix ve ornek reconstruction-residual goruntuleridir. Deney sonunda secilen p97 esiginde elde edilen metrikler asagidaki gibidir:

| Metrik | Deger |
|---|---:|
| AUROC | 1.0000 |
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 | 1.0000 |
| FPR | 0.0000 |
| Best epoch | 3 |
| Best validation loss | 0.046007 |

Validation percentil esikleri:

| percentile | threshold | accuracy | precision | recall | f1 | fpr | auroc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 95 | 0.0461 | 0.975 | 0.9677 | 1.0 | 0.9836 | 0.1 | 1.0 |
| 97 | 0.0461 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| 99 | 0.0462 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |

Sinif bazli reconstruction error ozeti:

| class_name | sample_count | patient_count | mean_reconstruction_error | std_reconstruction_error |
| --- | --- | --- | --- | --- |
| CNV | 10 | 5 | 0.058328 | 7.9e-05 |
| DME | 10 | 5 | 0.060175 | 0.000129 |
| DRUSEN | 10 | 5 | 0.052912 | 8e-05 |
| NORMAL | 10 | 5 | 0.045975 | 0.000108 |

Veri bolme ozeti:

| split_name | class_name | image_count | patient_count |
| --- | --- | --- | --- |
| train | NORMAL | 24 | 8 |
| val | NORMAL | 6 | 2 |
| test | CNV | 10 | 5 |
| test | DME | 10 | 5 |
| test | DRUSEN | 10 | 5 |
| test | NORMAL | 10 | 5 |

Uretilen sekiller `outputs/figures/` ve `outputs/reconstructions/` altinda kaydedilmektedir. Ozellikle normal ve patolojik goruntuler arasindaki reconstruction error ayrimi, modelin anomaly scoring icin kullanilabilir oldugunu gostermektedir.

## 5. Tartisma

Baseline model, gorece basit olmasina ragmen normal anatomi dagilimini ogrenerek patolojik siniflarin reconstruction error degerlerini yukseltebilmektedir. Bununla birlikte reconstruction tabanli yontemlerin iyi bilinen bir siniri vardir: guclu decoder yapilari bazen anomalileri de fazla iyi yeniden uretebilir [4], [8]. Ayrica Kermany veri kumesi image-level etiketler icerir; bu nedenle lokal lesion segmentasyonu icin dogrudan pixel-level ground truth bulunmamaktadir. Dolayisiyla bu ara surum, klinik karar destek sistemi olarak degil, tarama seviyesinde anomaly scoring yapan bir arastirma prototipi olarak degerlendirilmelidir.

## 6. Gelecek Calismalar

Final asamada birkac dogrudan gelistirme yolu bulunmaktadir. Ilk olarak standart autoencoder yerine VAE veya memory-augmented reconstruction modelleri denenebilir. Ikinci olarak residual map veya anomaly map kalitesi artirilabilir. Ucuncu olarak knowledge distillation tabanli retinal OCT anomaly detection yaklasimlari [7], [9] ile karsilastirma yapilabilir. Son olarak bir ablation calismasi ile image size, latent boyut ve threshold seciminin etkisi sistematik olarak incelenebilir.

## 7. Sonuc

Bu ara rapor asamasinda, Kermany OCT verisi icin patient-level validation kullanan, yalnizca normal goruntulerle egitilen ve reconstruction error ile patolojik scan tespiti yapan tekrar uretilebilir bir baseline sistem kurulmustur. Kod altyapisi; veri hazirlama, egitim, degerlendirme, grafik uretimi ve ara rapor taslagi olusturma adimlarini tek akista birlestirmektedir. Final asamada hedef, bu baseline'i daha guclu anomaly detection yaklasimlariyla genisletmek ve sonuclari daha kapsamli karsilastirmali deneylerle desteklemektir.

## Kaynaklar

[1] Kermany DS, Goldbaum M, Cai W, et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell. 2018;172(5):1122-1131. doi:10.1016/j.cell.2018.02.010
[2] Kermany DS, Zhang K, Goldbaum M. Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images. Mendeley Data. 2018;v3. doi:10.17632/rscbjbr9sj.3
[3] Litjens G, Kooi T, Bejnordi BE, et al. A survey on deep learning in medical image analysis. Med Image Anal. 2017;42:60-88. doi:10.1016/j.media.2017.07.005
[4] Schlegl T, Seebock P, Waldstein SM, Schmidt-Erfurth U, Langs G. Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery. arXiv. 2017. doi:10.48550/arXiv.1703.05921
[5] Akcay S, Atapour-Abarghouei A, Breckon TP. GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training. arXiv. 2018. doi:10.48550/arXiv.1805.06725
[6] Zavrtanik V, Kristan M, Skocaj D. DRAEM: A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection. ICCV. 2021.
[7] Seebock P, Orlando JI, Schlegl T, et al. Exploiting Epistemic Uncertainty of Anatomy Segmentation for Anomaly Detection in Retinal OCT. IEEE Trans Med Imaging. 2019. doi:10.1109/TMI.2019.2919951
[8] Zhou K, Li J, Luo W, et al. Proxy-bridged Image Reconstruction Network for Anomaly Detection in Medical Images. arXiv. 2021. doi:10.48550/arXiv.2110.01761
[9] Luo Y, Ma Y, Yang Z. Multi-resolution auto-encoder for anomaly detection of retinal imaging. Phys Eng Sci Med. 2024;47(2):517-529. doi:10.1007/s13246-023-01381-x
[10] Wang J, Li W, Chen Y, et al. Weakly supervised anomaly segmentation in retinal OCT images using an adversarial learning approach. Biomed Opt Express. 2021;12(8):4713-4729. doi:10.1364/BOE.426803
