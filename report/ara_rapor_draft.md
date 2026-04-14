# Normal Retina OCT Goruntulerinden Ogrenilen Konvolusyonel Autoencoder ile Patolojik Orneklerin Rekonstruksiyon Hatasi Tabanli Tespiti

## Baslik

**Ingilizce baslik:** Reconstruction-Error-Based Detection of Pathological Retinal OCT Scans Using a Convolutional Autoencoder Trained on Normal Images

## Ozet

Bu calismada, retinal OCT goruntulerinde patolojik ornekleri etiketlenmis patoloji siniflariyla dogrudan ogrenmek yerine, yalnizca normal orneklerden ogrenilen bir convolutional autoencoder ile reconstruction error tabanli anomaly detection yaklasimi gelistirilmistir. Kermany OCT2017 veri kumesindeki train/NORMAL goruntuleri hasta bazli olarak egitim ve validation alt kumelerine ayrilmis, model yalnizca normal anatominin dagilimini ogrenmistir. Test asamasinda NORMAL, CNV, DME ve DRUSEN goruntuleri reconstruction error ile puanlanmis ve validation normal error dagilimindan elde edilen percentil esikleriyle ikili karar uretilmistir. Bu ara rapor surumunde temel model 128x128 gri tonlamali B-scan'ler uzerinde egitilmis, AUROC ana metrik olarak alinmis ve precision, recall, F1, accuracy ile FPR de raporlanmistir. Gercek OCT2017 deneyi sonunda secilen p95 esiginde AUROC 0.9108 ve F1 0.8201 elde edilmistir. Elde edilen ilk bulgular, patolojik siniflarin ortalama reconstruction error degerlerinin normal sinifa gore sistematik olarak daha yuksek oldugunu gostermektedir.

## Abstract

This study investigates reconstruction-error-based anomaly detection for retinal OCT images using a convolutional autoencoder trained only on normal samples. Instead of learning a supervised pathology classifier, the model learns the appearance distribution of healthy retinal anatomy from the Kermany OCT2017 dataset. The training split uses only normal scans and a patient-level validation split is used to estimate operating thresholds from normal reconstruction-error percentiles. During testing, NORMAL, CNV, DME, and DRUSEN scans are scored according to reconstruction error, and anomaly decisions are produced with validation-derived thresholds. On the real OCT2017 experiment, the selected p95 operating point reaches AUROC 0.9108 and F1-score 0.8201. These findings indicate that a normal-only autoencoder can serve as a meaningful early baseline for retinal anomaly screening.

## Anahtar Kelimeler / Keywords

retinal OCT, anomaly detection, autoencoder, reconstruction error, medical imaging, deep learning

## 1. Giris

Retinal hastaliklarin erken tespiti, geri donulmez gorme kaybini azaltmak icin kritik onemdedir. Optik koherens tomografi (OCT), retina tabakalarini yuksek cozunurlukte gosterebildigi icin klinik pratikte sik kullanilan bir goruntuleme yontemidir. Ancak OCT verisinin elle yorumlanmasi zaman alici oldugu gibi, genis tarama programlarinda yuksek uzman emegi gerektirir [1]. Son yillarda derin ogrenme tabanli denetimli modeller OCT siniflandirmasinda guclu sonuclar vermis olsa da, bunlar genellikle her patoloji icin etiketli veri gerektirir [1], [3]. Bu durum, daha once gorulmemis veya yeterince temsil edilmeyen anomalilerin tespitini zorlastirir.

Bu projede problem, normal anatominin ogrenilmesi ve ondan sapmalarin reconstruction error ile yakalanmasi olarak ele alinmistir. Ara rapor kapsamindaki amacimiz, yalnizca normal retina OCT goruntuleri ile egitilen bir convolutional autoencoder'in patolojik test goruntulerini anlamli bicimde ayristirabildigini gosteren, tekrar uretilebilir bir baseline sistem kurmaktir. Bu calismanin farki; patient-level validation, validation-derived threshold secimi ve gercek OCT verisiyle uctan uca calisan deney boru hattini ayni raporda birlestirmesidir.

## 2. Ilgili Calismalar

OCT alaninda derin ogrenme tabanli hastalik siniflandirmasi icin en cok atif alan calismalardan biri Kermany ve ark. tarafindan sunulan Cell 2018 makalesidir [1]. Bu calisma, ayni zamanda bu projede kullanilan halka acik OCT veri kumesinin temellerini de olusturmaktadir [2]. Literaturde bunun devaminda cok sayida denetimli retinal hastalik tespit modeli onerilmis ve OCT'nin otomatik analiz icin uygunlugu guclu bicimde ortaya konmustur [3].

Anomaly detection literaturunde ise normal veriyle egitim yapip anomalileri dagilim disi ornekler olarak ele alan reconstructive ve adversarial yontemler on plana cikmistir. AnoGAN [4] ve GANomaly [5] gibi yaklasimlar normal dagilimi modelleme mantigini sistematiklestirmistir. DRAEM [6] ve ProxyAno [8] ise reconstruction tabanli yapilarin daha ayirt edici hale gelmesine odaklanmistir. Retinal OCT ozelinde Seebock ve ark. [7], Luo ve ark. [9] ve Wang ve ark. [10] gibi calismalar bu alanin artik yalnizca genel anomaly detection degil, retina anatomisine ozel cozumler de gerektirdigini gostermektedir.

Kim ne yapmis ve bu proje neyi farkli yapiyor sorusunu daha acik gostermek icin Tablo 1 verilmistir.

Tablo 1. Ilgili calismalar ve bu projeden farklari.

| study | focus | difference |
| --- | --- | --- |
| Kermany et al. [1] | Supervised OCT classification | Requires pathology labels, unlike our normal-only anomaly setting. |
| AnoGAN [4] | GAN-based anomaly detection | General anomaly detection reference, not retinal OCT-specific. |
| Seebock et al. [7] | Uncertainty-based OCT anomaly detection | Uses anatomy segmentation uncertainty instead of direct reconstruction error. |
| Luo et al. [9] | Multi-resolution retinal autoencoder | More advanced retinal anomaly model; our work is a simpler reproducible baseline. |
| This project | Normal-only OCT anomaly scoring | Patient-level validation split and percentile-based threshold selection on real OCT2017. |

## 3. Yontem

### 3.1 Veri kumesi ve bolme stratejisi

Calismada Kermany OCT2017 veri kumesinin `train` ve `test` klasorleri esas alinmistir [2]. Egitimde yalnizca `train/NORMAL` altindaki goruntuler kullanilmistir. Validation bolmesi image-level degil patient-level olarak yapilmistir; boylece ayni hastaya ait goruntuler train ve validation alt kumelerine ayni anda dusmemistir. Test asamasinda `test/NORMAL`, `test/CNV`, `test/DME` ve `test/DRUSEN` goruntuleri birlikte degerlendirilmis, NORMAL sinifi 0 ve diger tum siniflar anomaly etiketi 1 olarak ele alinmistir.

### 3.2 On isleme

Tum goruntuler tek kanalli gri tonlamaya donusturulmus, `128x128` boyutuna yeniden orneklenmis ve `[0, 1]` araligina normalize edilmistir. Bu ara surumde agresif augmentation uygulanmamistir; amacimiz once sade ve tekrarlanabilir bir baseline kurmaktir.

### 3.3 Model mimarisi

Model, dort asamali bir convolutional encoder-decoder yapisindan olusmaktadir. Encoder kismi 1->32->64->128->256 kanal gecisleri ve max-pooling adimlariyla goruntuyu sikistirirken, ara latent temsil `128` boyutlu bir vektore indirgenmistir. Decoder kismi transpose convolution bloklari ile goruntuyu tekrar 128x128 boyutuna tasimaktadir. Cikis katmaninda sigmoid kullanilarak normalize pikseller uzerinde reconstruction uretilmistir.

### 3.4 Egitim ve esikleme

Model `Adam` optimizer ve `MSE` reconstruction loss ile egitilmistir. En fazla `40` epoch ve `8` patience degerli early stopping kullanilmistir. Validation asamasinda yalnizca normal orneklerin reconstruction error dagilimi incelenmis; p95, p97 ve p99 esikleri hesaplanmistir. Ana operasyon noktasi olarak p95 secilmistir. Boylece threshold seciminde test verisi kullanilmamis ve leakage engellenmistir.

### 3.5 Degerlendirme olcutleri ve deney kurulumu

Ana basari olcutu olarak AUROC secilmistir; cunku anomaly detection senaryosunda threshold'dan bagimsiz ayristirma gucunu yansitir. Bunun yaninda accuracy, precision, recall, F1 ve false positive rate de raporlanmistir. Precision ve recall birlikte yorumlanmis, F1 ise dengeli operasyon noktasi seciminde kullanilmistir. Gercek deney kosusu yaklasik 109.7 dakika surmus ve en iyi validation sonucu 37. epoch'ta elde edilmistir. Deney kurulumu Tablo 2'de ozetlenmistir.

Tablo 2. Deney kurulumu ve temel hiperparametreler.

| setting | value |
| --- | --- |
| Train data | 40715 NORMAL images |
| Validation data | 10425 NORMAL images |
| Test data | 1000 images |
| Input size | 128x128 grayscale |
| Latent dimension | 128 |
| Optimizer | Adam, lr=0.001 |
| Max epochs / patience | 40 / 8 |
| Selected threshold | p95 |
| Training duration | 109.7 minutes |

### 3.6 Sistem akisi

Onerilen is akisi bes adimdan olusmaktadir: normal verinin secilmesi, on isleme, autoencoder egitimi, validation error dagilimindan esik secimi ve testte anomaly scoring. Raporun sonundaki Sekil 1 bu boru hattini gorsel olarak ozetlemektedir. Bu sema, odevde istenen sistem mimarisi beklentisini karsilamak icin eklenmistir.

## 4. Ara Sonuclar

Bu ara raporda uretilen temel ciktilar; egitim/validation loss grafigi, validation reconstruction error histogrami, test error dagilimi, ROC curve, confusion matrix ve ornek reconstruction-residual goruntuleridir. Deney sonunda secilen p95 esiginde elde edilen metrikler asagidaki gibidir:

| Metrik | Deger |
|---|---:|
| AUROC | 0.9108 |
| Accuracy | 0.7670 |
| Precision | 0.9743 |
| Recall | 0.7080 |
| F1 | 0.8201 |
| FPR | 0.0560 |
| Best epoch | 37 |
| Best validation loss | 0.000745 |

Validation percentil esikleri:

| percentile | threshold | accuracy | precision | recall | f1 | fpr | auroc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 95 | 0.0014 | 0.767 | 0.9743 | 0.708 | 0.8201 | 0.056 | 0.9108 |
| 97 | 0.0016 | 0.714 | 0.9813 | 0.6307 | 0.7679 | 0.036 | 0.9108 |
| 99 | 0.0021 | 0.593 | 0.9971 | 0.4587 | 0.6283 | 0.004 | 0.9108 |

Sinif bazli reconstruction error ozeti:

| class_name | sample_count | patient_count | mean_reconstruction_error | std_reconstruction_error |
| --- | --- | --- | --- | --- |
| CNV | 250 | 178 | 0.002921 | 0.001294 |
| DME | 250 | 167 | 0.002592 | 0.001554 |
| DRUSEN | 250 | 169 | 0.001317 | 0.000643 |
| NORMAL | 250 | 171 | 0.000864 | 0.000292 |

Veri bolme ozeti:

| split_name | class_name | image_count | patient_count |
| --- | --- | --- | --- |
| train | NORMAL | 40715 | 2747 |
| val | NORMAL | 10425 | 687 |
| test | CNV | 250 | 178 |
| test | DME | 250 | 167 |
| test | DRUSEN | 250 | 169 |
| test | NORMAL | 250 | 171 |

Sonuclar yalnizca tablo duzeyinde degil, yorum duzeyinde de anlamlidir. p95 esigi p97 ve p99'a gore daha yuksek recall ve F1 vermistir; bu nedenle ara rapor icin daha dengeli operasyon noktasi olarak secilmistir. CNV ve DME siniflari NORMAL goruntulerden belirgin sekilde ayrisirken, DRUSEN sinifinin error dagilimi normale daha yakindir. Bu durum, bazi patolojilerin reconstruction tabanli yaklasimlarda digerlerine gore daha zor ayristigini gostermektedir.

Rapor sonunda verilen Sekil 2, egitim egrisi, ROC performansi, error dagilimi ve reconstruction orneklerini bir araya getirerek ara sonuclarin gorsel ozetini sunmaktadir.

## 5. Tartisma

Baseline model, gorece basit olmasina ragmen normal anatomi dagilimini ogrenerek patolojik siniflarin reconstruction error degerlerini yukseltebilmektedir. Bununla birlikte reconstruction tabanli yontemlerin iyi bilinen bir siniri vardir: guclu decoder yapilari bazen anomalileri de fazla iyi yeniden uretebilir [4], [8]. Kermany veri kumesi image-level etiketler icerir; bu nedenle lokal lesion segmentasyonu icin dogrudan pixel-level ground truth bulunmamaktadir. Ayrica threshold seciminin precision-recall dengesi uzerinde guclu etkisi vardir. Bu nedenle tek bir metrik yerine percentile bazli karsilastirma tablosu korunmustur.

Hesaplama maliyeti de goz ardi edilemez. Egitim kosusu yerel ortamda uzun sayilabilecek bir surede tamamlanmistir ve bu durum veri yukleme ile on isleme hattinin da iyilestirme alani oldugunu gostermektedir. Dolayisiyla mevcut sistem klinik kullanimdan ziyade arastirma ve erken tarama mantiginda degerlendirilmelidir.

## 6. Gelecek Calismalar

Final asamada standart autoencoder yerine VAE veya memory-augmented reconstruction modelleri denenebilir. Residual map kalitesi artirilabilir, daha yuksek giris cozunurlugu ile DRUSEN gibi zorlayici siniflarda ayristirma gucu test edilebilir ve image size, latent boyut ile threshold seciminin etkisi sistematik ablation calismalariyla incelenebilir.

## 7. Sonuc

Bu ara rapor asamasinda, Kermany OCT verisi icin patient-level validation kullanan, yalnizca normal goruntulerle egitilen ve reconstruction error ile patolojik scan tespiti yapan tekrar uretilebilir bir baseline sistem kurulmustur. Gercek veri uzerinde elde edilen AUROC 0.9108 ve F1 0.8201 degerleri, projenin planlama asamasini gecip calisir ve savunulabilir bir noktaya geldigini gostermektedir. Final asamada hedef, bu baseline'i daha guclu anomaly detection yaklasimlariyla genisletmek ve sonuclari karsilastirmali deneylerle desteklemektir.

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
