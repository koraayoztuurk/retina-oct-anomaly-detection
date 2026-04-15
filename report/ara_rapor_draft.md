# Normal Retina OCT Görüntülerinden Öğrenilen Konvolüsyonel Autoencoder ile Patolojik Örneklerin Yeniden Oluşturma Hatası Tabanlı Tespiti

## Başlık

**İngilizce başlık:** Reconstruction-Error-Based Detection of Pathological Retinal OCT Images Using a Convolutional Autoencoder Trained on Normal Samples

## Özet

Bu çalışmada, retinal OCT görüntülerinde patolojik örnekleri etiketlenmiş patoloji sınıflarıyla doğrudan öğrenmek yerine, yalnızca normal örneklerden öğrenilen bir konvolüsyonel autoencoder ile yeniden oluşturma hatasına (reconstruction error) dayalı anomali tespiti yaklaşımı geliştirilmiştir. Kermany OCT2017 veri kümesindeki `train/NORMAL` görüntüleri hasta düzeyinde eğitim ve doğrulama alt kümelerine ayrılmış, model yalnızca normal anatominin dağılımını öğrenmiştir. Test aşamasında NORMAL, CNV, DME ve DRUSEN görüntüleri yeniden oluşturma hatasına göre puanlanmış ve doğrulama kümesindeki normal hata dağılımından elde edilen persentil eşikleriyle ikili karar üretilmiştir. Bu ara rapor sürümünde temel model 128x128 gri tonlamalı B-kesitleri üzerinde eğitilmiş, AUROC ana ölçüt olarak alınmış; doğruluk, kesinlik, duyarlılık, F1 ve yanlış pozitif oranı da raporlanmıştır. OCT2017 veri kümesi üzerinde yapılan deneylerde seçilen p95 eşiğinde AUROC 0.9108 ve F1 0.8201 elde edilmiştir.

## Abstract

This study investigates reconstruction-error-based anomaly detection for retinal OCT images using a convolutional autoencoder trained only on normal samples. Instead of learning a supervised pathology classifier, the model learns the appearance distribution of healthy retinal anatomy from the Kermany OCT2017 dataset. The training split uses only normal scans and a patient-level validation split is used to estimate operating thresholds from normal reconstruction-error percentiles. During testing, NORMAL, CNV, DME, and DRUSEN scans are scored according to reconstruction error, and anomaly decisions are produced with validation-derived thresholds. On the real OCT2017 experiment, the selected p95 operating point reaches AUROC 0.9108 and F1-score 0.8201. These findings indicate that a normal-only autoencoder can serve as a meaningful early baseline for retinal anomaly screening.

## Anahtar Kelimeler / Keywords

retinal OCT, anomali tespiti, autoencoder, yeniden oluşturma hatası, tıbbi görüntüleme, derin öğrenme

## 1. Giriş

Retinal hastalıkların erken tespiti, geri dönüşü olmayan görme kaybını azaltmak için kritik önemdedir. Optik koherens tomografi (OCT), retina tabakalarını yüksek çözünürlükte gösterebildiği için klinik pratikte sık kullanılan bir görüntüleme yöntemidir. Ancak OCT verisinin elle yorumlanması zaman alıcı olduğu gibi, geniş tarama programlarında yüksek uzman emeği gerektirir [1]. Son yıllarda derin öğrenme tabanlı denetimli modeller OCT sınıflandırmasında güçlü sonuçlar vermiş olsa da, bunlar genellikle her patoloji için etiketli veri gerektirir [1], [3]. Bu durum, daha önce görülmemiş veya yeterince temsil edilmeyen anomalilerin tespitini zorlaştırır.

Bu projede problem, normal anatominin öğrenilmesi ve ondan sapmaların yeniden oluşturma hatası ile yakalanması olarak ele alınmıştır. Ara rapor kapsamındaki amacımız, yalnızca normal retina OCT görüntüleri ile eğitilen bir konvolüsyonel autoencoder modelinin patolojik test görüntülerini anlamlı biçimde ayrıştırabildiğini gösteren, tekrar üretilebilir bir temel model kurmaktır. Bu çalışmada, hasta düzeyinde doğrulama, doğrulama dağılımından türetilen eşik seçimi ve gerçek OCT verisiyle uçtan uca çalışan bir deney hattını içeren tekrar üretilebilir bir temel sistem sunulmuştur.

## 2. İlgili Çalışmalar

OCT alanında derin öğrenme tabanlı hastalık sınıflandırması için en çok atıf alan çalışmalardan biri Kermany ve ark. tarafından sunulan Cell 2018 makalesidir [1]. Bu çalışma, aynı zamanda bu projede kullanılan halka açık OCT veri kümesinin temellerini de oluşturmaktadır [2]. Literatürde bunun devamında çok sayıda denetimli retinal hastalık tespit modeli önerilmiş ve OCT'nin otomatik analiz için uygunluğu güçlü biçimde ortaya konmuştur [3].

Anomali tespiti literatüründe ise normal veriyle eğitim yapıp anomalileri dağılım dışı örnekler olarak ele alan yeniden oluşturma temelli ve adversarial (çekişmeli) yöntemler ön plana çıkmıştır. AnoGAN [4] ve GANomaly [5] gibi yaklaşımlar normal dağılımı modelleme mantığını sistematikleştirmiştir. DRAEM [6] ve ProxyAno [8] ise yeniden oluşturma tabanlı yapıların daha ayırt edici hale gelmesine odaklanmıştır. Retinal OCT özelinde Seebock ve ark. [7], Luo ve ark. [9] ve Wang ve ark. [10] gibi çalışmalar bu alanın artık yalnızca genel amaçlı anomali tespiti değil, retina anatomisine özel çözümler de gerektirdiğini göstermektedir.

Kim ne yapmış ve bu proje neyi farklı yapıyor sorusunu daha açık göstermek için Tablo 1 verilmiştir.

Tablo 1. İlgili çalışmalar ve bu projeden farkları.

| Çalışma | Odak | Fark |
| --- | --- | --- |
| Kermany et al. [1] | Denetimli OCT sınıflandırması | Patoloji etiketleri gerektirir; bizim yaklaşımımız yalnızca normal görüntülerle anomali tespiti yapar. |
| AnoGAN [4] | GAN tabanlı anomali tespiti | Genel amaçlı anomali tespiti yaklaşımıdır; retinal OCT’ye özgü değildir. |
| Seebock et al. [7] | Belirsizlik tabanlı OCT anomali tespiti | Doğrudan rekonstrüksiyon hatası yerine anatomi segmentasyonu belirsizliği kullanır. |
| Luo et al. [9] | Çok çözünürlüklü retinal autoencoder | Daha gelişmiş retinal anomali modeli önerir; bizim çalışmamız ise daha sade ve tekrar üretilebilir bir temel model sunar. |
| Wang et al. [10] | Zayıf denetimli retinal OCT anomali bölütleme | Anomali bölgelerinin konumunu vurgular; bizim yaklaşımımız görüntü düzeyinde puanlama yapan daha sade bir temel modeldir. |
| Bu proje | Normal örneklerle OCT anomali puanlaması | Hasta düzeyinde doğrulama bölmesi ve gerçek OCT2017 verisi üzerinde persentil tabanlı eşik seçimi içerir. |

## 3. Yöntem

### 3.1 Veri kümesi ve bölme stratejisi

Çalışmada Kermany OCT2017 veri kümesinin `train` ve `test` klasörleri esas alınmıştır [2]. Eğitimde yalnızca `train/NORMAL` altındaki görüntüler kullanılmıştır. Doğrulama bölmesi görüntü düzeyinde değil hasta düzeyinde yapılmıştır; böylece aynı hastaya ait görüntüler eğitim ve doğrulama alt kümelerine aynı anda düşmemiştir. Hasta kimlikleri, veri kümesindeki dosya adlarında yer alan `hastalık-hastaID-görüntüNo` yapısından ayrıştırılmıştır. Test aşamasında `test/NORMAL`, `test/CNV`, `test/DME` ve `test/DRUSEN` görüntüleri birlikte değerlendirilmiş, NORMAL sınıfı 0 ve diğer tüm sınıflar anomali etiketi 1 olarak ele alınmıştır.

### 3.2 Ön işleme

Tüm görüntüler tek kanallı gri tonlamaya dönüştürülmüş, `128x128` boyutuna yeniden örneklenmiş ve `[0, 1]` aralığına normalize edilmiştir. Bu ara sürümde yoğun veri artırma uygulanmamıştır; amacımız önce sade ve tekrarlanabilir bir temel model kurmaktır.

### 3.3 Model mimarisi

Model, dört aşamalı bir konvolüsyonel autoencoder yapısından oluşmaktadır. Encoder kısmı `1->32->64->128->256` kanal geçişleri ve max-pooling adımlarıyla görüntüyü sıkıştırırken, ara gizil temsil `128` boyutlu bir vektöre indirgenmiştir. Decoder kısmı transpose convolution blokları ile görüntüyü tekrar `128x128` boyutuna taşımaktadır. Çıkış katmanında sigmoid kullanılarak normalize pikseller üzerinde yeniden oluşturma çıktısı üretilmiştir.

### 3.4 Eğitim ve eşikleme

Model `Adam` optimizer'ı ve `MSE` reconstruction loss ile eğitilmiştir. En fazla `40` epoch ve `8` patience değerli early stopping kullanılmıştır. Doğrulama aşamasında yalnızca normal örneklerin yeniden oluşturma hatası dağılımı incelenmiş; p95, p97 ve p99 eşikleri hesaplanmıştır. Ana operasyon noktası olarak p95 seçilmiştir. Böylece eşik seçiminde test verisi kullanılmamış ve data leakage engellenmiştir.

### 3.5 Değerlendirme ölçütleri ve deney kurulumu

Ana başarı ölçütü olarak AUROC seçilmiştir; çünkü anomali tespiti senaryosunda eşikten bağımsız ayrıştırma gücünü yansıtır. Bunun yanında doğruluk (accuracy), kesinlik (precision), duyarlılık (recall), F1 ve yanlış pozitif oranı (false positive rate, FPR) de raporlanmıştır. Kesinlik ve duyarlılık birlikte yorumlanmış, F1 ise dengeli operasyon noktası seçiminde kullanılmıştır. Gerçek deney koşusu yaklaşık 109.7 dakika sürmüş ve en iyi doğrulama sonucu 37. epoch'ta elde edilmiştir. Deney, NVIDIA GeForce RTX 4060 Laptop GPU içeren yerel bir PyTorch ortamında yürütülmüştür. Deney kurulumu Tablo 2'de özetlenmiştir.

Tablo 2. Deney kurulumu ve temel hiperparametreler.

| Ayar | Değer |
| --- | --- |
| Eğitim verisi | 40715 NORMAL görüntü |
| Doğrulama verisi | 10425 NORMAL görüntü |
| Test verisi | 1000 görüntü |
| Giriş boyutu | 128x128 gri seviye |
| Latent boyut | 128 |
| Optimizasyon | Adam, lr=0.001 |
| Maks. epoch / patience | 40 / 8 |
| Seçilen eşik | p95 |
| Eğitim süresi | 109.7 dakika |

### 3.6 Sistem akışı

Önerilen iş akışı beş adımdan oluşmaktadır: normal verinin seçilmesi, ön işleme, autoencoder eğitimi, doğrulama hata dağılımından eşik seçimi ve testte anomali puanlaması. Şekil 1, önerilen sistemin iş akışını özetlemektedir.

## 4. Ara Sonuçlar

Bu ara raporda üretilen temel çıktılar; eğitim ve doğrulama kayıp grafiği, doğrulama yeniden oluşturma hatası histogramı, test hata dağılımı, ROC eğrisi, karışıklık matrisi ve örnek yeniden oluşturma-kalıntı görüntüleridir. Deney sonunda seçilen p95 eşiğinde elde edilen metrikler aşağıdaki gibidir:

| Metrik | Değer |
|---|---:|
| AUROC | 0.9108 |
| Doğruluk | 0.7670 |
| Kesinlik | 0.9743 |
| Duyarlılık | 0.7080 |
| F1 | 0.8201 |
| FPR | 0.0560 |
| En iyi epoch | 37 |
| En iyi doğrulama kaybı | 0.000745 |

Doğrulama persentil eşikleri:

| Persentil | Eşik | Doğruluk | Kesinlik | Duyarlılık | F1 | FPR | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 95 | 0.0014 | 0.767 | 0.9743 | 0.708 | 0.8201 | 0.056 | 0.9108 |
| 97 | 0.0016 | 0.714 | 0.9813 | 0.6307 | 0.7679 | 0.036 | 0.9108 |
| 99 | 0.0021 | 0.593 | 0.9971 | 0.4587 | 0.6283 | 0.004 | 0.9108 |

Sınıf bazlı yeniden oluşturma hatası özeti:

| Sınıf | Örnek | Hasta | Ort. hata | Std. sapma |
| --- | --- | --- | --- | --- |
| CNV | 250 | 178 | 0.002921 | 0.001294 |
| DME | 250 | 167 | 0.002592 | 0.001554 |
| DRUSEN | 250 | 169 | 0.001317 | 0.000643 |
| NORMAL | 250 | 171 | 0.000864 | 0.000292 |

Veri bölme özeti:

| Bölme | Sınıf | Görüntü | Hasta |
| --- | --- | --- | --- |
| Eğitim | NORMAL | 40715 | 2747 |
| Doğrulama | NORMAL | 10425 | 687 |
| Test | CNV | 250 | 178 |
| Test | DME | 250 | 167 |
| Test | DRUSEN | 250 | 169 |
| Test | NORMAL | 250 | 171 |

Sonuçlar yorum düzeyinde de anlamlıdır. p95 eşiği p97 ve p99'a göre daha yüksek duyarlılık ve F1 vermiştir; bu nedenle ara rapor için daha dengeli operasyon noktası olarak seçilmiştir. CNV ve DME sınıfları NORMAL görüntülerden belirgin şekilde ayrışırken, DRUSEN sınıfının hata dağılımı normale daha yakındır. Bu durum, bazı patolojilerin yeniden oluşturma tabanlı yaklaşımlarda diğerlerine göre daha zor ayrıştığını göstermektedir.

Şekil 2'de eğitim eğrisi, ROC performansı, hata dağılımı ve yeniden oluşturma örnekleri bir arada verilerek ara sonuçlar görsel olarak özetlenmiştir.

## 5. Tartışma

Temel model, görece basit olmasına rağmen normal anatomi dağılımını öğrenerek patolojik sınıfların yeniden oluşturma hatalarını yükseltebilmektedir. Bununla birlikte yeniden oluşturma tabanlı yöntemlerin iyi bilinen bir sınırı vardır: güçlü çözücü yapıları bazen anomalileri de fazla iyi yeniden üretebilir [4], [8]. Kermany veri kümesi görüntü düzeyinde etiketler içerir; bu nedenle yerel lezyon bölütlemesi için doğrudan piksel düzeyinde gerçek etiket bulunmamaktadır. Ayrıca eşik seçiminin kesinlik-duyarlılık dengesi üzerinde güçlü etkisi vardır. Bu nedenle tek bir metrik yerine persentil bazlı karşılaştırma tablosu korunmuştur.

Hesaplama maliyeti de göz ardı edilemez. Eğitim koşusu yerel ortamda uzun sayılabilecek bir sürede tamamlanmıştır ve bu durum veri yükleme ile ön işleme hattının da iyileştirme alanı olduğunu göstermektedir. Dolayısıyla mevcut sistem klinik kullanımdan ziyade araştırma ve erken tarama mantığında değerlendirilmelidir.

## 6. Gelecek Çalışmalar

Final aşamada ilk geliştirme ekseni, mimari seviyesinde daha güçlü yeniden oluşturma modellerinin denenmesi olacaktır. Standart autoencoder yerine VAE, skip-connection içeren daha derin yapılar veya bellek destekli yeniden oluşturma modelleri uygulanabilir. Buna ek olarak MSE yanında L1 ve SSIM tabanlı kayıplar denenerek yeniden oluşturma kalitesi ile anomali duyarlılığı arasındaki denge incelenecektir.

İkinci eksen, veri ve deney tasarımına odaklanacaktır. Daha yüksek giriş çözünürlüğü ile özellikle DRUSEN gibi daha zor ayrıştırılan sınıfların daha iyi temsil edilip edilmediği test edilecek; gizil boyut, mini-batch boyutu (batch size), eşik seçimi ve görüntü boyutu gibi hiperparametreler sistematik bir ablasyon çalışması ile karşılaştırılacaktır.

Üçüncü eksen, yorumlanabilirlik ve karşılaştırmalı değerlendirmedir. Kalıntı (residual) haritaları ve hata haritası görselleştirmeleri kullanılarak modelin hangi bölgelerde sapma ürettiği incelenecek; en iyi ve en kötü örnekler ayrıca tartışılacaktır. Bunun yanında veri yükleme hattı ve eğitim süresi optimize edilerek mevcut temel model en az bir geliştirilmiş varyantla AUROC, F1, duyarlılık ve FPR açısından karşılaştırılacaktır.

## 7. Sonuç

Bu ara rapor aşamasında, Kermany OCT verisi için hasta düzeyinde doğrulama kullanan, yalnızca normal görüntülerle eğitilen ve yeniden oluşturma hatası ile patolojik OCT örneklerini tespit eden tekrar üretilebilir bir temel sistem kurulmuştur. Gerçek veri üzerinde elde edilen AUROC 0.9108 ve F1 0.8201 değerleri, yaklaşımın umut verici olduğunu göstermektedir. Final aşamada hedef, bu temel modeli daha güçlü anomali tespiti yaklaşımlarıyla genişletmek ve sonuçları karşılaştırmalı deneylerle desteklemektir.

## Kaynaklar

[1] D. S. Kermany et al., "Identifying medical diagnoses and treatable diseases by image-based deep learning," Cell, vol. 172, no. 5, pp. 1122-1131.e9, 2018, doi: 10.1016/j.cell.2018.02.010.
[2] D. S. Kermany, K. Zhang, and M. Goldbaum, "Large dataset of labeled optical coherence tomography (OCT) and chest X-ray images," Mendeley Data, ver. 3, 2018, doi: 10.17632/rscbjbr9sj.3.
[3] G. Litjens et al., "A survey on deep learning in medical image analysis," Med. Image Anal., vol. 42, pp. 60-88, 2017, doi: 10.1016/j.media.2017.07.005.
[4] T. Schlegl, P. Seebock, S. M. Waldstein, U. Schmidt-Erfurth, and G. Langs, "Unsupervised anomaly detection with generative adversarial networks to guide marker discovery," arXiv:1703.05921, 2017, doi: 10.48550/arXiv.1703.05921.
[5] S. Akcay, A. Atapour-Abarghouei, and T. P. Breckon, "GANomaly: Semi-supervised anomaly detection via adversarial training," in Proc. Asian Conf. Comput. Vis. (ACCV), pp. 622-637, 2018, doi: 10.1007/978-3-030-20893-6_39.
[6] V. Zavrtanik, M. Kristan, and D. Skocaj, "DRAEM - A discriminatively trained reconstruction embedding for surface anomaly detection," in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pp. 8330-8339, 2021.
[7] P. Seebock et al., "Exploiting epistemic uncertainty of anatomy segmentation for anomaly detection in retinal OCT," IEEE Trans. Med. Imaging, vol. 39, no. 1, pp. 87-98, 2020, doi: 10.1109/TMI.2019.2919951.
[8] K. Zhou et al., "Proxy-bridged image reconstruction network for anomaly detection in medical images," IEEE Trans. Med. Imaging, vol. 41, no. 3, pp. 582-594, 2022, doi: 10.1109/TMI.2021.3118223.
[9] Y. Luo, Y. Ma, and Z. Yang, "Multi-resolution auto-encoder for anomaly detection of retinal imaging," Phys. Eng. Sci. Med., vol. 47, no. 2, pp. 517-529, 2024, doi: 10.1007/s13246-023-01381-x.
[10] J. Wang, W. Li, Y. Chen, et al., "Weakly supervised anomaly segmentation in retinal OCT images using an adversarial learning approach," Biomed. Opt. Express, vol. 12, no. 8, pp. 4713-4729, 2021, doi: 10.1364/BOE.426803.
