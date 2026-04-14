# Normal Retina OCT Görüntülerinden Öğrenilen Konvolüsyonel Autoencoder ile Patolojik Örneklerin Rekonstrüksiyon Hatası Tabanlı Tespiti

## Başlık

**İngilizce başlık:** Reconstruction-Error-Based Detection of Pathological Retinal OCT Scans Using a Convolutional Autoencoder Trained on Normal Images

## Özet

Bu çalışmada, retinal OCT görüntülerinde patolojik örnekleri etiketlenmiş patoloji sınıflarıyla doğrudan öğrenmek yerine, yalnızca normal örneklerden öğrenilen bir convolutional autoencoder ile reconstruction error tabanlı anomaly detection yaklaşımı geliştirilmiştir. Kermany OCT2017 veri kümesindeki `train/NORMAL` görüntüleri hasta bazlı olarak eğitim ve doğrulama alt kümelerine ayrılmış, model yalnızca normal anatominin dağılımını öğrenmiştir. Test aşamasında NORMAL, CNV, DME ve DRUSEN görüntüleri reconstruction error ile puanlanmış ve doğrulama normal error dağılımından elde edilen persentil eşikleriyle ikili karar üretilmiştir. Bu ara rapor sürümünde temel model 128x128 gri tonlamalı B-scan'ler üzerinde eğitilmiş, AUROC ana metrik olarak alınmış ve precision, recall, F1, accuracy ile FPR de raporlanmıştır. Gerçek OCT2017 deneyi sonunda seçilen p95 eşiğinde AUROC 0.9108 ve F1 0.8201 elde edilmiştir. Elde edilen ilk bulgular, patolojik sınıfların ortalama reconstruction error değerlerinin normal sınıfa göre sistematik olarak daha yüksek olduğunu göstermektedir.

## Abstract

This study investigates reconstruction-error-based anomaly detection for retinal OCT images using a convolutional autoencoder trained only on normal samples. Instead of learning a supervised pathology classifier, the model learns the appearance distribution of healthy retinal anatomy from the Kermany OCT2017 dataset. The training split uses only normal scans and a patient-level validation split is used to estimate operating thresholds from normal reconstruction-error percentiles. During testing, NORMAL, CNV, DME, and DRUSEN scans are scored according to reconstruction error, and anomaly decisions are produced with validation-derived thresholds. On the real OCT2017 experiment, the selected p95 operating point reaches AUROC 0.9108 and F1-score 0.8201. These findings indicate that a normal-only autoencoder can serve as a meaningful early baseline for retinal anomaly screening.

## Anahtar Kelimeler / Keywords

retinal OCT, anomaly detection, autoencoder, reconstruction error, medical imaging, deep learning

## 1. Giriş

Retinal hastalıkların erken tespiti, geri dönüşü olmayan görme kaybını azaltmak için kritik önemdedir. Optik koherens tomografi (OCT), retina tabakalarını yüksek çözünürlükte gösterebildiği için klinik pratikte sık kullanılan bir görüntüleme yöntemidir. Ancak OCT verisinin elle yorumlanması zaman alıcı olduğu gibi, geniş tarama programlarında yüksek uzman emeği gerektirir [1]. Son yıllarda derin öğrenme tabanlı denetimli modeller OCT sınıflandırmasında güçlü sonuçlar vermiş olsa da, bunlar genellikle her patoloji için etiketli veri gerektirir [1], [3]. Bu durum, daha önce görülmemiş veya yeterince temsil edilmeyen anomalilerin tespitini zorlaştırır.

Bu projede problem, normal anatominin öğrenilmesi ve ondan sapmaların reconstruction error ile yakalanması olarak ele alınmıştır. Ara rapor kapsamındaki amacımız, yalnızca normal retina OCT görüntüleri ile eğitilen bir convolutional autoencoder'in patolojik test görüntülerini anlamlı biçimde ayrıştırabildiğini gösteren, tekrar üretilebilir bir baseline sistem kurmaktır. Bu çalışmada, hasta-bazlı doğrulama, doğrulama dağılımından türetilen eşik seçimi ve gerçek OCT verisiyle uçtan uca çalışan bir deney hattını içeren tekrar üretilebilir bir baseline sistem sunulmuştur.

## 2. İlgili Çalışmalar

OCT alanında derin öğrenme tabanlı hastalık sınıflandırması için en çok atıf alan çalışmalardan biri Kermany ve ark. tarafından sunulan Cell 2018 makalesidir [1]. Bu çalışma, aynı zamanda bu projede kullanılan halka açık OCT veri kümesinin temellerini de oluşturmaktadır [2]. Literatürde bunun devamında çok sayıda denetimli retinal hastalık tespit modeli önerilmiş ve OCT'nin otomatik analiz için uygunluğu güçlü biçimde ortaya konmuştur [3].

Anomaly detection literatüründe ise normal veriyle eğitim yapıp anomalileri dağılım dışı örnekler olarak ele alan reconstructive ve adversarial yöntemler ön plana çıkmıştır. AnoGAN [4] ve GANomaly [5] gibi yaklaşımlar normal dağılımı modelleme mantığını sistematikleştirmiştir. DRAEM [6] ve ProxyAno [8] ise reconstruction tabanlı yapıların daha ayırt edici hale gelmesine odaklanmıştır. Retinal OCT özelinde Seebock ve ark. [7], Luo ve ark. [9] ve Wang ve ark. [10] gibi çalışmalar bu alanın artık yalnızca genel anomaly detection değil, retina anatomisine özel çözümler de gerektirdiğini göstermektedir.

Kim ne yapmış ve bu proje neyi farklı yapıyor sorusunu daha açık göstermek için Tablo 1 verilmiştir.

Tablo 1. İlgili çalışmalar ve bu projeden farkları.

| çalışma | odak | fark |
| --- | --- | --- |
| Kermany et al. [1] | Denetimli OCT sınıflandırması | Patoloji etiketleri gerektirir; bizim yaklaşımımız yalnızca normal görüntülerle anomaly detection yapar. |
| AnoGAN [4] | GAN tabanlı anomali tespiti | Genel amaçlı anomaly detection yaklaşımıdır; retinal OCT’ye özgü değildir. |
| Seebock et al. [7] | Belirsizlik tabanlı OCT anomali tespiti | Doğrudan rekonstrüksiyon hatası yerine anatomi segmentasyonu belirsizliği kullanır. |
| Luo et al. [9] | Çok çözünürlüklü retinal autoencoder | Daha gelişmiş retinal anomali modeli önerir; bizim çalışmamız ise daha sade ve tekrar üretilebilir bir baseline sunar. |
| Bu proje | Normal-only OCT anomali puanlaması | Patient-level validation split ve gerçek OCT2017 verisi üzerinde percentile tabanlı threshold seçimi içerir. |

## 3. Yöntem

### 3.1 Veri kümesi ve bölme stratejisi

Çalışmada Kermany OCT2017 veri kümesinin `train` ve `test` klasörleri esas alınmıştır [2]. Eğitimde yalnızca `train/NORMAL` altındaki görüntüler kullanılmıştır. Validation bölmesi image-level değil patient-level olarak yapılmıştır; böylece aynı hastaya ait görüntüler train ve validation alt kümelerine aynı anda düşmemiştir. Hasta kimlikleri, veri kümesindeki dosya adlarında yer alan `hastalık-hastaID-görüntüNo` yapısından ayrıştırılmıştır. Test aşamasında `test/NORMAL`, `test/CNV`, `test/DME` ve `test/DRUSEN` görüntüleri birlikte değerlendirilmiş, NORMAL sınıfı 0 ve diğer tüm sınıflar anomaly etiketi 1 olarak ele alınmıştır.

### 3.2 Ön işleme

Tüm görüntüler tek kanallı gri tonlamaya dönüştürülmüş, `128x128` boyutuna yeniden örneklenmiş ve `[0, 1]` aralığına normalize edilmiştir. Bu ara sürümde agresif augmentation uygulanmamıştır; amacımız önce sade ve tekrarlanabilir bir baseline kurmaktır.

### 3.3 Model mimarisi

Model, dört aşamalı bir convolutional encoder-decoder yapısından oluşmaktadır. Encoder kısmı `1->32->64->128->256` kanal geçişleri ve max-pooling adımlarıyla görüntüyü sıkıştırırken, ara latent temsil `128` boyutlu bir vektöre indirgenmiştir. Decoder kısmı transpose convolution blokları ile görüntüyü tekrar `128x128` boyutuna taşımaktadır. Çıkış katmanında sigmoid kullanılarak normalize pikseller üzerinde reconstruction üretilmiştir.

### 3.4 Eğitim ve eşikleme

Model `Adam` optimizer ve `MSE` reconstruction loss ile eğitilmiştir. En fazla `40` epoch ve `8` patience değerli early stopping kullanılmıştır. Validation aşamasında yalnızca normal örneklerin reconstruction error dağılımı incelenmiş; p95, p97 ve p99 eşikleri hesaplanmıştır. Ana operasyon noktası olarak p95 seçilmiştir. Böylece threshold seçiminde test verisi kullanılmamış ve leakage engellenmiştir.

### 3.5 Değerlendirme ölçütleri ve deney kurulumu

Ana başarı ölçütü olarak AUROC seçilmiştir; çünkü anomaly detection senaryosunda threshold'dan bağımsız ayrıştırma gücünü yansıtır. Bunun yanında accuracy, precision, recall, F1 ve false positive rate de raporlanmıştır. Precision ve recall birlikte yorumlanmış, F1 ise dengeli operasyon noktası seçiminde kullanılmıştır. Gerçek deney koşusu yaklaşık 109.7 dakika sürmüş ve en iyi validation sonucu 37. epoch'ta elde edilmiştir. Deney, NVIDIA GeForce RTX 4060 Laptop GPU içeren yerel bir PyTorch ortamında yürütülmüştür. Deney kurulumu Tablo 2'de özetlenmiştir.

Tablo 2. Deney kurulumu ve temel hiperparametreler.

| ayar | değer |
| --- | --- |
| Eğitim verisi | 40715 NORMAL görüntü |
| Doğrulama verisi | 10425 NORMAL görüntü |
| Test verisi | 1000 görüntü |
| Giriş boyutu | 128x128 gri seviye |
| Latent boyut | 128 |
| Optimizasyon | Adam, lr=0.001 |
| Maks epoch / patience | 40 / 8 |
| Seçilen eşik | p95 |
| Eğitim süresi | 109.7 dakika |

### 3.6 Sistem akışı

Önerilen iş akışı beş adımdan oluşmaktadır: normal verinin seçilmesi, ön işleme, autoencoder eğitimi, validation error dağılımından eşik seçimi ve testte anomaly scoring. Raporun sonundaki Şekil 1 bu boru hattını görsel olarak özetlemektedir. Bu şema, ödevde istenen sistem mimarisi beklentisini karşılamak için eklenmiştir.

## 4. Ara Sonuçlar

Bu ara raporda üretilen temel çıktılar; eğitim/validation loss grafiği, validation reconstruction error histogramı, test error dağılımı, ROC curve, confusion matrix ve örnek reconstruction-residual görüntüleridir. Deney sonunda seçilen p95 eşiğinde elde edilen metrikler aşağıdaki gibidir:

| Metrik | Değer |
|---|---:|
| AUROC | 0.9108 |
| Accuracy | 0.7670 |
| Precision | 0.9743 |
| Recall | 0.7080 |
| F1 | 0.8201 |
| FPR | 0.0560 |
| Best epoch | 37 |
| Best validation loss | 0.000745 |

Validation persentil eşikleri:

| percentile | threshold | accuracy | precision | recall | f1 | fpr | auroc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 95 | 0.0014 | 0.767 | 0.9743 | 0.708 | 0.8201 | 0.056 | 0.9108 |
| 97 | 0.0016 | 0.714 | 0.9813 | 0.6307 | 0.7679 | 0.036 | 0.9108 |
| 99 | 0.0021 | 0.593 | 0.9971 | 0.4587 | 0.6283 | 0.004 | 0.9108 |

Sınıf bazlı reconstruction error özeti:

| class_name | sample_count | patient_count | mean_reconstruction_error | std_reconstruction_error |
| --- | --- | --- | --- | --- |
| CNV | 250 | 178 | 0.002921 | 0.001294 |
| DME | 250 | 167 | 0.002592 | 0.001554 |
| DRUSEN | 250 | 169 | 0.001317 | 0.000643 |
| NORMAL | 250 | 171 | 0.000864 | 0.000292 |

Veri bölme özeti:

| split_name | class_name | image_count | patient_count |
| --- | --- | --- | --- |
| train | NORMAL | 40715 | 2747 |
| val | NORMAL | 10425 | 687 |
| test | CNV | 250 | 178 |
| test | DME | 250 | 167 |
| test | DRUSEN | 250 | 169 |
| test | NORMAL | 250 | 171 |

Sonuçlar yalnızca tablo düzeyinde değil, yorum düzeyinde de anlamlıdır. p95 eşiği p97 ve p99'a göre daha yüksek recall ve F1 vermiştir; bu nedenle ara rapor için daha dengeli operasyon noktası olarak seçilmiştir. CNV ve DME sınıfları NORMAL görüntülerden belirgin şekilde ayrışırken, DRUSEN sınıfının error dağılımı normale daha yakındır. Bu durum, bazı patolojilerin reconstruction tabanlı yaklaşımlarda diğerlerine göre daha zor ayrıştığını göstermektedir.

Rapor sonunda verilen Şekil 2, eğitim eğrisi, ROC performansı, error dağılımı ve reconstruction örneklerini bir araya getirerek ara sonuçların görsel özetini sunmaktadır.

## 5. Tartışma

Baseline model, görece basit olmasına rağmen normal anatomi dağılımını öğrenerek patolojik sınıfların reconstruction error değerlerini yükseltebilmektedir. Bununla birlikte reconstruction tabanlı yöntemlerin iyi bilinen bir sınırı vardır: güçlü decoder yapıları bazen anomalileri de fazla iyi yeniden üretebilir [4], [8]. Kermany veri kümesi image-level etiketler içerir; bu nedenle lokal lesion segmentasyonu için doğrudan pixel-level ground truth bulunmamaktadır. Ayrıca threshold seçiminin precision-recall dengesi üzerinde güçlü etkisi vardır. Bu nedenle tek bir metrik yerine percentile bazlı karşılaştırma tablosu korunmuştur.

Hesaplama maliyeti de göz ardı edilemez. Eğitim koşusu yerel ortamda uzun sayılabilecek bir sürede tamamlanmıştır ve bu durum veri yükleme ile ön işleme hattının da iyileştirme alanı olduğunu göstermektedir. Dolayısıyla mevcut sistem klinik kullanımdan ziyade araştırma ve erken tarama mantığında değerlendirilmelidir.

## 6. Gelecek Çalışmalar

Final aşamada ilk geliştirme ekseni, mimari seviyesinde daha güçlü reconstruction modellerinin denenmesi olacaktır. Standart convolutional autoencoder yerine VAE, skip-connection içeren daha derin encoder-decoder yapıları veya memory-augmented reconstruction modelleri uygulanabilir. Bu sayede modelin normal anatomi dağılımını daha zengin bir latent temsille öğrenmesi ve özellikle sınıra yakın patolojik örneklerde daha ayırt edici reconstruction error üretmesi hedeflenmektedir. Buna ek olarak yalnızca MSE yerine L1, SSIM tabanlı kayıplar veya birleşik loss fonksiyonları denenerek yeniden oluşturma kalitesi ile anomaly sensitivity arasındaki denge incelenebilir.

İkinci geliştirme ekseni, veri ve deney tasarımı tarafında planlanmaktadır. Daha yüksek giriş çözünürlüğü ile deney yapılarak ince retinal yapıların ve özellikle DRUSEN gibi daha zor ayrıştırılan sınıfların model tarafından daha iyi temsil edilip edilmediği test edilecektir. Bunun yanında latent boyut, batch size, threshold seçimi ve image size gibi hiperparametreler sistematik bir ablation çalışması ile karşılaştırılacaktır. Böylece final raporda yalnızca tek bir model sonucu değil, tasarım kararlarının performansa etkisini gösteren daha akademik bir deney tablosu sunulabilecektir.

Üçüncü geliştirme ekseni, yorumlanabilirlik ve klinik anlamlandırma üzerine kurulacaktır. Mevcut residual map çıktıları daha detaylı incelenerek hata haritalarının retina üzerindeki hangi bölgelerde yoğunlaştığı analiz edilebilir. Eğer rekonstrüksiyon hatası belirli anatomik bozulmalarla tutarlı biçimde eşleşirse, modelin yalnızca sayısal anomaly skor üreten bir kara kutu olmaktan çıkması ve klinik olarak daha anlamlı hale gelmesi sağlanabilir. Bu nedenle final aşamada residual map görselleştirmeleri, en iyi ve en kötü örneklerin ayrı sunulması ve sınıf bazlı hata desenlerinin nitel olarak tartışılması planlanmaktadır.

Son olarak, hesaplama verimliliği ve karşılaştırmalı değerlendirme de gelecekteki temel adımlardan biridir. Veri yükleme hattının hızlandırılması, daha uygun batch boyutlarının seçilmesi ve eğitim süresinin optimize edilmesi ile tekrarlı deneyler daha verimli hale getirilecektir. Mevcut baseline sonucunun yanına en az bir geliştirilmiş model eklenerek AE ile geliştirilmiş varyantın AUROC, F1, recall ve FPR açısından doğrudan karşılaştırılması hedeflenmektedir. Bu geliştirmeler tamamlandığında proje, ara rapor seviyesindeki çalışan baseline'dan, karşılaştırmalı ve daha güçlü bir final proje yapısına taşınmış olacaktır.

## 7. Sonuç

Bu ara rapor aşamasında, Kermany OCT verisi için hasta-bazlı doğrulama kullanan, yalnızca normal görüntülerle eğitilen ve reconstruction error ile patolojik scan tespiti yapan tekrar üretilebilir bir baseline sistem kurulmuştur. Gerçek veri üzerinde elde edilen AUROC 0.9108 ve F1 0.8201 değerleri, yaklaşımın umut verici olduğunu göstermektedir. Final aşamada hedef, bu baseline'i daha güçlü anomaly detection yaklaşımlarıyla genişletmek ve sonuçları karşılaştırmalı deneylerle desteklemektir.

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
