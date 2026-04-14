from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches
from docx.shared import Pt

from utils import save_text


TITLE_TR = "Normal Retina OCT Görüntülerinden Öğrenilen Konvolüsyonel Autoencoder ile Patolojik Örneklerin Rekonstrüksiyon Hatası Tabanlı Tespiti"
TITLE_EN = "Reconstruction-Error-Based Detection of Pathological Retinal OCT Scans Using a Convolutional Autoencoder Trained on Normal Images"
AUTHOR_ENTRIES = [
    {
        "name": "Koray Öztürk",
        "department": "Bilgisayar Mühendisliği",
        "institution": "Eskişehir Osmangazi Üniversitesi",
        "location": "Eskişehir/Türkiye",
        "email": "korayoztuurk@gmail.com",
    },
    {
        "name": "Emir Alp İlhan",
        "department": "Bilgisayar Mühendisliği",
        "institution": "Eskişehir Osmangazi Üniversitesi",
        "location": "Eskişehir/Türkiye",
        "email": "emiralpilhan@gmail.com",
    },
]

REFERENCE_ENTRIES = [
    {"key": "[1]", "citation": "Kermany DS, Goldbaum M, Cai W, et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell. 2018;172(5):1122-1131. doi:10.1016/j.cell.2018.02.010", "url": "https://doi.org/10.1016/j.cell.2018.02.010", "group": "OCT classification"},
    {"key": "[2]", "citation": "Kermany DS, Zhang K, Goldbaum M. Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images. Mendeley Data. 2018;v3. doi:10.17632/rscbjbr9sj.3", "url": "https://doi.org/10.17632/rscbjbr9sj.3", "group": "Dataset"},
    {"key": "[3]", "citation": "Litjens G, Kooi T, Bejnordi BE, et al. A survey on deep learning in medical image analysis. Med Image Anal. 2017;42:60-88. doi:10.1016/j.media.2017.07.005", "url": "https://pubmed.ncbi.nlm.nih.gov/28778026/", "group": "Medical imaging survey"},
    {"key": "[4]", "citation": "Schlegl T, Seebock P, Waldstein SM, Schmidt-Erfurth U, Langs G. Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery. arXiv. 2017. doi:10.48550/arXiv.1703.05921", "url": "https://arxiv.org/abs/1703.05921", "group": "Medical anomaly detection"},
    {"key": "[5]", "citation": "Akcay S, Atapour-Abarghouei A, Breckon TP. GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training. arXiv. 2018. doi:10.48550/arXiv.1805.06725", "url": "https://arxiv.org/abs/1805.06725", "group": "Medical anomaly detection"},
    {"key": "[6]", "citation": "Zavrtanik V, Kristan M, Skocaj D. DRAEM: A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection. ICCV. 2021.", "url": "https://openaccess.thecvf.com/content/ICCV2021/html/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.html", "group": "Medical anomaly detection"},
    {"key": "[7]", "citation": "Seebock P, Orlando JI, Schlegl T, et al. Exploiting Epistemic Uncertainty of Anatomy Segmentation for Anomaly Detection in Retinal OCT. IEEE Trans Med Imaging. 2019. doi:10.1109/TMI.2019.2919951", "url": "https://arxiv.org/abs/1905.12806", "group": "Retinal OCT anomaly detection"},
    {"key": "[8]", "citation": "Zhou K, Li J, Luo W, et al. Proxy-bridged Image Reconstruction Network for Anomaly Detection in Medical Images. arXiv. 2021. doi:10.48550/arXiv.2110.01761", "url": "https://arxiv.org/abs/2110.01761", "group": "Medical anomaly detection"},
    {"key": "[9]", "citation": "Luo Y, Ma Y, Yang Z. Multi-resolution auto-encoder for anomaly detection of retinal imaging. Phys Eng Sci Med. 2024;47(2):517-529. doi:10.1007/s13246-023-01381-x", "url": "https://pubmed.ncbi.nlm.nih.gov/38285270/", "group": "Retinal OCT anomaly detection"},
    {"key": "[10]", "citation": "Wang J, Li W, Chen Y, et al. Weakly supervised anomaly segmentation in retinal OCT images using an adversarial learning approach. Biomed Opt Express. 2021;12(8):4713-4729. doi:10.1364/BOE.426803", "url": "https://pubmed.ncbi.nlm.nih.gov/34513220/", "group": "Retinal OCT anomaly detection"},
]


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def frame_to_markdown(frame: pd.DataFrame, decimals: int | None = None) -> str:
    if decimals is not None:
        frame = frame.copy()
        numeric_columns = frame.select_dtypes(include=["number"]).columns
        frame[numeric_columns] = frame[numeric_columns].round(decimals)

    headers = list(frame.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header_row, separator_row, *data_rows])


def threshold_table_markdown(threshold_table: pd.DataFrame) -> str:
    return frame_to_markdown(threshold_table, decimals=4)


def classwise_table_markdown(classwise_df: pd.DataFrame) -> str:
    return frame_to_markdown(classwise_df, decimals=6)


def dataset_table_markdown(dataset_summary: pd.DataFrame) -> str:
    return frame_to_markdown(dataset_summary)


def build_related_work_comparison_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"çalışma": "Kermany et al. [1]", "odak": "Denetimli OCT sınıflandırması", "fark": "Patoloji etiketleri gerektirir; bizim yaklaşımımız yalnızca normal görüntülerle anomaly detection yapar."},
            {"çalışma": "AnoGAN [4]", "odak": "GAN tabanlı anomali tespiti", "fark": "Genel amaçlı anomaly detection yaklaşımıdır; retinal OCT’ye özgü değildir."},
            {"çalışma": "Seebock et al. [7]", "odak": "Belirsizlik tabanlı OCT anomali tespiti", "fark": "Doğrudan rekonstrüksiyon hatası yerine anatomi segmentasyonu belirsizliği kullanır."},
            {"çalışma": "Luo et al. [9]", "odak": "Çok çözünürlüklü retinal autoencoder", "fark": "Daha gelişmiş retinal anomali modeli önerir; bizim çalışmamız ise daha sade ve tekrar üretilebilir bir baseline sunar."},
            {"çalışma": "Bu proje", "odak": "Normal-only OCT anomali puanlaması", "fark": "Patient-level validation split ve gerçek OCT2017 verisi üzerinde percentile tabanlı threshold seçimi içerir."},
        ]
    )


def build_experiment_setup_table(config: dict, dataset_summary: pd.DataFrame, history: dict) -> pd.DataFrame:
    train_count = int(dataset_summary.loc[(dataset_summary["split_name"] == "train") & (dataset_summary["class_name"] == "NORMAL"), "image_count"].iloc[0])
    val_count = int(dataset_summary.loc[(dataset_summary["split_name"] == "val") & (dataset_summary["class_name"] == "NORMAL"), "image_count"].iloc[0])
    test_count = int(dataset_summary.loc[dataset_summary["split_name"] == "test", "image_count"].sum())
    return pd.DataFrame(
        [
            {"ayar": "Eğitim verisi", "değer": f"{train_count} NORMAL görüntü"},
            {"ayar": "Doğrulama verisi", "değer": f"{val_count} NORMAL görüntü"},
            {"ayar": "Test verisi", "değer": f"{test_count} görüntü"},
            {"ayar": "Giriş boyutu", "değer": f"{config['image_size']}x{config['image_size']} gri seviye"},
            {"ayar": "Latent boyut", "değer": str(config["latent_dim"])},
            {"ayar": "Optimizasyon", "değer": f"Adam, lr={config['learning_rate']}"},
            {"ayar": "Maks epoch / patience", "değer": f"{config['epochs']} / {config['early_stopping_patience']}"},
            {"ayar": "Seçilen eşik", "değer": f"p{config['default_percentile']}"},
            {"ayar": "Eğitim süresi", "değer": f"{history['training_time_sec'] / 60:.1f} dakika"},
        ]
    )


def references_markdown() -> str:
    lines = ["# Literature Notes", ""]
    grouped_entries: dict[str, list[dict]] = {}
    for entry in REFERENCE_ENTRIES:
        grouped_entries.setdefault(entry["group"], []).append(entry)

    for group_name, entries in grouped_entries.items():
        lines.extend([f"## {group_name}", ""])
        for entry in entries:
            lines.append(f"- {entry['key']} {entry['citation']}  ")
            lines.append(f"  Link: {entry['url']}")
    return "\n".join(lines) + "\n"


def create_pipeline_figure(save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(12, 2.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    boxes = [
        (0.03, 0.28, 0.16, 0.42, "Yalnızca\nNORMAL eğitim"),
        (0.23, 0.28, 0.16, 0.42, "Yeniden boyutlandırma\n+ normalizasyon"),
        (0.43, 0.28, 0.16, 0.42, "Konvolüsyonel\nAutoencoder"),
        (0.63, 0.28, 0.16, 0.42, "Doğrulama\npersentilleri"),
        (0.83, 0.28, 0.14, 0.42, "Test anomali\npuanlaması"),
    ]

    for x, y, w, h, label in boxes:
        patch = plt.Rectangle((x, y), w, h, facecolor="#e9f2ff", edgecolor="#2d5f9a", linewidth=2)
        axis.add_patch(patch)
        axis.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

    for start_x, end_x in [(0.19, 0.23), (0.39, 0.43), (0.59, 0.63), (0.79, 0.83)]:
        axis.annotate("", xy=(end_x, 0.49), xytext=(start_x, 0.49), arrowprops={"arrowstyle": "->", "lw": 2})

    axis.text(0.5, 0.88, "Retina OCT anomali tespit hattı", ha="center", va="center", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_results_overview_figure(output_root: Path, save_path: Path) -> None:
    candidates = [
        ("Eğitim kaybı", output_root / "figures" / "training_loss.png"),
        ("ROC eğrisi", output_root / "figures" / "roc_curve.png"),
        ("Hata dağılımı", output_root / "figures" / "test_error_distribution.png"),
        ("Rekonstrüksiyonlar", output_root / "reconstructions" / "reconstruction_examples.png"),
    ]
    available = [(title, path) for title, path in candidates if path.exists()]
    if not available:
        return

    rows = 2 if len(available) > 2 else 1
    cols = 2 if len(available) > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(11, 7))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[axis] for axis in axes]

    flat_axes = [axis for row in axes for axis in row]
    for axis in flat_axes:
        axis.axis("off")

    for axis, (title, path) in zip(flat_axes, available):
        axis.imshow(mpimg.imread(path))
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_markdown_report(
    config: dict,
    metrics: dict,
    thresholds: dict[int, float],
    threshold_table: pd.DataFrame,
    classwise_df: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    history: dict,
    output_root: Path,
) -> str:
    related_work_table = build_related_work_comparison_table()
    setup_table = build_experiment_setup_table(config, dataset_summary, history)
    selected_percentile = config["default_percentile"]
    training_minutes = history["training_time_sec"] / 60

    return f"""# {TITLE_TR}

## Başlık

**İngilizce başlık:** {TITLE_EN}

## Özet

Bu çalışmada, retinal OCT görüntülerinde patolojik örnekleri etiketlenmiş patoloji sınıflarıyla doğrudan öğrenmek yerine, yalnızca normal örneklerden öğrenilen bir convolutional autoencoder ile reconstruction error tabanlı anomaly detection yaklaşımı geliştirilmiştir. Kermany OCT2017 veri kümesindeki `train/NORMAL` görüntüleri hasta bazlı olarak eğitim ve doğrulama alt kümelerine ayrılmış, model yalnızca normal anatominin dağılımını öğrenmiştir. Test aşamasında NORMAL, CNV, DME ve DRUSEN görüntüleri reconstruction error ile puanlanmış ve doğrulama normal error dağılımından elde edilen persentil eşikleriyle ikili karar üretilmiştir. Bu ara rapor sürümünde temel model 128x128 gri tonlamalı B-scan'ler üzerinde eğitilmiş, AUROC ana metrik olarak alınmış ve precision, recall, F1, accuracy ile FPR de raporlanmıştır. Gerçek OCT2017 deneyi sonunda seçilen p{selected_percentile} eşiğinde AUROC {metrics['auroc']:.4f} ve F1 {metrics['f1']:.4f} elde edilmiştir. Elde edilen ilk bulgular, patolojik sınıfların ortalama reconstruction error değerlerinin normal sınıfa göre sistematik olarak daha yüksek olduğunu göstermektedir.

## Abstract

This study investigates reconstruction-error-based anomaly detection for retinal OCT images using a convolutional autoencoder trained only on normal samples. Instead of learning a supervised pathology classifier, the model learns the appearance distribution of healthy retinal anatomy from the Kermany OCT2017 dataset. The training split uses only normal scans and a patient-level validation split is used to estimate operating thresholds from normal reconstruction-error percentiles. During testing, NORMAL, CNV, DME, and DRUSEN scans are scored according to reconstruction error, and anomaly decisions are produced with validation-derived thresholds. On the real OCT2017 experiment, the selected p{selected_percentile} operating point reaches AUROC {metrics['auroc']:.4f} and F1-score {metrics['f1']:.4f}. These findings indicate that a normal-only autoencoder can serve as a meaningful early baseline for retinal anomaly screening.

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

{frame_to_markdown(related_work_table)}

## 3. Yöntem

### 3.1 Veri kümesi ve bölme stratejisi

Çalışmada Kermany OCT2017 veri kümesinin `train` ve `test` klasörleri esas alınmıştır [2]. Eğitimde yalnızca `train/NORMAL` altındaki görüntüler kullanılmıştır. Validation bölmesi image-level değil patient-level olarak yapılmıştır; böylece aynı hastaya ait görüntüler train ve validation alt kümelerine aynı anda düşmemiştir. Hasta kimlikleri, veri kümesindeki dosya adlarında yer alan `hastalık-hastaID-görüntüNo` yapısından ayrıştırılmıştır. Test aşamasında `test/NORMAL`, `test/CNV`, `test/DME` ve `test/DRUSEN` görüntüleri birlikte değerlendirilmiş, NORMAL sınıfı 0 ve diğer tüm sınıflar anomaly etiketi 1 olarak ele alınmıştır.

### 3.2 Ön işleme

Tüm görüntüler tek kanallı gri tonlamaya dönüştürülmüş, `128x128` boyutuna yeniden örneklenmiş ve `[0, 1]` aralığına normalize edilmiştir. Bu ara sürümde agresif augmentation uygulanmamıştır; amacımız önce sade ve tekrarlanabilir bir baseline kurmaktır.

### 3.3 Model mimarisi

Model, dört aşamalı bir convolutional encoder-decoder yapısından oluşmaktadır. Encoder kısmı `1->32->64->128->256` kanal geçişleri ve max-pooling adımlarıyla görüntüyü sıkıştırırken, ara latent temsil `128` boyutlu bir vektöre indirgenmiştir. Decoder kısmı transpose convolution blokları ile görüntüyü tekrar `128x128` boyutuna taşımaktadır. Çıkış katmanında sigmoid kullanılarak normalize pikseller üzerinde reconstruction üretilmiştir.

### 3.4 Eğitim ve eşikleme

Model `Adam` optimizer ve `MSE` reconstruction loss ile eğitilmiştir. En fazla `40` epoch ve `8` patience değerli early stopping kullanılmıştır. Validation aşamasında yalnızca normal örneklerin reconstruction error dağılımı incelenmiş; p95, p97 ve p99 eşikleri hesaplanmıştır. Ana operasyon noktası olarak p{selected_percentile} seçilmiştir. Böylece threshold seçiminde test verisi kullanılmamış ve leakage engellenmiştir.

### 3.5 Değerlendirme ölçütleri ve deney kurulumu

Ana başarı ölçütü olarak AUROC seçilmiştir; çünkü anomaly detection senaryosunda threshold'dan bağımsız ayrıştırma gücünü yansıtır. Bunun yanında accuracy, precision, recall, F1 ve false positive rate de raporlanmıştır. Precision ve recall birlikte yorumlanmış, F1 ise dengeli operasyon noktası seçiminde kullanılmıştır. Gerçek deney koşusu yaklaşık {training_minutes:.1f} dakika sürmüş ve en iyi validation sonucu {history['best_epoch']}. epoch'ta elde edilmiştir. Deney, NVIDIA GeForce RTX 4060 Laptop GPU içeren yerel bir PyTorch ortamında yürütülmüştür. Deney kurulumu Tablo 2'de özetlenmiştir.

Tablo 2. Deney kurulumu ve temel hiperparametreler.

{frame_to_markdown(setup_table)}

### 3.6 Sistem akışı

Önerilen iş akışı beş adımdan oluşmaktadır: normal verinin seçilmesi, ön işleme, autoencoder eğitimi, validation error dağılımından eşik seçimi ve testte anomaly scoring. Raporun sonundaki Şekil 1 bu boru hattını görsel olarak özetlemektedir. Bu şema, ödevde istenen sistem mimarisi beklentisini karşılamak için eklenmiştir.

## 4. Ara Sonuçlar

Bu ara raporda üretilen temel çıktılar; eğitim/validation loss grafiği, validation reconstruction error histogramı, test error dağılımı, ROC curve, confusion matrix ve örnek reconstruction-residual görüntüleridir. Deney sonunda seçilen p{selected_percentile} eşiğinde elde edilen metrikler aşağıdaki gibidir:

| Metrik | Değer |
|---|---:|
| AUROC | {format_metric(metrics['auroc'])} |
| Accuracy | {format_metric(metrics['accuracy'])} |
| Precision | {format_metric(metrics['precision'])} |
| Recall | {format_metric(metrics['recall'])} |
| F1 | {format_metric(metrics['f1'])} |
| FPR | {format_metric(metrics['fpr'])} |
| Best epoch | {history['best_epoch']} |
| Best validation loss | {history['best_val_loss']:.6f} |

Validation persentil eşikleri:

{threshold_table_markdown(threshold_table)}

Sınıf bazlı reconstruction error özeti:

{classwise_table_markdown(classwise_df)}

Veri bölme özeti:

{dataset_table_markdown(dataset_summary)}

Sonuçlar yalnızca tablo düzeyinde değil, yorum düzeyinde de anlamlıdır. p{selected_percentile} eşiği p97 ve p99'a göre daha yüksek recall ve F1 vermiştir; bu nedenle ara rapor için daha dengeli operasyon noktası olarak seçilmiştir. CNV ve DME sınıfları NORMAL görüntülerden belirgin şekilde ayrışırken, DRUSEN sınıfının error dağılımı normale daha yakındır. Bu durum, bazı patolojilerin reconstruction tabanlı yaklaşımlarda diğerlerine göre daha zor ayrıştığını göstermektedir.

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

Bu ara rapor aşamasında, Kermany OCT verisi için hasta-bazlı doğrulama kullanan, yalnızca normal görüntülerle eğitilen ve reconstruction error ile patolojik scan tespiti yapan tekrar üretilebilir bir baseline sistem kurulmuştur. Gerçek veri üzerinde elde edilen AUROC {metrics['auroc']:.4f} ve F1 {metrics['f1']:.4f} değerleri, yaklaşımın umut verici olduğunu göstermektedir. Final aşamada hedef, bu baseline'i daha güçlü anomaly detection yaklaşımlarıyla genişletmek ve sonuçları karşılaştırmalı deneylerle desteklemektir.

## Kaynaklar

""" + "\n".join(f"{entry['key']} {entry['citation']}" for entry in REFERENCE_ENTRIES) + "\n"


def _clear_document(doc: Document) -> None:
    body = doc._element.body
    for element in list(body):
        if element.tag.endswith("sectPr"):
            continue
        body.remove(element)


def _set_section_columns(section, count: int) -> None:
    sect_pr = section._sectPr
    cols = sect_pr.xpath("./w:cols")
    if cols:
        cols_element = cols[0]
    else:
        cols_element = OxmlElement("w:cols")
        sect_pr.append(cols_element)
    cols_element.set(qn("w:num"), str(count))


def _set_table_borders_none(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "nil")


def _strip_heading_number(text: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*(?:\.)?\s+", "", text).strip()


def _parse_markdown_sections(markdown_text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for raw_line in markdown_text.strip().splitlines():
        if raw_line.startswith("# "):
            continue
        if raw_line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = raw_line.replace("## ", "", 1).strip()
            current_lines = []
            continue
        if current_heading is not None:
            current_lines.append(raw_line)

    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return sections


def _add_labeled_paragraph(doc: Document, label: str, content: str) -> None:
    paragraph = doc.add_paragraph()
    label_run = paragraph.add_run(f"{label} - ")
    label_run.bold = True
    paragraph.add_run(" ".join(line.strip() for line in content.splitlines() if line.strip()))


def _add_author_block(doc: Document) -> None:
    table = doc.add_table(rows=1, cols=len(AUTHOR_ENTRIES))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    _set_table_borders_none(table)

    for column, author in enumerate(AUTHOR_ENTRIES):
        cell = table.rows[0].cells[column]
        lines = [
            author["name"],
            author["department"],
            author["institution"],
            author["location"],
            author["email"],
        ]
        for index, line in enumerate(lines):
            paragraph = cell.paragraphs[0] if index == 0 else cell.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(line)
            if index == 0:
                run.bold = True


def _add_body_sections(doc: Document, sections: list[tuple[str, str]]) -> None:
    for heading, content in sections:
        doc.add_heading(_strip_heading_number(heading), level=1)
        if content.strip():
            _add_paragraphs(doc, content)


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    compact = "".join(cells).replace("-", "").replace(":", "").replace(" ", "")
    return compact == ""


def _add_markdown_table(doc: Document, lines: list[str]) -> None:
    parsed_rows = [_split_markdown_row(line) for line in lines if line.strip()]
    if len(parsed_rows) < 2:
        for line in lines:
            doc.add_paragraph(line)
        return

    header = parsed_rows[0]
    data_rows = [row for row in parsed_rows[1:] if not _is_separator_row(row)]
    table = doc.add_table(rows=1 + len(data_rows), cols=len(header))
    try:
        table.style = "Table Grid"
    except KeyError:
        pass

    for column, cell_text in enumerate(header):
        table.rows[0].cells[column].text = cell_text

    for row_index, row_values in enumerate(data_rows, start=1):
        normalized = row_values + [""] * (len(header) - len(row_values))
        for column, cell_text in enumerate(normalized[: len(header)]):
            table.rows[row_index].cells[column].text = cell_text


def _add_paragraphs(doc: Document, text: str) -> None:
    lines = text.strip().splitlines()
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.strip().replace("**", "")

        if not line:
            doc.add_paragraph("")
            index += 1
            continue

        if line.startswith("# "):
            index += 1
            continue

        if line.startswith("## "):
            doc.add_heading(_strip_heading_number(line.replace("## ", "", 1)), level=1)
            index += 1
            continue

        if line.startswith("### "):
            doc.add_heading(_strip_heading_number(line.replace("### ", "", 1)), level=2)
            index += 1
            continue

        if line.startswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            _add_markdown_table(doc, table_lines)
            continue

        if line.startswith("- "):
            doc.add_paragraph(line[2:], style="List Bullet")
            index += 1
            continue

        doc.add_paragraph(line)
        index += 1


def _append_report_figures(doc: Document, output_root: Path) -> None:
    figure_specs = [
        ("Şekil 1. Önerilen retina OCT anomali tespit boru hattı.", output_root / "figures" / "retina_oct_pipeline.png"),
        ("Şekil 2. Eğitim, ROC, error dağılımı ve reconstruction çıktılarının özet görseli.", output_root / "figures" / "report_results_overview.png"),
    ]
    available = [(caption, path) for caption, path in figure_specs if path.exists()]
    if not available:
        return

    figure_section = doc.add_section(WD_SECTION_START.CONTINUOUS)
    _set_section_columns(figure_section, 1)
    doc.add_heading("Şekiller", level=1)
    for caption, path in available:
        doc.add_picture(str(path), width=Inches(6.2))
        paragraph = doc.add_paragraph(caption)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def build_docx_report(markdown_text: str, template_path: Path, output_path: Path, output_root: Path) -> None:
    if template_path.exists():
        doc = Document(template_path)
        _clear_document(doc)
    else:
        doc = Document()

    _set_section_columns(doc.sections[0], 1)

    title_paragraph = doc.add_paragraph()
    title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_paragraph.add_run(TITLE_TR)
    title_run.bold = True
    title_run.font.size = Pt(18)
    title_paragraph.paragraph_format.space_after = Pt(6)

    english_title = doc.add_paragraph()
    english_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    english_run = english_title.add_run(TITLE_EN)
    english_run.font.size = Pt(15)
    english_title.paragraph_format.space_after = Pt(10)

    _add_author_block(doc)
    doc.add_paragraph("")

    body_section = doc.add_section(WD_SECTION_START.CONTINUOUS)
    _set_section_columns(body_section, 2)

    sections = _parse_markdown_sections(markdown_text)
    section_map = {heading: content for heading, content in sections}
    body_sections = [
        (heading, content)
        for heading, content in sections
        if heading not in {"Başlık", "Özet", "Abstract", "Anahtar Kelimeler / Keywords"}
    ]

    if section_map.get("Özet"):
        _add_labeled_paragraph(doc, "Özet", section_map["Özet"])
    if section_map.get("Abstract"):
        _add_labeled_paragraph(doc, "Abstract", section_map["Abstract"])
    if section_map.get("Anahtar Kelimeler / Keywords"):
        _add_labeled_paragraph(doc, "Anahtar Kelimeler / Keywords", section_map["Anahtar Kelimeler / Keywords"])

    _add_body_sections(doc, body_sections)
    _append_report_figures(doc, output_root)
    doc.save(output_path)


def build_report_assets(
    config: dict,
    metrics: dict,
    thresholds: dict[int, float],
    threshold_table: pd.DataFrame,
    classwise_df: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    history: dict,
    output_root: Path,
    report_root: Path,
    template_path: Path,
) -> dict:
    report_root.mkdir(parents=True, exist_ok=True)

    create_pipeline_figure(output_root / "figures" / "retina_oct_pipeline.png")
    create_results_overview_figure(output_root, output_root / "figures" / "report_results_overview.png")

    markdown_report = build_markdown_report(
        config=config,
        metrics=metrics,
        thresholds=thresholds,
        threshold_table=threshold_table,
        classwise_df=classwise_df,
        dataset_summary=dataset_summary,
        history=history,
        output_root=output_root,
    )
    save_text(markdown_report, report_root / "ara_rapor_draft.md")
    save_text(references_markdown(), report_root / "literature_notes.md")
    build_docx_report(markdown_report, template_path, report_root / "ara_rapor_draft.docx", output_root)

    return {
        "title_tr": TITLE_TR,
        "title_en": TITLE_EN,
        "metrics": metrics,
        "thresholds": thresholds,
        "history": history,
        "threshold_table": threshold_table.round(6).to_dict(orient="records"),
        "classwise_summary": classwise_df.round(6).to_dict(orient="records"),
        "dataset_summary": dataset_summary.to_dict(orient="records"),
        "references": REFERENCE_ENTRIES,
        "generated_files": [
            str(report_root / "ara_rapor_draft.md"),
            str(report_root / "ara_rapor_draft.docx"),
            str(report_root / "literature_notes.md"),
            str(output_root / "figures" / "retina_oct_pipeline.png"),
            str(output_root / "figures" / "report_results_overview.png"),
        ],
    }
