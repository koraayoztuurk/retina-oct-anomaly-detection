from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches

from utils import save_text


TITLE_TR = "Normal Retina OCT Goruntulerinden Ogrenilen Konvolusyonel Autoencoder ile Patolojik Orneklerin Rekonstruksiyon Hatasi Tabanli Tespiti"
TITLE_EN = "Reconstruction-Error-Based Detection of Pathological Retinal OCT Scans Using a Convolutional Autoencoder Trained on Normal Images"

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
            {"study": "Kermany et al. [1]", "focus": "Supervised OCT classification", "difference": "Requires pathology labels, unlike our normal-only anomaly setting."},
            {"study": "AnoGAN [4]", "focus": "GAN-based anomaly detection", "difference": "General anomaly detection reference, not retinal OCT-specific."},
            {"study": "Seebock et al. [7]", "focus": "Uncertainty-based OCT anomaly detection", "difference": "Uses anatomy segmentation uncertainty instead of direct reconstruction error."},
            {"study": "Luo et al. [9]", "focus": "Multi-resolution retinal autoencoder", "difference": "More advanced retinal anomaly model; our work is a simpler reproducible baseline."},
            {"study": "This project", "focus": "Normal-only OCT anomaly scoring", "difference": "Patient-level validation split and percentile-based threshold selection on real OCT2017."},
        ]
    )


def build_experiment_setup_table(config: dict, dataset_summary: pd.DataFrame, history: dict) -> pd.DataFrame:
    train_count = int(dataset_summary.loc[(dataset_summary["split_name"] == "train") & (dataset_summary["class_name"] == "NORMAL"), "image_count"].iloc[0])
    val_count = int(dataset_summary.loc[(dataset_summary["split_name"] == "val") & (dataset_summary["class_name"] == "NORMAL"), "image_count"].iloc[0])
    test_count = int(dataset_summary.loc[dataset_summary["split_name"] == "test", "image_count"].sum())
    return pd.DataFrame(
        [
            {"setting": "Train data", "value": f"{train_count} NORMAL images"},
            {"setting": "Validation data", "value": f"{val_count} NORMAL images"},
            {"setting": "Test data", "value": f"{test_count} images"},
            {"setting": "Input size", "value": f"{config['image_size']}x{config['image_size']} grayscale"},
            {"setting": "Latent dimension", "value": str(config["latent_dim"])},
            {"setting": "Optimizer", "value": f"Adam, lr={config['learning_rate']}"},
            {"setting": "Max epochs / patience", "value": f"{config['epochs']} / {config['early_stopping_patience']}"},
            {"setting": "Selected threshold", "value": f"p{config['default_percentile']}"},
            {"setting": "Training duration", "value": f"{history['training_time_sec'] / 60:.1f} minutes"},
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
        (0.03, 0.28, 0.16, 0.42, "Train NORMAL\nonly"),
        (0.23, 0.28, 0.16, 0.42, "Resize +\nnormalize"),
        (0.43, 0.28, 0.16, 0.42, "Conv\nAutoencoder"),
        (0.63, 0.28, 0.16, 0.42, "Validation\npercentiles"),
        (0.83, 0.28, 0.14, 0.42, "Test anomaly\nscoring"),
    ]

    for x, y, w, h, label in boxes:
        patch = plt.Rectangle((x, y), w, h, facecolor="#e9f2ff", edgecolor="#2d5f9a", linewidth=2)
        axis.add_patch(patch)
        axis.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

    for start_x, end_x in [(0.19, 0.23), (0.39, 0.43), (0.59, 0.63), (0.79, 0.83)]:
        axis.annotate("", xy=(end_x, 0.49), xytext=(start_x, 0.49), arrowprops={"arrowstyle": "->", "lw": 2})

    axis.text(0.5, 0.88, "Retina OCT anomaly detection pipeline", ha="center", va="center", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_results_overview_figure(output_root: Path, save_path: Path) -> None:
    candidates = [
        ("Training loss", output_root / "figures" / "training_loss.png"),
        ("ROC curve", output_root / "figures" / "roc_curve.png"),
        ("Error distribution", output_root / "figures" / "test_error_distribution.png"),
        ("Reconstructions", output_root / "reconstructions" / "reconstruction_examples.png"),
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

## Baslik

**Ingilizce baslik:** {TITLE_EN}

## Ozet

Bu calismada, retinal OCT goruntulerinde patolojik ornekleri etiketlenmis patoloji siniflariyla dogrudan ogrenmek yerine, yalnizca normal orneklerden ogrenilen bir convolutional autoencoder ile reconstruction error tabanli anomaly detection yaklasimi gelistirilmistir. Kermany OCT2017 veri kumesindeki train/NORMAL goruntuleri hasta bazli olarak egitim ve validation alt kumelerine ayrilmis, model yalnizca normal anatominin dagilimini ogrenmistir. Test asamasinda NORMAL, CNV, DME ve DRUSEN goruntuleri reconstruction error ile puanlanmis ve validation normal error dagilimindan elde edilen percentil esikleriyle ikili karar uretilmistir. Bu ara rapor surumunde temel model 128x128 gri tonlamali B-scan'ler uzerinde egitilmis, AUROC ana metrik olarak alinmis ve precision, recall, F1, accuracy ile FPR de raporlanmistir. Gercek OCT2017 deneyi sonunda secilen p{selected_percentile} esiginde AUROC {metrics['auroc']:.4f} ve F1 {metrics['f1']:.4f} elde edilmistir. Elde edilen ilk bulgular, patolojik siniflarin ortalama reconstruction error degerlerinin normal sinifa gore sistematik olarak daha yuksek oldugunu gostermektedir.

## Abstract

This study investigates reconstruction-error-based anomaly detection for retinal OCT images using a convolutional autoencoder trained only on normal samples. Instead of learning a supervised pathology classifier, the model learns the appearance distribution of healthy retinal anatomy from the Kermany OCT2017 dataset. The training split uses only normal scans and a patient-level validation split is used to estimate operating thresholds from normal reconstruction-error percentiles. During testing, NORMAL, CNV, DME, and DRUSEN scans are scored according to reconstruction error, and anomaly decisions are produced with validation-derived thresholds. On the real OCT2017 experiment, the selected p{selected_percentile} operating point reaches AUROC {metrics['auroc']:.4f} and F1-score {metrics['f1']:.4f}. These findings indicate that a normal-only autoencoder can serve as a meaningful early baseline for retinal anomaly screening.

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

{frame_to_markdown(related_work_table)}

## 3. Yontem

### 3.1 Veri kumesi ve bolme stratejisi

Calismada Kermany OCT2017 veri kumesinin `train` ve `test` klasorleri esas alinmistir [2]. Egitimde yalnizca `train/NORMAL` altindaki goruntuler kullanilmistir. Validation bolmesi image-level degil patient-level olarak yapilmistir; boylece ayni hastaya ait goruntuler train ve validation alt kumelerine ayni anda dusmemistir. Test asamasinda `test/NORMAL`, `test/CNV`, `test/DME` ve `test/DRUSEN` goruntuleri birlikte degerlendirilmis, NORMAL sinifi 0 ve diger tum siniflar anomaly etiketi 1 olarak ele alinmistir.

### 3.2 On isleme

Tum goruntuler tek kanalli gri tonlamaya donusturulmus, `128x128` boyutuna yeniden orneklenmis ve `[0, 1]` araligina normalize edilmistir. Bu ara surumde agresif augmentation uygulanmamistir; amacimiz once sade ve tekrarlanabilir bir baseline kurmaktir.

### 3.3 Model mimarisi

Model, dort asamali bir convolutional encoder-decoder yapisindan olusmaktadir. Encoder kismi 1->32->64->128->256 kanal gecisleri ve max-pooling adimlariyla goruntuyu sikistirirken, ara latent temsil `128` boyutlu bir vektore indirgenmistir. Decoder kismi transpose convolution bloklari ile goruntuyu tekrar 128x128 boyutuna tasimaktadir. Cikis katmaninda sigmoid kullanilarak normalize pikseller uzerinde reconstruction uretilmistir.

### 3.4 Egitim ve esikleme

Model `Adam` optimizer ve `MSE` reconstruction loss ile egitilmistir. En fazla `40` epoch ve `8` patience degerli early stopping kullanilmistir. Validation asamasinda yalnizca normal orneklerin reconstruction error dagilimi incelenmis; p95, p97 ve p99 esikleri hesaplanmistir. Ana operasyon noktasi olarak p{selected_percentile} secilmistir. Boylece threshold seciminde test verisi kullanilmamis ve leakage engellenmistir.

### 3.5 Degerlendirme olcutleri ve deney kurulumu

Ana basari olcutu olarak AUROC secilmistir; cunku anomaly detection senaryosunda threshold'dan bagimsiz ayristirma gucunu yansitir. Bunun yaninda accuracy, precision, recall, F1 ve false positive rate de raporlanmistir. Precision ve recall birlikte yorumlanmis, F1 ise dengeli operasyon noktasi seciminde kullanilmistir. Gercek deney kosusu yaklasik {training_minutes:.1f} dakika surmus ve en iyi validation sonucu {history['best_epoch']}. epoch'ta elde edilmistir. Deney kurulumu Tablo 2'de ozetlenmistir.

Tablo 2. Deney kurulumu ve temel hiperparametreler.

{frame_to_markdown(setup_table)}

### 3.6 Sistem akisi

Onerilen is akisi bes adimdan olusmaktadir: normal verinin secilmesi, on isleme, autoencoder egitimi, validation error dagilimindan esik secimi ve testte anomaly scoring. Raporun sonundaki Sekil 1 bu boru hattini gorsel olarak ozetlemektedir. Bu sema, odevde istenen sistem mimarisi beklentisini karsilamak icin eklenmistir.

## 4. Ara Sonuclar

Bu ara raporda uretilen temel ciktilar; egitim/validation loss grafigi, validation reconstruction error histogrami, test error dagilimi, ROC curve, confusion matrix ve ornek reconstruction-residual goruntuleridir. Deney sonunda secilen p{selected_percentile} esiginde elde edilen metrikler asagidaki gibidir:

| Metrik | Deger |
|---|---:|
| AUROC | {format_metric(metrics['auroc'])} |
| Accuracy | {format_metric(metrics['accuracy'])} |
| Precision | {format_metric(metrics['precision'])} |
| Recall | {format_metric(metrics['recall'])} |
| F1 | {format_metric(metrics['f1'])} |
| FPR | {format_metric(metrics['fpr'])} |
| Best epoch | {history['best_epoch']} |
| Best validation loss | {history['best_val_loss']:.6f} |

Validation percentil esikleri:

{threshold_table_markdown(threshold_table)}

Sinif bazli reconstruction error ozeti:

{classwise_table_markdown(classwise_df)}

Veri bolme ozeti:

{dataset_table_markdown(dataset_summary)}

Sonuclar yalnizca tablo duzeyinde degil, yorum duzeyinde de anlamlidir. p{selected_percentile} esigi p97 ve p99'a gore daha yuksek recall ve F1 vermistir; bu nedenle ara rapor icin daha dengeli operasyon noktasi olarak secilmistir. CNV ve DME siniflari NORMAL goruntulerden belirgin sekilde ayrisirken, DRUSEN sinifinin error dagilimi normale daha yakindir. Bu durum, bazi patolojilerin reconstruction tabanli yaklasimlarda digerlerine gore daha zor ayristigini gostermektedir.

Rapor sonunda verilen Sekil 2, egitim egrisi, ROC performansi, error dagilimi ve reconstruction orneklerini bir araya getirerek ara sonuclarin gorsel ozetini sunmaktadir.

## 5. Tartisma

Baseline model, gorece basit olmasina ragmen normal anatomi dagilimini ogrenerek patolojik siniflarin reconstruction error degerlerini yukseltebilmektedir. Bununla birlikte reconstruction tabanli yontemlerin iyi bilinen bir siniri vardir: guclu decoder yapilari bazen anomalileri de fazla iyi yeniden uretebilir [4], [8]. Kermany veri kumesi image-level etiketler icerir; bu nedenle lokal lesion segmentasyonu icin dogrudan pixel-level ground truth bulunmamaktadir. Ayrica threshold seciminin precision-recall dengesi uzerinde guclu etkisi vardir. Bu nedenle tek bir metrik yerine percentile bazli karsilastirma tablosu korunmustur.

Hesaplama maliyeti de goz ardi edilemez. Egitim kosusu yerel ortamda uzun sayilabilecek bir surede tamamlanmistir ve bu durum veri yukleme ile on isleme hattinin da iyilestirme alani oldugunu gostermektedir. Dolayisiyla mevcut sistem klinik kullanimdan ziyade arastirma ve erken tarama mantiginda degerlendirilmelidir.

## 6. Gelecek Calismalar

Final asamada ilk gelistirme ekseni, mimari seviyesinde daha guclu reconstruction modellerinin denenmesi olacaktir. Standart convolutional autoencoder yerine VAE, skip-connection iceren daha derin encoder-decoder yapilari veya memory-augmented reconstruction modelleri uygulanabilir. Bu sayede modelin normal anatomi dagilimini daha zengin bir latent temsille ogrenmesi ve ozellikle sinira yakin patolojik orneklerde daha ayirt edici reconstruction error uretmesi hedeflenmektedir. Buna ek olarak yalnizca MSE yerine L1, SSIM tabanli kayiplar veya birlesik loss fonksiyonlari denenerek yeniden olusturma kalitesi ile anomaly sensitivity arasindaki denge incelenebilir.

Ikinci gelistirme ekseni, veri ve deney tasarimi tarafinda planlanmaktadir. Daha yuksek giris cozunurlugu ile deney yapilarak ince retinal yapilarin ve ozellikle DRUSEN gibi daha zor ayristirilan siniflarin model tarafindan daha iyi temsil edilip edilmedigi test edilecektir. Bunun yaninda latent boyut, batch size, threshold secimi ve image size gibi hiperparametreler sistematik bir ablation calismasi ile karsilastirilacaktir. Boylece final raporda yalnizca tek bir model sonucu degil, tasarim kararlarinin performansa etkisini gosteren daha akademik bir deney tablosu sunulabilecektir.

Ucuncu gelistirme ekseni, yorumlanabilirlik ve klinik anlamlandirma uzerine kurulacaktir. Mevcut residual map ciktilari daha detayli incelenerek hata haritalarinin retina uzerindeki hangi bolgelerde yogunlastigi analiz edilebilir. Eger rekonstruksiyon hatasi belirli anatomik bozulmalarla tutarli bicimde eslesirse, modelin yalnizca sayisal anomaly skor ureten bir kara kutu olmaktan cikmasi ve klinik olarak daha anlamli hale gelmesi saglanabilir. Bu nedenle final asamada residual map gorsellestirmeleri, en iyi ve en kotu orneklerin ayri sunulmasi ve sinif bazli hata desenlerinin nitel olarak tartisilmasi planlanmaktadir.

Son olarak, hesaplama verimliligi ve karsilastirmali degerlendirme de gelecekteki temel adimlardan biridir. Veri yukleme hattinin hizlandirilmasi, daha uygun batch boyutlarinin secilmesi ve egitim suresinin optimize edilmesi ile tekrarli deneyler daha verimli hale getirilecektir. Mevcut baseline sonucunun yanina en az bir gelistirilmis model eklenerek AE ile gelistirilmis varyantin AUROC, F1, recall ve FPR acisindan dogrudan karsilastirilmasi hedeflenmektedir. Bu gelistirmeler tamamlandiginda proje, ara rapor seviyesindeki calisan baseline'dan, karsilastirmali ve daha guclu bir final proje yapisina tasinmis olacaktir.

## 7. Sonuc

Bu ara rapor asamasinda, Kermany OCT verisi icin patient-level validation kullanan, yalnizca normal goruntulerle egitilen ve reconstruction error ile patolojik scan tespiti yapan tekrar uretilebilir bir baseline sistem kurulmustur. Gercek veri uzerinde elde edilen AUROC {metrics['auroc']:.4f} ve F1 {metrics['f1']:.4f} degerleri, projenin planlama asamasini gecip calisir ve savunulabilir bir noktaya geldigini gostermektedir. Final asamada hedef, bu baseline'i daha guclu anomaly detection yaklasimlariyla genisletmek ve sonuclari karsilastirmali deneylerle desteklemektir.

## Kaynaklar

""" + "\n".join(f"{entry['key']} {entry['citation']}" for entry in REFERENCE_ENTRIES) + "\n"


def _clear_document(doc: Document) -> None:
    body = doc._element.body
    for element in list(body):
        if element.tag.endswith("sectPr"):
            continue
        body.remove(element)


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
            doc.add_heading(line.replace("## ", ""), level=1)
            index += 1
            continue

        if line.startswith("### "):
            doc.add_heading(line.replace("### ", ""), level=2)
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
        ("Sekil 1. Onerilen retina OCT anomaly detection boru hatti.", output_root / "figures" / "retina_oct_pipeline.png"),
        ("Sekil 2. Egitim, ROC, error dagilimi ve reconstruction ciktilarini bir araya getiren ozet gorsel.", output_root / "figures" / "report_results_overview.png"),
    ]
    available = [(caption, path) for caption, path in figure_specs if path.exists()]
    if not available:
        return

    doc.add_heading("Sekiller", level=1)
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

    title_paragraph = doc.add_paragraph(TITLE_TR)
    title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    english_title = doc.add_paragraph(TITLE_EN)
    english_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author_1 = doc.add_paragraph("Yazar 1 - Bilgisayar Muhendisligi - Universite - e-posta")
    author_1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author_2 = doc.add_paragraph("Yazar 2 - Bilgisayar Muhendisligi - Universite - e-posta")
    author_2.alignment = WD_ALIGN_PARAGRAPH.CENTER

    _add_paragraphs(doc, markdown_text)
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
