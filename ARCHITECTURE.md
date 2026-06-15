# Cortex Engine: Beyin Yapısına Benzer Modüler Yapay Zeka Mimarisi

Cortex Engine, geleneksel monolitik (tek parça) yapay zeka modellerinin aksine, insan beyninin çalışma prensiplerini taklit ederek **8GB VRAM** gibi kısıtlı donanımlarda yüksek verimle çalışacak şekilde tasarlanmış **sıfırdan eğitilebilir, dinamik ve modüler bir nöral grafik motorudur**.

Bu belge, sistemin biyolojik ilham kaynaklarını, kullanıcının vizyonunu ve kullanılacak matematiksel/algoritmik altyapıyı detaylandırmaktadır.

---

## 1. Biyolojik İlham ve Kullanıcı Vizyonu

### Atomik Bilgi Ayrıştırma ve Nöral Bağ Kurulumu
Geleneksel yapay zekalar verileri düz metin veya statik veri tabanlarında saklar. Cortex Engine ise verileri **nöronlar gibi milyarlarca küçük parçaya ayırır**:
- **Nöralleştirme (Neurogenesis)**: Girdi olarak verilen her bilgi (örn. bir Rust fonksiyonu, bir fizik teoremi) anlamlı en küçük mantıksal birimlerine bölünür. Her birim, kendi ağırlıklarına sahip bir mikro-nöral ağ olan **Alt Çekirdek (Sub-Nuclei)** veya bir **Bilgi Nöronu** haline getirilir.
- **Sinaptik Bağlantılar**: Bu nöronlar arasında anlamsal ve zamansal ilişkiler kurulur. Örneğin, "Rust" ve "Emniyetli Hafıza Yönetimi" kavramları ardışık işlendiğinde, aralarındaki sinaps gücü artar.
- **Konu Kümeleri (Loblar)**: Bu bağlar zamanla büyüyerek anlamsal kümeler oluşturur (Rust Kümesi, Kuantum Fizik Kümesi vb.).

### Çoklu Küme Aktivasyonu ve Dinamik VRAM Yükleme
Tüm beyni aynı anda VRAM'e yüklemek yerine, yalnızca o anki bilince (bağlama) hitap eden bölgeler uyarılır:
- Sistemde sadece **Talamus (Router)** ve temel **Glia** fonksiyonları VRAM'de sürekli uyanık kalır.
- Kullanıcı *"Rust"* hakkında konuştuğunda, Talamus bunu algılar ve sadece Rust Kümesi'ni diskten/RAM'den GPU VRAM'ine yükler.
- Eğer kullanıcı *"Rust ile kuantum fizik simülasyonu"* gibi karmaşık bir sorgu gönderirse, **hem Rust hem de Kuantum Fizik kümeleri eş zamanlı olarak VRAM'e yüklenir** ve aralarındaki geçiş bağlantıları üzerinden entegre şekilde çalışır.

### Glia Hücreleri (Metabolizma ve VRAM Kontrolü)
Biyolojik beyindeki glia hücreleri (özellikle astrositler) nöronların beslenmesini, sinaps temizliğini ve enerji dağıtımını yönetir. Cortex Engine'de Glia:
- **VRAM Monitörü**: Sürekli arka planda çalışarak GPU bellek durumunu izler.
- **Bellek Boşaltıcı (Garbage Collection)**: Ateşlenme sıklığı azalan veya uzun süredir uyarılmayan modülleri dinamik olarak VRAM'den RAM'e veya diske geri taşır. Bu sayede 8GB VRAM sınırı hiçbir zaman aşılmaz.

### Rüya Modu ve Rölanti Optimizasyonu (Dream-State Consolidation)
Sistem kullanıcıdan girdi almadığında (rölantideyken) uyku/rüya moduna geçer:
- **Sinaptik Budama (Pruning)**: Kullanılmayan veya ağırlığı çok düşük olan zayıf bağlantılar silinir.
- **Bellek Pekiştirme (Consolidation)**: Gün içinde öğrenilen yeni atomik bilgiler taranır, ilişkileri güçlendirilir ve kalıcı kümelerle entegre edilir.
- **Kümeleme Optimizasyonu**: Louvain veya K-Means benzeri topluluk algoritmaları çalıştırılarak grafik yapısındaki loblar yeniden düzenlenir ve sadeleştirilir.

---

## 2. Matematiksel ve Algoritmik Altyapı

### A. Mikro-Modül Yapısı (Sub-Nuclei)
Her bir modül (`SubNucleus`), Hugging Face'in `candle` kütüphanesini kullanan küçük bir yapay sinir ağıdır (örn. çok katmanlı bir MLP veya minik bir Recurrent-Attention katmanı).
- Her modülün girdi vektör boyutu $I$ ve çıktı vektör boyutu $O$'dur.
- Modüller kendi ağırlık matrislerini ($W$) ve bias değerlerini ($b$) taşırlar.
- Etkinleştirildiklerinde lokal aktivasyon değerlerini hesaplarlar:
  $$a_{out} = \sigma(W \cdot a_{in} + b)$$
  *(Burada $\sigma$ aktivasyon fonksiyonudur, örn. GeLU veya ReLU)*

### B. Aktivasyon Yayılımı (Spiking & Energy Flow)
Sistem, girdiyi semantik bir embedding vektörüne dönüştürür.
1. Talamus, bu vektöre en yakın düğümleri bulur ve onlara başlangıç "enerjisini" ($E_0$) verir.
2. Enerji, sinaptik ağırlıklar ($w_{ij}$) üzerinden komşu düğümlere yayılır:
   $$E_j = \sum_{i} E_i \cdot w_{ij}$$
3. Bir düğümün enerjisi belirli bir **Aktivasyon Eşiğini (Threshold $\theta$)** aşarsa, o düğüm (modül) "ateşlenir" ve VRAM'e yüklenmek üzere kuyruğa eklenir.

### C. Lokal Öğrenme ve Plastisite (Hebbian Learning)
Küresel backpropagation yerine, modüller arası ilişkiler **Hebbian Plastisite** formülüyle güncellenir:
$$w_{ij}^{(t+1)} = w_{ij}^{(t)} + \eta \cdot a_i \cdot a_j - \lambda \cdot w_{ij}^{(t)}$$
- $\eta$: Öğrenme oranı (learning rate).
- $a_i, a_j$: $i$ ve $j$ modüllerinin aktivasyon seviyeleri.
- $\lambda$: Unutma/bozulma katsayısı (decay rate), aktif olmayan bağların zamanla zayıflamasını sağlar.

Modüllerin kendi iç ağırlıkları ise yerel tahmin hatalarına göre (Predictive Coding) güncellenir. Her modül bir sonraki modülün durumunu tahmin etmeye çalışır ve hata ($e$) oranına göre yerel gradyan hesaplar:
$$\Delta W_{local} = -\eta_{local} \cdot \frac{\partial \mathcal{L}(e)}{\partial W}$$

---

## 3. Sistem Mimarisi Detayları (Rust & Candle)

```
+-------------------------------------------------------+
|                    KULLANICI ARAYÜZÜ (CLI / Ham Veri)  |
+--------------------------+----------------------------+
                           | (Ham Metin)
                           v
+--------------------------+----------------------------+
|             VERİ GİRİŞİ (Ingestion Pipeline)          |
|  - TextChunker: Cümle/Mantıksal Birim Ayrıştırma      |
|  - Sıralı & Semantik Sinaps Kurulumu                  |
+--------------------------+----------------------------+
                           |
                           v
+--------------------------+----------------------------+
|             TALAMUS (Semantik Yönlendirici)           |
|  - Girdi Analizi         - Aktif Küme Belirleme       |
+--------------------------+----------------------------+
                           |
            +--------------+--------------+
            |                             | (Aktivasyon Sinyali)
            v                             v
+--------------------------+  +-------------------------+
|     GLİA SİSTEMİ         |  |   CORTEX GRAFİK MOTORU  |
|  - VRAM Yönetimi         |  |  - Nöron & Sinaps Ağı   |
|  - RAM/GPU Hot-Swap      |  |  - Kümeleme Yapısı      |
+-------------+------------+  +------------+------------+
              |                            |
              +--------------+-------------+
                             |
                             v
+----------------------------+--------------------------+
|                  GPU VRAM (CANDLE MOTORU)             |
|  - Aktif Kümeler (örn. Rust + Kuantum)                |
|  - Paralel Matris Hesaplamaları (Lokal Eğitim/Tahmin) |
+----------------------------+--------------------------+
                             |
                             v
+----------------------------+--------------------------+
|                  RÜYA MODU OPTİMİZASYONU              |
|  - Çevrimdışı Budama       - Bilgi Pekiştirme         |
+-------------------------------------------------------+
```

### Veri Giriş ve Ayrıştırma Modülü (`src/ingestion.rs`)
Verilerin sisteme yüklenmesi ve nöral ağ yapısına dönüştürülmesi tamamen modüler bir boru hattı üzerinden yürütülür:
- **`TextChunker` Arayüzü**: Ham girdileri belirli kurallarla cümle veya mantıksal veri bloklarına (nöron adaylarına) böler. Farklı döküman tipleri için özelleştirilmiş chunker'lar bu trait ile sisteme dahil edilebilir.
- **`SentenceChunker`**: Noktalama ve uzunluk limitlerine göre metinleri atomik anlam birimlerine bölen varsayılan modüldür.
- **Dosya Değişim Takip Mekanizması (State Registry)**:
  - Klasördeki dosyaların tekrar tekrar işlenmesini engellemek için `cortex_storage/ingested_files.json` dosyasında bir durum kayıt defteri tutulur.
  - Her dosyanın içeriği `DefaultHasher` (64-bit FNV-1a benzeri) ile hash'lenir. Sadece yeni veya hash değeri değişmiş (güncellenmiş) dosyalar işlenerek sisteme yüklenir, değişmeyen dosyalar atlanır.
- **`IngestionPipeline`**:
  1. Ayrıştırılan her bir parça için grafik üzerinde mükerrerlik kontrolü yapar. Eğer nöron yoksa oluşturur.
  2. Nörona ait semantik `concept_vector` Talamus ile çıkartılarak düğüme atanır.
  3. Yeni nöronun aktivasyonu `1.0` (tam uyarılmış) olarak başlatılır ve Talamus ile benzer konseptteki komşu nöronlara enerji yayılır.
  4. Glia uyarılıp aktifleşen modülleri VRAM'e taşır.
  5. Aktifleşen modül üzerinde **Lokal Tahmin/Oto-Kodlama Eğitimi** (`train_local`) çalıştırılarak modülün kendi semantik kimliğini kavraması sağlanır.
  6. Peş peşe gelen cümleler arasında yönlendirilmiş sıralı sinapslar ($A \to B$) kurulurken, eş-zamanlı uyanan benzer nöronlar da Hebbian Plastisitesi aracılığıyla yatay olarak birbirine bağlanır.

### Bellek Dağılım Tahmini (8GB VRAM)
- **Glia & Talamus Çekirdeği (CPU/VRAM tabanlı)**: ~200 MB
- **Semantik Arama Modeli (Embeddings için)**: ~500 MB
- **GPU Tensor Belleği (Aktif Modüller)**: ~4.0 GB (Maksimum 500 aktif mikro-modülün aynı anda yüklenmesi için ayrılan dinamik alan)
- **Bağlam Belleği (Kombine Çalışma Alanı)**: ~2.0 GB
- **Geriye Kalan Güvenli Alan**: ~1.3 GB
