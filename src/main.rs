mod cortex_graph;
mod thalamus_router;
mod glial_system;
mod dream_cycle;
mod ingestion;
mod synthesizer;
mod morphology;
mod pdf_ingestor;

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;
use cortex_graph::{CortexGraph, SynapseType};
use thalamus_router::ThalamusRouter;
use glial_system::GlialSystem;
use dream_cycle::DreamCycle;
use ingestion::{SentenceChunker, IngestionPipeline};
use synthesizer::Synthesizer;

fn main() -> anyhow::Result<()> {
    println!("========================================================");
    println!("     CORTEX ENGINE: SAF GRAFİK & BÖLÜMLENMİŞ BELLEK    ");
    println!("========================================================");

    let storage_dir = Path::new("cortex_storage");
    let inputs_dir = Path::new("cortex_inputs");

    std::fs::create_dir_all(storage_dir)?;
    std::fs::create_dir_all(inputs_dir)?;

    println!("[Sistem] Bellek Deposu: {:?}", storage_dir);
    println!("[Sistem] Giriş Klasörü: {:?}", inputs_dir);

    // 1. Çekirdek Sistemlerin Başlatılması
    // Saf Grafik Motoru
    let mut cortex = CortexGraph::new(storage_dir);

    // Talamus Kelime Yönlendiricisi
    let mut router = ThalamusRouter::new();
    router.reload_mappings(storage_dir)?;

    // Glia RAM Yöneticisi (Maksimum 18 harici lob yüklenir, sönümlenme oranı 0.05)
    let glia = GlialSystem::new(18, 0.05);

    // Rüya İşçisi (Budama eşiği 0.15)
    let dream_worker = DreamCycle::new(0.15);

    // Veri Giriş Hattı
    let chunker = SentenceChunker::default();
    let pipeline = IngestionPipeline::new(chunker);

    // Her zaman uyanık kalan Core Language lobunu ve general lobunu yükle
    cortex.load_lobe("core_language")?;
    cortex.load_lobe("general")?;

    // Başlangıç Kurulumu (Grafik boşsa)
    let mut empty = true;
    for l in &["rust", "quantum", "math", "brain"] {
        let lobe_path = storage_dir.join(format!("{}_lobe.json", l));
        if lobe_path.exists() {
            empty = false;
            break;
        }
    }

    if empty && cortex.graph.node_count() == 0 {
        println!("[Sistem] Başlangıç nöronları oluşturuluyor ve loblara bölünüyor...");

        // Rust Lobu Nöronları
        cortex.load_lobe("rust")?;
        let r1 = cortex.add_node(
            "Rust bellek güvenliğini ve yüksek performansı derleme aşamasında garanti eder.",
            create_tag_map(&["rust", "memory", "performance", "derleme"]),
            "rust"
        );
        let r2 = cortex.add_node(
            "Borrow Checker sistemi, hafıza sızıntılarını ve veri yarışlarını önler.",
            create_tag_map(&["borrow", "checker", "memory", "rust"]),
            "rust"
        );
        cortex.add_synapse(r1, r2, 0.8, SynapseType::Sequential);
        cortex.save_lobe("rust")?;
        cortex.unload_lobe("rust")?;

        // Kuantum Lobu Nöronları
        cortex.load_lobe("quantum")?;
        let q1 = cortex.add_node(
            "Kuantum dolanıklığı, iki parçacığın spin durumlarının birbirine bağlı olmasıdır.",
            create_tag_map(&["quantum", "entanglement", "dolanıklık", "spin"]),
            "quantum"
        );
        let q2 = cortex.add_node(
            "Schrodinger'in kedisi deneyi, kuantum süperpozisyon durumunu temsil eder.",
            create_tag_map(&["schrodinger", "kedi", "superposition", "quantum"]),
            "quantum"
        );
        cortex.add_synapse(q1, q2, 0.85, SynapseType::Sequential);
        cortex.save_lobe("quantum")?;
        cortex.unload_lobe("quantum")?;

        // Dil Şablonları (Core Language Lobe)
        cortex.add_node(
            "öncelikle sistem mimarisini anlamak gerekir.",
            create_tag_map(&["system", "architecture"]),
            "core_language"
        );
        cortex.save_lobe("core_language")?;

        println!("[Sistem] Başlangıç lobları başarıyla diske kaydedildi.");
    }

    // İnteraktif Döngü
    loop {
        println!("\n========================================================");
        println!("  CORTEX MENU (Saf Grafik & Sıfır Yapay Zeka Ağırlığı):");
        println!("  [1] Klasör Yapısından Verileri Tara ve Yükle (cortex_inputs/)");
        println!("  [2] Elle Metin Girişi Yap (Data Ingestion & Connection)");
        println!("  [3] Sorgu Gönder ve Yanıt Üret (Semantic Recall & Synthesis)");
        println!("  [4] Rüya/Rölanti Modu Optimizasyonu (Budama & Otomatik Loblama)");
        println!("  [5] Grafik Durumunu ve RAM'deki Aktif Lobları Göster");
        println!("  [6] Çıkış (Exit)");
        println!("========================================================");
        print!("Seçiminiz [1-6]: ");
        io::stdout().flush()?;

        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;
        let choice = choice.trim();

        match choice {
            "1" => {
                println!("\n--- KLASÖR TARAMA MODU ---");
                if let Err(e) = pipeline.ingest_directory(&mut cortex, &router, &glia, inputs_dir) {
                    println!("[Hata] Klasör işlenemedi: {:?}", e);
                }
                let _ = router.reload_mappings(storage_dir);
            }
            "2" => {
                println!("\n--- ELLE METIN GİRİŞ MODU ---");
                println!("Metninizi girin (boş satırla sonlandırın):");
                let mut text_lines = Vec::new();
                loop {
                    let mut line = String::new();
                    io::stdin().read_line(&mut line)?;
                    if line.trim().is_empty() {
                        break;
                    }
                    text_lines.push(line);
                }
                let full_text = text_lines.join(" ");
                if !full_text.trim().is_empty() {
                    if let Err(e) = pipeline.ingest_text(&mut cortex, &router, &glia, &full_text, None) {
                        println!("[Hata] Metin girişi başarısız: {:?}", e);
                    }
                }
                let _ = router.reload_mappings(storage_dir);
            }
            "3" => {
                println!("\n--- SORGU VE ANLAMLI METİN SENTEZİ MODU ---");
                print!("Sorgunuzu girin: ");
                io::stdout().flush()?;
                let mut query = String::new();
                io::stdin().read_line(&mut query)?;
                let query = query.trim();
                if query.is_empty() {
                    continue;
                }

                // A. Talamus sorguyu inceler ve yüklenmesi gereken lobları belirler
                let target_lobes = router.route_query_lobes(query, storage_dir);
                println!("  -> Talamus Hedef Lobları Saptadı: {:?}", target_lobes);

                // B. İlgili lobları diskten RAM'e çek ve kilitle (Lobe Locking)
                cortex.locked_lobes.clear();
                cortex.locked_lobes.extend(target_lobes.clone());
                for lobe in &target_lobes {
                    if let Err(e) = cortex.load_lobe(lobe) {
                        println!("[Hata] Lobe yüklenemedi {:?}: {:?}", lobe, e);
                    }
                }

                // C. Uyarım yayılımı gerçekleştir
                let query_tags = router.tokenize_query(query);
                println!("  -> Uyarım Haritası: {:?}", query_tags);
                cortex.propagate_activation(&query_tags, 0.05, 2);

                // Lobe-Wide Spiking uyarımı gerçekleştir (Teknik sorgularda temel kod düğümlerini zorla uyar)
                router.perform_lobe_wide_spiking(&mut cortex, query);

                // E. Aktif uyanık nöronları topla (Proxy olmayan ve aktivasyonu yüksek olanlar)
                let mut active_nodes = Vec::new();
                for idx in cortex.graph.node_indices() {
                    let node = &cortex.graph[idx];
                    if !node.is_proxy && node.activation_level > 0.15 {
                        active_nodes.push(node.clone());
                    }
                }

                // Uyarım derecelerine göre sırala
                let mut sorted_active = active_nodes.clone();
                sorted_active.sort_by(|a, b| b.activation_level.partial_cmp(&a.activation_level).unwrap_or(std::cmp::Ordering::Equal));

                println!("  -> Aktif Bellek Nöronları: {}", active_nodes.len());
                let print_limit = 50;
                for node in sorted_active.iter().take(print_limit) {
                    println!("     * [{}] LOB: {} (Uyarım: {:.3}) - \"{}\"", node.id, node.lobe_name, node.activation_level, node.content);
                }
                if active_nodes.len() > print_limit {
                    println!("     ... (ve {} adet diğer uyarılmış nöron listelenmedi)", active_nodes.len() - print_limit);
                }

                // F. Kural Tabanlı Metin Sentezleyiciyi çalıştır
                let response = Synthesizer::synthesize_response(query, active_nodes, &cortex.graph);
                println!("\n{}", response);

                // G. Hebbian Öğrenme güncellemesi yap (Çağrışımları güçlendir)
                cortex.hebbian_update(0.1, 0.02);

                // H. Glia RAM bütçesini kontrol et ve sönümle (Sorgu bittikten sonra) ve kilitleri kaldır
                glia.regulate_and_optimize(&mut cortex)?;
                cortex.locked_lobes.clear();
            }
            "4" => {
                println!("\n--- RÖLANTİ MODU: UYKU VE RÜYA DÖNGÜSÜ ---");
                // Tüm lobları diske kaydet ve RAM'den temizle (General ve core_language hariç)
                let loaded_lobes_list: Vec<String> = cortex.loaded_lobes.iter().cloned().collect();
                for lobe in loaded_lobes_list {
                    cortex.save_lobe(&lobe)?;
                }

                if let Err(e) = dream_worker.run_sleep_cycle(&mut cortex) {
                    println!("[Hata] Uyku döngüsü hatası: {:?}", e);
                }
                let _ = router.reload_mappings(storage_dir);
            }
            "5" => {
                println!("\n--- GRAFİK DURUMU VE BELLEK ANALİZİ ---");
                println!("Toplam Düğüm Sayısı (RAM'de): {}", cortex.graph.node_count());
                println!("RAM'deki Yüklü Loblar: {:?}", cortex.loaded_lobes);
                
                println!("\n[Aktif Çalışma Belleği Düğümleri]:");
                let node_count = cortex.graph.node_count();
                let print_limit = 50;
                for (count, idx) in cortex.graph.node_indices().enumerate() {
                    if count >= print_limit {
                        println!("  ... (ve {} adet diğer düğüm listelenmedi)", node_count - print_limit);
                        break;
                    }
                    let node = &cortex.graph[idx];
                    let proxy_str = if node.is_proxy { " (Sanal ID Köprüsü)" } else { "" };
                    println!("  - [{}] \"{}\" [Lob: {}] - Akt: {:.3}{}", node.id, node.content, node.lobe_name, node.activation_level, proxy_str);
                }

                println!("\n[Sinaptik Bağlantılar]:");
                let edge_count = cortex.graph.edge_count();
                let mut has_synapse = false;
                for (count, idx) in cortex.graph.edge_indices().enumerate() {
                    if count >= print_limit {
                        println!("  ... (ve {} adet diğer sinaps listelenmedi)", edge_count - print_limit);
                        break;
                    }
                    if let Some((u_idx, v_idx)) = cortex.graph.edge_endpoints(idx) {
                        let u = &cortex.graph[u_idx];
                        let v = &cortex.graph[v_idx];
                        let synapse = &cortex.graph[idx];
                        let syn_type_str = match synapse.synapse_type {
                            SynapseType::Sequential => "Sequential",
                            SynapseType::Semantic => "Semantic",
                        };
                        println!("  - [{}] -> [{}] ({}) (Ağırlık: {:.3}, Tip: {}, Co-Firings: {})", 
                            u.id, v.id, v.content, synapse.weight, syn_type_str, synapse.co_firings);
                        has_synapse = true;
                    }
                }
                if !has_synapse {
                    println!("  (Aktif bir sinaptik bağ yok.)");
                }
            }
            "6" => {
                println!("\n[Sistem] Kapatılıyor. Tüm loblar diske kaydediliyor...");
                let loaded_lobes_list: Vec<String> = cortex.loaded_lobes.iter().cloned().collect();
                for lobe in loaded_lobes_list {
                    let _ = cortex.save_lobe(&lobe);
                }
                println!("Hoşça kalın!");
                break;
            }
            _ => {
                println!("[Hata] Geçersiz seçim!");
            }
        }
    }

    Ok(())
}

fn create_tag_map(words: &[&str]) -> HashMap<String, f32> {
    let mut map = HashMap::new();
    for w in words {
        map.insert(w.to_string(), 1.0f32);
    }
    map
}
