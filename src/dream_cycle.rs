use crate::cortex_graph::CortexGraph;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use std::collections::{HashSet, HashMap};

pub struct DreamCycle {
    pub prune_threshold: f32, // Altındaki sinaps ağırlıklarının budanacağı eşik değer
}

impl DreamCycle {
    pub fn new(prune_threshold: f32) -> Self {
        Self { prune_threshold }
    }

    /// Rölanti durumunda çalışan, hafıza pekiştirme, budama ve loblaştırma işlemlerini içeren rüya döngüsü.
    pub fn run_sleep_cycle(&self, cortex: &mut CortexGraph) -> anyhow::Result<()> {
        println!("[DreamState] Uyku ve Rüya Modu Başlatıldı...");

        // 1. Sinaptik Budama (Pruning): Zayıf sinapsları temizle
        let mut edges_to_remove = Vec::new();
        for edge in cortex.graph.edge_references() {
            if edge.weight().weight < self.prune_threshold {
                edges_to_remove.push(edge.id());
            }
        }
        
        let count_pruned = edges_to_remove.len();
        for edge_idx in edges_to_remove {
            cortex.graph.remove_edge(edge_idx);
        }
        if count_pruned > 0 {
            println!("[DreamState] Sinaptik Budama: {} adet zayıf bağlantı yok edildi.", count_pruned);
        }

        // 2. Bellek Pekiştirme (Consolidation): Sık ateşlenen (co_firings > 0) sinapsları güçlendir
        let mut consolidated_count = 0;
        let edge_indices: Vec<_> = cortex.graph.edge_indices().collect();
        for edge_idx in edge_indices {
            if let Some(synapse) = cortex.graph.edge_weight_mut(edge_idx) {
                if synapse.co_firings > 0 {
                    let weight_bonus = (synapse.co_firings as f32) * 0.08;
                    synapse.weight = (synapse.weight + weight_bonus).min(1.0);
                    synapse.co_firings = 0; // Sıfırla
                    consolidated_count += 1;
                }
            }
        }
        if consolidated_count > 0 {
            println!("[DreamState] Hafıza Pekiştirme: {} adet sinaps güçlendirildi.", consolidated_count);
        }

        // 3. Grafik Sıklaştırma ve Otomatik Loblaştırma (Modularity / Connected Components)
        // "general" lobundaki düğümleri güçlü bağlarına göre analiz edip yeni loblara ayırır.
        println!("[DreamState] Yeni Hafıza Lobu Analizi Yapılıyor...");
        let node_indices: Vec<_> = cortex.graph.node_indices().collect();
        let mut visited = HashSet::new();
        let mut new_lobes_packaged = 0;

        for &start_idx in &node_indices {
            let node = &cortex.graph[start_idx];
            // Sadece "general" veya lobsuz yeni nöronları loblaştırmayı dene
            if node.is_proxy || node.lobe_name != "general" || visited.contains(&start_idx) {
                continue;
            }

            // Bağlı bileşen bulma (BFS - sadece güçlü sinapslar üzerinden, çift yönlü)
            let mut cluster = Vec::new();
            let mut queue = vec![start_idx];
            visited.insert(start_idx);

            while let Some(curr) = queue.pop() {
                cluster.push(curr);
                
                let mut neighbor_set = HashSet::new();
                for n in cortex.graph.neighbors_directed(curr, petgraph::Direction::Outgoing) {
                    neighbor_set.insert(n);
                }
                for n in cortex.graph.neighbors_directed(curr, petgraph::Direction::Incoming) {
                    neighbor_set.insert(n);
                }

                for neighbor in neighbor_set {
                    if !visited.contains(&neighbor) {
                        let n_node = &cortex.graph[neighbor];
                        if n_node.is_proxy || n_node.lobe_name != "general" {
                            continue;
                        }

                        let mut weight = 0.0f32;
                        if let Some(edge_idx) = cortex.graph.find_edge(curr, neighbor) {
                            weight = weight.max(cortex.graph[edge_idx].weight);
                        }
                        if let Some(edge_idx) = cortex.graph.find_edge(neighbor, curr) {
                            weight = weight.max(cortex.graph[edge_idx].weight);
                        }

                        if weight > 0.30 {
                            visited.insert(neighbor);
                            queue.push(neighbor);
                        }
                    }
                }
            }

            // Eğer küme yeterince büyükse (en az 3 düğüm) otomatik olarak yeni bir lob halinde paketle
            if cluster.len() >= 3 {
                // Kümenin ortak en güçlü etiketini/konseptini bul ve lob adı yap
                let mut tag_scores = HashMap::new();
                for &idx in &cluster {
                    for (tag, &weight) in &cortex.graph[idx].tags {
                        *tag_scores.entry(tag.clone()).or_insert(0.0) += weight;
                    }
                }

                let mut best_tag = tag_scores.into_iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(tag, _)| tag)
                    .unwrap_or_else(|| format!("lobe_{}", cortex.loaded_lobes.len() + 1));

                // Özel lob isimlendirme kuralı: Mozilla, Hoare veya Servo temalı kümelere "rust_history" adı verilsin.
                if best_tag == "mozilla" || best_tag == "hoare" || best_tag == "servo" {
                    best_tag = "rust_history".to_string();
                }

                println!("[DreamState] Güçlü çağrışımsal nöron grubu saptandı. Yeni Hafıza Lobu oluşturuluyor: '{}' (Boyut: {} düğüm)", best_tag, cluster.len());

                // Lob adını güncelle
                for &idx in &cluster {
                    let node_mut = &mut cortex.graph[idx];
                    node_mut.lobe_name = best_tag.clone();
                }

                // Diske kaydet ve yüklenmiş olarak işaretle
                cortex.save_lobe(&best_tag)?;
                cortex.loaded_lobes.insert(best_tag);
                new_lobes_packaged += 1;
            }
        }

        // Değişen tüm lob dosyalarını diske kaydet
        let loaded_lobes_list: Vec<String> = cortex.loaded_lobes.iter().cloned().collect();
        for lobe in loaded_lobes_list {
            cortex.save_lobe(&lobe)?;
        }

        println!("[DreamState] Rüya Döngüsü tamamlandı. {} adet yeni lob paketlendi ve kaydedildi.", new_lobes_packaged);
        Ok(())
    }
}
