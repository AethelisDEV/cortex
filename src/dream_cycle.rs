use crate::cortex_graph::CortexGraph;
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
        println!("[DreamState] Uyku ve Rüya Modu Başlatıldı (Otonom Plastisite ve Budama)...");

        // 1. Veritabanındaki tüm lob listesini al
        let lobes: std::collections::HashSet<String> = if let Some(bytes) = cortex.db.get("__lobes__")? {
            bincode::deserialize(&bytes).unwrap_or_default()
        } else {
            std::collections::HashSet::new()
        };

        let mut total_pruned_synapses = 0;
        let mut total_consolidated_synapses = 0;
        let mut total_decayed_synapses = 0;

        // 2. Her bir lobu tek tek yükle, buda ve pekiştir, kaydet ve RAM'den temizle
        for lobe_name in &lobes {
            // Lobu yükle
            cortex.load_lobe(lobe_name)?;

            // Bu lobdaki kenarları topla ve filtrele
            let mut edges_to_remove = Vec::new();
            let edge_indices: Vec<_> = cortex.graph.edge_indices().collect();

            for edge_idx in edge_indices {
                if let Some((u, _)) = cortex.graph.edge_endpoints(edge_idx) {
                    let source_node = &cortex.graph[u];
                    if source_node.lobe_name == *lobe_name && !source_node.is_proxy {
                        if let Some(synapse) = cortex.graph.edge_weight_mut(edge_idx) {
                            if synapse.co_firings > 0 {
                                // Birlikte uyarılmış/ateşlenmiş: Pekiştir
                                let weight_bonus = (synapse.co_firings as f32) * 0.1;
                                synapse.weight = (synapse.weight + weight_bonus).min(1.0);
                                synapse.co_firings = 0; // Sıfırla
                                total_consolidated_synapses += 1;
                            } else {
                                // Ateşlenmemiş: Zayıflat
                                synapse.weight -= 0.05;
                                total_decayed_synapses += 1;
                            }

                            if synapse.weight < self.prune_threshold {
                                edges_to_remove.push(edge_idx);
                            }
                        }
                    }
                }
            }

            // Zayıf sinapsları sil
            let count_pruned = edges_to_remove.len();
            total_pruned_synapses += count_pruned;
            for edge_idx in edges_to_remove {
                cortex.graph.remove_edge(edge_idx);
            }

            // Lobu diske kaydet ve RAM'den temizle (core_language ve general hariç)
            cortex.save_lobe(lobe_name)?;
            if lobe_name != "core_language" && lobe_name != "general" {
                cortex.unload_lobe(lobe_name)?;
            }
        }

        println!(
            "[DreamState] Otonom Plastisite: {} adet sinaps pekiştirildi, {} adet sinaps zayıflatıldı, {} adet zayıf bağlantı budandı.",
            total_consolidated_synapses, total_decayed_synapses, total_pruned_synapses
        );

        // 3. Grafik Sıklaştırma ve Otomatik Loblaştırma (general lobundaki düğümleri güçlü bağlarına göre analiz edip yeni loblara ayırır)
        cortex.load_lobe("general")?;
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
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(tag, _)| tag)
                    .unwrap_or_else(|| format!("lobe_{}", lobes.len() + 1));

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
                cortex.loaded_lobes.insert(best_tag.clone());
                new_lobes_packaged += 1;
            }
        }

        // Değişen tüm lob dosyalarını diske kaydet ve gereksizleri RAM'den temizle
        let loaded_lobes_list: Vec<String> = cortex.loaded_lobes.iter().cloned().collect();
        for lobe in loaded_lobes_list {
            cortex.save_lobe(&lobe)?;
            if lobe != "core_language" && lobe != "general" {
                cortex.unload_lobe(&lobe)?;
            }
        }

        crate::morphology::update_stats_from_graph(&cortex.graph);
        println!("[DreamState] Rüya Döngüsü tamamlandı. {} adet yeni lob paketlendi ve kaydedildi.", new_lobes_packaged);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cortex_graph::SynapseType;
    use std::collections::HashMap;

    #[test]
    fn test_dream_cycle_autonomous_pruning_and_consolidation() {
        // Create a temporary database
        let db = sled::Config::new().temporary(true).open().unwrap();
        let mut cortex = CortexGraph::new(db.clone());
        let dream = DreamCycle::new(0.15);

        // 1. Create a dummy lobe "test_lobe" with some nodes and synapses
        cortex.load_lobe("test_lobe").unwrap();
        let n1 = cortex.add_node("Sentence 1", HashMap::new(), "test_lobe");
        let n2 = cortex.add_node("Sentence 2", HashMap::new(), "test_lobe");
        let n3 = cortex.add_node("Sentence 3", HashMap::new(), "test_lobe");

        // Synapse with co_firings > 0 (should be consolidated/strengthened)
        cortex.add_synapse(n1, n2, 0.5, SynapseType::Semantic);
        if let Some(edge_idx) = cortex.graph.find_edge(*cortex.id_to_index.get(&n1).unwrap(), *cortex.id_to_index.get(&n2).unwrap()) {
            if let Some(synapse) = cortex.graph.edge_weight_mut(edge_idx) {
                synapse.co_firings = 2;
            }
        }

        // Synapse with co_firings == 0 and low weight (should be decayed and pruned)
        cortex.add_synapse(n2, n3, 0.18, SynapseType::Semantic);

        // Save and unload lobe to simulate database-wide scenario
        cortex.save_lobe("test_lobe").unwrap();
        cortex.unload_lobe("test_lobe").unwrap();

        // 2. Run sleep cycle
        dream.run_sleep_cycle(&mut cortex).unwrap();

        // 3. Re-load the lobe to assert modifications
        cortex.load_lobe("test_lobe").unwrap();

        // n1 -> n2 should have co_firings = 0 and weight = 0.5 + 2 * 0.1 = 0.7
        let u1 = *cortex.id_to_index.get(&n1).unwrap();
        let u2 = *cortex.id_to_index.get(&n2).unwrap();
        let u3 = *cortex.id_to_index.get(&n3).unwrap();

        let edge_1_2 = cortex.graph.find_edge(u1, u2).expect("Synapse between n1 and n2 should exist");
        let synapse_1_2 = &cortex.graph[edge_1_2];
        assert_eq!(synapse_1_2.co_firings, 0, "co_firings should be reset to 0");
        assert!((synapse_1_2.weight - 0.7).abs() < 1e-5, "Weight should be consolidated to 0.7");

        // n2 -> n3 had weight 0.18. It had 0 co_firings, so decayed by 0.05 -> 0.13.
        // Since 0.13 < prune_threshold (0.15), it should have been pruned/deleted.
        assert!(cortex.graph.find_edge(u2, u3).is_none(), "Synapse between n2 and n3 should be pruned");
    }
}

