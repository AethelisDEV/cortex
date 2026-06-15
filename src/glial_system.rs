use crate::cortex_graph::CortexGraph;
use std::collections::HashMap;

pub struct GlialSystem {
    pub max_loaded_lobes: usize,
    pub decay_rate: f32,
}

impl GlialSystem {
    pub fn new(max_loaded_lobes: usize, decay_rate: f32) -> Self {
        Self {
            max_loaded_lobes,
            decay_rate,
        }
    }

    /// RAM'deki aktif lob sayısını ve uyarım sönümlenmesini yönetir.
    pub fn regulate_and_optimize(&self, cortex: &mut CortexGraph) -> anyhow::Result<()> {
        // 1. Aktivasyon Sönümleme (Damping)
        let node_indices: Vec<_> = cortex.graph.node_indices().collect();
        for node_idx in node_indices {
            if let Some(node) = cortex.graph.node_weight_mut(node_idx) {
                if !node.is_proxy {
                    node.activation_level = (node.activation_level - self.decay_rate).max(0.0);
                }
            }
        }

        // 2. RAM Lob Limiti Kontrolü
        // Yüklü lobları kontrol et. Eğer limit aşılmışsa, en az uyarılmış (en düşük toplam aktivasyonlu) lobu diske boşalt.
        let loaded_count = cortex.loaded_lobes.iter()
            .filter(|l| l.as_str() != "core_language" && l.as_str() != "general")
            .count();

        if loaded_count > self.max_loaded_lobes {
            println!("[Glia] RAM Lob Limiti aşıldı (Aktif Lob Sayısı: {}/{}). En az uyarılmış lob tahliye ediliyor...", 
                cortex.loaded_lobes.len(), self.max_loaded_lobes + 2 // +2 general ve core_language için
            );

            // Tahliye edilebilecek loblar: ne core_language/general ne de locked olanlar
            let mut eviction_candidates: Vec<String> = cortex.loaded_lobes.iter()
                .filter(|l| l.as_str() != "core_language" && l.as_str() != "general" && !cortex.locked_lobes.contains(*l))
                .cloned()
                .collect();

            if !eviction_candidates.is_empty() {
                // Her lobun toplam aktivasyon seviyesini hesapla
                let mut lobe_activations = HashMap::new();
                for l in &eviction_candidates {
                    lobe_activations.insert(l.clone(), 0.0f32);
                }

                for node_idx in cortex.graph.node_indices() {
                    let node = &cortex.graph[node_idx];
                    if !node.is_proxy {
                        if let Some(act_sum) = lobe_activations.get_mut(&node.lobe_name) {
                            *act_sum += node.activation_level;
                        }
                    }
                }

                // Aktivasyon seviyesine göre sırala (en düşük olan en başta)
                eviction_candidates.sort_by(|a, b| {
                    let act_a = lobe_activations.get(a).cloned().unwrap_or(0.0);
                    let act_b = lobe_activations.get(b).cloned().unwrap_or(0.0);
                    act_a.partial_cmp(&act_b).unwrap_or(std::cmp::Ordering::Equal)
                });

                // En düşük aktivasyonlu lobu RAM'den tahliye et
                if let Some(lobe_to_unload) = eviction_candidates.first() {
                    cortex.unload_lobe(lobe_to_unload)?;
                }
            }
        }

        Ok(())
    }
}
