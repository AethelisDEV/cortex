use std::collections::HashMap;

pub struct ThalamusRouter {
    #[allow(dead_code)]
    pub lobe_keywords: HashMap<String, Vec<String>>, // keyword -> list of lobe_names (Geriye uyumluluk için boş tutulur)
    pub all_lobes: std::collections::HashSet<String>, // Veritabanındaki tüm lob isimlerinin RAM cache'i
}

impl ThalamusRouter {
    pub fn new() -> Self {
        Self {
            lobe_keywords: HashMap::new(),
            all_lobes: std::collections::HashSet::new(),
        }
    }

    /// Veritabanındaki tüm lob isimlerini RAM cache'ine yükler (Performans optimizasyonu).
    pub fn reload_mappings(&mut self, db: &sled::Db) -> anyhow::Result<()> {
        let mut lobes: std::collections::HashSet<String> = if let Some(bytes) = db.get("__lobes__")? {
            bincode::deserialize(&bytes).unwrap_or_default()
        } else {
            std::collections::HashSet::new()
        };

        if lobes.is_empty() {
            println!("[Talamus] Lobe dizini oluşturuluyor (Bir defaya mahsus tarama, lütfen bekleyin)...");
            for item in db.iter() {
                if let Ok((key, _)) = item {
                    if let Ok(lobe_name) = std::str::from_utf8(&key) {
                        if lobe_name != "core_language" && lobe_name != "general" && lobe_name != "__registry__" && lobe_name != "__lobes__" {
                            lobes.insert(lobe_name.to_string());
                        }
                    }
                }
            }
            let lobes_bytes = bincode::serialize(&lobes)?;
            db.insert("__lobes__", lobes_bytes)?;
            db.flush()?;
            println!("[Talamus] Lobe dizini başarıyla kaydedildi: {} adet lob.", lobes.len());
        }

        self.all_lobes = lobes;
        Ok(())
    }

    /// Türkçe kelimelerin İngilizce kavram köprülerini (eş-anlamlılarını) döner.
    pub fn get_synonyms(word: &str) -> Vec<String> {
        match word {
            "başlat" | "çalıştır" | "yap" => {
                vec!["run".to_string(), "start".to_string(), "execute".to_string(), "main".to_string(), "init".to_string()]
            }
            "kodla" | "yaz" | "oluştur" => {
                vec!["code".to_string(), "write".to_string(), "create".to_string(), "struct".to_string(), "fn".to_string()]
            }
            "göster" | "yazdır" | "bas" => {
                vec!["print".to_string(), "println".to_string(), "show".to_string(), "display".to_string()]
            }
            _ => vec![]
        }
    }

    /// Girdiyi kelimelerine bölerek frekans haritası (Tag Map) çıkartır.
    pub fn tokenize_query(&self, text: &str) -> HashMap<String, f32> {
        let mut tag_map = HashMap::new();
        let words = clean_text_to_words(text);
        
        if words.is_empty() {
            return tag_map;
        }

        let total_lobes = self.all_lobes.len();

        for word in &words {
            let match_count = self.count_matching_lobes(word);

            let ratio = match_count as f32 / total_lobes.max(1) as f32;
            // Güvenli Logaritma Kalkanı:
            let activation = 1.0 / (1.0 + ratio + ((match_count + 1) as f32).ln());

            *tag_map.entry(word.clone()).or_insert(0.0) += activation;
            
            // Eş-Anlamlı Kavram Köprüleri (Concept Synonyms)
            for syn in Self::get_synonyms(word) {
                tag_map.entry(syn).or_insert(activation * 0.8);
            }
        }

        // Değerleri normalize et (maksimum frekans 1.0 olacak şekilde)
        let max_val = tag_map.values().cloned().fold(0.0f32, f32::max);
        if max_val > 0.0 {
            for val in tag_map.values_mut() {
                *val /= max_val;
            }
        }

        tag_map
    }

    /// Sorgudan yola çıkarak yüklenmesi (diskten çekilmesi) gereken hedef hafıza loblarını belirler.
    pub fn route_query_lobes(&self, text: &str, db: &sled::Db) -> Vec<String> {
        let matched = self.find_dynamic_matching_lobes(text, db);
        if !matched.is_empty() {
            return matched;
        }

        // Eğer hiçbir lob eşleşmediyse, varsayılan lobu döneriz
        vec!["general".to_string()]
    }

    /// Dinamik olarak eşleşen tüm loblardaki proxy olmayan düğümleri 1.000 tam enerji seviyesine getirir.
    pub fn perform_lobe_wide_spiking(&self, cortex: &mut crate::cortex_graph::CortexGraph, query: &str) {
        let matched = self.find_dynamic_matching_lobes(query, &cortex.db);

        if !matched.is_empty() {
            // Eşleşen tüm lobların düğümlerini 1.0 uyarım seviyesine getir
            for lobe in &matched {
                let mut count = 0;
                let indices: Vec<_> = cortex.graph.node_indices().collect();
                for idx in indices {
                    let node = &cortex.graph[idx];
                    if !node.is_proxy && node.lobe_name == *lobe {
                        if let Some(node_mut) = cortex.graph.node_weight_mut(idx) {
                            node_mut.activation_level = 1.0;
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    println!("[Talamus] Akıllı Dinamik Tetikleyici: '{}' lobundaki {} adet düğüm 1.000 uyarım seviyesine yükseltildi.", lobe, count);
                }
            }
        }
    }

    /// Kelimenin kaç adet lob ile eşleştiğini dinamik olarak hesaplar.
    pub fn count_matching_lobes(&self, word: &str) -> usize {
        let norm_word = normalize_for_match(word);
        let (query_stem, _) = crate::morphology::parse_noun_suffix(&norm_word);
        let query_stem_norm = normalize_for_match(&query_stem);
        if query_stem_norm.is_empty() {
            return 0;
        }

        let mut match_count = 0;
        for lobe in &self.all_lobes {
            let norm_lobe = normalize_for_match(lobe);
            let components: Vec<&str> = norm_lobe.split('_').collect();
            let is_match = components.iter().any(|comp| {
                if query_stem_norm.len() <= 4 {
                    comp == &query_stem_norm
                } else {
                    comp == &query_stem_norm || comp.starts_with(&query_stem_norm)
                }
            });
            if is_match {
                match_count += 1;
            }
        }
        match_count
    }

    pub fn find_dynamic_matching_lobes(&self, query: &str, _db: &sled::Db) -> Vec<String> {
        let query_lower = query.to_lowercase();
        // Boşluklar ve temel noktalama işaretlerine göre bölüp temizle
        let words: Vec<&str> = query_lower.split(|c: char| c.is_whitespace() || c == '?' || c == '!' || c == '.' || c == ',')
            .map(|w| w.trim())
            .filter(|w| !w.is_empty() && w.len() > 1)
            .collect();

        if words.is_empty() {
            return Vec::new();
        }

        // 1. Sorgudaki kelimelerin normalize edilmiş köklerini hesapla
        let mut query_stems = Vec::new();
        for word in &words {
            let norm_word = normalize_for_match(word);
            let (query_stem, _) = crate::morphology::parse_noun_suffix(&norm_word);
            let query_stem_norm = normalize_for_match(&query_stem);
            if !query_stem_norm.is_empty() && !query_stems.contains(&query_stem_norm) {
                query_stems.push(query_stem_norm);
            }
        }

        if query_stems.is_empty() {
            return Vec::new();
        }

        // 2. Tek geçişte:
        // - Kelimelerin kaç farklı lobda geçtiğini say (TF-IDF Paydası)
        // - Lobların hangi kelimelerle eşleştiğini kaydet
        let mut match_counts = vec![0usize; query_stems.len()];
        let mut lobe_matches = HashMap::new();

        for lobe in &self.all_lobes {
            let norm_lobe = normalize_for_match(lobe);
            let components: Vec<&str> = norm_lobe.split('_').collect();
            
            let mut matched_indices = Vec::new();
            for (idx, query_stem_norm) in query_stems.iter().enumerate() {
                let is_match = components.iter().any(|comp| {
                    if query_stem_norm.len() <= 4 {
                        comp == query_stem_norm
                    } else {
                        comp == query_stem_norm || comp.starts_with(query_stem_norm)
                    }
                });
                if is_match {
                    matched_indices.push(idx);
                }
            }
            
            if !matched_indices.is_empty() {
                for &idx in &matched_indices {
                    match_counts[idx] += 1;
                }
                lobe_matches.insert(lobe.clone(), matched_indices);
            }
        }

        // 3. Her bir kelime için TF-IDF (Logaritmik Sönümleme) aktivasyonunu hesapla
        let total_lobes = self.all_lobes.len();
        let mut stem_activations = vec![0.0f32; query_stems.len()];
        for idx in 0..query_stems.len() {
            let match_count = match_counts[idx];
            let ratio = match_count as f32 / total_lobes.max(1) as f32;
            let activation = 1.0 / (1.0 + ratio + ((match_count + 1) as f32).ln());
            stem_activations[idx] = activation;
        }

        // 4. Eşleşen lobları, eşleşen kelimelerin toplam TF-IDF ağırlığına göre puanla
        let mut lobe_scores = Vec::with_capacity(lobe_matches.len());
        for (lobe, matched_indices) in lobe_matches {
            let mut score = 0.0f32;
            for idx in matched_indices {
                score += stem_activations[idx];
            }
            lobe_scores.push((lobe, score));
        }

        // 5. Puanlara göre azalan sırada sırala ve en alakalı ilk 15 lobu dön
        lobe_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        
        let matched_lobes: Vec<String> = lobe_scores.into_iter()
            .take(15)
            .map(|(lobe, _)| lobe)
            .collect();

        matched_lobes
    }
}

/// Türkçe/İngilizce karakter esnekliğini sağlamak için normalizasyon köprüsü
pub fn normalize_for_match(s: &str) -> String {
    crate::morphology::lowercase_tr(s)
        .chars()
        .map(|c| match c {
            'ı' => 'i',
            'ğ' | 'Ğ' => 'g',
            'ü' | 'Ü' => 'u',
            'ş' | 'Ş' => 's',
            'ö' | 'Ö' => 'o',
            'ç' | 'Ç' => 'c',
            _ => c,
        })
        .collect()
}

/// Metni temizleyip küçük harfli kelime dizisine çeviren yardımcı fonksiyon.
pub fn clean_text_to_words(text: &str) -> Vec<String> {
    crate::morphology::lowercase_tr(text)
        .split(|c: char| !c.is_alphabetic() && c != '_' && c != '$' && c != '-')
        .map(|w| w.trim())
        .filter(|w| !w.is_empty() && w.len() > 1)
        .map(|w| w.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_stopword_filtering() {
        let mut router = ThalamusRouter::new();
        // Insert 200 lobes
        for i in 0..200 {
            router.all_lobes.insert(format!("lobe_{}", i));
        }

        // Add a common word to more than 0.5% of lobes
        router.all_lobes.insert("lobe_ortak_1".to_string());
        router.all_lobes.insert("lobe_ortak_2".to_string());
        router.all_lobes.insert("lobe_ortak_3".to_string());
        router.all_lobes.insert("lobe_ortak_4".to_string());
        router.all_lobes.insert("lobe_ortak_5".to_string());

        // A rare word
        router.all_lobes.insert("lobe_nadir".to_string());

        // Tokenize query with a dummy database
        let tag_map = router.tokenize_query("ortak nadir");

        // Common word 'ortak' is not completely discarded, but has significantly decayed weight
        let ortak_weight = tag_map.get("ortak").cloned().unwrap_or(0.0);
        let nadir_weight = tag_map.get("nadir").cloned().unwrap_or(0.0);
        assert!(ortak_weight < nadir_weight, "Common word 'ortak' should have lower weight than rare word 'nadir'.");
        assert!(tag_map.contains_key("nadir"), "Rare word 'nadir' should be kept.");
    }

    #[test]
    fn test_safe_logarithm_weight_shield() {
        let mut router = ThalamusRouter::new();
        router.all_lobes.insert("lobe_gta_6".to_string());
        router.all_lobes.insert("lobe_zaman".to_string());
        router.all_lobes.insert("other_zaman".to_string());

        // Total lobes <= 100, so no stop-word filtering activates.
        // But activation values should differ based on match count (rarity).
        let tag_map = router.tokenize_query("gta zaman");

        let gta_weight = tag_map.get("gta").cloned().unwrap_or(0.0);
        let zaman_weight = tag_map.get("zaman").cloned().unwrap_or(0.0);

        assert_eq!(gta_weight, 1.0);
        assert!(zaman_weight < 1.0, "Common word 'zaman' should have lower weight than rare word 'gta'");
    }
}
