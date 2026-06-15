use std::collections::HashMap;

pub struct ThalamusRouter {
    #[allow(dead_code)]
    pub lobe_keywords: HashMap<String, Vec<String>>, // keyword -> list of lobe_names (Geriye uyumluluk için boş tutulur)
}

impl ThalamusRouter {
    pub fn new() -> Self {
        Self {
            lobe_keywords: HashMap::new(),
        }
    }

    /// Geriye uyumluluk için arayüzde kalan ama artık bir işlev yapmayan fonksiyon.
    pub fn reload_mappings(&mut self, _db: &sled::Db) -> anyhow::Result<()> {
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

        for word in &words {
            *tag_map.entry(word.clone()).or_insert(0.0) += 1.0;
            // Eş-Anlamlı Kavram Köprüleri (Concept Synonyms)
            for syn in Self::get_synonyms(word) {
                tag_map.entry(syn).or_insert(0.8);
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

    /// Sorgudaki kelimelerle mevcut lob isimlerini dinamik olarak karşılaştırıp eşleşen lob adlarını döner.
    pub fn find_dynamic_matching_lobes(&self, query: &str, db: &sled::Db) -> Vec<String> {
        // Soru ve Bağlaç Filtresi (Stop-Words Exclusion)
        let stopwords = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "from",
            "is", "was", "were", "are", "be", "been", "this", "that", "these", "those", "it", "its", "they",
            "their", "them", "he", "she", "his", "her", "as", "what", "how", "why", "where", "when", "who", "which",
            "ve", "veya", "ile", "için", "göre", "tarafından", "bir", "bu", "o", "şu", "da", "de", "ki", "en",
            "daha", "çok", "her", "ise", "gibi", "kadar", "olan", "olarak", "nedir", "nasil", "nasıl", "ne"
        ];

        let query_lower = query.to_lowercase();
        // Boşluklar ve temel noktalama işaretlerine göre bölüp temizle
        let words: Vec<&str> = query_lower.split(|c: char| c.is_whitespace() || c == '?' || c == '!' || c == '.' || c == ',')
            .map(|w| w.trim())
            .filter(|w| !w.is_empty() && w.len() > 1 && !stopwords.contains(w))
            .collect();

        let mut matched_lobes = Vec::new();
        if words.is_empty() {
            return matched_lobes;
        }

        // Mevcut tüm lob isimlerini veritabanından toplayalım
        let mut all_lobes = std::collections::HashSet::new();
        for item in db.iter() {
            if let Ok((key, _)) = item {
                if let Ok(lobe_name) = std::str::from_utf8(&key) {
                    if lobe_name != "core_language" && lobe_name != "general" && lobe_name != "__registry__" {
                        all_lobes.insert(lobe_name.to_string());
                    }
                }
            }
        }

        // Kelimeler ile karşılaştır (Component-based matching: components must start with the query word)
        for word in &words {
            let norm_word = normalize_for_match(word);
            for lobe in &all_lobes {
                let norm_lobe = normalize_for_match(lobe);
                
                // Lobe adını bileşenlerine ayır
                let components: Vec<&str> = norm_lobe.split('_').collect();
                
                // Eğer herhangi bir bileşen sorgu kelimesi ile başlıyorsa
                let is_match = components.iter().any(|comp| comp.starts_with(&norm_word));
                
                if is_match {
                    if !matched_lobes.contains(lobe) {
                        matched_lobes.push(lobe.clone());
                    }
                }
            }
        }

        matched_lobes
    }
}

/// Türkçe/İngilizce karakter esnekliğini sağlamak için normalizasyon köprüsü
pub fn normalize_for_match(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| match c {
            'ı' | 'İ' => 'i',
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
    let stopwords = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "is", "was", "were", "are", "be", "been", "this", "that", "these", "those", "it", "its", "they",
        "their", "them", "he", "she", "his", "her", "as",
        "ve", "veya", "ile", "için", "göre", "tarafından", "bir", "bu", "o", "şu", "da", "de", "ki", "en",
        "daha", "çok", "her", "ise", "gibi", "kadar", "olan", "olarak"
    ];
    text.to_lowercase()
        .split(|c: char| !c.is_alphabetic() && c != '_' && c != '$' && c != '-')
        .map(|w| w.trim())
        .filter(|w| !w.is_empty() && w.len() > 1 && !stopwords.contains(w))
        .map(|w| w.to_string())
        .collect()
}
