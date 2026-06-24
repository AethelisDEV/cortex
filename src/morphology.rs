use std::collections::HashMap;
use std::sync::RwLock;
use petgraph::visit::IntoEdgeReferences;

pub static STATS_CACHE: RwLock<Option<HashMap<String, (f32, f32)>>> = RwLock::new(None);

pub fn lowercase_tr(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            'İ' => result.push('i'),
            'I' => result.push('ı'),
            other => {
                for lower_c in other.to_lowercase() {
                    result.push(lower_c);
                }
            }
        }
    }
    result
}

pub fn update_stats_from_graph(graph: &petgraph::stable_graph::StableDiGraph<crate::cortex_graph::ConceptNode, crate::cortex_graph::Synapse>) {
    let mut cache = HashMap::new();
    let mut idx_to_info = HashMap::with_capacity(graph.node_count());
    
    for idx in graph.node_indices() {
        let node = &graph[idx];
        let norm = lowercase_tr(&node.content);
        idx_to_info.insert(idx, (node.content.clone(), norm.clone()));
        
        let entry = cache.entry(norm.clone()).or_insert((0.0, 0.0));
        entry.0 += 1.0;
        if node.content != norm {
            let entry_exact = cache.entry(node.content.clone()).or_insert((0.0, 0.0));
            entry_exact.0 += 1.0;
        }
    }
    
    use petgraph::visit::EdgeRef;
    for edge in graph.edge_references() {
        if let Some((source_content, norm_src)) = idx_to_info.get(&edge.source()) {
            let entry = cache.entry(norm_src.clone()).or_insert((0.0, 0.0));
            entry.1 += edge.weight().weight;
            if source_content != norm_src {
                let entry_exact = cache.entry(source_content.clone()).or_insert((0.0, 0.0));
                entry_exact.1 += edge.weight().weight;
            }
        }
        if let Some((target_content, norm_tgt)) = idx_to_info.get(&edge.target()) {
            let entry = cache.entry(norm_tgt.clone()).or_insert((0.0, 0.0));
            entry.1 += edge.weight().weight;
            if target_content != norm_tgt {
                let entry_exact = cache.entry(target_content.clone()).or_insert((0.0, 0.0));
                entry_exact.1 += edge.weight().weight;
            }
        }
    }
    
    if let Ok(mut write_guard) = STATS_CACHE.write() {
        *write_guard = Some(cache);
    }
}

pub fn get_word_stats(word: &str) -> (f32, f32) {
    if let Ok(read_guard) = STATS_CACHE.read() {
        if let Some(ref cache) = *read_guard {
            let norm = lowercase_tr(word);
            if let Some(&(freq, syn)) = cache.get(&norm) {
                return (freq, syn);
            }
            if let Some(&(freq, syn)) = cache.get(word) {
                return (freq, syn);
            }
        }
    }
    (0.0, 0.0)
}

pub fn is_cache_populated() -> bool {
    if let Ok(read_guard) = STATS_CACHE.read() {
        if let Some(ref cache) = *read_guard {
            return !cache.is_empty();
        }
    }
    false
}

pub fn is_stemming_valid(original: &str, stem: &str) -> bool {
    let (freq_orig, syn_orig) = get_word_stats(original);
    let (freq_stem, syn_stem) = get_word_stats(stem);
    if freq_orig > 0.0 && freq_stem == 0.0 {
        return false;
    }
    if freq_orig > 0.0 && freq_stem > 0.0 {
        if syn_orig > 2.0 * syn_stem {
            return false;
        }
    }
    true
}

pub const VERB_STEMS: &[&str] = &[
    "sağla", "et", "engelle", "önle", "kullan", "incele", "hesapla", "bulun", "çalış",
    "tanımla", "oluştur", "yönet", "yaz", "oku", "çöz", "geliştir", "destekle", "göster",
    "yap", "arttır", "azalt", "sil", "ekle", "güncelle", "başlat", "bitir", "yükle", "kaydet",
    "temizle", "duraklat", "devam", "seç", "geç", "gir", "çık", "ver", "al", "işle", "üret"
];

pub struct ConceptSplitter;

impl ConceptSplitter {
    /// Parçalanan metinlerin içindeki eylemleri, isim köklerini ayıklar ve soyut şablonu çıkarır.
    pub fn split_to_concepts(text: &str) -> (Vec<String>, String) {
        let text_trimmed = text.trim();
        
        if is_operator(text_trimmed) || is_equation(text_trimmed) {
            return (vec![text_trimmed.to_string()], "".to_string());
        }

        // Eğer girdi kod bloğu ise şablon çıkarma, düz listele
        if is_code(text_trimmed) {
            let mut concepts = Vec::new();
            for word in text_trimmed.split_whitespace() {
                let cleaned = clean_punctuation(word);
                if !cleaned.is_empty() {
                    concepts.push(cleaned);
                }
            }
            return (concepts, "".to_string());
        }

        let words: Vec<&str> = text_trimmed.split_whitespace().collect();
        let mut concepts = Vec::new();
        let mut template_parts = Vec::new();
        
        let mut has_subject = false;
        let conj_preps = ["ve", "veya", "ile", "için", "göre", "tarafından"];

        for &word in &words {
            let cleaned = clean_punctuation(word);
            if cleaned.is_empty() {
                continue;
            }

            // Bağlaç veya edat ise şablona doğrudan sabit kelime olarak ekle
            let cleaned_lower = lowercase_tr(&cleaned);
            if conj_preps.contains(&cleaned_lower.as_str()) {
                template_parts.push(cleaned_lower);
                continue;
            }

            // Eylem kontrolü
            if let Some((verb_stem, suffix_type)) = detect_verb(&cleaned) {
                let verb_repr = format!("[{}]", verb_stem);
                if !concepts.contains(&verb_repr) {
                    concepts.push(verb_repr);
                }
                
                let verb_slot = if suffix_type.is_empty() {
                    "[Eylem]".to_string()
                } else {
                    format!("[Eylem]{}", suffix_type)
                };
                template_parts.push(verb_slot);
                continue;
            }

            // İsim kontrolü ve durum eki çıkarma
            let (noun_stem, suffix_type) = parse_noun_suffix(&cleaned);
            if !concepts.contains(&noun_stem) {
                concepts.push(noun_stem.clone());
            }

            let noun_slot = if !has_subject {
                has_subject = true;
                if suffix_type.is_empty() {
                    "[Özne]".to_string()
                } else {
                    format!("[Özne]{}", suffix_type)
                }
            } else {
                if suffix_type.is_empty() {
                    "[Nesne]".to_string()
                } else {
                    format!("[Nesne]{}", suffix_type)
                }
            };
            template_parts.push(noun_slot);
        }

        let template_str = template_parts.join(" + ");
        (concepts, template_str)
    }
}

pub struct MorphologicalSynthesizer;

impl MorphologicalSynthesizer {
    /// Türkçe ünlü uyumu kurallarına göre durum eki ekler.
    /// case parametreleri: "-i", "-e", "-de", "-den", "-in", "-ler"
    pub fn add_case_suffix(word: &str, case: &str) -> String {
        if word.is_empty() {
            return word.to_string();
        }

        let is_proper = word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
            || word.chars().any(|c| c.is_numeric())
            || word.chars().any(|c| matches!(c, 'w' | 'q' | 'x' | 'W' | 'Q' | 'X'));

        let last_vowel = get_last_vowel(word).unwrap_or('e');
        let ends_in_vowel = ends_with_vowel(word);

        let suffix = match case {
            "-i" => {
                // Belirtme durumu: ı, u, i, ü (ünlü ile bitiyorsa y kaynaştırma)
                let harmonized = match last_vowel {
                    'a' | 'ı' => "ı",
                    'o' | 'u' => "u",
                    'e' | 'i' => "i",
                    'ö' | 'ü' => "ü",
                    _ => "i",
                };
                if ends_in_vowel {
                    format!("y{}", harmonized)
                } else {
                    harmonized.to_string()
                }
            }
            "-e" => {
                // Yönelme durumu: a, e (ünlü ile bitiyorsa y kaynaştırma)
                let harmonized = match last_vowel {
                    'a' | 'ı' | 'o' | 'u' => "a",
                    _ => "e",
                };
                if ends_in_vowel {
                    format!("y{}", harmonized)
                } else {
                    harmonized.to_string()
                }
            }
            "-de" => {
                // Bulunma durumu: da, de, ta, te (ünsüz sertleşmesi dahil)
                let ends_in_unvoiced = ends_with_unvoiced(word);
                let prefix = if ends_in_unvoiced { "t" } else { "d" };
                let harmonized = match last_vowel {
                    'a' | 'ı' | 'o' | 'u' => "a",
                    _ => "e",
                };
                format!("{}{}", prefix, harmonized)
            }
            "-den" => {
                // Ayrılma durumu: dan, den, tan, ten (ünsüz sertleşmesi dahil)
                let ends_in_unvoiced = ends_with_unvoiced(word);
                let prefix = if ends_in_unvoiced { "t" } else { "d" };
                let harmonized = match last_vowel {
                    'a' | 'ı' | 'o' | 'u' => "an",
                    _ => "en",
                };
                format!("{}{}", prefix, harmonized)
            }
            "-in" => {
                // Tamlayan durumu: ın, un, in, ün (ünlü ile bitiyorsa n kaynaştırma)
                let harmonized = match last_vowel {
                    'a' | 'ı' => "ın",
                    'o' | 'u' => "un",
                    'e' | 'i' => "in",
                    'ö' | 'ü' => "ün",
                    _ => "in",
                };
                if ends_in_vowel {
                    format!("n{}", harmonized)
                } else {
                    harmonized.to_string()
                }
            }
            "-ler" => {
                // Çoğul eki: lar, ler
                match last_vowel {
                    'a' | 'ı' | 'o' | 'u' => "lar".to_string(),
                    _ => "ler".to_string(),
                }
            }
            _ => "".to_string(),
        };

        if is_proper {
            format!("{}'{}", word, suffix) // Kesme işareti ile birleştir
        } else {
            // Cins isimlerde, ünlü ile başlayan eklerde ünsüz yumuşaması uygula
            let needs_voicing = case == "-i" || case == "-e" || case == "-in";
            if needs_voicing && !ends_in_vowel {
                let voiced_base = apply_voicing(word);
                format!("{}{}", voiced_base, suffix)
            } else {
                format!("{}{}", word, suffix)
            }
        }
    }

    /// Eylemi geniş zaman, gereklilik kipi veya geçmiş zaman varyasyonlarına göre çekimler.
    pub fn conjugate_verb(verb: &str, suffix_type: &str) -> String {
        let last_vowel = get_last_vowel(verb).unwrap_or('e');
        
        match suffix_type {
            "-meli" => {
                let suffix = match last_vowel {
                    'a' | 'ı' | 'o' | 'u' => "malı",
                    _ => "meli",
                };
                format!("{}{}", verb, suffix)
            }
            "-di" => {
                let last_char = verb.chars().last().unwrap_or(' ');
                let is_unvoiced = "fstkçşhpFSTKÇŞHP".contains(last_char);
                let prefix = if is_unvoiced { "t" } else { "d" };
                let vowel = match last_vowel {
                    'a' | 'ı' => "ı",
                    'o' | 'u' => "u",
                    'e' | 'i' => "i",
                    'ö' | 'ü' => "ü",
                    _ => "i",
                };
                format!("{}{}{}", verb, prefix, vowel)
            }
            "-r" | _ => {
                // Geniş zaman
                if "aeıioöuü".contains(verb.chars().last().unwrap_or(' ')) {
                    format!("{}r", verb)
                } else {
                    let exceptions = ["et", "al", "ver", "bul", "gel", "gör", "kal", "ol", "dur", "vur"];
                    let is_exception = exceptions.contains(&verb);
                    let vowels_count = verb.chars().filter(|c| "aeıioöuüAEIİOÖUÜ".contains(*c)).count();
                    
                    if vowels_count > 1 || is_exception {
                        let suffix = match last_vowel {
                            'a' | 'ı' => "ır",
                            'o' | 'u' => "ur",
                            'e' | 'i' => "ir",
                            'ö' | 'ü' => "ür",
                            _ => "ir",
                        };
                        if verb == "et" {
                            "eder".to_string()
                        } else {
                            format!("{}{}", verb, suffix)
                        }
                    } else {
                        let suffix = match last_vowel {
                            'a' | 'ı' | 'o' | 'u' => "ar",
                            _ => "er",
                        };
                        format!("{}{}", verb, suffix)
                    }
                }
            }
        }
    }

    /// Şablon, isimler ve eylemleri ağırlıklarına göre birleştirerek yeni bir cümle sentezler.
    pub fn generative_synthesis(
        subjects: &[(String, f32)],
        objects: &[(String, f32)],
        verbs: &[(String, f32)],
        template: &str,
        lang: &str,
    ) -> String {
        if template.is_empty() {
            return "".to_string();
        }

        let parts: Vec<&str> = template.split('+').map(|s| s.trim()).collect();
        let mut result_words = Vec::new();
        
        let mut subject_idx = 0;
        let mut object_idx = 0;
        let mut verb_idx = 0;

        for part in parts {
            if part.starts_with("[Özne]") {
                let suffix = if part.contains('-') {
                    part.split_once('-').map(|(_, s)| format!("-{}", s)).unwrap_or_default()
                } else {
                    "".to_string()
                };

                let base_noun = if let Some((noun, _)) = subjects.get(subject_idx) {
                    subject_idx += 1;
                    noun.clone()
                } else if let Some((noun, _)) = objects.get(object_idx) {
                    object_idx += 1;
                    noun.clone()
                } else {
                    "sistem".to_string()
                };

                let word = if suffix.is_empty() || lang == "en" {
                    base_noun
                } else {
                    Self::add_case_suffix(&base_noun, &suffix)
                };
                result_words.push(word);
            } else if part.starts_with("[Nesne]") {
                let suffix = if part.contains('-') {
                    part.split_once('-').map(|(_, s)| format!("-{}", s)).unwrap_or_default()
                } else {
                    "".to_string()
                };

                let base_noun = if let Some((noun, _)) = objects.get(object_idx) {
                    object_idx += 1;
                    noun.clone()
                } else if let Some((noun, _)) = subjects.get(subject_idx) {
                    subject_idx += 1;
                    noun.clone()
                } else {
                    "veri".to_string()
                };

                let word = if suffix.is_empty() || lang == "en" {
                    base_noun
                } else {
                    Self::add_case_suffix(&base_noun, &suffix)
                };
                result_words.push(word);
            } else if part.starts_with("[Eylem]") {
                let suffix = if part.contains('-') {
                    part.split_once('-').map(|(_, s)| format!("-{}", s)).unwrap_or_default()
                } else {
                    "-r".to_string()
                };

                let base_verb = if let Some((verb, _)) = verbs.get(verb_idx) {
                    verb_idx += 1;
                    verb.trim_matches(|c| c == '[' || c == ']').to_string()
                } else {
                    "sağla".to_string()
                };

                let word = if lang == "en" {
                    base_verb
                } else {
                    Self::conjugate_verb(&base_verb, &suffix)
                };
                result_words.push(word);
            } else {
                // Sabit kelimeler (bağlaçlar vb.)
                result_words.push(part.to_string());
            }
        }

        let mut sentence = result_words.join(" ");
        if !sentence.is_empty() {
            // İlk harfi büyüt ve sonuna nokta koy
            let mut chars = sentence.chars();
            sentence = chars.next().unwrap().to_uppercase().collect::<String>() + chars.as_str() + ".";
        }
        sentence
    }
}

// ==================== Yardımcı Fonksiyonlar ====================

fn is_operator(text: &str) -> bool {
    let t = text.trim();
    t == "+" || t == "-" || t == "*" || t == "/" || t == "="
}

fn is_equation(text: &str) -> bool {
    let t = text.trim();
    if t.contains('=') {
        return true;
    }
    let op_count = t.chars().filter(|&c| c == '+' || c == '-' || c == '*' || c == '/').count();
    op_count >= 2 && t.chars().any(|c| c.is_ascii_digit())
}

fn is_code(text: &str) -> bool {
    text.starts_with("fn ") 
        || text.starts_with("let ") 
        || text.starts_with("struct ") 
        || text.starts_with("impl ")
        || text.starts_with("use ")
        || text.starts_with("pub ")
        || text.starts_with("$ ")
        || text.starts_with("> ")
        || text.contains("println!")
        || text.contains('{') && text.contains('}') && text.contains('\n')
}

fn clean_punctuation(word: &str) -> String {
    word.trim_matches(|c: char| c.is_ascii_punctuation() && c != '\'' && c != '[' && c != ']')
        .to_string()
}

pub fn get_last_vowel(word: &str) -> Option<char> {
    let lower = lowercase_tr(word);
    if lower == "rust" {
        return Some('a'); // pronounced "rast"
    }
    if lower == "wgpu" {
        return Some('u'); // pronounced "ve-ge-pe-u"
    }

    let vowels = "aeıioöuüAEIİOÖUÜ";
    for c in word.chars().rev() {
        if vowels.contains(c) {
            return Some(match c {
                'A' | 'a' => 'a',
                'E' | 'e' => 'e',
                'I' | 'ı' => 'ı',
                'İ' | 'i' => 'i',
                'O' | 'o' => 'o',
                'Ö' | 'ö' => 'ö',
                'U' | 'u' => 'u',
                'Ü' | 'ü' => 'ü',
                _ => 'e',
            });
        }
    }
    None
}

fn ends_with_vowel(word: &str) -> bool {
    let vowels = "aeıioöuüAEIİOÖUÜ";
    word.chars().last().map(|c| vowels.contains(c)).unwrap_or(false)
}

fn ends_with_unvoiced(word: &str) -> bool {
    let unvoiced = "fstkçşhpFSTKÇŞHP";
    word.chars().last().map(|c| unvoiced.contains(c)).unwrap_or(false)
}

fn restore_voicing(stem: &str) -> String {
    if stem.is_empty() {
        return stem.to_string();
    }
    let mut chars: Vec<char> = stem.chars().collect();
    let last_idx = chars.len() - 1;
    match chars[last_idx] {
        'ğ' => chars[last_idx] = 'k',
        'd' => chars[last_idx] = 't',
        'b' => chars[last_idx] = 'p',
        'c' => chars[last_idx] = 'ç',
        _ => {}
    }
    chars.into_iter().collect()
}

fn apply_voicing(word: &str) -> String {
    if word.is_empty() { return word.to_string(); }
    let mut chars: Vec<char> = word.chars().collect();
    let last_idx = chars.len() - 1;
    match chars[last_idx] {
        'k' => chars[last_idx] = 'ğ',
        't' => chars[last_idx] = 'd',
        'p' => chars[last_idx] = 'b',
        'ç' => chars[last_idx] = 'c',
        _ => {}
    }
    chars.into_iter().collect()
}

pub fn parse_noun_suffix(word: &str) -> (String, String) {
    if let Some(pos) = word.find('\'') {
        let stem = word[..pos].to_string();
        let suffix_part = &word[pos + 1..];
        let suffix_type = match suffix_part {
            s if s.contains("da") || s.contains("de") || s.contains("ta") || s.contains("te") => "-de".to_string(),
            s if s.contains("dan") || s.contains("den") || s.contains("tan") || s.contains("ten") => "-den".to_string(),
            s if s.contains("in") || s.contains("ın") || s.contains("un") || s.contains("ün") => "-in".to_string(),
            s if s.contains("i") || s.contains("ı") || s.contains("u") || s.contains("ü") => "-i".to_string(),
            s if s.contains("e") || s.contains("a") => "-e".to_string(),
            _ => "".to_string(),
        };
        return (stem, suffix_type);
    }

    let mut current_word = word.to_string();
    let mut current_lower = lowercase_tr(word);
    let mut final_suffix = "".to_string();
    
    let suffix_rules = [
        ("-i", vec!["lerini", "larını"]),
        ("-e", vec!["lerine", "larına"]),
        ("-de", vec!["lerinde", "larında"]),
        ("-den", vec!["lerinden", "larından"]),
        
        ("-in", vec!["inin", "ının", "unun", "ünün", "nin", "nın", "nun", "nün"]),
        ("-i", vec!["ini", "ını", "unu", "ünü", "yi", "yı", "yu", "yü", "ni", "nı", "nu", "nü"]),
        ("-e", vec!["ye", "ya", "ne", "na"]),
        ("-de", vec!["nde", "nda"]),
        ("-den", vec!["nden", "ndan"]),
        ("-ler", vec!["ler", "lar"]),
        
        ("-i", vec!["i", "ı", "u", "ü"]),
        ("-e", vec!["e", "a"]),
        ("-de", vec!["de", "da", "te", "ta"]),
        ("-den", vec!["den", "dan", "ten", "tan"]),
        ("-in", vec!["in", "ın", "un", "ün"]),
    ];

    let mut stripped_any = true;
    while stripped_any {
        let (freq_curr, _) = get_word_stats(&current_lower);
        stripped_any = false;
        
        for (suffix_type, endings) in &suffix_rules {
            let mut matched = false;
            for ending in endings {
                if current_lower.ends_with(ending) {
                    let is_single_vowel = (suffix_type == &"-i" && (ending == &"i" || ending == &"ı" || ending == &"u" || ending == &"ü"))
                        || (suffix_type == &"-e" && (ending == &"e" || ending == &"a"));

                    if is_single_vowel && !is_cache_populated() {
                        continue;
                    }

                    if is_single_vowel {
                        let char_count = current_lower.chars().count();
                        if char_count >= 2 {
                            let last_two: String = current_lower.chars().skip(char_count - 2).collect();
                            if matches!(last_two.as_str(), "me" | "ma" | "ki" | "si" | "sı" | "su" | "sü" | "ci" | "cı" | "cu" | "cü" | "çi" | "çı" | "çu" | "çü" | "li" | "lı" | "lu" | "lü" | "gi" | "gı" | "gu" | "gü" | "ti" | "tı" | "tu" | "tü" | "ri" | "rı" | "ru" | "rü" | "di" | "dı" | "du" | "dü") {
                                continue;
                            }
                        }
                    }

                    let ending_char_count = ending.chars().count();
                    let word_char_count = current_word.chars().count();
                    if word_char_count >= ending_char_count + 3 {
                        let stem_raw: String = current_word.chars().take(word_char_count - ending_char_count).collect();
                        let candidate = restore_voicing(&stem_raw);
                        let candidate_lower = lowercase_tr(&candidate);
                        
                        if is_stemming_valid(&current_lower, &candidate_lower) {
                            current_word = candidate;
                            current_lower = candidate_lower;
                            if final_suffix.is_empty() {
                                final_suffix = suffix_type.to_string();
                            }
                            matched = true;
                            stripped_any = true;
                            break;
                        }
                    }
                }
            }
            if matched {
                break;
            }
        }
        if freq_curr > 0.0 {
            break;
        }
    }

    (current_word, final_suffix)
}

pub fn detect_verb(word: &str) -> Option<(String, String)> {
    let lower = lowercase_tr(word);
    for &stem in VERB_STEMS {
        let match_stem = if stem == "et" && lower.starts_with("ed") {
            "ed"
        } else if lower.starts_with(stem) {
            stem
        } else {
            continue;
        };
        
        let suffix = &lower[match_stem.len()..];
        let mut detected_suffix_type = None;
        if suffix.is_empty() {
            detected_suffix_type = Some("-r".to_string());
        } else if suffix.contains("meli") || suffix.contains("malı") {
            detected_suffix_type = Some("-meli".to_string());
        } else if suffix.contains("di") || suffix.contains("dı") || suffix.contains("du") || suffix.contains("dü") ||
                  suffix.contains("ti") || suffix.contains("tı") || suffix.contains("tu") || suffix.contains("tü") {
            detected_suffix_type = Some("-di".to_string());
        } else if suffix.contains("mek") || suffix.contains("mak") {
            detected_suffix_type = Some("-mek".to_string());
        } else if suffix.contains('r') || suffix.contains("ar") || suffix.contains("er") ||
                  suffix.contains("ır") || suffix.contains("ir") || suffix.contains("ur") || suffix.contains("ür") {
            detected_suffix_type = Some("-r".to_string());
        }

        if let Some(suf_type) = detected_suffix_type {
            let mut has_longer_db_prefix = false;
            for (byte_idx, ch) in lower.char_indices() {
                let len = byte_idx + ch.len_utf8();
                if len > match_stem.len() && len <= lower.len() {
                    let prefix = &lower[..len];
                    let (freq, _) = get_word_stats(prefix);
                    if freq > 0.0 {
                        has_longer_db_prefix = true;
                        break;
                    }
                }
            }
            if has_longer_db_prefix {
                continue;
            }

            if is_stemming_valid(&lower, match_stem) {
                return Some((stem.to_string(), suf_type));
            }
        }
    }
    None
}

// ==================== BİRİM TESTLERİ ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vowel_harmony() {
        assert_eq!(get_last_vowel("hafıza"), Some('a'));
        assert_eq!(get_last_vowel("güvenlik"), Some('i'));
    }

    #[test]
    fn test_add_case_suffix() {
        // Cins isimler + Yumuşama
        assert_eq!(MorphologicalSynthesizer::add_case_suffix("hafıza", "-e"), "hafızaya");
        assert_eq!(MorphologicalSynthesizer::add_case_suffix("güvenlik", "-i"), "güvenliği");
        assert_eq!(MorphologicalSynthesizer::add_case_suffix("sistem", "-de"), "sistemde");
        assert_eq!(MorphologicalSynthesizer::add_case_suffix("kitap", "-den"), "kitaptan");
        
        // Özel İsimler + Kesme işareti
        assert_eq!(MorphologicalSynthesizer::add_case_suffix("Rust", "-in"), "Rust'ın");
        assert_eq!(MorphologicalSynthesizer::add_case_suffix("wgpu", "-e"), "wgpu'ya");
    }

    #[test]
    fn test_conjugate_verb() {
        assert_eq!(MorphologicalSynthesizer::conjugate_verb("sağla", "-r"), "sağlar");
        assert_eq!(MorphologicalSynthesizer::conjugate_verb("et", "-r"), "eder");
        assert_eq!(MorphologicalSynthesizer::conjugate_verb("engelle", "-meli"), "engellemeli");
        assert_eq!(MorphologicalSynthesizer::conjugate_verb("kullan", "-di"), "kullandı");
        assert_eq!(MorphologicalSynthesizer::conjugate_verb("çalış", "-di"), "çalıştı");
    }

    fn seed_all() {
        let mut cache = HashMap::new();
        let words = vec![
            "hafıza", "güvenlik", "sağla", "işletim", "isletim", "sistem", "sistemi",
            "indirme", "şekildeki", "kümeleme", "mekanik", "davranış"
        ];
        for word in words {
            cache.insert(word.to_string(), (1.0, 1.0));
            cache.insert(lowercase_tr(word), (1.0, 1.0));
        }
        if let Ok(mut write_guard) = STATS_CACHE.write() {
            *write_guard = Some(cache);
        }
    }

    #[test]
    fn test_split_to_concepts() {
        seed_all();
        let text = "Rust hafıza güvenliğini sağlar.";
        let (concepts, template) = ConceptSplitter::split_to_concepts(text);
        assert!(concepts.contains(&"Rust".to_string()));
        assert!(concepts.contains(&"hafıza".to_string()));
        assert!(concepts.contains(&"güvenlik".to_string()));
        assert!(concepts.contains(&"[sağla]".to_string()));
        assert_eq!(template, "[Özne] + [Nesne] + [Nesne]-i + [Eylem]-r");
    }

    #[test]
    fn test_math_and_formula_shield() {
        seed_all();
        let eq = "E = mc^2";
        let (concepts, template) = ConceptSplitter::split_to_concepts(eq);
        assert_eq!(concepts, vec!["E = mc^2".to_string()]);
        assert_eq!(template, "");

        let op = "+";
        let (concepts, template) = ConceptSplitter::split_to_concepts(op);
        assert_eq!(concepts, vec!["+".to_string()]);
        assert_eq!(template, "");
    }

    #[test]
    fn test_generative_synthesis() {
        let subjects = vec![("Rust".to_string(), 1.0)];
        let objects = vec![("güvenlik".to_string(), 1.0)];
        let verbs = vec![("[sağla]".to_string(), 1.0)];
        let template = "[Özne] + [Nesne]-i + [Eylem]-r";
        
        let sentence = MorphologicalSynthesizer::generative_synthesis(&subjects, &objects, &verbs, template, "tr");
        assert_eq!(sentence, "Rust güvenliği sağlar.");
    }

    #[test]
    fn test_split_to_concepts_edge_cases() {
        seed_all();
        let (stem1, suffix1) = parse_noun_suffix("indirme");
        assert_eq!(stem1, "indirme");
        assert_eq!(suffix1, "");

        let (stem2, suffix2) = parse_noun_suffix("şekildeki");
        assert_eq!(stem2, "şekildeki");
        assert_eq!(suffix2, "");

        let (stem3, suffix3) = parse_noun_suffix("indirmeden");
        assert_eq!(stem3, "indirme");
        assert_eq!(suffix3, "-den");

        let (stem4, suffix4) = parse_noun_suffix("şekildekinden");
        assert_eq!(stem4, "şekildeki");
        assert_eq!(suffix4, "-den");

        let (stem5, suffix5) = parse_noun_suffix("kümeleme");
        assert_eq!(stem5, "kümeleme");
        assert_eq!(suffix5, "");

        let (stem6, suffix6) = parse_noun_suffix("mekaniği");
        assert_eq!(stem6, "mekanik");
        assert_eq!(suffix6, "-i");

        let (stem7, suffix7) = parse_noun_suffix("davranışını");
        assert_eq!(stem7, "davranış");
        assert_eq!(suffix7, "-i");

        let (stem8, suffix8) = parse_noun_suffix("Hamburg'daki");
        assert_eq!(stem8, "Hamburg");
        assert_eq!(suffix8, "-de");

        let (stem9, suffix9) = parse_noun_suffix("proton'larla");
        assert_eq!(stem9, "proton");
        assert_eq!(suffix9, "-e");
    }

    #[test]
    fn test_isletim_behavior() {
        seed_all();
        let (stem, suffix) = parse_noun_suffix("işletim");
        assert_eq!(stem, "işletim");
        assert_eq!(suffix, "");
        let (stem_norm, suffix_norm) = parse_noun_suffix("isletim");
        assert_eq!(stem_norm, "isletim");
        assert_eq!(suffix_norm, "");
        let verb = detect_verb("işletim");
        assert_eq!(verb, None);
    }

    #[test]
    fn test_unicode_slicing_safety() {
        seed_all();
        // These inputs would previously trigger char boundary panics in detect_verb
        let _ = detect_verb("işleşti");
        let _ = detect_verb("çalışı");
        let _ = detect_verb("engellediğ");
        
        // This input tests parse_noun_suffix with Turkish unicode characters
        let (_stem, suffix) = parse_noun_suffix("çalışacağı");
        assert_eq!(suffix, "-i");
    }
}

