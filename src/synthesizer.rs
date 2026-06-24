use crate::cortex_graph::{ConceptNode, Synapse, SynapseType};
use petgraph::stable_graph::{StableDiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::{HashSet, HashMap};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

#[derive(Debug, PartialEq)]
pub enum SynthesizerMode {
    Deterministic,
    Creative,
}

pub struct Synthesizer;

impl Synthesizer {
    /// Sorgu niteliğine göre deterministik kalkanı veya yaratıcı mod yürüyüşlerini kullanarak metin sentezler.
    pub fn synthesize_response(
        query: &str,
        active_nodes: Vec<ConceptNode>,
        graph: &StableDiGraph<ConceptNode, Synapse>,
    ) -> String {
        let mut active_nodes: Vec<ConceptNode> = active_nodes.into_iter()
            .filter(|n| !is_template(&n.content))
            .collect();

        if active_nodes.is_empty() {
            return "Konuyla ilgili aktifleşen bir bellek kaydı bulunamadı. Lütfen daha fazla veri besleyin.".to_string();
        }

        // En aktif 50 düğümü tut (Hem performansı hem de çıktı okunabilirliğini korumak için)
        active_nodes.sort_by(|a, b| b.activation_level.total_cmp(&a.activation_level));
        active_nodes.truncate(50);

        let lang = detect_language(&active_nodes);
        println!("[Synthesizer] Saptanan Dil: {}", lang);

        // 1. Mod Belirleme (Deterministik Kalkan / Deterministic Shield)
        let mode = detect_mode(query);
        println!("[Synthesizer] Algılanan Çalışma Modu: {:?}", mode);

        if mode == SynthesizerMode::Deterministic {
            let sentences = run_neural_pathfinding_walk(graph, &active_nodes, true, 0);
            let mut results = Vec::new();
            for sentence_path in &sentences {
                let sentence_text = weave_sentence_from_path(sentence_path, graph, lang);
                if !sentence_text.is_empty() {
                    results.push(sentence_text);
                }
            }
            let response_text = results.join(" ");
            return format!(
                "[Mod: DETERMINISTIC]\nSorgu: \"{}\"\n[Çağrışımsal Bellek Sentezi]:\n{}",
                query, response_text
            );
        }

        // 2. Yaratıcı Mod: Spekülatif Nöral Yürüyüşler ve Generative Şablon Sentezi
        println!("[Synthesizer] Yaratıcı adaylar üretiliyor...");
        let mut candidates: Vec<(String, f32)> = Vec::new();

        // A. Spekülatif Nöral Yürüyüş Adayları
        for i in 0..3 {
            let sentences = run_neural_pathfinding_walk(graph, &active_nodes, false, i);
            let mut results = Vec::new();
            for sentence_path in &sentences {
                let sentence_text = weave_sentence_from_path(sentence_path, graph, lang);
                if !sentence_text.is_empty() {
                    results.push(sentence_text);
                }
            }
            let text = results.join(" ");
            if !text.is_empty() {
                let score = evaluate_walk_from_paths(&sentences, graph);
                let decorated_text = format!(
                    "Sorgu: \"{}\"\n[Morfolojik Yaratıcı Sentez]:\n{}",
                    query, text
                );
                candidates.push((decorated_text, score));
            }
        }

        // B. Türkçe Morfolojik Şablon Sentez Adayları
        let mut active_verbs: Vec<&ConceptNode> = active_nodes.iter()
            .filter(|n| n.content.starts_with('[') && n.content.ends_with(']') && !n.activation_level.is_nan())
            .collect();

        // En yüksek uyarılmış ilk 5 eylemi seç
        active_verbs.sort_by(|a, b| b.activation_level.total_cmp(&a.activation_level));
        active_verbs.truncate(5);

        let active_nouns: Vec<&ConceptNode> = active_nodes.iter()
            .filter(|n| !n.content.starts_with('[') && !n.content.contains(" + ") && !is_code(&n.content))
            .collect();

        // Lobe templates toplama
        let mut templates = Vec::new();
        for node_idx in graph.node_indices() {
            let node = &graph[node_idx];
            if node.lobe_name == "core_language" && node.content.contains(" + ") {
                templates.push(node.content.clone());
            }
        }

        // Şablon havuzunu sınırla (konsolun kilitlenmesini engellemek için)
        if templates.len() > 10 {
            let mut rng = thread_rng();
            templates.shuffle(&mut rng);
            templates.truncate(10);
        }

        if templates.is_empty() {
            templates.push("[Özne] + [Nesne]-i + [Eylem]-r".to_string());
            templates.push("[Özne] + [Nesne]-e + [Eylem]-meli".to_string());
        }

        for verb in &active_verbs {
            let mut subjects = Vec::new();
            let mut objects = Vec::new();

            if let Some(v_idx) = graph.node_indices().find(|&idx| graph[idx].id == verb.id) {
                // Outgoing
                let mut neighbors = graph.neighbors_directed(v_idx, petgraph::Direction::Outgoing).detach();
                while let Some((edge_idx, neighbor_idx)) = neighbors.next(graph) {
                    let neighbor = &graph[neighbor_idx];
                    if !neighbor.content.starts_with('[') {
                        let edge = &graph[edge_idx];
                        if let Some(ref role) = edge.role {
                            if role == "Özne" {
                                subjects.push((neighbor.content.clone(), edge.weight));
                            } else if role == "Nesne" {
                                objects.push((neighbor.content.clone(), edge.weight));
                            }
                        }
                    }
                }

                // Incoming
                let mut neighbors = graph.neighbors_directed(v_idx, petgraph::Direction::Incoming).detach();
                while let Some((edge_idx, neighbor_idx)) = neighbors.next(graph) {
                    let neighbor = &graph[neighbor_idx];
                    if !neighbor.content.starts_with('[') {
                        let edge = &graph[edge_idx];
                        if let Some(ref role) = edge.role {
                            if role == "Özne" {
                                subjects.push((neighbor.content.clone(), edge.weight));
                            } else if role == "Nesne" {
                                objects.push((neighbor.content.clone(), edge.weight));
                            }
                        }
                    }
                }
            }

            // Fallback if no specific role connections are found
            if subjects.is_empty() {
                for noun in &active_nouns {
                    subjects.push((noun.content.clone(), noun.activation_level));
                }
            }
            if objects.is_empty() {
                for noun in &active_nouns {
                    objects.push((noun.content.clone(), noun.activation_level));
                }
            }

            subjects.retain(|x| !x.1.is_nan());
            objects.retain(|x| !x.1.is_nan());
            subjects.sort_by(|a, b| b.1.total_cmp(&a.1));
            objects.sort_by(|a, b| b.1.total_cmp(&a.1));

            for template in &templates {
                let text = crate::morphology::MorphologicalSynthesizer::generative_synthesis(
                    &subjects,
                    &objects,
                    &[(verb.content.clone(), verb.activation_level)],
                    template,
                    lang,
                );
                
                if !text.is_empty() {
                    let score = evaluate_generative_sentence(&text, graph);
                    let decorated_text = format!(
                        "Sorgu: \"{}\"\n[Morfolojik Yaratıcı Sentez]:\n{}",
                        query, text
                    );
                    candidates.push((decorated_text, score));
                }
            }
        }

        // Skorlara göre sırala
        candidates.retain(|x| !x.1.is_nan());
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

        if candidates.is_empty() {
            println!("[Synthesizer] Aday üretilemedi. Deterministik yola dönülüyor.");
            let sentences = run_neural_pathfinding_walk(graph, &active_nodes, true, 0);
            let mut results = Vec::new();
            for sentence_path in &sentences {
                let sentence_text = weave_sentence_from_path(sentence_path, graph, lang);
                if !sentence_text.is_empty() {
                    results.push(sentence_text);
                }
            }
            let response_text = results.join(" ");
            return format!(
                "[Mod: DETERMINISTIC FALLBACK]\nSorgu: \"{}\"\n[Çağrışımsal Bellek Sentezi]:\n{}",
                query, response_text
            );
        }

        let best_candidate = &candidates[0];
        let best_text = &best_candidate.0;
        let best_score = best_candidate.1;

        println!("[PrefrontalValidator] En yüksek yaratıcı aday skoru: {:.3}", best_score);

        // 3. Eşik Kuralı (Hard Threshold 0.4)
        if best_score < 0.4 {
            println!(
                "[PrefrontalValidator] UYARI: Yaratıcı sentez güvenilirlik eşiğinin ({:.1}) altında kaldı (Skor: {:.3}). Halüsinasyon reddedildi. Deterministik yola geri dönülüyor...",
                0.4, best_score
            );
            let sentences = run_neural_pathfinding_walk(graph, &active_nodes, true, 0);
            let mut results = Vec::new();
            for sentence_path in &sentences {
                let sentence_text = weave_sentence_from_path(sentence_path, graph, lang);
                if !sentence_text.is_empty() {
                    results.push(sentence_text);
                }
            }
            let response_text = results.join(" ");
            return format!(
                "[Mod: DETERMINISTIC FALLBACK (Skor {:.3} < 0.4)]\nSorgu: \"{}\"\n[Çağrışımsal Bellek Sentezi]:\n{}",
                best_score, query, response_text
            );
        }

        format!(
            "[Mod: CREATIVE (Skor: {:.3})]\n{}",
            best_score,
            best_text
        )
    }
}

/// Sorgunun deterministik mi yoksa yaratıcı mı olacağını saptayan Deterministik Kalkan
fn detect_mode(query: &str) -> SynthesizerMode {
    let q_lower = query.to_lowercase();
    
    // Güçlü kilit karakterleri ([det] veya !)
    if q_lower.starts_with("[det]") || q_lower.starts_with("!") {
        return SynthesizerMode::Deterministic;
    }

    // Matematiksel semboller
    let math_symbols = ['+', '-', '*', '/', '=', '<', '>', '%'];
    if q_lower.chars().any(|c| math_symbols.contains(&c)) {
        return SynthesizerMode::Deterministic;
    }

    // Factual/Definition sorguları için anahtar kelimeler
    let factual_keywords = vec!["nedir", "kimdir", "nelerdir", "açıkla", "ne demek", "tanımı"];
    for keyword in &factual_keywords {
        if q_lower.contains(keyword) {
            return SynthesizerMode::Deterministic;
        }
    }

    // Teknik ve deterministik kelimeler
    let tech_keywords = vec![
        "fn", "let", "struct", "impl", "cargo", "unsafe", "mut", "pub", "match", 
        "main", "loop", "crate", "math", "matrix", "equation", "algebra", "algorithm", 
        "big o", "sorting", "binary", "boolean", "logical", "compile", "error", "rendering", 
        "wgpu", "pointer", "reference", "vector", "türev", "integral", "limit",
        "başlat", "çalıştır", "yap", "run", "start", "execute", "init", "kodla", "yaz", "oluştur",
        "code", "write", "create", "rust", "projects"
    ];

    for word in tech_keywords {
        if q_lower.contains(word) {
            return SynthesizerMode::Deterministic;
        }
    }

    SynthesizerMode::Creative
}

/// Nöral İz Sürme (Neural Path-Finding) Algoritması
/// Aktif nöronlar ve onların sinaptik bağlantıları takip edilerek cümle öbekleri çıkarılır.
fn run_neural_pathfinding_walk(
    graph: &StableDiGraph<ConceptNode, Synapse>,
    active_nodes: &[ConceptNode],
    deterministic: bool,
    seed_offset: usize,
) -> Vec<Vec<NodeIndex>> {
    let mut visited = HashSet::new();
    let mut sentences = Vec::new();
    
    let mut active_list = active_nodes.to_vec();
    active_list.retain(|n| !n.activation_level.is_nan() && !is_template(&n.content));
    
    if active_list.is_empty() {
        return Vec::new();
    }
    
    // Sort active list by activation level descending
    active_list.sort_by(|a, b| b.activation_level.total_cmp(&a.activation_level));
    
    let mut id_to_index = HashMap::with_capacity(graph.node_count());
    for idx in graph.node_indices() {
        id_to_index.insert(graph[idx].id, idx);
    }
    
    let mut rng = thread_rng();
    
    while visited.len() < active_list.len() {
        let unvisited_nodes: Vec<_> = active_list.iter()
            .filter(|n| !visited.contains(&n.id))
            .collect();
            
        if unvisited_nodes.is_empty() {
            break;
        }
        
        let start_concept = if !deterministic && seed_offset > 0 {
            let limit = unvisited_nodes.len().min(3);
            let idx = seed_offset % limit;
            unvisited_nodes[idx]
        } else {
            unvisited_nodes[0]
        };
        
        let mut start_idx = match id_to_index.get(&start_concept.id) {
            Some(&idx) => idx,
            None => {
                visited.insert(start_concept.id);
                continue;
            }
        };
        
        // Sıralı tamlamaları bozmamak için zincirin en başına (köküne) kadar geri git (Back-tracing)
        let mut trace_idx = start_idx;
        let mut visited_back = HashSet::new();
        visited_back.insert(trace_idx);
        
        loop {
            let mut parent_idx = None;
            let mut highest_weight = 0.0;
            for edge_ref in graph.edges_directed(trace_idx, petgraph::Direction::Incoming) {
                let source_idx = edge_ref.source();
                let source_node = &graph[source_idx];
                if !visited.contains(&source_node.id) && !visited_back.contains(&source_idx) {
                    let edge = edge_ref.weight();
                    if edge.synapse_type == SynapseType::Sequential && edge.weight > highest_weight {
                        highest_weight = edge.weight;
                        parent_idx = Some(source_idx);
                    }
                }
            }
            if let Some(p_idx) = parent_idx {
                trace_idx = p_idx;
                visited_back.insert(trace_idx);
            } else {
                break;
            }
        }
        start_idx = trace_idx;
        
        let mut current_idx = start_idx;
        let mut sentence_path = vec![current_idx];
        visited.insert(graph[current_idx].id);
        
        let mut sentence_len = 1;
        loop {
            let mut candidates = Vec::new();
            
            for edge_ref in graph.edges(current_idx) {
                let neighbor_idx = edge_ref.target();
                let neighbor = &graph[neighbor_idx];
                
                if visited.contains(&neighbor.id) || neighbor.is_proxy || is_template(&neighbor.content) {
                    continue;
                }
                
                let edge = edge_ref.weight();
                let co_firing_factor = 1.0 + (edge.co_firings as f32).ln_1p();
                let base_weight = edge.weight * co_firing_factor;
                let act = neighbor.activation_level;
                
                let mut score = base_weight * act;
                
                // Hiç uyarılmamış olsa da sequential komşuya küçük bir şans tanı
                if edge.synapse_type == SynapseType::Sequential && act == 0.0 {
                    score = base_weight * 0.05;
                }
                
                if score > 0.0 {
                    candidates.push((neighbor_idx, score));
                }
            }
            
            if candidates.is_empty() || sentence_len >= 8 {
                break;
            }
            
            candidates.retain(|x| !x.1.is_nan());
            candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
            
            let next_idx = if !deterministic && !candidates.is_empty() {
                if rng.gen_bool(0.3) && candidates.len() > 1 {
                    let limit = candidates.len().min(3);
                    let chosen = candidates[..limit].choose(&mut rng).unwrap();
                    chosen.0
                } else {
                    candidates[0].0
                }
            } else {
                candidates[0].0
            };
            
            sentence_path.push(next_idx);
            visited.insert(graph[next_idx].id);
            current_idx = next_idx;
            sentence_len += 1;
        }
        
        sentences.push(sentence_path);
    }
    
    sentences
}

/// Weave a sentence from node indexes using morphological synthesis
fn weave_sentence_from_path(
    path: &[NodeIndex],
    graph: &StableDiGraph<ConceptNode, Synapse>,
    lang: &str,
) -> String {
    if path.is_empty() {
        return String::new();
    }
    
    let first_node = &graph[path[0]];
    let first_clean = clean_mediawiki_remnants(&first_node.content);
    
    if is_code(&first_node.content) {
        return format!("```rust\n{}\n```", first_node.content.trim());
    }
    
    let mut current_sentence = first_clean;
    let mut ends_in_verb = false;
    
    for i in 0..path.len() - 1 {
        let u_idx = path[i];
        let v_idx = path[i + 1];
        let v = &graph[v_idx];
        
        let v_clean = clean_mediawiki_remnants(&v.content);
        if is_code(&v.content) {
            current_sentence = finalize_sentence(&current_sentence, lang, ends_in_verb);
            return format!("{}\n```rust\n{}\n```", current_sentence, v.content.trim());
        }
        
        let is_v_verb = v_clean.starts_with('[') && v_clean.ends_with(']');
        
        if is_v_verb {
            let verb_stem = v_clean.trim_matches(|c| c == '[' || c == ']');
            ends_in_verb = true;
            
            let mut role = None;
            if let Some(edge_idx) = graph.find_edge(u_idx, v_idx) {
                role = graph[edge_idx].role.clone();
            } else if let Some(edge_idx) = graph.find_edge(v_idx, u_idx) {
                role = graph[edge_idx].role.clone();
            }
            
            if role.as_deref() == Some("Nesne") && lang == "tr" {
                current_sentence = crate::morphology::MorphologicalSynthesizer::add_case_suffix(&current_sentence, "-i");
                let conjugated = crate::morphology::MorphologicalSynthesizer::conjugate_verb(verb_stem, "-r");
                current_sentence = format!("{} {}", current_sentence, conjugated);
            } else if role.as_deref() == Some("Özne") && lang == "tr" {
                let conjugated = crate::morphology::MorphologicalSynthesizer::conjugate_verb(verb_stem, "-r");
                current_sentence = format!("{} {}", current_sentence, conjugated);
            } else {
                let conjugated = crate::morphology::MorphologicalSynthesizer::conjugate_verb(verb_stem, "-r");
                current_sentence = format!("{} {}", current_sentence, conjugated);
            }
        } else {
            ends_in_verb = false;
            if let Some(edge_idx) = graph.find_edge(u_idx, v_idx) {
                let edge = &graph[edge_idx];
                if edge.synapse_type == SynapseType::Sequential {
                    if edge.weight >= 0.85 {
                        current_sentence = format!("{} {}", current_sentence, v_clean);
                    } else if edge.weight >= 0.7 {
                        if lang == "tr" {
                            current_sentence = crate::morphology::MorphologicalSynthesizer::add_case_suffix(&current_sentence, "-in");
                            current_sentence = format!("{} {}", current_sentence, v_clean);
                        } else {
                            current_sentence = format!("{} {}", current_sentence, v_clean);
                        }
                    } else {
                        if lang == "tr" {
                            current_sentence = format!("{} olan {}", current_sentence, v_clean);
                        } else {
                            current_sentence = format!("{} and {}", current_sentence, v_clean);
                        }
                    }
                } else {
                    if lang == "tr" {
                        current_sentence = format!("{} ile {}", current_sentence, v_clean);
                    } else {
                        current_sentence = format!("{}, {}", current_sentence, v_clean);
                    }
                }
            } else {
                if lang == "tr" {
                    current_sentence = format!("{}, {}", current_sentence, v_clean);
                } else {
                    current_sentence = format!("{} {}", current_sentence, v_clean);
                }
            }
        }
    }
    
    finalize_sentence(&current_sentence, lang, ends_in_verb)
}

/// Helper to add copula suffix
fn add_copula_suffix(word: &str) -> String {
    if word.is_empty() {
        return word.to_string();
    }
    let last_vowel = crate::morphology::get_last_vowel(word).unwrap_or('e');
    let last_char = word.chars().last().unwrap_or(' ');
    let is_unvoiced = "fstkçşhpFSTKÇŞHP".contains(last_char);
    let prefix = if is_unvoiced { "t" } else { "d" };
    let vowel = match last_vowel {
        'a' | 'ı' => "ır",
        'e' | 'i' => "ir",
        'o' | 'u' => "ur",
        'ö' | 'ü' => "ür",
        _ => "ir",
    };
    format!("{}{}{}", word, prefix, vowel)
}

/// Apply copula if needed
fn apply_copula_if_needed(sentence: &str, lang: &str) -> String {
    if lang != "tr" || sentence.is_empty() {
        return sentence.to_string();
    }
    let words: Vec<&str> = sentence.split_whitespace().collect();
    if let Some(&last_word) = words.last() {
        let last_word_clean = last_word.trim_matches(|c: char| c.is_ascii_punctuation());
        if !last_word_clean.is_empty() 
            && !last_word_clean.ends_with("dir") && !last_word_clean.ends_with("dır") 
            && !last_word_clean.ends_with("dur") && !last_word_clean.ends_with("dür") 
            && !last_word_clean.ends_with("tir") && !last_word_clean.ends_with("tır") 
            && !last_word_clean.ends_with("tur") && !last_word_clean.ends_with("tür")
            && !last_word_clean.starts_with('[') 
            && !crate::morphology::VERB_STEMS.contains(&last_word_clean) {
            
            let copula = add_copula_suffix(last_word_clean);
            let mut words_owned: Vec<String> = words.iter().map(|w| w.to_string()).collect();
            if let Some(last) = words_owned.last_mut() {
                *last = copula;
            }
            return words_owned.join(" ");
        }
    }
    sentence.to_string()
}

/// Finalize a sentence with capitalization and period
fn finalize_sentence(sentence: &str, lang: &str, ends_in_verb: bool) -> String {
    let mut s = sentence.trim().to_string();
    if s.is_empty() {
        return s;
    }
    
    if lang == "tr" && !ends_in_verb {
        s = apply_copula_if_needed(&s, lang);
    }
    
    let mut chars = s.chars();
    let mut finalized = chars.next().unwrap().to_uppercase().collect::<String>() + chars.as_str();
    if !finalized.ends_with('.') && !finalized.ends_with('!') && !finalized.ends_with('?') {
        finalized.push('.');
    }
    finalized
}

/// Otonom MediaWiki/XML Kalıntı Temizliği
fn clean_mediawiki_remnants(text: &str) -> String {
    let mut s = text.to_string();
    
    // Wiki Linkleri Temizle: [[Dosya:resim.jpg|küçükresim|sağ|Açıklama]] -> Açıklama
    // veya [[Kategori:Fizik]] -> Kategori:Fizik (veya boşluk)
    while let Some(start) = s.find("[[") {
        if let Some(end) = s[start..].find("]]") {
            let abs_end = start + end;
            let link_content = &s[start + 2..abs_end];
            
            let replacement = if link_content.to_lowercase().starts_with("dosya:") 
                || link_content.to_lowercase().starts_with("resim:") 
                || link_content.to_lowercase().starts_with("kategori:") 
                || link_content.to_lowercase().starts_with("category:") 
                || link_content.to_lowercase().starts_with("file:") 
                || link_content.to_lowercase().starts_with("image:") {
                
                let parts: Vec<&str> = link_content.split('|').collect();
                if parts.len() > 1 {
                    let last = parts.last().unwrap().trim();
                    let last_lower = last.to_lowercase();
                    let is_layout = last_lower == "thumb" || last_lower == "thumbnail" || last_lower == "küçükresim" 
                        || last_lower == "right" || last_lower == "left" || last_lower == "center" || last_lower == "none"
                        || last_lower == "sağ" || last_lower == "sol" || last_lower == "orta" || last_lower.contains("px");
                    if is_layout { "" } else { last }
                } else {
                    ""
                }
            } else if let Some(pipe_pos) = link_content.rfind('|') {
                &link_content[pipe_pos + 1..]
            } else {
                link_content
            };
            
            s = format!("{}{}{}", &s[..start], replacement, &s[abs_end + 2..]);
        } else {
            break;
        }
    }

    // Bilgi Kutusu ve Şablonları {{...}} Temizle
    while let Some(start) = s.find("{{") {
        if let Some(end) = s[start..].find("}}") {
            let abs_end = start + end;
            s = format!("{}{}{}", &s[..start], "", &s[abs_end + 2..]);
        } else {
            break;
        }
    }
    
    s = s.replace("'''", "").replace("''", "");
    s = s.replace(")|", "").replace("(|", "");
    s = s.replace('\n', " ").replace('\r', "");
    
    while s.contains("  ") {
        s = s.replace("  ", " ");
    }
    
    let chars_to_trim: &[char] = &[' ', '\t', '\n', '\r', '|', '*', '(', ')', '[', ']', '{', '}'];
    s.trim_matches(chars_to_trim).to_string()
}

/// Flat path connection evaluation
fn evaluate_walk_from_paths(
    sentences: &[Vec<NodeIndex>],
    graph: &StableDiGraph<ConceptNode, Synapse>,
) -> f32 {
    let mut total_score = 0.5f32;
    let mut total_checks = 0;
    
    for path in sentences {
        if path.len() < 2 {
            continue;
        }
        for i in 0..path.len() - 1 {
            let u = path[i];
            let v = path[i + 1];
            total_checks += 1;
            
            if let Some(edge_idx) = graph.find_edge(u, v) {
                let edge = &graph[edge_idx];
                match edge.synapse_type {
                    SynapseType::Sequential => total_score += edge.weight * 0.2,
                    SynapseType::Semantic => total_score += edge.weight * 0.15,
                }
            } else if let Some(edge_idx) = graph.find_edge(v, u) {
                let edge = &graph[edge_idx];
                match edge.synapse_type {
                    SynapseType::Sequential => total_score += edge.weight * 0.15,
                    SynapseType::Semantic => total_score += edge.weight * 0.1,
                }
            } else {
                total_score -= 0.2;
            }
        }
    }
    
    if total_checks > 0 {
        total_score.clamp(-1.0, 1.0)
    } else {
        0.5
    }
}

fn evaluate_generative_sentence(
    sentence: &str,
    graph: &StableDiGraph<ConceptNode, Synapse>,
) -> f32 {
    let (concepts, _) = crate::morphology::ConceptSplitter::split_to_concepts(sentence);
    if concepts.len() < 2 {
        return 0.5;
    }

    let mut content_to_index = HashMap::with_capacity(graph.node_count());
    for idx in graph.node_indices() {
        content_to_index.insert(&graph[idx].content, idx);
    }
    
    let mut total_score = 0.5f32;
    let mut total_checks = 0;
    
    for i in 0..concepts.len() {
        for j in i+1..concepts.len() {
            let c1 = &concepts[i];
            let c2 = &concepts[j];
            
            let idx1 = content_to_index.get(c1).cloned();
            let idx2 = content_to_index.get(c2).cloned();
            
            total_checks += 1;
            if let (Some(u), Some(v)) = (idx1, idx2) {
                let pair_score = if let Some(edge) = graph.find_edge(u, v) {
                    match graph[edge].synapse_type {
                        SynapseType::Sequential => graph[edge].weight * 0.2,
                        SynapseType::Semantic => graph[edge].weight * 0.15,
                    }
                } else if let Some(edge) = graph.find_edge(v, u) {
                    match graph[edge].synapse_type {
                        SynapseType::Sequential => graph[edge].weight * 0.15,
                        SynapseType::Semantic => graph[edge].weight * 0.1,
                    }
                } else if has_path_within_hops(graph, u, v, 3) {
                    0.05
                } else {
                    let overlap = calculate_tag_overlap(&graph[u].tags, &graph[v].tags);
                    if overlap > 0.1 {
                        overlap * 0.05
                    } else {
                        -0.2
                    }
                };
                total_score += pair_score;
            } else {
                total_score -= 0.1;
            }
        }
    }
    
    if total_checks > 0 {
        total_score.clamp(-1.0, 1.0)
    } else {
        0.5
    }
}

/// Check if path exists using BFS
fn has_path_within_hops(
    graph: &StableDiGraph<ConceptNode, Synapse>,
    start_idx: NodeIndex,
    end_idx: NodeIndex,
    max_hops: usize,
) -> bool {
    let mut queue = vec![(start_idx, 0)];
    let mut visited = HashSet::new();
    visited.insert(start_idx);
    
    while let Some((curr, hops)) = queue.pop() {
        if curr == end_idx {
            return true;
        }
        if hops < max_hops {
            for neighbor in graph.neighbors(curr) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push((neighbor, hops + 1));
                }
            }
        }
    }
    false
}

fn calculate_tag_overlap(
    tags_a: &HashMap<String, f32>,
    tags_b: &HashMap<String, f32>,
) -> f32 {
    let mut overlap = 0.0f32;
    for (tag, &weight_a) in tags_a {
        if let Some(&weight_b) = tags_b.get(tag) {
            overlap += weight_a.min(weight_b);
        }
    }
    overlap
}

fn is_template(text: &str) -> bool {
    text.contains(" + ") 
        || text.contains("[Özne]") 
        || text.contains("[Nesne]") 
        || text.contains("[Eylem]")
}

fn detect_language(nodes: &[ConceptNode]) -> &'static str {
    let mut en_count = 0;
    let mut tr_count = 0;

    let en_words = [
        "the", "of", "and", "to", "in", "is", "you", "that", "it", "he", "was", "for", 
        "on", "are", "as", "with", "his", "they", "at", "be", "this", "have", "from", 
        "not", "by", "or", "but", "an", "this", "your", "which", "their", "will"
    ];
    let tr_words = [
        "ve", "veya", "ile", "için", "göre", "tarafından", "bir", "bu", "o", "da", "de", 
        "ki", "ama", "fakat", "lakin", "ise", "en", "daha", "her", "gibi", "yok", "var",
        "ile", "kendi", "biz", "siz", "onlar", "bunu", "buna", "bunda", "bundan"
    ];

    for node in nodes {
        let content_lower = node.content.to_lowercase();
        for word in content_lower.split_whitespace() {
            let cleaned = word.trim_matches(|c: char| c.is_ascii_punctuation());
            if en_words.contains(&cleaned) {
                en_count += 1;
            }
            if tr_words.contains(&cleaned) {
                tr_count += 1;
            }
        }
    }

    if en_count > tr_count {
        "en"
    } else {
        "tr"
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cortex_graph::{ConceptNode, Synapse, SynapseType};
    use petgraph::stable_graph::StableDiGraph;
    use std::collections::HashMap;

    #[test]
    fn test_neural_pathfinding_walk() {
        let mut graph = StableDiGraph::new();
        
        let n1 = ConceptNode {
            id: 0,
            content: "parçacık".to_string(),
            tags: HashMap::new(),
            activation_level: 0.8,
            lobe_name: "test".to_string(),
            is_proxy: false,
            proxy_target: None,
            node_type: None,
        };
        let n2 = ConceptNode {
            id: 1,
            content: "hızlandırıcı".to_string(),
            tags: HashMap::new(),
            activation_level: 0.9,
            lobe_name: "test".to_string(),
            is_proxy: false,
            proxy_target: None,
            node_type: None,
        };
        
        let u1 = graph.add_node(n1.clone());
        let u2 = graph.add_node(n2.clone());
        
        // Add sequential edge n1 -> n2
        graph.add_edge(u1, u2, Synapse {
            weight: 0.8,
            co_firings: 0,
            synapse_type: SynapseType::Sequential,
            role: None,
        });
        
        let active_nodes = vec![n2.clone(), n1.clone()];
        let sentences = run_neural_pathfinding_walk(&graph, &active_nodes, true, 0);
        
        assert!(!sentences.is_empty());
        let first_sentence = &sentences[0];
        assert_eq!(first_sentence.len(), 2);
        assert_eq!(graph[first_sentence[0]].id, 0); // starts with n1 because it is connected sequentially to n2
        assert_eq!(graph[first_sentence[1]].id, 1);
    }
}
