use crate::cortex_graph::{ConceptNode, Synapse, SynapseType};
use petgraph::stable_graph::{StableDiGraph, NodeIndex};
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
        let active_nodes: Vec<ConceptNode> = active_nodes.into_iter()
            .filter(|n| !is_template(&n.content))
            .collect();

        if active_nodes.is_empty() {
            return "Konuyla ilgili aktifleşen bir bellek kaydı bulunamadı. Lütfen daha fazla veri besleyin.".to_string();
        }

        let lang = detect_language(&active_nodes);
        println!("[Synthesizer] Saptanan Dil: {}", lang);

        // 1. Mod Belirleme (Deterministik Kalkan / Deterministic Shield)
        let mode = detect_mode(query);
        println!("[Synthesizer] Algılanan Çalışma Modu: {:?}", mode);

        if mode == SynthesizerMode::Deterministic {
            let walk = run_deterministic_walk(active_nodes, graph);
            return format!(
                "[Mod: DETERMINISTIC]\n{}",
                build_text_from_walk(query, &walk, graph, true, lang)
            );
        }

        // 2. Yaratıcı Mod: Spekülatif Nöral Yürüyüşler ve Generative Şablon Sentezi
        println!("[Synthesizer] Yaratıcı adaylar üretiliyor...");
        let mut candidates: Vec<(String, f32)> = Vec::new();

        // A. Spekülatif Nöral Yürüyüş Adayları
        for i in 0..3 {
            if let Some(walk) = run_speculative_walk(&active_nodes, graph, i) {
                let score = evaluate_walk(&walk, graph);
                let text = build_text_from_walk(query, &walk, graph, false, lang);
                candidates.push((text, score));
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
            let walk = run_deterministic_walk(active_nodes, graph);
            return format!("[Mod: DETERMINISTIC FALLBACK]\n{}", build_text_from_walk(query, &walk, graph, true, lang));
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
            let walk = run_deterministic_walk(active_nodes, graph);
            return format!(
                "[Mod: DETERMINISTIC FALLBACK (Skor {:.3} < 0.4)]\n{}",
                best_score,
                build_text_from_walk(query, &walk, graph, true, lang)
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

/// Tamamen sıralı ve uyarımı en yüksek düğümleri takip eden deterministik sıralama
fn run_deterministic_walk(
    mut active_nodes: Vec<ConceptNode>,
    graph: &StableDiGraph<ConceptNode, Synapse>,
) -> Vec<ConceptNode> {
    active_nodes.retain(|n| !n.activation_level.is_nan());
    
    let mut id_to_index = HashMap::with_capacity(graph.node_count());
    for idx in graph.node_indices() {
        id_to_index.insert(graph[idx].id, idx);
    }

    active_nodes.sort_by(|a, b| {
        let index_a = id_to_index.get(&a.id).cloned();
        let index_b = id_to_index.get(&b.id).cloned();
        
        if let (Some(idx_a), Some(idx_b)) = (index_a, index_b) {
            if let Some(edge) = graph.find_edge(idx_a, idx_b) {
                if graph[edge].synapse_type == SynapseType::Sequential {
                    return std::cmp::Ordering::Less;
                }
            }
            if let Some(edge) = graph.find_edge(idx_b, idx_a) {
                if graph[edge].synapse_type == SynapseType::Sequential {
                    return std::cmp::Ordering::Greater;
                }
            }
        }
        b.activation_level.total_cmp(&a.activation_level)
    });
    active_nodes
}

/// Dolaylı bağlar ve etiket benzerliğiyle yaratıcı/spekülatif yürüyüş yapar.
fn run_speculative_walk(
    active_nodes: &[ConceptNode],
    graph: &StableDiGraph<ConceptNode, Synapse>,
    seed_offset: usize,
) -> Option<Vec<ConceptNode>> {
    let mut active_nodes_filtered = active_nodes.to_vec();
    active_nodes_filtered.retain(|n| !n.activation_level.is_nan());

    if active_nodes_filtered.is_empty() {
        return None;
    }

    let mut id_to_index = HashMap::with_capacity(graph.node_count());
    for idx in graph.node_indices() {
        id_to_index.insert(graph[idx].id, idx);
    }

    let mut rng = thread_rng();
    let mut walk = Vec::new();
    let mut visited = HashSet::new();

    // Seed düğümü seç (En uyarık 3 düğümden biri)
    let mut sorted_active = active_nodes_filtered.clone();
    sorted_active.sort_by(|a, b| b.activation_level.total_cmp(&a.activation_level));
    
    let seed_index = seed_offset % sorted_active.len().min(3);
    let mut current = sorted_active[seed_index].clone();
    
    walk.push(current.clone());
    visited.insert(current.id);

    // Yürüyüş adımları
    for _ in 0..5 {
        let current_idx = *id_to_index.get(&current.id)?;

        // Aday seçimi (Doğrudan komşular + 2/3 adım dolaylılar + benzer etiketliler)
        let mut candidates = Vec::new();

        for candidate_node in &active_nodes_filtered {
            if visited.contains(&candidate_node.id) {
                continue;
            }

            let cand_idx = *id_to_index.get(&candidate_node.id)?;

            let is_direct = graph.find_edge(current_idx, cand_idx).is_some() 
                || graph.find_edge(cand_idx, current_idx).is_some();
            
            let is_indirect = !is_direct && has_path_within_hops(graph, current_idx, cand_idx, 3);

            let tag_overlap = calculate_tag_overlap(&current.tags, &candidate_node.tags);

            // Adaya skor ata
            let mut score = 0.0f32;
            if is_direct {
                score += 1.0;
            } else if is_indirect {
                score += 0.5;
            }
            score += tag_overlap * 0.4;

            if score > 0.1 {
                candidates.push((candidate_node.clone(), score));
            }
        }

        if candidates.is_empty() {
            break;
        }

        // Adaylar arasından skora göre ağırlıklı rastgele veya en iyisini seç
        candidates.retain(|x| !x.1.is_nan());
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        
        // Rastgele spekülatif sapma olasılığı (%40)
        let next_node = if rng.gen_bool(0.4) && candidates.len() > 1 {
            candidates.choose(&mut rng).unwrap().0.clone()
        } else {
            candidates[0].0.clone()
        };

        walk.push(next_node.clone());
        visited.insert(next_node.id);
        current = next_node;
    }

    Some(walk)
}

/// Metin/kod tutarlılığını Hebbian bağlarına ve Rust sentaks kurallarına göre değerlendiren Prefrontal Validator
fn evaluate_walk(
    walk: &[ConceptNode],
    graph: &StableDiGraph<ConceptNode, Synapse>,
) -> f32 {
    if walk.len() < 2 {
        return 1.0;
    }

    let mut id_to_index = HashMap::with_capacity(graph.node_count());
    for idx in graph.node_indices() {
        id_to_index.insert(graph[idx].id, idx);
    }

    let mut score = 0.5f32;
    let mut total_checks = 0;

    for i in 0..walk.len() - 1 {
        let u = &walk[i];
        let v = &walk[i + 1];

        let u_idx = match id_to_index.get(&u.id).cloned() {
            Some(idx) => idx,
            None => continue,
        };
        let v_idx = match id_to_index.get(&v.id).cloned() {
            Some(idx) => idx,
            None => continue,
        };

        total_checks += 1;

        // Doğrudan sinaps var mı?
        if let Some(edge_idx) = graph.find_edge(u_idx, v_idx) {
            let edge = &graph[edge_idx];
            match edge.synapse_type {
                SynapseType::Sequential => score += edge.weight * 0.2,
                SynapseType::Semantic => score += edge.weight * 0.15,
            }
        } else if let Some(edge_idx) = graph.find_edge(v_idx, u_idx) {
            let edge = &graph[edge_idx];
            match edge.synapse_type {
                SynapseType::Sequential => score += edge.weight * 0.15,
                SynapseType::Semantic => score += edge.weight * 0.1,
            }
        } else {
            // Dolaylı yol var mı?
            if has_path_within_hops(graph, u_idx, v_idx, 3) {
                score += 0.05;
            } else {
                // Sadece etiket benzerliği mi var?
                let overlap = calculate_tag_overlap(&u.tags, &v.tags);
                if overlap > 0.1 {
                    score += overlap * 0.05;
                } else {
                    // Tamamen kopukluk cezası
                    score -= 0.2;
                }
            }
        }
    }

    // Sentaks Güvencesi (Brackets/Syntax Assurance)
    // Kod nöronlarının birleştirilmiş hallerinde açılan süslü parantez, parantezlerin dengesini test et
    let mut code_accumulator = String::new();
    for node in walk {
        let content_trimmed = node.content.trim();
        if is_code(content_trimmed) {
            code_accumulator.push_str(content_trimmed);
            code_accumulator.push('\n');
        }
    }

    if !code_accumulator.is_empty() && !validate_code_syntax(&code_accumulator) {
        println!("   -> [Validator] Sentaks hatası saptandı! Ağır ceza veriliyor (-1.0).");
        return -1.0;
    }

    if total_checks > 0 {
        score = score.clamp(-1.0, 1.0);
    }
    score
}

/// Basit BFS ile düğümler arası yol kontrolü
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

fn validate_code_syntax(text: &str) -> bool {
    let mut parens = 0i32;
    let mut braces = 0i32;
    let mut brackets = 0i32;
    
    for c in text.chars() {
        match c {
            '(' => parens += 1,
            ')' => { parens -= 1; if parens < 0 { return false; } }
            '{' => braces += 1,
            '}' => { braces -= 1; if braces < 0 { return false; } }
            '[' => brackets += 1,
            ']' => { brackets -= 1; if brackets < 0 { return false; } }
            _ => {}
        }
    }
    
    if parens != 0 || braces != 0 || brackets != 0 {
        return false;
    }
    
    if text.contains("fn ") && !text.contains('{') {
        return false;
    }
    
    true
}

fn build_text_from_walk(
    query: &str,
    walk: &[ConceptNode],
    graph: &StableDiGraph<ConceptNode, Synapse>,
    deterministic: bool,
    lang: &str,
) -> String {
    let mut synthesized_text = String::new();
    synthesized_text.push_str(&format!("Sorgu: \"{}\"\n[Çağrışımsal Bellek Sentezi]:\n", query));

    let mut prev_node: Option<&ConceptNode> = None;
    let mut visited_ids = HashSet::new();

    for node in walk {
        if visited_ids.contains(&node.id) {
            continue;
        }
        visited_ids.insert(node.id);

        let content_trimmed = node.content.trim();
        let current_is_code = is_code(content_trimmed);

        if synthesized_text.lines().count() > 2 {
            if deterministic {
                synthesized_text.push_str("\n");
            } else if current_is_code {
                synthesized_text.push_str("\n\n");
            } else if let Some(prev) = prev_node {
                let prev_is_code = is_code(&prev.content);
                if prev_is_code {
                    synthesized_text.push_str("\n\n");
                } else {
                    let prev_idx = graph.node_indices().find(|&idx| graph[idx].id == prev.id);
                    let curr_idx = graph.node_indices().find(|&idx| graph[idx].id == node.id);
                    
                    let mut connection_type = None;
                    if let (Some(p_idx), Some(c_idx)) = (prev_idx, curr_idx) {
                        if let Some(edge) = graph.find_edge(p_idx, c_idx) {
                            connection_type = Some(graph[edge].synapse_type.clone());
                        } else if let Some(edge) = graph.find_edge(c_idx, p_idx) {
                            connection_type = Some(graph[edge].synapse_type.clone());
                        }
                    }

                    if lang == "en" {
                        match connection_type {
                            Some(SynapseType::Sequential) => {
                                synthesized_text.push_str(" then, ");
                            }
                            Some(SynapseType::Semantic) => {
                                synthesized_text.push_str(". In relation to this, ");
                            }
                            None => {
                                synthesized_text.push_str(". Additionally, ");
                            }
                        }
                    } else {
                        match connection_type {
                            Some(SynapseType::Sequential) => {
                                synthesized_text.push_str("... ardından ");
                            }
                            Some(SynapseType::Semantic) => {
                                synthesized_text.push_str(". Bununla ilişkili olarak, ");
                            }
                            None => {
                                synthesized_text.push_str(". Ayrıca, ");
                            }
                        }
                    }
                }
            }
        }

        if current_is_code {
            synthesized_text.push_str("```rust\n");
            synthesized_text.push_str(content_trimmed);
            synthesized_text.push_str("\n```");
        } else {
            let mut chars = content_trimmed.chars();
            let ends_with_then = synthesized_text.ends_with(" then, ") || synthesized_text.ends_with(" ardından ");
            if ends_with_then {
                if let Some(first) = chars.next() {
                    synthesized_text.push_str(&first.to_lowercase().to_string());
                }
            } else {
                if let Some(first) = chars.next() {
                    synthesized_text.push_str(&first.to_uppercase().to_string());
                }
            }
            synthesized_text.push_str(chars.as_str());
        }

        prev_node = Some(node);
    }

    synthesized_text
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
