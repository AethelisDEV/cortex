use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use bzip2::read::BzDecoder;
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use rayon::prelude::*;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};

use crate::cortex_graph::{CortexGraph, SerializedLobe, SerializedEdge, SynapseType};
use crate::morphology::ConceptSplitter;
use crate::thalamus_router::clean_text_to_words;
use crate::ingestion::{SentenceChunker, TextChunker};

pub struct WikiPage {
    pub title: String,
    pub text: String,
}

/// Wikipedia XML veya sıkıştırılmış XML.bz2 dump dosyasını okuyup nöral loblara dönüştürür.
pub fn parse_and_ingest_dump(file_path: &Path, db: &sled::Db) -> anyhow::Result<(usize, usize)> {
    let file = File::open(file_path)?;
    let decoder: Box<dyn Read + Send> = if file_path.extension().map_or(false, |ext| ext.to_string_lossy().to_lowercase() == "bz2") {
        Box::new(BzDecoder::new(file))
    } else {
        Box::new(file)
    };

    let mut reader = Reader::from_reader(BufReader::new(decoder));
    reader.config_mut().trim_text(true);

    let mut buf = Vec::new();
    let mut current_title = String::new();
    let mut current_text = String::new();
    let mut in_title = false;
    let mut in_text = false;

    let mut batch = Vec::with_capacity(1000);
    let mut processed_count = 0;
    let mut skipped_count = 0;

    println!("[WikiParser] Wikipedia Dump dosyası okunmaya başlanıyor...");

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                match e.name().as_ref() {
                    b"title" => in_title = true,
                    b"text" => in_text = true,
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                match e.name().as_ref() {
                    b"title" => in_title = false,
                    b"text" => in_text = false,
                    b"page" => {
                        let title = std::mem::take(&mut current_title);
                        let text = std::mem::take(&mut current_text);
                        
                        // Yönlendirme Kontrolü (Redirect check on raw text)
                        let text_upper = text.to_uppercase();
                        if text_upper.contains("#REDIRECT") || text_upper.contains("#YÖNLENDİRME") {
                            skipped_count += 1;
                        } else {
                            batch.push(WikiPage { title, text });
                            if batch.len() >= 1000 {
                                let (p, s) = process_batch(&batch, db)?;
                                processed_count += p;
                                skipped_count += s;
                                batch.clear();
                                println!("[WikiParser] İlerleme: {} makale işlendi, {} makale atlandı.", processed_count, skipped_count);
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                if in_title {
                    if let Ok(unescaped) = e.unescape() {
                        current_title.push_str(&unescaped);
                    }
                } else if in_text {
                    if let Ok(unescaped) = e.unescape() {
                        current_text.push_str(&unescaped);
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(anyhow::anyhow!("XML Ayrıştırma Hatası: {:?}", e));
            }
            _ => {}
        }
        buf.clear();
    }

    // Kalan son paketi işleme al
    if !batch.is_empty() {
        let (p, s) = process_batch(&batch, db)?;
        processed_count += p;
        skipped_count += s;
    }

    println!("[WikiParser] Tamamlandı! Toplam işlenen makale: {}, Atlanan: {}", processed_count, skipped_count);
    Ok((processed_count, skipped_count))
}

/// Belirtilen paketi paralel olarak işler ve sırayla diske yazar.
fn process_batch(batch: &[WikiPage], db: &sled::Db) -> anyhow::Result<(usize, usize)> {
    // 1. Paralel Ingestion (CPU Yoğun Morfolojik Ayrıştırma)
    let results: Vec<(String, Option<SerializedLobe>)> = batch.par_iter().map(|page| {
        let lobe_name = derive_lobe_name_from_title(&page.title);
        if lobe_name.is_empty() {
            return (lobe_name, None);
        }

        // MediaWiki Temizleyici
        let cleaned_text = clean_mediawiki_syntax(&page.text);

        // Taslak Filtresi (100 karakterden kısa ise atla)
        if cleaned_text.len() < 100 {
            return (lobe_name, None);
        }

        // Bellek içi graf oluştur ve ingestion'ı gerçekleştir
        match ingest_single_page_in_memory(db.clone(), &lobe_name, &cleaned_text) {
            Ok(serialized_lobe) => (lobe_name, Some(serialized_lobe)),
            Err(e) => {
                eprintln!("[Hata] Makale ingeste edilemedi ({}): {:?}", page.title, e);
                (lobe_name, None)
            }
        }
    }).collect();

    let mut processed = 0;
    let mut skipped = 0;

    // 2. Toplu Disk Yazma Optimizasyonu (sled::Batch)
    let mut batch_write = sled::Batch::default();
    for (lobe_name, serialized_opt) in results {
        if let Some(serialized) = serialized_opt {
            match serde_json::to_vec(&serialized) {
                Ok(bytes) => {
                    batch_write.insert(lobe_name.as_bytes(), bytes);
                    processed += 1;
                }
                Err(e) => {
                    eprintln!("[Hata] Lobe serileştirilemedi ({}): {:?}", lobe_name, e);
                    skipped += 1;
                }
            }
        } else {
            skipped += 1;
        }
    }

    db.apply_batch(batch_write)?;
    db.flush()?;

    Ok((processed, skipped))
}

/// Tek bir makaleyi bellek içinde işleyerek SerializedLobe yapısına çevirir.
fn ingest_single_page_in_memory(db: sled::Db, lobe_name: &str, text: &str) -> anyhow::Result<SerializedLobe> {
    let mut cortex = CortexGraph::new(db);
    let chunker = SentenceChunker::default();
    let chunks = chunker.chunk(text);

    let mut prev_node_id: Option<usize> = None;

    for chunk in &chunks {
        let existing_id = cortex.content_to_id.get(chunk).cloned();
        let id = match existing_id {
            Some(id_val) => id_val,
            None => {
                let words = clean_text_to_words(chunk);
                let mut tags = HashMap::new();
                for w in words {
                    *tags.entry(w).or_insert(0.0) += 1.0;
                }
                cortex.add_node(chunk, tags, lobe_name)
            }
        };

        if let Some(prev_id) = prev_node_id {
            cortex.add_synapse(prev_id, id, 0.85, SynapseType::Sequential);
        }
        prev_node_id = Some(id);

        // Türkçe Morfoloji ve Şablon Çıkarma
        let (concepts, _template) = ConceptSplitter::split_to_concepts(chunk);

        let mut concept_ids = Vec::new();
        for concept in &concepts {
            let existing_concept_id = cortex.content_to_id.get(concept).cloned();
            let concept_id = match existing_concept_id {
                Some(cid) => cid,
                None => {
                    let mut tags = HashMap::new();
                    let cleaned_concept = concept.trim_matches(|c| c == '[' || c == ']');
                    tags.insert(cleaned_concept.to_lowercase(), 1.0);
                    cortex.add_node(concept, tags, lobe_name)
                }
            };
            concept_ids.push(concept_id);
        }

        // Sıralı bağlantı (Sequential)
        let mut prev_concept_id: Option<usize> = None;
        for &cid in &concept_ids {
            if let Some(prev_cid) = prev_concept_id {
                cortex.add_synapse(prev_cid, cid, 0.85, SynapseType::Sequential);
            }
            prev_concept_id = Some(cid);
        }

        // Rol Etiketli Sinapslar
        if let Some((verb_idx, &verb_id)) = concept_ids.iter().enumerate().find(|(idx, _)| concepts[*idx].starts_with('[') && concepts[*idx].ends_with(']')) {
            let mut found_first_noun = false;
            for (idx, &cid) in concept_ids.iter().enumerate() {
                if idx == verb_idx { continue; }
                let is_verb = concepts[idx].starts_with('[') && concepts[idx].ends_with(']');
                if !is_verb {
                    let role = if !found_first_noun {
                        found_first_noun = true;
                        "Özne".to_string()
                    } else {
                        "Nesne".to_string()
                    };
                    cortex.add_synapse_with_role(cid, verb_id, 0.8, SynapseType::Semantic, Some(role));
                }
            }
        }
    }

    Ok(extract_serialized_lobe(&cortex, lobe_name))
}

/// CortexGraph grafiğinden SerializedLobe nesnesini çıkarır.
fn extract_serialized_lobe(cortex: &CortexGraph, lobe_name: &str) -> SerializedLobe {
    let mut nodes_to_save = Vec::new();
    for node_idx in cortex.graph.node_indices() {
        let node = &cortex.graph[node_idx];
        if node.lobe_name == lobe_name && !node.is_proxy {
            nodes_to_save.push(node.clone());
        }
    }

    let mut edges_to_save = Vec::new();
    for edge in cortex.graph.edge_references() {
        let u_node = &cortex.graph[edge.source()];
        let v_node = &cortex.graph[edge.target()];
        
        if u_node.lobe_name == lobe_name && !u_node.is_proxy {
            let target_lobe = if v_node.is_proxy {
                v_node.proxy_target.as_ref().map(|t| t.0.clone()).unwrap_or_else(|| v_node.lobe_name.clone())
            } else {
                v_node.lobe_name.clone()
            };

            edges_to_save.push(SerializedEdge {
                source_id: u_node.id,
                target_id: v_node.id,
                target_lobe,
                synapse: edge.weight().clone(),
            });
        }
    }

    SerializedLobe {
        lobe_name: lobe_name.to_string(),
        nodes: nodes_to_save,
        edges: edges_to_save,
    }
}

/// Başlıktan temizlenmiş ve normalize edilmiş dosya adı formatında lob adı türetir.
pub fn derive_lobe_name_from_title(title: &str) -> String {
    let mut name = title.to_lowercase();
    // Türkçe karakter normalizasyonu (ı -> i, ğ -> g, ü -> u, ş -> s, ö -> o, ç -> c)
    name = name.chars()
        .map(|c| match c {
            'ı' | 'İ' => 'i',
            'ğ' | 'Ğ' => 'g',
            'ü' | 'Ü' => 'u',
            'ş' | 'Ş' => 's',
            'ö' | 'Ö' => 'o',
            'ç' | 'Ç' => 'c',
            _ => c,
        })
        .collect();

    // Boşlukları ve tireleri alt tireye çevir
    name = name.replace(' ', "_");
    name = name.replace('-', "_");
    
    name = name.chars()
        .filter(|&c| c.is_alphanumeric() || c == '_')
        .collect();
    
    let mut clean = String::new();
    let mut last_was_under = false;
    for c in name.chars() {
        if c == '_' {
            if !last_was_under {
                clean.push(c);
                last_was_under = true;
            }
        } else {
            clean.push(c);
            last_was_under = false;
        }
    }
    let result = clean.trim_matches('_').to_string();
    if result.is_empty() {
        "general".to_string()
    } else {
        result
    }
}

/// MediaWiki biçimlendirmelerini, bilgi kutularını, bağlantıları ve HTML'leri temizleyen fonksiyon.
pub fn clean_mediawiki_syntax(text: &str) -> String {
    // 1. HTML Yorumlarını temizle
    let mut text = remove_pattern(text, "<!--", "-->");
    
    // 2. <ref>...</ref> etiketlerini temizle
    text = remove_pattern(&text, "<ref", "</ref>");
    text = remove_single_tags(&text, "<ref", "/>");

    // 3. HTML etiketlerini temizle
    text = remove_html_tags(&text);

    // 4. MediaWiki şablonlarını {{...}} temizle
    text = remove_nested_braces(&text, '{', '}');

    // 5. Wiki tablolarını {|...|} temizle
    text = remove_nested_tables(&text);

    // 6. Wiki linklerini [[...]] temizle (görünen metni koru)
    text = clean_wiki_links(&text);

    // 7. Başlık çizgilerini ==...== temizle
    text = clean_headers(&text);

    // 8. Fazla boşlukları normalize et
    normalize_whitespace(&text)
}

fn remove_pattern(text: &str, start_token: &str, end_token: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start_idx) = remaining.find(start_token) {
        result.push_str(&remaining[..start_idx]);
        let search_from = start_idx + start_token.len();
        if let Some(end_idx) = remaining[search_from..].find(end_token) {
            remaining = &remaining[search_from + end_idx + end_token.len()..];
        } else {
            remaining = "";
            break;
        }
    }
    result.push_str(remaining);
    result
}

fn remove_single_tags(text: &str, start_token: &str, end_token: &str) -> String {
    remove_pattern(text, start_token, end_token)
}

fn remove_html_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_tag = false;
    for c in text.chars() {
        if c == '<' {
            in_tag = true;
        } else if c == '>' {
            in_tag = false;
        } else if !in_tag {
            result.push(c);
        }
    }
    result
}

fn remove_nested_braces(s: &str, open: char, close: char) -> String {
    let mut result = String::with_capacity(s.len());
    let mut depth = 0;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == open && chars.peek() == Some(&open) {
            chars.next();
            depth += 1;
        } else if c == close && chars.peek() == Some(&close) {
            chars.next();
            if depth > 0 {
                depth -= 1;
            }
        } else if depth == 0 {
            result.push(c);
        }
    }
    result
}

fn remove_nested_tables(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut depth = 0;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' && chars.peek() == Some(&'|') {
            chars.next();
            depth += 1;
        } else if c == '|' && chars.peek() == Some(&'}') {
            chars.next();
            if depth > 0 {
                depth -= 1;
            }
        } else if depth == 0 {
            result.push(c);
        }
    }
    result
}

fn clean_wiki_links(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '[' && chars.peek() == Some(&'[') {
            chars.next(); // ikinci '[' karakterini tüket
            let mut inside = String::new();
            while let Some(inner_c) = chars.next() {
                if inner_c == ']' && chars.peek() == Some(&']') {
                    chars.next(); // ikinci ']' karakterini tüket
                    break;
                }
                inside.push(inner_c);
            }
            if let Some(pipe_pos) = inside.find('|') {
                result.push_str(&inside[pipe_pos + 1..]);
            } else {
                result.push_str(&inside);
            }
        } else {
            result.push(c);
        }
    }
    result
}

fn clean_headers(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for line in s.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('=') && trimmed.ends_with('=') {
            let cleaned = trimmed.trim_matches('=');
            result.push_str(cleaned.trim());
        } else {
            result.push_str(line);
        }
        result.push('\n');
    }
    result
}

fn normalize_whitespace(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut last_was_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !last_was_space {
                result.push(' ');
                last_was_space = true;
            }
        } else {
            result.push(c);
            last_was_space = false;
        }
    }
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_lobe_name_from_title() {
        assert_eq!(derive_lobe_name_from_title("Kuantum Fiziği"), "kuantum_fizigi");
        assert_eq!(derive_lobe_name_from_title("VM 2. Hafta"), "vm_2_hafta");
        assert_eq!(derive_lobe_name_from_title("Rust (programlama dili)"), "rust_programlama_dili");
        assert_eq!(derive_lobe_name_from_title("  Yapay Zeka--Modeli "), "yapay_zeka_modeli");
    }

    #[test]
    fn test_clean_mediawiki_syntax() {
        let raw_wiki = "== Giriş ==\n[[Kuantum mekaniği|Kuantum]] kuramı, atomik ve atom altı {{bilgi kutusu | veri = 123}} seviyedeki maddelerin davranışlarını inceler. <ref>Kaynak 1</ref>";
        let cleaned = clean_mediawiki_syntax(raw_wiki);
        assert!(!cleaned.contains("=="));
        assert!(!cleaned.contains("{{"));
        assert!(!cleaned.contains("[["));
        assert!(!cleaned.contains("<ref>"));
        assert!(cleaned.contains("Giriş"));
        assert!(cleaned.contains("Kuantum kuramı"));
    }
}
