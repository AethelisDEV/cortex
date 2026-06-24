use std::collections::HashMap;
use crate::cortex_graph::{CortexGraph, SynapseType};
use crate::thalamus_router::{ThalamusRouter, clean_text_to_words};
use crate::glial_system::GlialSystem;

/// Metin ayrıştırma arayüzü. Modülerlik için tasarlanmıştır.
pub trait TextChunker {
    fn chunk(&self, text: &str) -> Vec<String>;
}

/// Varsayılan cümle/paragraf bazlı metin ayrıştırıcı.
pub struct SentenceChunker {
    pub min_length: usize,
    pub max_length: usize,
}

impl Default for SentenceChunker {
    fn default() -> Self {
        Self {
            min_length: 5,
            max_length: 150,
        }
    }
}

impl TextChunker for SentenceChunker {
    fn chunk(&self, text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        // Cümle bitiş karakterlerine ve satır sonlarına göre böl
        let raw_splits = text.split(|c| c == '.' || c == '?' || c == '!' || c == '\n' || c == ';');
        
        for part in raw_splits {
            let trimmed = part.trim();
            // Çok kısa veya çok uzun olmayan mantıksal cümle birimlerini al
            if trimmed.len() >= self.min_length && trimmed.len() <= self.max_length {
                chunks.push(trimmed.to_string());
            } else if trimmed.len() > self.max_length {
                // Çok uzunsa kelime bazlı bölüp küçük parçalara ayır
                let words: Vec<&str> = trimmed.split_whitespace().collect();
                let mut current_chunk = String::new();
                for word in words {
                    if current_chunk.len() + word.len() + 1 > self.max_length {
                        if !current_chunk.is_empty() {
                            chunks.push(current_chunk.trim().to_string());
                        }
                        current_chunk = word.to_string();
                    } else {
                        if current_chunk.is_empty() {
                            current_chunk = word.to_string();
                        } else {
                            current_chunk.push(' ');
                            current_chunk.push_str(word);
                        }
                    }
                }
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                }
            }
        }
        chunks
    }
}

/// Modüler Veri Giriş ve Hafıza Yapılandırma Hattı (Ingestion Pipeline)
pub struct IngestionPipeline<C: TextChunker> {
    pub chunker: C,
}

impl<C: TextChunker> IngestionPipeline<C> {
    pub fn new(chunker: C) -> Self {
        Self { chunker }
    }

    /// Ham metni sisteme yükler, ayrıştırır, nöronları/sinapsları oluşturur ve lobları kaydeder.
    pub fn ingest_text(
        &self,
        cortex: &mut CortexGraph,
        router: &ThalamusRouter,
        glia: &GlialSystem,
        text: &str,
        lobe_name_override: Option<&str>,
    ) -> anyhow::Result<Vec<usize>> {
        let chunks = self.chunker.chunk(text);
        println!("[Ingestion] Metin {} mantıksal cümle/parçaya ayrıştırıldı.", chunks.len());

        let mut ingested_ids = Vec::new();
        let mut prev_node_id: Option<usize> = None;
        let mut affected_lobes = std::collections::HashSet::new();

        for (i, chunk) in chunks.iter().enumerate() {
            let show_progress = i % 100 == 0 || i == chunks.len() - 1;
            if show_progress {
                println!("\n  [{}/{}] İşlenen Birim: \"{}\"", i + 1, chunks.len(), chunk);
            }

            // 1. Hedef Loba karar ver
            let target_lobe = if let Some(override_name) = lobe_name_override {
                override_name.to_string()
            } else {
                let routed_lobes = router.route_query_lobes(chunk, &cortex.db);
                routed_lobes.first().cloned().unwrap_or_else(|| "general".to_string())
            };
            affected_lobes.insert(target_lobe.clone());
            affected_lobes.insert("core_language".to_string());

            // Hedef lobun RAM'e yüklenmesini sağla
            cortex.load_lobe(&target_lobe)?;
            cortex.load_lobe("core_language")?;

            // 2. Düğüm kontrolü veya oluşturma (Mükerrerlik engelle)
            let existing_id = cortex.content_to_id.get(chunk).cloned();

            let id = match existing_id {
                Some(id_val) => {
                    if show_progress {
                        println!("   -> Bu nöron zaten mevcut (ID: {}).", id_val);
                    }
                    id_val
                }
                None => {
                    // Anahtar kelime haritası çıkar (Frekans analizi)
                    let words = clean_text_to_words(chunk);
                    let mut tags = HashMap::new();
                    for w in words {
                        *tags.entry(w).or_insert(0.0) += 1.0;
                    }

                    let new_id = cortex.add_node(chunk, tags, &target_lobe);
                    if show_progress {
                        println!("   -> Yeni nöron oluşturuldu (ID: {}, Lob: {})", new_id, target_lobe);
                    }
                    new_id
                }
            };
            ingested_ids.push(id);

            // Türkçe Morfoloji ve Şablon Çıkarma Adımı
            let (concepts, template) = crate::morphology::ConceptSplitter::split_to_concepts(chunk);
            if !template.is_empty() {
                // Şablonu core_language lobuna kaydet
                let existing_template_id = cortex.content_to_id.get(&template).cloned();
                let template_node_id = match existing_template_id {
                    Some(tid) => tid,
                    None => {
                        let mut tags = HashMap::new();
                        tags.insert("template".to_string(), 1.0);
                        tags.insert("şablon".to_string(), 1.0);
                        cortex.add_node(&template, tags, "core_language")
                    }
                };
                if show_progress {
                    println!("   -> Şablon Kaydedildi: \"{}\" (ID: {})", template, template_node_id);
                }

                // Kavram düğümlerini oluştur/bul ve sequential bağla
                let mut concept_ids = Vec::new();
                for concept in &concepts {
                    let existing_concept_id = cortex.content_to_id.get(concept).cloned();
                    let concept_id = match existing_concept_id {
                        Some(cid) => cid,
                        None => {
                            let mut tags = HashMap::new();
                            let cleaned_concept = concept.trim_matches(|c| c == '[' || c == ']');
                            tags.insert(cleaned_concept.to_lowercase(), 1.0);
                            cortex.add_node(concept, tags, &target_lobe)
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

                // Rol Etiketli Sinapslar (Role Tagging)
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
                            cortex.add_synapse_with_role(cid, verb_id, 0.8, SynapseType::Semantic, Some(role.clone()));
                            if show_progress {
                                println!("     -> Rol Etiketli Sinaps ({}): {} -> {}", role, concepts[idx], concepts[verb_idx]);
                            }
                        }
                    }
                }
            }

            // 3. Sinaps Oluşturma (Orijinal Sıralı Bağ)
            if let Some(prev_id) = prev_node_id {
                cortex.add_synapse(prev_id, id, 0.85, SynapseType::Sequential);
                if show_progress {
                    println!("   -> Orijinal Sıralı Sinaps: {} -> {}", prev_id, id);
                }
            }

            prev_node_id = Some(id);
        }

        // 4. Glia RAM Bütçelendirmesi (Döngü sonunda bir kez çalıştırılır)
        glia.regulate_and_optimize(cortex)?;

        // 7. Etkilenen lobları diske kaydet
        for lobe in affected_lobes {
            cortex.save_lobe(&lobe)?;
        }

        crate::morphology::update_stats_from_graph(&cortex.graph);
        println!("\n[Ingestion] Veri girişi ve lob güncellemeleri tamamlandı.");
        Ok(ingested_ids)
    }

    /// Klasördeki dosyaları tarar ve yükler.
    pub fn ingest_directory(
        &self,
        cortex: &mut CortexGraph,
        router: &ThalamusRouter,
        glia: &GlialSystem,
        dir_path: &std::path::Path,
    ) -> anyhow::Result<()> {
        println!("[Ingestion] Klasör taranıyor: {:?}", dir_path);

        // Kayıt defterini yükle (cortex.db üzerinden "__registry__" key'inden)
        let registry_bytes = cortex.db.get("__registry__")?;
        let mut registry: HashMap<String, String> = if let Some(bytes) = registry_bytes {
            bincode::deserialize(&bytes).unwrap_or_default()
        } else {
            HashMap::new()
        };

        let mut files_to_ingest = Vec::new();
        find_ingestible_files(dir_path, &mut files_to_ingest)?;
        println!("[Ingestion] Toplam {} adet (.txt/.pdf) dosya saptandı.", files_to_ingest.len());

        let mut processed = 0;
        let mut skipped = 0;

        for file_path in files_to_ingest {
            let relative_path = file_path.to_string_lossy().to_string();
            let is_pdf = file_path.extension().map_or(false, |ext| ext.to_string_lossy().to_lowercase() == "pdf");

            let content = if is_pdf {
                match crate::pdf_ingestor::PdfIngestor::extract_text_from_pdf(&file_path) {
                    Ok(c) => c,
                    Err(e) => {
                        println!("[Hata] PDF okunamadı {:?}: {:?}", file_path, e);
                        continue;
                    }
                }
            } else {
                match std::fs::read_to_string(&file_path) {
                    Ok(c) => c,
                    Err(e) => {
                        println!("[Hata] Dosya okunamadı {:?}: {:?}", file_path, e);
                        continue;
                    }
                }
            };

            // Basit hash
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            content.hash(&mut hasher);
            let current_hash = format!("{:x}", hasher.finish());

            if let Some(existing_hash) = registry.get(&relative_path) {
                if existing_hash == &current_hash {
                    skipped += 1;
                    continue;
                }
            }

            let filename = file_path.file_name().and_then(|f| f.to_str()).unwrap_or("");
            let derived_lobe = derive_lobe_name_from_filename(filename);

            println!("\n[Ingestion] Dosya işleniyor: {:?} (LOB: {})", file_path, derived_lobe);
            if let Err(e) = self.ingest_text(cortex, router, glia, &content, Some(&derived_lobe)) {
                println!("[Hata] Dosya işlenirken hata oluştu {:?}: {:?}", file_path, e);
                continue;
            }

            registry.insert(relative_path, current_hash);
            processed += 1;

            let serialized_registry = bincode::serialize(&registry)?;
            cortex.db.insert("__registry__", serialized_registry)?;
            cortex.db.flush()?;
        }

        println!("\n[Ingestion] Klasör taraması bitti. İşlenen: {}, Atlanan: {}", processed, skipped);
        Ok(())
    }
}

fn find_ingestible_files(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) -> anyhow::Result<()> {
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                find_ingestible_files(&path, files)?;
            } else if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if ext_str == "txt" || ext_str == "pdf" {
                        files.push(path);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Dosya adını temizleyerek küçük harf, özel karakterlerden arındırılmış ve boşlukları alt tireye çevrilmiş lob adını türetir.
pub fn derive_lobe_name_from_filename(filename: &str) -> String {
    let mut name = filename.to_lowercase();
    // Uzantıyı kes
    if let Some(pos) = name.rfind('.') {
        name.truncate(pos);
    }
    // "rust_" önekini kaldır
    if name.starts_with("rust_") {
        name = name["rust_".len()..].to_string();
    }
    
    // Boşlukları alt tireye çevir
    name = name.replace(' ', "_");
    
    // Nokta, parantez, tire gibi özel karakterleri kaldır, sadece alfanümerik ve alt tireyi tut
    name = name.chars()
        .filter(|&c| c.is_alphanumeric() || c == '_')
        .collect();

    // Çift veya mükerrer alt tireleri temizle
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
    
    let clean = clean.trim_matches('_').to_string();
    
    // Bilinen ortak kalıpları eşleştir/grupla
    if clean.contains("ownership") {
        "ownership".to_string()
    } else if clean.contains("borrowing") || clean.contains("reference") {
        "borrowing".to_string()
    } else if clean.contains("slice") {
        "slices".to_string()
    } else if clean.contains("guessing_game") {
        "guessing_game".to_string()
    } else if clean == "programming_language" {
        "rust".to_string()
    } else if clean.contains("hellocargo") {
        "cargo".to_string()
    } else if clean.contains("helloworld") {
        "hello_world".to_string()
    } else if clean.contains("control_flow") {
        "control_flow".to_string()
    } else if clean.contains("data_types") {
        "data_types".to_string()
    } else if clean.contains("comments") {
        "comments".to_string()
    } else if clean.contains("functions") {
        "functions".to_string()
    } else if clean.contains("quantum_physics") {
        "quantum".to_string()
    } else {
        clean
    }
}
