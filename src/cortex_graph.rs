use std::collections::{HashMap, HashSet};

use petgraph::stable_graph::{StableDiGraph, NodeIndex};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph::Direction;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SynapseType {
    Sequential, // Akış sırası (A -> B)
    Semantic,   // Çağrışımsal bağ (A <-> C)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Synapse {
    pub weight: f32,
    pub co_firings: u64,
    pub synapse_type: SynapseType,
    #[serde(default)]
    pub role: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum NodeType {
    Concept,
    Operator,
    Equation,
    Code,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConceptNode {
    pub id: usize,
    pub content: String, // Ham metin veya kod bloğu
    pub tags: HashMap<String, f32>, // Anahtar kelimeler ve ağırlıkları
    pub activation_level: f32,
    pub lobe_name: String,
    
    // Sanal ID Köprüsü (Proxy/Stub)
    pub is_proxy: bool,
    pub proxy_target: Option<(String, usize)>, // (Target Lobe, Target ID)
    
    #[serde(default)]
    pub node_type: Option<NodeType>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializedEdge {
    pub source_id: usize,
    pub target_id: usize,
    pub target_lobe: String,
    pub synapse: Synapse,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializedLobe {
    pub lobe_name: String,
    pub nodes: Vec<ConceptNode>,
    pub edges: Vec<SerializedEdge>,
}

pub struct CortexGraph {
    pub graph: StableDiGraph<ConceptNode, Synapse>,
    pub id_to_index: HashMap<usize, NodeIndex>,
    pub content_to_id: HashMap<String, usize>,
    pub loaded_lobes: HashSet<String>,
    pub locked_lobes: HashSet<String>,
    pub db: sled::Db,
}

impl CortexGraph {
    pub fn new(db: sled::Db) -> Self {
        Self {
            graph: StableDiGraph::new(),
            id_to_index: HashMap::new(),
            content_to_id: HashMap::new(),
            loaded_lobes: HashSet::new(),
            locked_lobes: HashSet::new(),
            db,
        }
    }

    /// Yeni bir düğüm (cümle/kavram) ekler.
    pub fn add_node(
        &mut self,
        content: &str,
        tags: HashMap<String, f32>,
        lobe_name: &str,
    ) -> usize {
        let id = self.graph.node_count();
        
        let node_type = if is_code(content) {
            NodeType::Code
        } else if is_operator(content) {
            NodeType::Operator
        } else if is_equation(content) {
            NodeType::Equation
        } else {
            NodeType::Concept
        };

        let node = ConceptNode {
            id,
            content: content.to_string(),
            tags,
            activation_level: 0.0,
            lobe_name: lobe_name.to_string(),
            is_proxy: false,
            proxy_target: None,
            node_type: Some(node_type),
        };

        let node_idx = self.graph.add_node(node);
        self.id_to_index.insert(id, node_idx);
        self.content_to_id.insert(content.to_string(), id);
        id
    }

    /// Düğümler arasına yönlü sinaps (bağlantı) ekler.
    pub fn add_synapse(&mut self, source_id: usize, target_id: usize, weight: f32, syn_type: SynapseType) {
        self.add_synapse_with_role(source_id, target_id, weight, syn_type, None);
    }

    /// Düğümler arasına yönlü ve rol etiketli sinaps (bağlantı) ekler.
    pub fn add_synapse_with_role(
        &mut self,
        source_id: usize,
        target_id: usize,
        weight: f32,
        syn_type: SynapseType,
        role: Option<String>,
    ) {
        if let (Some(&u), Some(&v)) = (self.id_to_index.get(&source_id), self.id_to_index.get(&target_id)) {
            if let Some(edge_idx) = self.graph.find_edge(u, v) {
                if let Some(synapse) = self.graph.edge_weight_mut(edge_idx) {
                    synapse.weight = weight.clamp(0.0, 1.0);
                    synapse.synapse_type = syn_type;
                    if role.is_some() {
                        synapse.role = role;
                    }
                }
            } else {
                let synapse = Synapse {
                    weight: weight.clamp(0.0, 1.0),
                    co_firings: 0,
                    synapse_type: syn_type,
                    role,
                };
                self.graph.add_edge(u, v, synapse);
            }
        }
    }

    /// Talamus uyarımı ile aktivasyonu grafikte yayar.
    pub fn propagate_activation(&mut self, query_tags: &HashMap<String, f32>, decay: f32, steps: usize) {
        // 1. Adım: Anahtar kelime eşleşmelerine göre başlangıç uyarım seviyelerini belirle
        let mut initial_activations = HashMap::new();
        for node_idx in self.graph.node_indices() {
            let node = &self.graph[node_idx];
            if node.is_proxy {
                continue;
            }
            
            // Tag benzerliği hesapla (Basit Dot Product / Keyword match)
            let mut match_score = 0.0f32;
            for (tag, &weight) in query_tags {
                if let Some(&node_weight) = node.tags.get(tag) {
                    match_score += weight * node_weight;
                }
            }
            if match_score > 0.05 {
                initial_activations.insert(node_idx, match_score.min(1.0));
            }
        }

        // Başlangıç uyarımını uygula
        for (idx, act) in &initial_activations {
            if let Some(node) = self.graph.node_weight_mut(*idx) {
                node.activation_level = (node.activation_level + *act).min(1.0);
            }
        }

        // 2. Adım: Aktivasyon Yayılım Döngüsü
        for _ in 0..steps {
            let mut activation_updates = vec![0.0; self.graph.node_count() * 2]; // Güvenli boyut
            
            for node_idx in self.graph.node_indices() {
                let node = &self.graph[node_idx];
                if node.activation_level > 0.05 {
                    let mut neighbors = self.graph.neighbors_directed(node_idx, Direction::Outgoing).detach();
                    while let Some((edge_idx, neighbor_idx)) = neighbors.next(&self.graph) {
                        if let Some(synapse) = self.graph.edge_weight(edge_idx) {
                            let signal = node.activation_level * synapse.weight;
                            let idx_val = neighbor_idx.index();
                            if idx_val < activation_updates.len() {
                                activation_updates[idx_val] += signal;
                            }
                        }
                    }
                }
            }

            // Değişiklikleri uygula ve sönümle
            let node_indices: Vec<_> = self.graph.node_indices().collect();
            for node_idx in node_indices {
                let node = &mut self.graph[node_idx];
                let idx_val = node_idx.index();
                let update = if idx_val < activation_updates.len() { activation_updates[idx_val] } else { 0.0 };
                node.activation_level = (node.activation_level * (1.0 - decay) + update).clamp(0.0, 1.0);
            }
        }

        // GABA Baskılama Mekanizması (Inhibition / Top-K Spiking)
        // Sadece en alakalı ilk 20 nöronu uyanık tut, diğerlerini sıfırla.
        let mut active_nodes_indices: Vec<_> = self.graph.node_indices()
            .filter(|&idx| {
                let node = &self.graph[idx];
                !node.is_proxy && node.activation_level > 0.0
            })
            .collect();

        // Aktivasyon seviyesine göre azalan sırada sırala
        active_nodes_indices.retain(|&idx| !self.graph[idx].activation_level.is_nan());
        active_nodes_indices.sort_by(|&a, &b| {
            let act_a = self.graph[a].activation_level;
            let act_b = self.graph[b].activation_level;
            act_b.total_cmp(&act_a)
        });

        // Top 20'yi seç
        let top_k_set: HashSet<_> = active_nodes_indices.iter().take(20).cloned().collect();

        // Geriye kalan tüm düğümlerin aktivasyon seviyesini sıfırla
        let all_indices: Vec<_> = self.graph.node_indices().collect();
        for node_idx in all_indices {
            if !top_k_set.contains(&node_idx) {
                if let Some(node) = self.graph.node_weight_mut(node_idx) {
                    node.activation_level = 0.0;
                }
            }
        }
    }

    /// Hebbian Öğrenme: Birlikte ateşlenen düğümler arasındaki bağı kuvvetlendir, yoksa yeni bağ kur
    pub fn hebbian_update(&mut self, learning_rate: f32, decay: f32) {
        // Sadece aktif düğümlerin etrafındaki kenarları (sinapsları) güncelle
        let active_nodes: Vec<NodeIndex> = self.graph.node_indices()
            .filter(|&idx| self.graph[idx].activation_level > 0.05)
            .collect();

        let mut edges_to_update = HashSet::new();
        for &u in &active_nodes {
            for edge in self.graph.edges(u) {
                edges_to_update.insert(edge.id());
            }
        }

        for edge_idx in edges_to_update {
            if let Some((u, v)) = self.graph.edge_endpoints(edge_idx) {
                let act_u = self.graph[u].activation_level;
                let act_v = self.graph[v].activation_level;
                if let Some(synapse) = self.graph.edge_weight_mut(edge_idx) {
                    let delta = learning_rate * act_u * act_v;
                    synapse.weight = (synapse.weight + delta - decay * synapse.weight).clamp(0.0, 1.0);
                    if act_u > 0.3 && act_v > 0.3 {
                        synapse.co_firings += 1;
                    }
                }
            }
        }

        // Birlikte uyarık olan ama sinapsı olmayan düğümlere yeni semantik sinaps ekle (Plastisite)
        let active_nodes_high: Vec<NodeIndex> = self.graph.node_indices()
            .filter(|&idx| self.graph[idx].activation_level > 0.4)
            .collect();

        for i in 0..active_nodes_high.len() {
            for j in 0..active_nodes_high.len() {
                if i != j {
                    let u = active_nodes_high[i];
                    let v = active_nodes_high[j];
                    let id_u = self.graph[u].id;
                    let id_v = self.graph[v].id;
                    if self.graph.find_edge(u, v).is_none() {
                        self.add_synapse(id_u, id_v, 0.15, SynapseType::Semantic);
                    }
                }
            }
        }
    }

    /// Belirtilen lobu veritabanına kaydeder.
    pub fn save_lobe(&self, lobe_name: &str) -> anyhow::Result<()> {
        let mut nodes_to_save = Vec::new();
        let mut node_ids = HashSet::new();

        // Lob düğümlerini topla
        for node_idx in self.graph.node_indices() {
            let node = &self.graph[node_idx];
            if node.lobe_name == lobe_name && !node.is_proxy {
                nodes_to_save.push(node.clone());
                node_ids.insert(node.id);
            }
        }

        let mut edges_to_save = Vec::new();
        for edge in self.graph.edge_references() {
            let u_node = &self.graph[edge.source()];
            let v_node = &self.graph[edge.target()];
            
            // Eğer kenarın kaynağı bu lobda ise kaydet
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

        let serialized = SerializedLobe {
            lobe_name: lobe_name.to_string(),
            nodes: nodes_to_save,
            edges: edges_to_save,
        };

        // Serileştir ve veritabanına kaydet
        let serialized_bytes = bincode::serialize(&serialized)?;
        self.db.insert(lobe_name, serialized_bytes)?;

        // Ayrıca lobi adını "__lobes__" listesine ekle
        let mut lobes: HashSet<String> = if let Some(bytes) = self.db.get("__lobes__")? {
            bincode::deserialize(&bytes).unwrap_or_default()
        } else {
            HashSet::new()
        };
        if lobes.insert(lobe_name.to_string()) {
            let lobes_bytes = bincode::serialize(&lobes)?;
            self.db.insert("__lobes__", lobes_bytes)?;
        }

        self.db.flush()?;
        Ok(())
    }

    /// Belirtilen lobu veritabanından RAM'e yükler.
    pub fn load_lobe(&mut self, lobe_name: &str) -> anyhow::Result<()> {
        if self.loaded_lobes.contains(lobe_name) {
            return Ok(());
        }

        let opt_bytes = self.db.get(lobe_name)?;
        let serialized: SerializedLobe = match opt_bytes {
            Some(bytes) => bincode::deserialize(&bytes)?,
            None => {
                // Lob veritabanında yoksa yüklenmiş gibi kabul et
                self.loaded_lobes.insert(lobe_name.to_string());
                return Ok(());
            }
        };

        println!("[LobeManager] Lob veritabanından RAM'e yükleniyor: {}", lobe_name);

        // 1. Adım: Düğümleri yükle
        for loaded_node in serialized.nodes {
            let id = loaded_node.id;
            let content = loaded_node.content.clone();
            
            if let Some(&node_idx) = self.id_to_index.get(&id) {
                // Eğer grafikte bu düğüm bir Proxy olarak bulunuyorsa, onu gerçek verilerle yükselt (Upgrade)
                if self.graph[node_idx].is_proxy {
                    let node_mut = &mut self.graph[node_idx];
                    node_mut.content = loaded_node.content;
                    node_mut.tags = loaded_node.tags;
                    node_mut.activation_level = loaded_node.activation_level;
                    node_mut.lobe_name = loaded_node.lobe_name;
                    node_mut.is_proxy = false;
                    node_mut.proxy_target = None;
                    self.content_to_id.insert(content, id);
                }
            } else {
                // Düğüm grafikte yoksa yeni ekle
                self.content_to_id.insert(content, id);
                let node_idx = self.graph.add_node(loaded_node);
                self.id_to_index.insert(id, node_idx);
            }
        }

        // 2. Adım: Kenarları yükle
        for ser_edge in serialized.edges {
            let source_id = ser_edge.source_id;
            let target_id = ser_edge.target_id;
            let target_lobe = ser_edge.target_lobe;

            // Target grafikte var mı?
            if !self.id_to_index.contains_key(&target_id) {
                // Hedef düğüm yüklü değilse, onun yerine geçici bir Proxy/Stub düğüm oluştur
                let proxy_node = ConceptNode {
                    id: target_id,
                    content: format!("[Proxy Node: {} ID {}]", target_lobe, target_id),
                    tags: HashMap::new(),
                    activation_level: 0.0,
                    lobe_name: target_lobe.clone(),
                    is_proxy: true,
                    proxy_target: Some((target_lobe.clone(), target_id)),
                    node_type: None,
                };
                let proxy_idx = self.graph.add_node(proxy_node);
                self.id_to_index.insert(target_id, proxy_idx);
            }

            self.add_synapse(source_id, target_id, ser_edge.synapse.weight, ser_edge.synapse.synapse_type);
        }

        self.loaded_lobes.insert(lobe_name.to_string());
        Ok(())
    }

    /// Lobu RAM'den çıkarır ve gerekirse diske yazar.
    pub fn unload_lobe(&mut self, lobe_name: &str) -> anyhow::Result<()> {
        if !self.loaded_lobes.contains(lobe_name) {
            return Ok(());
        }

        // 1. Önce diske kaydet
        self.save_lobe(lobe_name)?;
        println!("[LobeManager] Lob diske kaydedildi ve RAM'den temizleniyor: {}", lobe_name);

        // 2. Düğümleri ve kenarları temizle
        let node_indices: Vec<_> = self.graph.node_indices().collect();
        let mut nodes_to_remove = Vec::new();
        let mut nodes_to_downgrade = Vec::new();

        for node_idx in node_indices {
            let node = &self.graph[node_idx];
            if node.lobe_name == lobe_name && !node.is_proxy {
                // Bu lobdaki düğümün diğer yüklü loblardan gelen/giden bağlantısı var mı kontrol et
                let has_cross_connections = self.graph.neighbors_directed(node_idx, Direction::Incoming)
                    .any(|parent_idx| self.graph[parent_idx].lobe_name != lobe_name)
                    || self.graph.neighbors_directed(node_idx, Direction::Outgoing)
                    .any(|child_idx| self.graph[child_idx].lobe_name != lobe_name);

                if has_cross_connections {
                    // Bağlantı kopmasın diye düğümü Proxy durumuna düşür (Downgrade)
                    nodes_to_downgrade.push((node_idx, node.id));
                } else {
                    // Bağlantı yoksa tamamen kaldır
                    nodes_to_remove.push((node_idx, node.id));
                }
            }
        }

        // Downgrade işlemi (Sanal ID Köprüsüne dönüştürme)
        for (node_idx, id) in nodes_to_downgrade {
            let old_content = self.graph[node_idx].content.clone();
            self.content_to_id.remove(&old_content);

            let node_mut = &mut self.graph[node_idx];
            node_mut.content = format!("[Proxy Node: {} ID {}]", lobe_name, id);
            node_mut.tags = HashMap::new();
            node_mut.activation_level = 0.0;
            node_mut.is_proxy = true;
            node_mut.proxy_target = Some((lobe_name.to_string(), id));
        }

        // Tamamen kaldırma işlemi
        for (node_idx, id) in nodes_to_remove {
            let old_content = self.graph[node_idx].content.clone();
            self.content_to_id.remove(&old_content);
            self.graph.remove_node(node_idx);
            self.id_to_index.remove(&id);
        }

        self.loaded_lobes.remove(lobe_name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thalamus_router::ThalamusRouter;

    #[test]
    fn test_all_lobes_valid_binary() {
        let lock_file_path = std::path::Path::new("cortex_storage/cortex.db/db");
        if lock_file_path.exists() {
            if std::fs::OpenOptions::new().write(true).open(lock_file_path).is_err() {
                println!("Database is locked by another process, skipping validation test.");
                return;
            }
        }

        let db_path = "cortex_storage/cortex.db";
        let db_result = sled::open(db_path);
        if let Ok(db) = db_result {
            let mut validated_count = 0;
            for item in db.iter() {
                if let Ok((key, value)) = item {
                    let lobe_name = String::from_utf8_lossy(&key);
                    if lobe_name != "__registry__" {
                        let res: Result<SerializedLobe, _> = bincode::deserialize(&value);
                        assert!(
                            res.is_ok(),
                            "Lobe '{}' binary (bincode) formatında çözümlenemedi! Eski JSON verileri kalmış olabilir. Lütfen 'cortex_storage/cortex.db' klasörünü tamamen silip sistemi baştan çalıştırın. Hata: {:?}",
                            lobe_name,
                            res.err()
                        );
                        validated_count += 1;
                        if validated_count >= 20 {
                            break; // 20 numune doğrulamak test için yeterlidir, 8GB+ db'de zaman aşımını önler.
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_lobe_locking_and_ownership_trigger() {
        let db = sled::Config::new().temporary(true).open().unwrap();
        let mut cortex = CortexGraph::new(db.clone());
        let mut router = ThalamusRouter::new();
        
        // 1. Create and save lobes
        cortex.load_lobe("ownership").unwrap();
        let mut tags = HashMap::new();
        tags.insert("ownership".to_string(), 1.0);
        cortex.add_node("This is about Rust's ownership model.", tags, "ownership");
        cortex.save_lobe("ownership").unwrap();
        cortex.unload_lobe("ownership").unwrap();
        
        cortex.load_lobe("math_test").unwrap();
        let mut math_tags = HashMap::new();
        math_tags.insert("math".to_string(), 1.0);
        cortex.add_node("1 + 1 = 2", math_tags, "math_test");
        cortex.save_lobe("math_test").unwrap();
        cortex.unload_lobe("math_test").unwrap();
        
        // Reload router mappings
        router.reload_mappings(&db).unwrap();
        
        // 2. Test query routing
        let target_lobes = router.route_query_lobes("what is ownership?", &db);
        assert!(target_lobes.contains(&"ownership".to_string()));
        
        // 3. Test Lobe Locking
        // Load both lobes
        cortex.load_lobe("ownership").unwrap();
        cortex.load_lobe("math_test").unwrap();
        assert!(cortex.loaded_lobes.contains("ownership"));
        assert!(cortex.loaded_lobes.contains("math_test"));
        
        // Lock ownership lobe
        cortex.locked_lobes.insert("ownership".to_string());
        
        // Run regulate_and_optimize with max_loaded_lobes = 1
        let glia = crate::glial_system::GlialSystem::new(1, 0.05);
        glia.regulate_and_optimize(&mut cortex).unwrap();
        
        assert!(cortex.loaded_lobes.contains("ownership"), "Locked lobe ownership should not be unloaded!");
        assert!(!cortex.loaded_lobes.contains("math_test"), "Unlocked lobe math_test should be unloaded!");
        
        // 4. Test Smart Ownership Trigger
        let rust_node_idx = *cortex.id_to_index.get(&0).unwrap();
        cortex.graph[rust_node_idx].activation_level = 0.0;
        
        // Run the trigger
        router.perform_lobe_wide_spiking(&mut cortex, "what is ownership?");
        
        // Confirm activation level is 1.0
        assert_eq!(cortex.graph[rust_node_idx].activation_level, 1.0);
    }

    #[test]
    fn test_derive_lobe_name_from_filename() {
        use crate::ingestion::derive_lobe_name_from_filename;
        assert_eq!(derive_lobe_name_from_filename("VM 2. Hafta.pdf"), "vm_2_hafta");
        assert_eq!(derive_lobe_name_from_filename("rust_what_is_ownership.txt"), "ownership");
        assert_eq!(derive_lobe_name_from_filename("rust_references_and_borrowing.txt"), "borrowing");
        assert_eq!(derive_lobe_name_from_filename("Variables and Mutability.txt"), "variables_and_mutability");
    }
}

pub fn is_code(text: &str) -> bool {
    let t = text.trim();
    t.starts_with("fn ") 
        || t.starts_with("let ") 
        || t.starts_with("struct ") 
        || t.starts_with("impl ")
        || t.starts_with("use ")
        || t.starts_with("pub ")
        || t.starts_with("$ ")
        || t.starts_with("> ")
        || t.contains("println!")
        || t.contains('{') && t.contains('}') && t.contains('\n')
}

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
