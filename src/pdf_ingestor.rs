use std::path::Path;
use std::collections::{HashMap, HashSet};

pub struct PdfIngestor;

impl PdfIngestor {
    /// PDF dosyasındaki metinleri temiz bir şekilde ayıklar.
    /// Sayfa numaralarını, başlık/dipnot tekrarlarını, mizanpaj çökmelerini ve tablo/indeks satırlarını temizler.
    pub fn extract_text_from_pdf(path: &Path) -> Result<String, String> {
        let pages = pdf_extract::extract_text_by_pages(path)
            .map_err(|e| format!("PDF metni ayıklanamadı: {}", e))?;
        
        println!("RAW PAGES: {:?}", pages);
        if pages.is_empty() {
            return Ok(String::new());
        }

        // 1. Sayfa sayfa satırları ayır ve temizle
        let mut pages_lines: Vec<Vec<String>> = Vec::new();
        for page_text in &pages {
            let lines: Vec<String> = page_text
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect();
            pages_lines.push(lines);
        }

        // 2. Başlık ve dipnot (header/footer) tekrarlarını bul (ilk ve son 2 satırdaki tekrarlar)
        let mut header_footer_freq = HashMap::new();
        let total_pages = pages_lines.len();

        for lines in &pages_lines {
            let n = lines.len();
            let mut page_candidates = HashSet::new();
            // İlk 2 satır (Header adayları)
            for i in 0..std::cmp::min(2, n) {
                page_candidates.insert(lines[i].clone());
            }
            // Son 2 satır (Footer adayları)
            for i in (std::cmp::max(0, n as i32 - 2) as usize)..n {
                page_candidates.insert(lines[i].clone());
            }
            for candidate in page_candidates {
                *header_footer_freq.entry(candidate).or_insert(0) += 1;
            }
        }

        // Sayfa sayısının %30'undan fazlasında tekrarlanan satırları header/footer olarak işaretle (en az 2 sayfa olması gerekir)
        let mut header_footers = HashSet::new();
        if total_pages >= 2 {
            let threshold = std::cmp::max(2, (total_pages as f32 * 0.3).ceil() as usize);
            for (line, freq) in header_footer_freq {
                if freq >= threshold {
                    header_footers.insert(line);
                }
            }
        }

        // 3. Metinleri temizleyerek birleştir
        let mut cleaned_text = String::new();
        for lines in pages_lines {
            for line in lines {
                // Sayfa numarası filtreleme
                if is_page_number(&line) {
                    continue;
                }
                // Başlık/dipnot tekrarlarını temizle
                if header_footers.contains(&line) {
                    continue;
                }
                // Tablo ve İndeks filtresi (yoğun nokta tekrarları veya dizin satırları)
                if is_toc_line(&line) {
                    continue;
                }
                // Mizanpaj çökmesi / anlamsız karakter filtresi
                if is_layout_collapsed(&line) {
                    continue;
                }

                // Karakter bazlı temizleme (ASCII kontrol karakterleri vb.)
                let cleaned_line = clean_meaningless_chars(&line);
                if cleaned_line.trim().is_empty() {
                    continue;
                }

                cleaned_text.push_str(&cleaned_line);
                cleaned_text.push(' ');
            }
            cleaned_text.push('\n');
        }

        Ok(cleaned_text.trim().to_string())
    }
}

fn is_page_number(line: &str) -> bool {
    let l_lower = line.to_lowercase();
    // Sadece rakam, boşluk veya / - içeren çok kısa satırlar sayfa numarasıdır
    if line.chars().all(|c| c.is_ascii_digit() || c.is_whitespace() || c == '-' || c == '/') {
        if line.chars().any(|c| c.is_ascii_digit()) && line.len() <= 10 {
            return true;
        }
    }
    // "sayfa 1", "page 1 of 5", "sayfa 1/5" vb.
    if l_lower.starts_with("sayfa") || l_lower.starts_with("page") {
        return true;
    }
    false
}

fn is_toc_line(line: &str) -> bool {
    // Yoğun nokta tekrarları (Örn: Giriş .............. 5)
    if line.contains("...") || line.contains("..") {
        return true;
    }
    let dot_count = line.chars().filter(|&c| c == '.').count();
    if dot_count >= 5 {
        return true;
    }
    false
}

fn is_layout_collapsed(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return true;
    }
    // Sadece özel simgelerden oluşan anlamsız satırlar
    if trimmed.len() < 3 && !trimmed.chars().all(|c| c.is_alphanumeric()) {
        return true;
    }
    // Harf veya rakam dışındaki karakter oranı çok yüksekse mizanpaj çökmüştür
    let special_chars = trimmed.chars()
        .filter(|&c| !c.is_alphanumeric() && !c.is_whitespace() && c != '\'' && c != '.' && c != ',' && c != '?' && c != '!' && c != '-' && c != '+' && c != '*' && c != '/' && c != '=')
        .count();
    if (special_chars as f32 / trimmed.len() as f32) > 0.35 {
        return true;
    }
    false
}

fn clean_meaningless_chars(line: &str) -> String {
    let mut cleaned = String::new();
    let mut last_char = None;
    let mut repeat_count = 0;

    for c in line.chars() {
        if c.is_ascii_control() && c != '\n' && c != '\t' {
            continue;
        }
        
        // Aşırı tekrarlayan sembolleri sınırla
        if let Some(lc) = last_char {
            if c == lc && !c.is_alphanumeric() && !c.is_whitespace() {
                repeat_count += 1;
                if repeat_count > 3 {
                    continue;
                }
            } else {
                repeat_count = 0;
            }
        }
        last_char = Some(c);
        cleaned.push(c);
    }
    cleaned
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_page_number() {
        assert!(is_page_number("1"));
        assert!(is_page_number("Sayfa 5"));
        assert!(is_page_number("Page 10 of 20"));
        assert!(is_page_number("12/45"));
        assert!(!is_page_number("Rust Programlama"));
    }

    #[test]
    fn test_is_toc_line() {
        assert!(is_toc_line("1. Giriş ................ 5"));
        assert!(is_toc_line("Bölüm 2 ..... 12"));
        assert!(!is_toc_line("Bu bir normal cümledir."));
    }

    #[test]
    fn test_is_layout_collapsed() {
        assert!(is_layout_collapsed(""));
        assert!(is_layout_collapsed("#$"));
        assert!(!is_layout_collapsed("Rust hafıza güvenliğini sağlar."));
    }

    #[test]
    fn test_clean_meaningless_chars() {
        assert_eq!(clean_meaningless_chars("normal text"), "normal text");
        assert_eq!(clean_meaningless_chars("text ------ test"), "text ---- test");
    }

    #[test]
    fn test_extract_text_from_pdf() {
        use lopdf::{dictionary, Document, Object, Stream};
        use lopdf::content::{Content, Operation};

        let temp_dir = std::env::temp_dir();
        let pdf_path = temp_dir.join("test_extract.pdf");

        // Create a simple PDF
        let mut doc = Document::with_version("1.5");

        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Courier",
        });

        let resources_id = doc.add_object(dictionary! {
            "Font" => dictionary! {
                "F1" => font_id,
            },
        });

        let content = Content {
            operations: vec![
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec!["F1".into(), 12.into()]),
                Operation::new("Td", vec![50.into(), 700.into()]),
                Operation::new("Tj", vec![Object::string_literal("Rust hafiza guvenligi saglar")]),
                Operation::new("ET", vec![]),
            ],
        };

        let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));

        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Contents" => content_id,
        });

        let pages_id = doc.add_object(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
            "Resources" => resources_id,
            "MediaBox" => vec![0.into(), 0.into(), 500.into(), 800.into()],
        });

        if let Ok(page) = doc.get_object_mut(page_id) {
            if let Object::Dictionary(ref mut dict) = *page {
                dict.set("Parent", pages_id);
            }
        }

        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });

        doc.trailer.set("Root", catalog_id);

        doc.save(&pdf_path).unwrap();

        // Now extract
        let result = PdfIngestor::extract_text_from_pdf(&pdf_path);
        assert!(result.is_ok());
        let text = result.unwrap();
        println!("EXTRACTED TEXT: {:?}", text);
        assert!(text.contains("Rust hafiza guvenligi saglar"));

        // Clean up
        let _ = std::fs::remove_file(pdf_path);
    }
}
