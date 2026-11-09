//! Grouped vector search with zvec.
//!
//! A plain vector search returns one flat list, which a single popular category
//! can dominate. A grouped search buckets the nearest matches by a scalar
//! field, so every bucket gets its own share of the results.

use zvec_rust::*;

fn main() -> zvec_rust::Result<()> {
    println!("zvec version: {}", version());
    initialize(None)?;

    let data_path = "./zvec_group_by_search_example_data";

    let schema = CollectionSchema::builder("articles")
        .add_field(FieldSchema::new("id", DataType::String, false, 0)?)
        .add_field(FieldSchema::new("title", DataType::String, true, 0)?)
        .add_field(FieldSchema::new("category", DataType::String, true, 0)?)
        .add_vector_field(
            "embedding",
            DataType::VectorFp32,
            4,
            IndexParams::hnsw(MetricType::Cosine, 16, 100)?,
        )
        .build()?;

    let collection = Collection::create_and_open(data_path, &schema, None)?;
    println!("Collection created successfully!");

    let titles = [
        "Vector indexes in practice",
        "Choosing an ANN algorithm",
        "Quantization for embeddings",
        "Hybrid search with filters",
        "Ranking beyond cosine",
        "Fuzzy matching at scale",
        "Training embeddings",
        "Evaluating retrieval quality",
        "Serving models online",
    ];
    let categories = [
        "databases",
        "databases",
        "databases",
        "search",
        "search",
        "search",
        "ml",
        "ml",
        "ml",
    ];
    let vectors: Vec<[f32; 4]> = vec![
        [0.10, 0.20, 0.30, 0.40],
        [0.12, 0.22, 0.28, 0.42],
        [0.50, 0.60, 0.70, 0.80],
        [0.14, 0.24, 0.32, 0.44],
        [0.16, 0.26, 0.34, 0.46],
        [0.90, 0.10, 0.20, 0.30],
        [0.18, 0.28, 0.36, 0.48],
        [0.40, 0.50, 0.60, 0.70],
        [0.20, 0.30, 0.38, 0.50],
    ];

    let mut docs = Vec::new();
    for i in 0..titles.len() {
        let mut doc = Doc::new()?;
        let pk = format!("doc_{}", i);
        doc.set_pk(&pk);
        doc.add_string("id", &pk)?;
        doc.add_string("title", titles[i])?;
        doc.add_string("category", categories[i])?;
        doc.add_vector_f32("embedding", &vectors[i])?;
        docs.push(doc);
    }

    let doc_refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&doc_refs)?;
    println!("Inserted {} sample documents", docs.len());

    let query_vector = [0.15, 0.25, 0.33, 0.45];

    // Example 1: the same search without grouping
    println!("\n=== Example 1: Flat Search (top 6) ===");
    let flat_query = SearchQuery::new("embedding", &query_vector, 6)?;
    let flat_results = collection.query(&flat_query)?;
    for (i, result) in flat_results.iter().enumerate() {
        println!(
            "  #{} {} [{}] score {:.4}",
            i + 1,
            result.get_pk().unwrap_or("<unknown>"),
            result.get_string("category")?.unwrap_or_default(),
            result.get_score()
        );
    }
    println!("  -> one category can own the whole page");

    // Example 2: grouped search, three groups of at most two documents
    println!("\n=== Example 2: Grouped Search (3 groups, 2 per group) ===");
    let mut group_query = GroupBySearchQuery::new("embedding", "category", &query_vector, 3, 2)?;
    group_query.set_output_fields(&["id", "title", "category"])?;

    let groups = collection.group_by_query(&group_query)?;
    println!("  {} groups returned:", groups.len());
    for group in &groups {
        println!(
            "  [{}] {} document(s)",
            group.group_by_value,
            group.doc_count()
        );
        for doc in &group.docs {
            println!(
                "      - {} \"{}\" score {:.4}",
                doc.get_pk().unwrap_or("<unknown>"),
                doc.get_string("title")?.unwrap_or_default(),
                doc.get_score()
            );
        }
    }

    // Example 3: narrow the search with a filter and ask for the vectors back
    println!("\n=== Example 3: Grouped Search With a Filter ===");
    let mut filtered_query = GroupBySearchQuery::new("embedding", "category", &query_vector, 3, 1)?;
    filtered_query.set_output_fields(&["title"])?;
    filtered_query.set_filter("category != 'ml'")?;
    filtered_query.set_include_vector(true)?;

    for group in &collection.group_by_query(&filtered_query)? {
        let doc = &group.docs[0];
        let vector = doc.get_vector_f32("embedding")?.unwrap_or_default();
        println!(
            "  [{}] {} \"{}\" vector {:?}",
            group.group_by_value,
            doc.get_pk().unwrap_or("<unknown>"),
            doc.get_string("title")?.unwrap_or_default(),
            vector
        );
    }

    // The groups own their documents, so they can be inspected after the
    // collection is closed.
    let kept = collection.group_by_query(&group_query)?;
    collection.close()?;
    shutdown()?;
    println!(
        "\nGroups still readable after close: {:?}",
        kept.iter()
            .map(|group| group.group_by_value.as_str())
            .collect::<Vec<_>>()
    );

    let _ = std::fs::remove_dir_all(data_path);
    println!("\nDone! All grouped search examples demonstrated successfully.");

    Ok(())
}
