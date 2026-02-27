use surrealdb_types::SurrealValue;
use burn::tensor::backend::Backend;
use crate::metadata_provider::model::{BaseImageWithImage, Metadata, MetadataProvider};
use ai_models::image_embedder::ImageEmbedder;
use burn::prelude::Device;
use serde::{Deserialize, Serialize};
use surrealdb::{Connection, Surreal};

pub struct ImageEmbeddingMetadataProvider<B: Backend> {
    image_embedder: ImageEmbedder<B>,
}

impl<B: Backend> ImageEmbeddingMetadataProvider<B> {
    pub fn new(
        device: Device<B>,
        image_embedder: &str,
    ) -> Self {
        Self {
            image_embedder: ImageEmbedder::new(image_embedder, device),
        }
    }
}

#[derive(Debug, Serialize, SurrealValue, Deserialize, Clone)]
pub struct ImageEmbedding {
    pub embedding: Vec<f32>,
}

impl<B: Backend> MetadataProvider<BaseImageWithImage, ImageEmbedding> for ImageEmbeddingMetadataProvider<B> {
    fn extract(
        &self,
        images: &[BaseImageWithImage],
    ) -> anyhow::Result<Vec<Metadata<ImageEmbedding>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let image_refs: Vec<&image::DynamicImage> = images.iter().map(|img| &img.image).collect();
        let embeddings = self.image_embedder.embed_batch(&image_refs);

        let results: Vec<Metadata<ImageEmbedding>> = images
            .iter()
            .zip(embeddings)
            .map(|(image, embedding)| Metadata {
                id: None,
                metadata: Some(ImageEmbedding { embedding }),
                base: Some(image.base_image.id.clone().unwrap()),
            })
            .collect();

        Ok(results)
    }
}
pub struct ImageEmbeddingMetadataRepository<C: Connection> {
    db: Surreal<C>,
}
static IMAGE_EMBEDDING_VECTOR_DATA_NAME: &str = "image_embedding_vector";
static IMAGE_EMBEDDING_VECTOR_RELATION_NAME: &str = "has_image_embedding_vector";

impl <C: Connection>ImageEmbeddingMetadataRepository<C> {
    pub async fn new(db: Surreal<C>) -> Self {
        Self::prepare_repository(&db)
            .await
            .expect("cannot prepare repository with indexes");
        Self { db }
    }
    async fn prepare_repository(db: &Surreal<C>) -> anyhow::Result<()> {
        db.query(format!(
            r#"
            DEFINE INDEX IF NOT EXISTS {IMAGE_EMBEDDING_VECTOR_DATA_NAME}_base_unique
            ON {IMAGE_EMBEDDING_VECTOR_DATA_NAME}
            FIELDS base
            UNIQUE;

            DEFINE INDEX IF NOT EXISTS {IMAGE_EMBEDDING_VECTOR_DATA_NAME}_hnsw
            ON {IMAGE_EMBEDDING_VECTOR_DATA_NAME}
            FIELDS embedding
            HNSW
            DIMENSION 768
            DIST EUCLIDEAN;
            "#
        ))
        .await?;
        Ok(())
    }
    pub async fn rebuild_index(&self) -> anyhow::Result<()> {
        self.db
            .query(format!(
                r#"
            REBUILD INDEX IF EXISTS {IMAGE_EMBEDDING_VECTOR_DATA_NAME}_hnsw
            ON {IMAGE_EMBEDDING_VECTOR_DATA_NAME};
            "#
            ))
            .await?;
        Ok(())
    }
    pub async fn insert_many_image_embeddings(
        &self,
        items: &Vec<Metadata<ImageEmbedding>>,
    ) -> anyhow::Result<Vec<Metadata<ImageEmbedding>>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let mut inserted = Vec::new();

        for item in items {
            let base_id = item
                .base
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Base ID missing"))?;
            let embedding = item
                .metadata
                .clone()
                .ok_or(anyhow::anyhow!("Metadata missing"))?
                .embedding;

            let mut response = self
                .db
                .query(format!(
                    r#"
                LET $tmp = (
                    UPSERT {IMAGE_EMBEDDING_VECTOR_DATA_NAME}
                    SET embedding = $embedding,
                        base = $base
                    WHERE base = $base
                );
                LET $id = $tmp[0].id;
                RELATE $base -> {IMAGE_EMBEDDING_VECTOR_RELATION_NAME} -> $id
                "#
                ))
                .bind(("base", base_id.clone()))
                .bind(("embedding", embedding))
                .await?;

            if let Ok(mut rows) = response.take::<Vec<Metadata<ImageEmbedding>>>(0) {
                inserted.append(&mut rows);
            }
        }

        Ok(inserted)
    }
}
