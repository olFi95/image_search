use crate::metadata_provider::metadata_provider::{BaseImageWithImage, Metadata, MetadataProvider};
use clip::ImageEmbedder;
use serde::{Deserialize, Serialize};
use surrealdb::engine::remote::ws::Client;
use surrealdb::Surreal;

pub struct ImageEmbeddingMetadataProvider{
    image_embedder: ImageEmbedder,
}

impl ImageEmbeddingMetadataProvider {
    pub fn new(device: std::sync::Arc<Box<burn::tensor::Device<burn_wgpu::Wgpu>>>, image_embedder: &str) -> Self {
        Self {
            image_embedder: ImageEmbedder::new(image_embedder, device),
        }
    }}


#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageEmbedding{
    pub vector: Vec<f32>,
}

impl MetadataProvider<BaseImageWithImage, ImageEmbedding> for ImageEmbeddingMetadataProvider {
    fn extract(&self, images: &Vec<BaseImageWithImage>) -> anyhow::Result<Vec<Metadata<ImageEmbedding>>> {
        let mut results: Vec<Metadata<ImageEmbedding>> = vec![];
        for image in images {
            let embedding = self.image_embedder
                .embed(&image.image);
            results.push(Metadata{
                id: None,
                metadata: Some(ImageEmbedding {
                    vector: embedding,
                }),
                base: Some(image.base_image.id.clone().unwrap()),
            });
        }
        Ok(results)
    }
}
pub struct ImageEmbeddingMetadataRepository {
    db: Surreal<Client>,
}
static IMAGE_EMBEDDING_VECTOR_REPOSITORY_NAME: &str = "image_embedding_vector";

impl ImageEmbeddingMetadataRepository {
    pub async fn new(db: Surreal<Client>) -> Self {
        Self::prepare_repository(&db).await.expect("cannot prepare repository with indexes");
        Self { db }
    }
    async fn prepare_repository(
        db: &Surreal<Client>,
    ) -> anyhow::Result<()> {
        db.query(format!(r#"
            DEFINE INDEX IF NOT EXISTS {IMAGE_EMBEDDING_VECTOR_REPOSITORY_NAME}_base_unique
            ON {IMAGE_EMBEDDING_VECTOR_REPOSITORY_NAME}
            FIELDS base
            UNIQUE;
            "#),
        ).query(format!(r#"
            DEFINE INDEX IF NOT EXISTS {IMAGE_EMBEDDING_VECTOR_REPOSITORY_NAME}_mtree
            ON {IMAGE_EMBEDDING_VECTOR_REPOSITORY_NAME}
            FIELDS embedding MTREE DIMENSION 768 DIST COSINE TYPE F32;
            "#),
        ).await?;
        Ok(())
    }
    pub async fn rebuild_index(
        &self,
    ) -> anyhow::Result<()> {
        self.db.query(format!(r#"
            REBUILD INDEX IF EXISTS {IMAGE_EMBEDDING_VECTOR_REPOSITORY_NAME}_mtree
            ON {IMAGE_EMBEDDING_VECTOR_REPOSITORY_NAME};
            "#),
        ).await?;
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
            let base_id = item.base.clone().ok_or_else(|| anyhow::anyhow!("Base ID missing"))?;
            let embedding = item.metadata.clone().ok_or(anyhow::anyhow!("Metadata missing"))?.vector;

            let mut response = self.db
                .query(format!(r#"
                UPSERT {IMAGE_EMBEDDING_VECTOR_REPOSITORY_NAME}
                SET base = $base, embedding = $embedding
                WHERE base = $base;
                "#))
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