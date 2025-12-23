use log::error;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use crate::metadata_provider::metadata_provider::{BaseImage, BaseImageWithImage, Metadata, MetadataProvider};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use surrealdb::engine::remote::ws::Client;
use surrealdb::{RecordId, Surreal};

pub struct ImageHashMetadataProvider;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageHashMetadata{
    pub hash_type: String,
    pub hash: [u8; 32]
}

impl MetadataProvider<BaseImageWithImage, ImageHashMetadata> for ImageHashMetadataProvider {
    fn extract(&self, base_images: &Vec<BaseImageWithImage>) -> anyhow::Result<Vec<Metadata<ImageHashMetadata>>> {
        // Parallel über die Bilder iterieren
        let results: Vec<Metadata<ImageHashMetadata>> = base_images
            .par_iter()  // rayon parallel iterator
            .map(|base_image| {
                let rgb = base_image.image.to_rgb8();
                let mut hasher = Sha256::new();
                hasher.update(rgb.as_raw());
                let hash_result = hasher.finalize();

                let mut hash_bytes = [0u8; 32];
                hash_bytes.copy_from_slice(&hash_result[..]);

                Metadata {
                    id: None,
                    metadata: Some(ImageHashMetadata {
                        hash: hash_bytes,
                        hash_type: "SHA256".to_string(),
                    }),
                    base: base_image.base_image.id.clone(),
                }
            })
            .collect();

        Ok(results)
    }
}

static IMAGE_HASH_REPOSITORY_NAME: &str = "image_hash_metadata";
pub struct ImageHashMetadataRepository{
    db: Surreal<Client>,
}
impl ImageHashMetadataRepository {
    pub async fn new(db: Surreal<Client>) -> Self{
        Self::prepare_repository(&db).await.expect("cannot prepare repository with indexes");
        Self { db }
    }
    async fn prepare_repository(
        db: &Surreal<Client>,
    ) -> anyhow::Result<()> {
        db.query(format!(r#"
            DEFINE INDEX IF NOT EXISTS {IMAGE_HASH_REPOSITORY_NAME}_base_unique
            ON {IMAGE_HASH_REPOSITORY_NAME}
            FIELDS base
            UNIQUE;
            "#),
        )
            .await?;

        Ok(())
    }

    pub async fn insert_many(
        &self,
        items: &Vec<Metadata<ImageHashMetadata>>,
    ) -> anyhow::Result<Vec<Metadata<ImageHashMetadata>>> {

        if items.is_empty() {
            return Ok(Vec::new());
        }

        let mut inserted = Vec::new();

        for item in items {
            let base_id = item.base.clone().ok_or_else(|| anyhow::anyhow!("Base ID missing"))?;

            let metadata = match item.metadata.clone(){
                Some(metadata) => metadata,
                None => {
                    error!("ImageHashMetadata is missing for ID {:?}", item.id);
                    continue;
                }
            };

            let mut response = self.db
                .query(format!(r#"
                UPSERT {IMAGE_HASH_REPOSITORY_NAME}
                SET base = $base,
                    hash = $hash,
                    hash_type = $hash_type
                WHERE base = $base;
                "#))
                .bind(("base", base_id.clone()))
                .bind(("hash", metadata.hash))
                .bind(("hash_type", metadata.hash_type))
                .await?;

            // SurrealDB liefert die erstellten oder vorhandenen Datensätze zurück
            if let Ok(mut rows) = response.take::<Vec<Metadata<ImageHashMetadata>>>(0) {
                inserted.append(&mut rows);
            }
        }

        Ok(inserted)
    }

    pub async fn find_by_bases(
        &self,
        base_ids: Vec<RecordId>,
    ) -> anyhow::Result<Vec<Metadata<ImageHashMetadata>>> {

        if base_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut response = self.db
            .query(format!("SELECT *
             FROM {IMAGE_HASH_REPOSITORY_NAME}
             WHERE base IN $base_ids"))
            .bind(("base_ids", base_ids))
            .await?;

        let items: Vec<Metadata<ImageHashMetadata>> = response.take(0)?;
        Ok(items)
    }
    pub async fn find_by_base_images(
        &self,
        images: &[BaseImage],
    ) -> anyhow::Result<Vec<Metadata<ImageHashMetadata>>> {

        let base_ids: Vec<RecordId> = images
            .iter()
            .filter_map(|img| img.id.clone())
            .collect();

        self.find_by_bases(base_ids).await
    }
}

#[cfg(test)]
mod tests {
    use crate::metadata_provider::metadata_provider::{BaseImage, BaseImageWithImage, MetadataProvider};
    use image::{ColorType, DynamicImage};

    #[test]
    fn test_image_hash_metadata_provider() {
        let images = vec![
            BaseImageWithImage{
                base_image: BaseImage {
                    id: None,
                    path: String::from("/test1.jpg"),
                },
                image: DynamicImage::new(10, 10, ColorType::Rgb8)
            },
                BaseImageWithImage{
                base_image: BaseImage {
                    id: None,
                    path: String::from("/test2.jpg"),
                },
                image: DynamicImage::new(10, 10, ColorType::Rgb16)
            },
            BaseImageWithImage{
                base_image: BaseImage {
                    id: None,
                    path: String::from("/test3.jpg"),
                },
                image: DynamicImage::new(20, 10, ColorType::Rgb8)
            }
        ];
        let hash_provider = super::ImageHashMetadataProvider{};
        let results = hash_provider.extract(&images).unwrap();
        assert_eq!(results.len(), 3);
        assert!(&results[0].metadata.is_some());
        assert!(&results[1].metadata.is_some());
        assert!(&results[2].metadata.is_some());
    }
}