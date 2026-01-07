use crate::metadata_provider::metadata_provider::{
    BaseImage, BaseImageWithImage, Metadata, MetadataProvider,
};
use log::error;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use surrealdb::{Connection, RecordId, Surreal};

pub struct ImageHashMetadataProvider;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageHashMetadata {
    pub hash_type: String,
    pub hash: [u8; 32],
}

impl MetadataProvider<BaseImageWithImage, ImageHashMetadata> for ImageHashMetadataProvider {
    fn extract(
        &self,
        base_images: &Vec<BaseImageWithImage>,
    ) -> anyhow::Result<Vec<Metadata<ImageHashMetadata>>> {
        // Parallel über die Bilder iterieren
        let results: Vec<Metadata<ImageHashMetadata>> = base_images
            .par_iter() // rayon parallel iterator
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

static IMAGE_HASH_DATA_NAME: &str = "image_hash_metadata";
static IMAGE_HASH_RELATION_NAME: &str = "has_image_hash_metadata";
pub struct ImageHashMetadataRepository<C> where C: Connection {
    db: Surreal<C>,
}

impl <C: Connection>ImageHashMetadataRepository<C> {
    pub async fn new(db: Surreal<C>) -> Self {
        Self::prepare_repository(&db)
            .await
            .expect("cannot prepare repository with indexes");
        Self { db }
    }
    async fn prepare_repository(db: &Surreal<C>) -> anyhow::Result<()> {
        db.query(format!(
            r#"
            DEFINE INDEX IF NOT EXISTS {IMAGE_HASH_DATA_NAME}_base_unique
            ON {IMAGE_HASH_DATA_NAME}
            FIELDS base
            UNIQUE;
            "#
        ))
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
            let base_id = item
                .base
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Base ID missing"))?;

            let metadata = match item.metadata.clone() {
                Some(metadata) => metadata,
                None => {
                    error!("ImageHashMetadata is missing for ID {:?}", item.id);
                    continue;
                }
            };

            let mut response = self
                .db
                .query(format!(
                    r#"
                LET $tmp = (
                    UPSERT {IMAGE_HASH_DATA_NAME}
                    SET hash = $hash,
                        hash_type = $hash_type
                    WHERE base = $base
                );
                LET $id = $tmp[0].id;
                RELATE $base -> {IMAGE_HASH_RELATION_NAME} -> $id;
                "#
                ))
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

    #[allow(dead_code)]
    pub(crate) async fn get_image_hashes_for_base_images(
        &self,
        base_images: &[&BaseImage],
    ) -> Vec<ImageHashMetadata> {
        let base_ids: Vec<RecordId> = base_images
            .iter()
            .filter_map(|img| img.id.clone())
            .collect();

        let mut response = self
            .db
            .query(format!(
                r#"
            SELECT VALUE ->{IMAGE_HASH_RELATION_NAME}-> {IMAGE_HASH_DATA_NAME}.*
            FROM $base_ids
            "#
            ))
            .bind(("base_ids", base_ids))
            .await
            .expect("cannot query image hashes for base images");

        let nested: Vec<Vec<ImageHashMetadata>> = response
            .take(0)
            .expect("cannot take image hashes from response");

        nested.into_iter().flatten().collect()
    }

}

#[cfg(test)]
mod tests {
    use crate::metadata_provider::metadata_provider::{BaseImage, BaseImageRepository, BaseImageWithImage, MetadataProvider};
    use image::{ColorType, DynamicImage};
    use std::path::PathBuf;
    use surrealdb::engine::local::Mem;
    use surrealdb::Surreal;

    #[test]
    fn test_image_hash_metadata_provider() {
        let images = vec![
            BaseImageWithImage {
                base_image: BaseImage {
                    id: None,
                    path: String::from("/test1.jpg"),
                },
                image: DynamicImage::new(10, 10, ColorType::Rgb8),
            },
            BaseImageWithImage {
                base_image: BaseImage {
                    id: None,
                    path: String::from("/test2.jpg"),
                },
                image: DynamicImage::new(10, 10, ColorType::Rgb16),
            },
            BaseImageWithImage {
                base_image: BaseImage {
                    id: None,
                    path: String::from("/test3.jpg"),
                },
                image: DynamicImage::new(20, 10, ColorType::Rgb8),
            },
        ];
        let hash_provider = super::ImageHashMetadataProvider {};
        let results = hash_provider.extract(&images).unwrap();
        assert_eq!(results.len(), 3);
        assert!(&results[0].metadata.is_some());
        assert!(&results[1].metadata.is_some());
        assert!(&results[2].metadata.is_some());
    }

    #[tokio::test]
    async fn test_image_hash_calculation() {
        let db = Surreal::new::<Mem>(()).await.unwrap();
        db.use_ns("test").use_db("test").await.unwrap();
        let image_hash_metadata_repository =
            super::ImageHashMetadataRepository::new(db.clone()).await;
        let base_image_repository = BaseImageRepository::new(db.clone()).await;
        let image_hash_metadata_provider = super::ImageHashMetadataProvider{};
        let image_path = "../test_pictures/0_1.jpg";
        let base_images: Vec<BaseImage> = vec![image_path]
            .iter()
            .map(|path| BaseImage::new(PathBuf::from(path)))
            .collect();
        let base_images = base_image_repository
            .insert_many(base_images)
            .await
            .expect("Inserting base image failed");
        let base_images_with_image: Vec<_> = base_images
            .iter()
            .cloned()
            .map(|bi| bi.try_into())
            .filter(|biwi| biwi.is_ok())
            .map(|biwi| biwi.unwrap())
            .collect();


        let hashes = image_hash_metadata_provider.extract(&base_images_with_image).expect("cannot calculate hashes images");
        let _ = image_hash_metadata_repository
            .insert_many(&hashes)
            .await
            .expect("cannot insert hashes");
        let base_image_entry_result = base_image_repository.get_base_image_by_path(image_path).await;
        assert!(base_image_entry_result.is_some(), "No base_image was found even though it was just inserted");
        let base_image_entry = base_image_entry_result.unwrap();
        assert_eq!(base_image_entry.path, image_path);
        let image_hash_entries = image_hash_metadata_repository.get_image_hashes_for_base_images(&[&base_image_entry]).await;
        assert_eq!(image_hash_entries.len(), 1, "No image_hash was found even though it was just inserted");
        let image_hash_entry = &image_hash_entries[0];
        assert_eq!(image_hash_entry.hash_type, "SHA256");
        assert_eq!(hex::encode(image_hash_entry.hash), "d78f6226b8b5bab6ba377b9de4f2d7172336a82688e288fbfa85533d73dcd3c6");
    }
}
