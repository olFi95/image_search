use crate::metadata_provider::metadata_provider::{BaseImage, Metadata, MetadataProvider};
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

impl MetadataProvider<BaseImage, ImageHashMetadata> for ImageHashMetadataProvider {
    fn extract(&self, base_images: &Vec<BaseImage>) -> anyhow::Result<Vec<Metadata<ImageHashMetadata>>> {

        let mut results = Vec::with_capacity(base_images.len());

        for base_image in base_images {
            let rgb = base_image.image.to_rgb8();
            let mut hasher = Sha256::new();
            hasher.update(rgb.as_raw());
            let result = hasher.finalize();
            let mut hash_bytes = [0u8; 32];
            hash_bytes.copy_from_slice(&result[..]);
            results.push(
                Metadata{
                    id: None,
                    metadata: Some(ImageHashMetadata{hash:hash_bytes, hash_type: String::from("SHA256")}),
                    base: base_image.id.clone(),
                }
            )
        }

        Ok(results)
    }
}


pub struct ImageHashMetadataRepository;
impl ImageHashMetadataRepository {
    pub async fn insert_many(
        db: &Surreal<Client>,
        items: Vec<Metadata<ImageHashMetadata>>,
    ) -> anyhow::Result<Vec<Metadata<ImageHashMetadata>>
    > {
        Ok(db.insert::<Vec<Metadata<ImageHashMetadata>>>("image_hash_metadata")
            .content(items)
            .await?)
    }

    pub async fn find_by_bases(
        db: &Surreal<Client>,
        base_ids: Vec<RecordId>,
    ) -> anyhow::Result<Vec<Metadata<ImageHashMetadata>>> {

        if base_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut response = db
            .query(
                "SELECT *
             FROM image_hash_metadata
             WHERE base IN $base_ids"
            )
            .bind(("base_ids", base_ids))
            .await?;

        let items: Vec<Metadata<ImageHashMetadata>> = response.take(0)?;
        Ok(items)
    }
    pub async fn find_by_base_images(
        db: &Surreal<Client>,
        images: &[BaseImage],
    ) -> anyhow::Result<Vec<Metadata<ImageHashMetadata>>> {

        let base_ids: Vec<RecordId> = images
            .iter()
            .filter_map(|img| img.id.clone())
            .collect();

        Self::find_by_bases(db, base_ids).await
    }
}

#[cfg(test)]
mod tests {
    use crate::metadata_provider::metadata_provider::{BaseImage, MetadataProvider};
    use image::{ColorType, DynamicImage};

    #[test]
    fn test_image_hash_metadata_provider() {
        let images = vec![
            BaseImage{
                id: None,
                path: String::from("/test1.jpg"),
                image: DynamicImage::new(10, 10, ColorType::Rgb8)
            },
            BaseImage{
                id: None,
                path: String::from("/test2.jpg"),
                image: DynamicImage::new(10, 10, ColorType::Rgb16)
            },
            BaseImage{
                id: None,
                path: String::from("/test3.jpg"),
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