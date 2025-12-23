use crate::metadata_provider::metadata_provider::{BaseImage, BaseImageWithImage, Metadata, MetadataProvider};
use log::error;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;
use surrealdb::engine::remote::ws::Client;
use surrealdb::{RecordId, Surreal};
use crate::metadata_provider::image_hash_metadata_provider::{ImageHashMetadata, ImageHashMetadataRepository};

pub struct BasicMetadataProvider;


#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BasicMetadata{
    pub file_extension: Option<String>,
    pub height: u32,
    pub width: u32,
    pub size_in_bytes: u64,
    pub created: Option<SystemTime>,
}


impl MetadataProvider<BaseImageWithImage, BasicMetadata> for BasicMetadataProvider {
    fn extract(&self, base_images: &Vec<BaseImageWithImage>) -> anyhow::Result<Vec<Metadata<BasicMetadata>>> {
        let results: Vec<Metadata<BasicMetadata>> = base_images
            .par_iter()
            .map(|base_image| {
                let metadata_result = fs::metadata(&base_image.base_image.path);

                if metadata_result.is_ok() {
                    let file_extension = match PathBuf::from(&base_image.base_image.path).extension(){
                        Some(ext) => Some(ext.to_string_lossy().to_string()),
                        None => None,
                    };

                    let metadata = metadata_result.unwrap();
                    Ok(Metadata {
                        id: None,
                        metadata: Some(BasicMetadata {
                            file_extension,
                            height: base_image.image.height(),
                            width: base_image.image.width(),
                            size_in_bytes: metadata.len(),
                            created: match metadata.created(){
                                Ok(time) => Some(time),
                                Err(_) => None,
                            },
                        }),
                        base: base_image.base_image.id.clone(),
                    })
                } else {
                    error!("unable to get file metadata for image {}", &base_image.base_image.path);
                    Err(metadata_result.unwrap_err())
                }

            })
            .filter(|metadata_result| metadata_result.is_ok())
            .map(|metadata_result| metadata_result.unwrap())
            .collect();

        Ok(results)
    }
}

static BASIC_METADATA_REPOSITORY_NAME: &str = "basic_metadata";

pub struct BasicMetadataRepository {
    db: Surreal<Client>,
}

impl BasicMetadataRepository {
    pub async fn new(db: Surreal<Client>) -> Self{
        Self::prepare_repository(&db).await.expect("cannot prepare repository with indexes");
        Self { db }
    }
    async fn prepare_repository(
        db: &Surreal<Client>,
    ) -> anyhow::Result<()> {
        db.query(format!(r#"
            DEFINE INDEX IF NOT EXISTS {BASIC_METADATA_REPOSITORY_NAME}_base_unique
            ON {BASIC_METADATA_REPOSITORY_NAME}
            FIELDS base
            UNIQUE;
            "#),
        )
            .await?;

        Ok(())
    }

    pub async fn insert_many(
        &self,
        items: &Vec<Metadata<BasicMetadata>>,
    ) -> anyhow::Result<Vec<Metadata<BasicMetadata>>> {

        if items.is_empty() {
            return Ok(Vec::new());
        }

        let mut inserted = Vec::new();

        for item in items {
            let base_id = item.base.clone().ok_or_else(|| anyhow::anyhow!("Base ID missing"))?;

            let metadata = match item.metadata.clone(){
                Some(metadata) => metadata,
                None => {
                    error!("BasicMetadata is missing for ID {:?}", item.id);
                    continue;
                }
            };
            let mut response = self.db
                .query(format!(r#"
                UPSERT {BASIC_METADATA_REPOSITORY_NAME}
                SET base = $base,
                    file_extension = $file_extension,
                    height = $height,
                    width = $width,
                    size_in_bytes = $size_in_bytes,
                    created = $created
                WHERE base = $base;
                "#))
                .bind(("base", base_id.clone()))
                .bind(("file_extension", metadata.file_extension))
                .bind(("height", metadata.height))
                .bind(("width", metadata.width))
                .bind(("size_in_bytes", metadata.size_in_bytes))
                .bind(("created", metadata.created))
                .await?;

            if let Ok(mut rows) = response.take::<Vec<Metadata<BasicMetadata>>>(0) {
                inserted.append(&mut rows);
            }
        }

        Ok(inserted)
    }
}
