use crate::metadata_provider::model::{
    BaseImageWithImage, Metadata, MetadataProvider,
};
use log::error;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use surrealdb::{Connection, Surreal};
use surrealdb::types::{Datetime, SurrealValue};
pub struct BasicMetadataProvider;

#[derive(Debug, Serialize, SurrealValue, Deserialize, Clone)]
pub struct BasicMetadata {
    pub file_extension: Option<String>,
    pub height: u32,
    pub width: u32,
    pub size_in_bytes: u64,
    pub created: Option<Datetime>,
}

impl MetadataProvider<Arc<BaseImageWithImage>, BasicMetadata> for BasicMetadataProvider {
    fn extract(
        &self,
        base_images: &[Arc<BaseImageWithImage>],
    ) -> anyhow::Result<Vec<Metadata<BasicMetadata>>> {
        let results: Vec<Metadata<BasicMetadata>> = base_images
            .par_iter()
            .map(|base_image| {
                let metadata_result = fs::metadata(&base_image.base_image.path);

                match metadata_result {
                    Ok(metadata) => {
                        let file_extension = PathBuf::from(&base_image.base_image.path)
                            .extension()
                            .map(|ext| ext.to_string_lossy().to_string());

                        Ok(Metadata {
                            id: None,
                            metadata: Some(BasicMetadata {
                                file_extension,
                                height: base_image.image.height(),
                                width: base_image.image.width(),
                                size_in_bytes: metadata.len(),
                                created: metadata
                                    .created()
                                    .ok()
                                    .map(|t| {
                                        let dt: chrono::DateTime<chrono::Utc> = t.into();
                                        dt.into()
                                    }),
                            }),
                            base: base_image.base_image.id.clone(),
                        })
                    }
                    Err(_) => {
                        error!(
                        "unable to get file metadata for image {}",
                        &base_image.base_image.path
                    );
                        Err(metadata_result.unwrap_err())
                    }
                }
            })
            .flat_map(|metadata_result| metadata_result.ok())
            .collect();

        Ok(results)
    }
}

static BASIC_METADATA_DATA_NAME: &str = "basic_metadata";
static BASIC_METADATA_RELATION_NAME: &str = "has_basic_metadata";

pub struct BasicMetadataRepository<C: Connection> {
    db: Surreal<C>,
}

impl <C: Connection>BasicMetadataRepository<C> {
    pub async fn new(db: Surreal<C>) -> Self {
        Self::prepare_repository(&db)
            .await
            .expect("cannot prepare repository with indexes");
        Self { db }
    }
    async fn prepare_repository(db: &Surreal<C>) -> anyhow::Result<()> {
        db.query(format!(
            r#"
            DEFINE INDEX IF NOT EXISTS {BASIC_METADATA_DATA_NAME}_base_unique
            ON {BASIC_METADATA_DATA_NAME}
            FIELDS base
            UNIQUE;
            "#
        ))
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
            let base_id = item
                .base
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Base ID missing"))?;

            let metadata = match item.metadata.clone() {
                Some(metadata) => metadata,
                None => {
                    error!("BasicMetadata is missing for ID {:?}", item.id);
                    continue;
                }
            };
            let mut response = self
                .db
                .query(format!(
                    r#"
                LET $tmp = (
                    UPSERT {BASIC_METADATA_DATA_NAME}
                    SET file_extension = $file_extension,
                        height = $height,
                        width = $width,
                        size_in_bytes = $size_in_bytes,
                        created = $created,
                        base = $base
                    WHERE base = $base
                );
                LET $id = $tmp[0].id;
                RELATE $base -> {BASIC_METADATA_RELATION_NAME} -> $id;
                "#
                ))
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
