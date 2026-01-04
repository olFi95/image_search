use image::{open, DynamicImage};
use log::error;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use surrealdb::{Connection, RecordId, Surreal};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BaseImage {
    pub id: Option<RecordId>,
    pub path: String,
}

impl TryInto<BaseImageWithImage> for BaseImage {
    type Error = ();

    fn try_into(self) -> Result<BaseImageWithImage, Self::Error> {
        let image_loading_result = open(&self.path);
        if image_loading_result.is_ok() {
            Ok(BaseImageWithImage {
                base_image: self.clone(),
                image: image_loading_result.unwrap(),
            })
        } else {
            error!("Failed to load base image: {}", self.path);
            Err(())
        }
    }
}

pub struct BaseImageWithImage {
    pub base_image: BaseImage,
    pub image: DynamicImage,
}

impl BaseImage {
    pub(crate) fn new(image_path: PathBuf) -> Self {
        Self {
            id: None,
            path: image_path
                .into_os_string()
                .to_str()
                .expect("cannot convert provided path to string")
                .to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Metadata<M> {
    pub id: Option<RecordId>,
    pub metadata: Option<M>,
    pub base: Option<RecordId>,
}

pub trait MetadataProvider<B, M> {
    fn extract(&self, base_data_elements: &Vec<B>) -> anyhow::Result<Vec<Metadata<M>>>;
}

pub struct BaseImageRepository<C: Connection> {
    db: Surreal<C>,
}

impl<C: Connection> BaseImageRepository<C> {
    pub(crate) async fn get_base_image_by_path(&self, path: &str) -> Option<BaseImage> {
        let mut response = self.db.query("SELECT * FROM base_image WHERE path = $path")
            .bind(("path", path.to_string())).await.expect("failed to get base image");
        let result: Vec<BaseImage> = response.take(0).unwrap_or_default();
        result.first().cloned()
    }
}

impl <C: Connection>BaseImageRepository<C> {
    pub async fn new(db: Surreal<C>) -> Self {
        Self::prepare_repository(&db)
            .await
            .expect("cannot prepare repository with indexes");
        Self { db }
    }
    async fn prepare_repository(db: &Surreal<C>) -> anyhow::Result<()> {
        db.query(
            r#"
            DEFINE INDEX IF NOT EXISTS base_image_unique_path
            ON base_image
            FIELDS path
            UNIQUE;
            "#,
        )
        .await?;

        Ok(())
    }
    pub async fn insert_many(&self, items: Vec<BaseImage>) -> anyhow::Result<Vec<BaseImage>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(items.len());

        for item in items {
            // UPSERT anhand des Unique-Fields `path`
            let mut response = self
                .db
                .query(
                    r#"
                    UPSERT base_image
                    SET path = $path
                    WHERE path = $path;
                    "#,
                )
                .bind(("path", item.path.clone()))
                .await?;

            // SurrealDB liefert das Resultat zurück
            if let Ok(mut rows) = response.take::<Vec<BaseImage>>(0) {
                results.append(&mut rows);
            }
        }

        Ok(results)
    }
}
