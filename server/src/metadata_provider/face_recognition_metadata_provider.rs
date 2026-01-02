use crate::metadata_provider::metadata_provider::{BaseImageWithImage, Metadata, MetadataProvider};
use burn::tensor::Device;
use burn_wgpu::Wgpu;
use face_detection::face_detector::FaceDetector;
use face_detection::face_embedder::FaceEmbedder;
use image::DynamicImage;
use log::error;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use surrealdb::{Connection, Surreal};

pub struct FaceRecognitionMetadataProvider {
    face_detector: FaceDetector,
    face_embedder: FaceEmbedder,
}

impl FaceRecognitionMetadataProvider {
    pub fn new(device: Arc<Box<Device<Wgpu>>>, face_detector: &str, face_embedder: &str) -> Self {
        Self {
            face_detector: FaceDetector::new(face_detector, device.clone()),
            face_embedder: FaceEmbedder::new(face_embedder, device),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FaceInPicture {
    pub top_left_x: f32,
    pub top_left_y: f32,
    pub bottom_right_x: f32,
    pub bottom_right_y: f32,
    pub confidence: f32,
    #[serde(skip)]
    pub face: Option<DynamicImage>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FaceInPictureVector {
    pub vector: Vec<f32>,
}

static FACE_IN_PICTURE_DATA_NAME: &str = "face_in_picture";
static FACE_IN_PICTURE_RELATION_NAME: &str = "has_face_in_picture";
static FACE_IN_PICTURE_VECTOR_DATA_NAME: &str = "face_in_picture_vector";
static FACE_IN_PICTURE_VECTOR_RELATION_NAME: &str = "has_face_in_picture_vector";

impl MetadataProvider<BaseImageWithImage, FaceInPicture> for FaceRecognitionMetadataProvider {
    fn extract(
        &self,
        base_images: &Vec<BaseImageWithImage>,
    ) -> anyhow::Result<Vec<Metadata<FaceInPicture>>> {
        let mut results: Vec<Metadata<FaceInPicture>> = vec![];
        for base_image in base_images {
            let image_height = base_image.image.height() as f32;
            let image_width = base_image.image.width() as f32;
            let detected_faces = self.face_detector.detect(&base_image.image);
            for face in detected_faces {
                results.push(Metadata {
                    id: base_image.base_image.id.clone(),
                    metadata: Some(FaceInPicture {
                        top_left_x: face.bbox.xmin / image_width,
                        top_left_y: face.bbox.ymin / image_height,
                        bottom_right_x: face.bbox.xmax / image_width,
                        bottom_right_y: face.bbox.ymax / image_height,
                        confidence: face.bbox.score,
                        face: Some(face.face_image),
                    }),
                    base: base_image.base_image.id.clone(),
                });
            }
        }
        Ok(results)
    }
}
impl MetadataProvider<Metadata<FaceInPicture>, FaceInPictureVector>
    for FaceRecognitionMetadataProvider
{
    fn extract(
        &self,
        face_in_picture: &Vec<Metadata<FaceInPicture>>,
    ) -> anyhow::Result<Vec<Metadata<FaceInPictureVector>>> {
        let mut results: Vec<Metadata<FaceInPictureVector>> = vec![];
        for face_in_picture_metadata in face_in_picture {
            let face_in_picture = face_in_picture_metadata.clone();
            let face_in_picture_metadata = match face_in_picture.metadata {
                Some(metadata) => metadata,
                None => {
                    error!(
                        "FaceInPicture metadata is missing for ID {:?}",
                        face_in_picture.id
                    );
                    continue;
                }
            };
            let face = match face_in_picture_metadata.face {
                Some(face) => face,
                None => {
                    error!(
                        "Face image is missing in FaceInPicture metadata for ID {:?}",
                        face_in_picture.id
                    );
                    continue;
                }
            };
            let face_in_picture_id = match face_in_picture.id {
                None => {
                    error!("FaceInPicture ID is missing");
                    continue;
                }
                Some(id) => id,
            };
            let embedding = self.face_embedder.embed(face);
            results.push(Metadata {
                id: None,
                metadata: Some(FaceInPictureVector { vector: embedding }),
                base: Some(face_in_picture_id),
            });
        }
        Ok(results)
    }
}

pub struct FaceRecognitionMetadataRepository<C: Connection> {
    db: Surreal<C>,
}

impl <C: Connection>FaceRecognitionMetadataRepository<C> {
    pub async fn new(db: Surreal<C>) -> Self {
        Self::prepare_repository(&db)
            .await
            .expect("cannot prepare repository with indexes");
        Self { db }
    }
    async fn prepare_repository(db: &Surreal<C>) -> anyhow::Result<()> {
        db.query(format!(
            r#"
            DEFINE INDEX IF NOT EXISTS {FACE_IN_PICTURE_DATA_NAME}_base_unique
            ON {FACE_IN_PICTURE_DATA_NAME}
            FIELDS base
            UNIQUE;
            "#
        ))
        .query(format!(
            r#"
            DEFINE INDEX IF NOT EXISTS {FACE_IN_PICTURE_VECTOR_DATA_NAME}_base_unique
            ON {FACE_IN_PICTURE_VECTOR_DATA_NAME}
            FIELDS base
            UNIQUE;
            "#
        ))
        .await?;
        Ok(())
    }

    pub async fn insert_many_face_in_picture(
        &self,
        items: &Vec<Metadata<FaceInPicture>>,
    ) -> anyhow::Result<Vec<Metadata<FaceInPicture>>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let mut inserted = Vec::new();

        for item in items {
            let face_in_picture_metadata = match item.metadata.clone() {
                Some(metadata) => metadata,
                None => {
                    error!("FaceInPicture metadata is missing for ID {:?}", item.id);
                    continue;
                }
            };
            if item.id.is_none() {
                error!("FaceInPicture ID is missing");
                continue;
            }

            let mut response = self
                .db
                .query(format!(
                    r#"
                LET $tmp = (
                    UPSERT {FACE_IN_PICTURE_DATA_NAME}
                    SET top_left_x = $top_left_x,
                        top_left_y = $top_left_y,
                        bottom_right_x = $bottom_right_x,
                        bottom_right_y = $bottom_right_y,
                        confidence = $confidence
                    WHERE base = $base
                );
                "#
                ))
                .query(format!(
                    r#"
                LET $id = $tmp[0].id;
                RELATE $base -> {FACE_IN_PICTURE_RELATION_NAME} -> $id;
                "#
                ))
                .query(format!(
                    r#"
                $tmp[0];
                "#
                ))
                .bind(("base", item.id.clone()))
                .bind(("top_left_x", face_in_picture_metadata.top_left_x))
                .bind(("top_left_y", face_in_picture_metadata.top_left_y))
                .bind(("bottom_right_x", face_in_picture_metadata.bottom_right_x))
                .bind(("bottom_right_y", face_in_picture_metadata.bottom_right_y))
                .bind(("confidence", face_in_picture_metadata.confidence))
                .await?;

            if let Ok(mut rows) = response.take::<Vec<Metadata<FaceInPicture>>>(3) {
                rows[0].metadata = item.metadata.clone();
                inserted.append(&mut rows);
            }
        }

        Ok(inserted)
    }
    pub async fn insert_many_face_embeddings(
        &self,
        items: &Vec<Metadata<FaceInPictureVector>>,
    ) -> anyhow::Result<Vec<Metadata<FaceInPictureVector>>> {
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
                .vector;

            let mut response = self
                .db
                .query(format!(
                    r#"
                LET $tmp = (
                    UPSERT {FACE_IN_PICTURE_VECTOR_DATA_NAME}
                    SET embedding = $embedding
                    WHERE base = $base
                );
                LET $id = $tmp[0].id;
                RELATE $base -> {FACE_IN_PICTURE_VECTOR_RELATION_NAME} -> $id;
                $tmp[0];
                "#
                ))
                .bind(("base", base_id.clone()))
                .bind(("embedding", embedding))
                .await?;

            if let Ok(mut rows) = response.take::<Vec<Metadata<FaceInPictureVector>>>(3) {
                inserted.append(&mut rows);
            }
        }

        Ok(inserted)
    }
}
