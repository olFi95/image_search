use crate::metadata_provider::model::{BaseImageWithImage, Metadata, MetadataProvider};
use burn::tensor::Device;
use ai_models::face_detector::FaceDetector;
use ai_models::face_embedder::FaceEmbedder;
use image::DynamicImage;
use log::error;
use serde::{Deserialize, Serialize};
use burn::prelude::Backend;
use surrealdb::{Connection, Surreal};
use surrealdb::types::SurrealValue;

pub struct FaceRecognitionMetadataProvider<B: Backend> {
    face_detector: FaceDetector<B>,
    face_embedder: FaceEmbedder<B>,
}

impl <B>FaceRecognitionMetadataProvider<B> where B: Backend {
    pub fn new(device: Device<B>, face_detector: &str, face_embedder: &str) -> Self {
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
impl SurrealValue for FaceInPicture {
    fn kind_of() -> ::surrealdb::types::Kind {
        {
            let mut map = std::collections::BTreeMap::new();
            map.insert("top_left_x".to_string(), <f32 as SurrealValue>::kind_of());
            map.insert("top_left_y".to_string(), <f32 as SurrealValue>::kind_of());
            map.insert("bottom_right_x".to_string(), <f32 as SurrealValue>::kind_of());
            map.insert("bottom_right_y".to_string(), <f32 as SurrealValue>::kind_of());
            map.insert("confidence".to_string(), <f32 as SurrealValue>::kind_of());
            ::surrealdb::types::Kind::Literal(::surrealdb::types::KindLiteral::Object(map))
        }
    }
    fn is_value(value: &::surrealdb::types::Value) -> bool {
        if let ::surrealdb::types::Value::Object(map) = value {
            {
                let mut valid = true;
                if valid { if let Some(v) = map.get("top_left_x") { if !<f32 as SurrealValue>::is_value(v) { valid = false; } } else { valid = false; } }
                if valid { if let Some(v) = map.get("top_left_y") { if !<f32 as SurrealValue>::is_value(v) { valid = false; } } else { valid = false; } }
                if valid { if let Some(v) = map.get("bottom_right_x") { if !<f32 as SurrealValue>::is_value(v) { valid = false; } } else { valid = false; } }
                if valid { if let Some(v) = map.get("bottom_right_y") { if !<f32 as SurrealValue>::is_value(v) { valid = false; } } else { valid = false; } }
                if valid { if let Some(v) = map.get("confidence") { if !<f32 as SurrealValue>::is_value(v) { valid = false; } } else { valid = false; } }
                if valid { return true; }
            }
        }
        false
    }
    fn into_value(self) -> ::surrealdb::types::Value {
        let Self { top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence, face: _ } = self;
        {
            let mut map = ::surrealdb::types::Object::new();
            map.insert("top_left_x".to_string(), top_left_x.into_value());
            map.insert("top_left_y".to_string(), top_left_y.into_value());
            map.insert("bottom_right_x".to_string(), bottom_right_x.into_value());
            map.insert("bottom_right_y".to_string(), bottom_right_y.into_value());
            map.insert("confidence".to_string(), confidence.into_value());
            ::surrealdb::types::Value::Object(map)
        }
    }
    fn from_value(value: ::surrealdb::types::Value) -> std::result::Result<Self, ::surrealdb::types::Error> {
        if let ::surrealdb::types::Value::Object(mut map) = value {
            {
                let field_value = map.remove("top_left_x").unwrap_or_default();
                let top_left_x = <f32 as SurrealValue>::from_value(field_value).map_err(|e| ::surrealdb::types::Error::internal(format!("Failed to deserialize field '{}' on type '{}': {}", "top_left_x", "FaceInPicture", e)))?;
                let field_value = map.remove("top_left_y").unwrap_or_default();
                let top_left_y = <f32 as SurrealValue>::from_value(field_value).map_err(|e| ::surrealdb::types::Error::internal(format!("Failed to deserialize field '{}' on type '{}': {}", "top_left_y", "FaceInPicture", e)))?;
                let field_value = map.remove("bottom_right_x").unwrap_or_default();
                let bottom_right_x = <f32 as SurrealValue>::from_value(field_value).map_err(|e| ::surrealdb::types::Error::internal(format!("Failed to deserialize field '{}' on type '{}': {}", "bottom_right_x", "FaceInPicture", e)))?;
                let field_value = map.remove("bottom_right_y").unwrap_or_default();
                let bottom_right_y = <f32 as SurrealValue>::from_value(field_value).map_err(|e| ::surrealdb::types::Error::internal(format!("Failed to deserialize field '{}' on type '{}': {}", "bottom_right_y", "FaceInPicture", e)))?;
                let field_value = map.remove("confidence").unwrap_or_default();
                let confidence = <f32 as SurrealValue>::from_value(field_value).map_err(|e| ::surrealdb::types::Error::internal(format!("Failed to deserialize field '{}' on type '{}': {}", "confidence", "FaceInPicture", e)))?;
                Ok(Self { top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence, face: None })
            }
        } else {
            let err = ::surrealdb::types::ConversionError::from_value(::surrealdb::types::Kind::Object, &value);
            Err(err.into())
        }
    }
}


#[derive(Debug, Serialize, SurrealValue, Deserialize, Clone)]
pub struct FaceInPictureVector {
    pub embedding: Vec<f32>,
}

static FACE_IN_PICTURE_DATA_NAME: &str = "face_in_picture";
static FACE_IN_PICTURE_RELATION_NAME: &str = "has_face_in_picture";
static FACE_IN_PICTURE_VECTOR_DATA_NAME: &str = "face_in_picture_vector";
static FACE_IN_PICTURE_VECTOR_RELATION_NAME: &str = "has_face_in_picture_vector";

impl<B: Backend> MetadataProvider<BaseImageWithImage, FaceInPicture> for FaceRecognitionMetadataProvider<B> {
    fn extract(
        &self,
        base_images: &[BaseImageWithImage],
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
impl<B: Backend> MetadataProvider<Metadata<FaceInPicture>, FaceInPictureVector>
    for FaceRecognitionMetadataProvider<B>
{
    fn extract(
        &self,
        face_in_picture: &[Metadata<FaceInPicture>],
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
                metadata: Some(FaceInPictureVector { embedding }),
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

            DEFINE INDEX IF NOT EXISTS {FACE_IN_PICTURE_VECTOR_DATA_NAME}_base_unique
            ON {FACE_IN_PICTURE_VECTOR_DATA_NAME}
            FIELDS base
            UNIQUE;

            DEFINE INDEX IF NOT EXISTS {FACE_IN_PICTURE_VECTOR_DATA_NAME}_hnsw
            ON {FACE_IN_PICTURE_VECTOR_DATA_NAME}
            FIELDS embedding
            HNSW
            DIMENSION 512
            DIST COSINE;
            "#
        ))
        .await?;
        Ok(())
    }
    pub async fn rebuild_index(&self) -> anyhow::Result<()> {
        self.db
            .query(format!(
                r#"
            REBUILD INDEX IF EXISTS {FACE_IN_PICTURE_VECTOR_DATA_NAME}_hnsw
            ON {FACE_IN_PICTURE_VECTOR_DATA_NAME};
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

                LET $id = $tmp[0].id;
                RELATE $base -> {FACE_IN_PICTURE_RELATION_NAME} -> $id;

                $tmp[0];
                "#))
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
                .embedding;

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
