use crate::metadata_provider::model::{BaseImageWithImage, Metadata, MetadataProvider};
use burn::tensor::Device;
use ai_models::face_detector::FaceDetector;
use ai_models::face_embedder::FaceEmbedder;
use image::DynamicImage;
use log::error;
use serde::{Deserialize, Serialize};
use burn::prelude::Backend;
use surrealdb::{Connection, Surreal};
use surrealdb::types::{RecordId, SurrealValue};

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
        if base_images.is_empty() {
            return Ok(Vec::new());
        }

        let image_refs: Vec<&image::DynamicImage> = base_images.iter().map(|bi| &bi.image).collect();
        let batch_results = self.face_detector.detect_batch(&image_refs);

        let mut results: Vec<Metadata<FaceInPicture>> = vec![];
        for (base_image, detected_faces) in base_images.iter().zip(batch_results.into_iter()) {
            let image_height = base_image.image.height() as f32;
            let image_width = base_image.image.width() as f32;
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
        if face_in_picture.is_empty() {
            return Ok(Vec::new());
        }

        // Collect valid faces and their indices
        let mut valid_faces: Vec<DynamicImage> = Vec::new();
        let mut valid_ids: Vec<RecordId> = Vec::new();

        for fip_metadata in face_in_picture {
            let fip = fip_metadata.clone();
            let metadata = match fip.metadata {
                Some(m) => m,
                None => {
                    error!("FaceInPicture metadata is missing for ID {:?}", fip.id);
                    continue;
                }
            };
            let face = match metadata.face {
                Some(f) => f,
                None => {
                    error!("Face image is missing in FaceInPicture metadata for ID {:?}", fip.id);
                    continue;
                }
            };
            let id = match fip.id {
                Some(id) => id,
                None => {
                    error!("FaceInPicture ID is missing");
                    continue;
                }
            };
            valid_faces.push(face);
            valid_ids.push(id);
        }

        if valid_faces.is_empty() {
            return Ok(Vec::new());
        }

        let embeddings = self.face_embedder.embed_batch(&valid_faces);

        let results: Vec<Metadata<FaceInPictureVector>> = valid_ids
            .into_iter()
            .zip(embeddings.into_iter())
            .map(|(id, embedding)| Metadata {
                id: None,
                metadata: Some(FaceInPictureVector { embedding }),
                base: Some(id),
            })
            .collect();

        Ok(results)
    }
}

pub struct FaceRecognitionMetadataRepository<C: Connection> {
    db: Surreal<C>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata_provider::model::{BaseImage, BaseImageRepository, Metadata};
    use crate::metadata_provider::metadata_query_engine::MetadataQueryEngine;

    #[tokio::test]
    async fn test_insert_and_read_single_face() {
        use surrealdb::engine::local::Mem;
        use surrealdb::Surreal;

        let db = Surreal::new::<Mem>(()).await.unwrap();
        db.use_ns("test").use_db("test").await.unwrap();

        // Setup repositories
        let base_image_repo = BaseImageRepository::new(db.clone()).await;
        let face_repo = FaceRecognitionMetadataRepository::new(db.clone()).await;
        let query_engine = MetadataQueryEngine::new(db.clone());

        // Insert a base image
        let base_images = base_image_repo
            .insert_many(vec![BaseImage { id: None, path: "/test/1_1.jpg".to_string() }])
            .await
            .expect("insert base image");
        assert_eq!(base_images.len(), 1);
        let base_image = &base_images[0];
        assert!(base_image.id.is_some());

        // Insert one face for this base image
        let faces = vec![Metadata {
            id: base_image.id.clone(),
            metadata: Some(FaceInPicture {
                top_left_x: 0.37,
                top_left_y: 0.12,
                bottom_right_x: 0.56,
                bottom_right_y: 0.31,
                confidence: 0.65,
                face: None,
            }),
            base: base_image.id.clone(),
        }];

        let inserted = face_repo
            .insert_many_face_in_picture(&faces)
            .await
            .expect("insert face");
        assert_eq!(inserted.len(), 1, "Expected 1 inserted face, got {}", inserted.len());

        // Read back via query engine
        let metadata = query_engine
            .get_all_metadata_attached_to_base_image(base_image)
            .await
            .expect("query metadata");
        assert_eq!(metadata.faces.len(), 1, "Expected 1 face from query, got {}", metadata.faces.len());
        assert!((metadata.faces[0].confidence - 0.65).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_insert_and_read_multiple_faces() {
        use surrealdb::engine::local::Mem;
        use surrealdb::Surreal;

        let db = Surreal::new::<Mem>(()).await.unwrap();
        db.use_ns("test").use_db("test").await.unwrap();

        let base_image_repo = BaseImageRepository::new(db.clone()).await;
        let face_repo = FaceRecognitionMetadataRepository::new(db.clone()).await;
        let query_engine = MetadataQueryEngine::new(db.clone());

        // Insert a base image
        let base_images = base_image_repo
            .insert_many(vec![BaseImage { id: None, path: "/test/3_1.jpg".to_string() }])
            .await
            .expect("insert base image");
        let base_image = &base_images[0];

        // Insert 3 faces for this base image
        let faces = vec![
            Metadata {
                id: base_image.id.clone(),
                metadata: Some(FaceInPicture {
                    top_left_x: 0.1, top_left_y: 0.2,
                    bottom_right_x: 0.3, bottom_right_y: 0.4,
                    confidence: 0.9, face: None,
                }),
                base: base_image.id.clone(),
            },
            Metadata {
                id: base_image.id.clone(),
                metadata: Some(FaceInPicture {
                    top_left_x: 0.4, top_left_y: 0.2,
                    bottom_right_x: 0.6, bottom_right_y: 0.4,
                    confidence: 0.85, face: None,
                }),
                base: base_image.id.clone(),
            },
            Metadata {
                id: base_image.id.clone(),
                metadata: Some(FaceInPicture {
                    top_left_x: 0.7, top_left_y: 0.2,
                    bottom_right_x: 0.9, bottom_right_y: 0.4,
                    confidence: 0.8, face: None,
                }),
                base: base_image.id.clone(),
            },
        ];

        let inserted = face_repo
            .insert_many_face_in_picture(&faces)
            .await
            .expect("insert faces");
        assert_eq!(inserted.len(), 3, "Expected 3 inserted faces, got {}", inserted.len());

        // Read back via query engine
        let metadata = query_engine
            .get_all_metadata_attached_to_base_image(base_image)
            .await
            .expect("query metadata");
        assert_eq!(metadata.faces.len(), 3, "Expected 3 faces from query, got {}", metadata.faces.len());
    }
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
            DEFINE INDEX IF NOT EXISTS {FACE_IN_PICTURE_DATA_NAME}_base_idx
            ON {FACE_IN_PICTURE_DATA_NAME}
            FIELDS base;

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
                    CREATE {FACE_IN_PICTURE_DATA_NAME}
                    SET top_left_x = $top_left_x,
                        top_left_y = $top_left_y,
                        bottom_right_x = $bottom_right_x,
                        bottom_right_y = $bottom_right_y,
                        confidence = $confidence,
                        base = $base
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
                    SET embedding = $embedding,
                        base = $base
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
