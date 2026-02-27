use burn::prelude::{Backend, Device};
use crate::metadata_provider::face_recognition_metadata_provider::FaceInPicture;
use crate::metadata_provider::model::{Metadata, MetadataProvider};
use ai_models::face_age_and_gender_estimator::FaceAgeAndGenderEstimator;
use log::error;
use serde::{Deserialize, Serialize};
use surrealdb::{Connection, Surreal};
use surrealdb::types::SurrealValue;

#[derive(Debug, Serialize, SurrealValue, Deserialize, Clone)]
pub struct FaceAgeAndGender {
    pub gender: f32,
    pub age: f32,
}

pub struct AgeAndGenderMetadataProvider<B: Backend> {
    face_age_and_gender_estimator: FaceAgeAndGenderEstimator<B>,
}

impl<B: Backend> AgeAndGenderMetadataProvider<B> {
    pub fn new(
        device: Device<B>,
        age_and_gender_model: &str,
    ) -> Self {
        Self {
            face_age_and_gender_estimator: FaceAgeAndGenderEstimator::new(
                age_and_gender_model,
                device,
            ),
        }
    }
}

impl<B: Backend> MetadataProvider<Metadata<FaceInPicture>, FaceAgeAndGender> for AgeAndGenderMetadataProvider<B> {
    fn extract(
        &self,
        face_in_picture: &[Metadata<FaceInPicture>],
    ) -> anyhow::Result<Vec<Metadata<FaceAgeAndGender>>> {
        if face_in_picture.is_empty() {
            return Ok(Vec::new());
        }

        // Collect valid faces and their IDs
        let mut valid_face_images: Vec<image::DynamicImage> = Vec::new();
        let mut valid_ids: Vec<surrealdb::types::RecordId> = Vec::new();

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
                None => {
                    error!("FaceInPicture ID is missing");
                    continue;
                }
                Some(id) => id,
            };
            valid_face_images.push(face);
            valid_ids.push(id);
        }

        if valid_face_images.is_empty() {
            return Ok(Vec::new());
        }

        let face_refs: Vec<&image::DynamicImage> = valid_face_images.iter().collect();
        let batch_results = self.face_age_and_gender_estimator.embed_batch(&face_refs);

        let results: Vec<Metadata<FaceAgeAndGender>> = valid_ids
            .into_iter()
            .zip(batch_results.into_iter())
            .map(|(id, embedding)| Metadata {
                id: None,
                metadata: Some(FaceAgeAndGender {
                    age: embedding[0],
                    gender: embedding[1],
                }),
                base: Some(id),
            })
            .collect();

        Ok(results)
    }
}

pub struct FaceAgeAndGenderMetadataRepository<C: Connection> {
    db: Surreal<C>,
}
static FACE_AGE_AND_GENDER_DATA_NAME: &str = "face_age_and_gender_estimation";
static FACE_AGE_AND_GENDER_RELATION_NAME: &str = "has_face_age_and_gender_estimation";
impl<C: Connection> FaceAgeAndGenderMetadataRepository<C> {
    pub async fn new(db: Surreal<C>) -> Self {
        Self::prepare_repository(&db)
            .await
            .expect("cannot prepare repository with indexes");
        Self { db }
    }
    async fn prepare_repository(db: &Surreal<C>) -> anyhow::Result<()> {
        db.query(format!(
            r#"
            DEFINE INDEX IF NOT EXISTS {FACE_AGE_AND_GENDER_DATA_NAME}_base_unique
            ON {FACE_AGE_AND_GENDER_DATA_NAME}
            FIELDS base
            UNIQUE;
            "#
        ))
        .await?;
        Ok(())
    }
    pub async fn insert_many_age_and_gender(
        &self,
        items: &Vec<Metadata<FaceAgeAndGender>>,
    ) -> anyhow::Result<Vec<Metadata<FaceAgeAndGender>>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let mut inserted = Vec::new();

        for item in items {
            let base_id = item
                .base
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Base ID missing"))?;
            let age_and_gender = item
                .metadata
                .clone()
                .ok_or(anyhow::anyhow!("Metadata missing"))?;

            let mut response = self
                .db
                .query(format!(
                    r#"
                LET $tmp = (
                    UPSERT {FACE_AGE_AND_GENDER_DATA_NAME}
                    SET age = $age,
                        gender = $gender,
                        base = $base
                    WHERE base = $base
                );
                LET $id = $tmp[0].id;
                RELATE $base-> {FACE_AGE_AND_GENDER_RELATION_NAME} -> $id;
                "#
                ))
                .bind(("base", base_id.clone()))
                .bind(("age", age_and_gender.age))
                .bind(("gender", age_and_gender.gender))
                .await?;

            if let Ok(mut rows) = response.take::<Vec<Metadata<FaceAgeAndGender>>>(0) {
                inserted.append(&mut rows);
            }
        }

        Ok(inserted)
    }
}
