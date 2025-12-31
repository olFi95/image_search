use crate::metadata_provider::face_recognition_metadata_provider::FaceInPicture;
use crate::metadata_provider::metadata_provider::{Metadata, MetadataProvider};
use face_detection::face_age_and_gender_estimator::FaceAgeAndGenderEstimator;
use log::error;
use serde::{Deserialize, Serialize};
use surrealdb::engine::remote::ws::Client;
use surrealdb::Surreal;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FaceAgeAndGender {
    pub gender: f32,
    pub age: f32,
}

pub struct AgeAndGenderMetadataProvider {
    face_age_and_gender_estimator: FaceAgeAndGenderEstimator,
}

impl AgeAndGenderMetadataProvider {
    pub fn new(
        device: std::sync::Arc<Box<burn::tensor::Device<burn_wgpu::Wgpu>>>,
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

impl MetadataProvider<Metadata<FaceInPicture>, FaceAgeAndGender> for AgeAndGenderMetadataProvider {
    fn extract(
        &self,
        face_in_picture: &Vec<Metadata<FaceInPicture>>,
    ) -> anyhow::Result<Vec<Metadata<FaceAgeAndGender>>> {
        let mut results: Vec<Metadata<FaceAgeAndGender>> = vec![];
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
            let embedding = self.face_age_and_gender_estimator.embed(face);
            results.push(Metadata {
                id: None,
                metadata: Some(FaceAgeAndGender {
                    age: embedding[0],
                    gender: embedding[1],
                }),
                base: Some(face_in_picture_id),
            });
        }
        Ok(results)
    }
}

pub struct FaceAgeAndGenderMetadataRepository {
    db: Surreal<Client>,
}

static FACE_AGE_AND_GENDER_DATA_NAME: &str = "face_age_and_gender_estimation";
static FACE_AGE_AND_GENDER_RELATION_NAME: &str = "has_face_age_and_gender_estimation";
impl FaceAgeAndGenderMetadataRepository {
    pub async fn new(db: Surreal<Client>) -> Self {
        Self::prepare_repository(&db)
            .await
            .expect("cannot prepare repository with indexes");
        Self { db }
    }
    async fn prepare_repository(db: &Surreal<Client>) -> anyhow::Result<()> {
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
                        gender = $gender
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
