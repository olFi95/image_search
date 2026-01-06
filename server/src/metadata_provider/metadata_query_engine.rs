use serde::{Deserialize, Deserializer, Serialize};
use surrealdb::{Connection, RecordId, Surreal};
use crate::metadata_provider::age_and_gender_metadata_provider::FaceAgeAndGender;
use crate::metadata_provider::basic_metadata_provider::BasicMetadata;
use crate::metadata_provider::face_recognition_metadata_provider::{FaceInPicture, FaceInPictureVector};
use crate::metadata_provider::image_embedding_metadata_provider::ImageEmbedding;
use crate::metadata_provider::image_hash_metadata_provider::ImageHashMetadata;
use crate::metadata_provider::metadata_provider::BaseImage;

pub struct MetadataQueryEngine<C: Connection> {
    db: Surreal<C>
}

fn vec_f64_to_f32<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let v = Vec::<f64>::deserialize(deserializer)?;
    Ok(v.into_iter().map(|x| x as f32).collect())
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FaceInPictureWithMetadata {
    pub top_left_x: f32,
    pub top_left_y: f32,
    pub bottom_right_x: f32,
    pub bottom_right_y: f32,
    pub confidence: f32,
    pub embedding: Vec<FaceInPictureVector>,
    pub age_and_gender: Vec<FaceAgeAndGender>,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BaseImageWithMetadata {
    pub id: Option<RecordId>,
    pub path: String,
    pub basic_metadata: Vec<BasicMetadata>,
    pub faces: Vec<FaceInPictureWithMetadata>,
    pub image_embedding: Vec<ImageEmbedding>,
    pub image_hash: Vec<ImageHashMetadata>,
}

impl <C: Connection> MetadataQueryEngine<C> {
    pub fn new(db: Surreal<C>) -> Self {
        Self { db }
    }
    pub async fn get_all_metadata_attached_to_base_image(&self, base_image: &BaseImage) -> anyhow::Result<BaseImageWithMetadata> {
        if base_image.id.is_none() {
            return Err(anyhow::anyhow!("BaseImage ID is None"));
        }
        let base_image_id = base_image.id.clone().unwrap();
        let mut response = self.db.query(r#"
        SELECT *,
            (
                SELECT
                    *,
                    ->has_face_in_picture_vector->face_in_picture_vector.* AS embedding,
                    ->has_face_age_and_gender_estimation->face_age_and_gender_estimation.* AS age_and_gender
                FROM ->has_face_in_picture->face_in_picture
                ORDER BY top_left_x
            ) AS faces,
            ->has_basic_metadata->basic_metadata.* AS basic_metadata,
            ->has_image_embedding_vector->image_embedding_vector.* AS image_embedding,
            ->has_image_hash_metadata->image_hash_metadata.* AS image_hash
        FROM base_image
        WHERE id = $id
        LIMIT 1;
        "#)
            .bind(("id", base_image_id)).await.expect("cannot query all metadata for base image");
        let result = response.take::<Vec<BaseImageWithMetadata>>(0).expect("Error reading metadata for base_image");
        if result.len()!=1 {
            Err(anyhow::anyhow!("Expected exactly one BaseImageWithMetadata, got {}", result.len()))
        } else {
            Ok(result[0].clone())
        }
    }

}