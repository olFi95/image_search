use std::marker::PhantomData;
use burn::prelude::Backend;
use log::{debug, info, trace};
use serde::Serialize;
use surrealdb::Surreal;
use surrealdb::engine::any::Any;
use surrealdb_types::SurrealValue;
use data::{FaceBoundingBox, FacesRequest, FacesResponse, ImageReference, NumberOfImagesResponse, SearchParams, SearchResponse};
use crate::DbImage;
use crate::metadata_provider::metadata_query_engine::MetadataQueryEngine;
use crate::metadata_provider::model::BaseImage;
use crate::search::ImageType;
use anyhow::{Context, Result};

#[derive(Clone)]
pub struct QueryService<B: Backend>{
    _marker: PhantomData<B>,
}

impl<B: Backend> QueryService<B> {

    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    pub async fn search_text(
        &self,
        db: &Surreal<Any>,
        text_embedding: Vec<f32>,
        params: SearchParams,
        media_dir_str: &str,
    ) -> Result<SearchResponse> {
        debug!("Handle Search with params: {:?}", params);

        let mut query_vector = text_embedding;

        info!("image_paths: {:?}", params.referenced_images);

        if !params.referenced_images.is_empty() {
            let image_paths: Vec<String> = params
                .referenced_images
                .into_iter()
                .filter(|img| img.starts_with("media/"))
                .map(|img| img.replacen("media/", media_dir_str, 1))
                .collect::<Vec<String>>();
            trace!("image_paths: {image_paths:?}");

            let mut marked_image_embeddings_response = db
                .query(
                    r#"
                    SELECT
                        id,
                        path AS image_path,
                        ->has_image_embedding_vector->image_embedding_vector[0].embedding AS embedding
                    FROM base_image
                    WHERE path IN $image_paths
                "#,
                )
                .bind(("image_paths", image_paths))
                .await
                .context("DB query error")?;
            let marked_image: Vec<ImageType> =
                marked_image_embeddings_response.take(0)
                    .context("Failed to deserialize response")?;
            debug!("marked_image_embeddings {}", marked_image.len());
            if !marked_image.is_empty() {
                let slices = marked_image
                    .iter()
                    .map(|embedding| &embedding.embedding)
                    .collect::<Vec<&Vec<f32>>>();
                let selected_images_average = Self::average_slices(&slices);
                query_vector = Self::average_slices(&vec![&selected_images_average, &query_vector]);
            }
        }

        let query = r#"
        LET $similar_vectors = (
            SELECT
                id,
                vector::distance::knn() AS similarity
            FROM image_embedding_vector
            WHERE embedding <|500, 150|> $reference
            ORDER BY similarity ASC
        );


        SELECT
            similarity,
            <-has_image_embedding_vector<-base_image[0].id AS id,
            <-has_image_embedding_vector<-base_image[0].path AS image_path
        FROM $similar_vectors;
    "#;

        let mut response = db
            .query(query)
            .bind(("reference", query_vector))
            .await
            .context("DB query error")?;

        let db_images: Vec<DbImage> = response.take(1)
            .context("Failed to deserialize response")?;

        let images: Vec<ImageReference> = db_images
            .into_iter()
            .map(|img| ImageReference {
                id: img.id_string(),
                image_path: img.image_path.replace(media_dir_str, "media/"),
            })
            .collect();

        Ok(SearchResponse { images })
    }

    pub async fn get_faces(
        &self,
        db: &Surreal<Any>,
        params: FacesRequest,
        media_dir_str: &str,
    ) -> Result<FacesResponse> {
        debug!("Handle get_faces for image: {:?}", params.image_path);

        let absolute_path = if params.image_path.starts_with("media/") {
            params.image_path.replacen("media/", media_dir_str, 1)
        } else {
            params.image_path.clone()
        };

        debug!("Resolved absolute_path for faces: {:?}", absolute_path);


        let mut response = db
            .query("SELECT * FROM base_image WHERE path = $path")
            .bind(("path", absolute_path))
            .await
            .context("DB query error")?;

        let base_images: Vec<BaseImage> = response.take(0)
            .context("Failed to deserialize base image")?;

        let base_image = match base_images.first() {
            Some(img) => img,
            None => {
                debug!("No base image found for path: {:?}", params.image_path);
                return Ok(FacesResponse { faces: vec![] });
            }
        };

        let query_engine = MetadataQueryEngine::new(db.clone());
        let metadata = query_engine
            .get_all_metadata_attached_to_base_image(base_image)
            .await
            .context("Failed to get metadata")?;

        let faces: Vec<FaceBoundingBox> = metadata
            .faces
            .into_iter()
            .map(|face| {
                let age_gender = face.age_and_gender.first();
                FaceBoundingBox {
                    top_left_x: face.top_left_x,
                    top_left_y: face.top_left_y,
                    bottom_right_x: face.bottom_right_x,
                    bottom_right_y: face.bottom_right_y,
                    confidence: face.confidence,
                    age: age_gender.map(|ag| ag.age),
                    gender: age_gender.map(|ag| ag.gender),
                }
            })
            .collect();

        Ok(FacesResponse { faces })
    }

    pub async fn get_number_of_images(
        &self,
        db: &Surreal<Any>,
    ) -> Result<NumberOfImagesResponse> {

        let mut response = db
            .query("SELECT count() AS total FROM base_image GROUP ALL;")
            .await
            .context("DB query error")?;

        let rows: Vec<NumOfImagesResult> = response.take(0)
            .context("Failed to deserialize response")?;

        let total = rows.first().map(|r| r.total).unwrap_or(0);

        Ok(NumberOfImagesResponse { images: total })
    }

    fn average_slices(vectors: &Vec<&Vec<f32>>) -> Vec<f32> {
        assert!(!vectors.is_empty(), "Input must not be empty");

        let len = vectors[0].len();
        assert!(
            vectors.iter().all(|v| v.len() == len),
            "All vectors must have the same length"
        );

        let mut result = vec![0.0; len];

        for v in vectors {
            for (i, val) in v.iter().enumerate() {
                result[i] += val;
            }
        }

        let count = vectors.len() as f32;
        for val in &mut result {
            *val /= count;
        }

        result
    }
}

#[derive(Debug, Serialize, SurrealValue)]
struct NumOfImagesResult {
    total: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tes_average_vector() {
        let a = vec![1.0, 2.0, 4.0, 4.0, 10.0];
        let b = vec![1.0, 1.0, 2.0, 4.0, 0.0];
        let result = QueryService::average_slices(&vec![&a, &b]);
        assert_eq!(result, vec![1.0, 1.5, 3.0, 4.0, 5.0]);
    }
}
