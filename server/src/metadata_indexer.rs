use std::mem::take;
use std::path::PathBuf;
use std::sync::Arc;
use burn::tensor::Device;
use burn_wgpu::Wgpu;
use log::Level::Info;
use log::{info, trace};
use rayon::iter::IntoParallelRefIterator;
use surrealdb::engine::remote::ws::Client;
use surrealdb::Surreal;
use crate::clip::get_all_directories_in_dir;
use crate::metadata_provider::image_hash_metadata_provider::{ImageHashMetadataProvider, ImageHashMetadataRepository};
use crate::metadata_provider::metadata_provider::{BaseImage, BaseImageRepository, MetadataProvider};
use rayon::iter::ParallelIterator;
use face_detection::face_detector::FaceDetector;
use face_detection::face_embedder::FaceEmbedder;
use crate::metadata_provider::basic_metadata_provider::{BasicMetadataProvider, BasicMetadataRepository};
use crate::metadata_provider::face_recognition_metadata_provider::{FaceRecognitionMetadataProvider, FaceRecognitionMetadataRepository};
use crate::metadata_provider::image_embedding_metadata_provider::{ImageEmbeddingMetadataProvider, ImageEmbeddingMetadataRepository};

pub struct MetadataIndexer {
    db: Surreal<Client>,
    device: Arc<Box<Device<Wgpu>>>,
    face_detector: String,
    face_embedder: String,
    image_embedder: String,
}

impl MetadataIndexer {
    pub fn new(db: Surreal<Client>, device: Arc<Box<Device<Wgpu>>>, face_embedder: String, face_detector: String, image_embedder: String) -> Self {
        MetadataIndexer { db, device, face_embedder, face_detector, image_embedder }
    }

    pub async fn index_metadata(&self, path: PathBuf) -> anyhow::Result<()> {
        // Metadata Provider
        let image_hash_metadata_provider = ImageHashMetadataProvider;
        let basic_metadata_provider = BasicMetadataProvider;
        let face_recognition_metadata_provider = FaceRecognitionMetadataProvider::new(
            self.device.clone(), self.face_detector.as_str(), self.face_embedder.as_str()
        );
        let image_embedding_metadata_provider = ImageEmbeddingMetadataProvider::new(self.device.clone(), self.image_embedder.as_str());
        // Metadata Repositories
        let image_hash_metadata_repository = ImageHashMetadataRepository::new(self.db.clone()).await;
        let basic_metadata_repository = BasicMetadataRepository::new(self.db.clone()).await;
        let face_recognition_metadata_repository = FaceRecognitionMetadataRepository::new(self.db.clone()).await;
        let image_embedding_metadata_repository = ImageEmbeddingMetadataRepository::new(self.db.clone()).await;

        let base_image_repository = BaseImageRepository::new(self.db.clone()).await;
        let all_image_paths: Vec<PathBuf> = get_all_directories_in_dir(&path)
            .par_iter()
            .map(PathBuf::from)
            .collect();
        let chunk_size = 2;
        for image_paths in all_image_paths.chunks(chunk_size) {
            // Convert Path Strings into PathBufs and then into BaseImages
            trace!("converting {chunk_size} paths to base_images");
            let mut base_images: Vec<BaseImage> = image_paths.par_iter().map(|path| BaseImage::new(PathBuf::from(path))).collect();
            trace!("Loading inserting {} base_images", &base_images.len());

            // Save BaseImages to the repository
            base_images = base_image_repository.insert_many(base_images).await.expect("Inserting base image failed");

            // Now actually load the images. Only continue if the image could be loaded.
            trace!("loading images into {} base_images", &base_images.len());
            let base_images_with_image: Vec<_> = base_images.par_iter().cloned()
                .map(|bi| bi.try_into())
                .filter(|biwi| biwi.is_ok())
                .map(|biwi| biwi.unwrap())
                .collect();

            // Calculate hashes of the image data.
            trace!("calculating hashes for {} base_images", &base_images_with_image.len());
            let hashes = image_hash_metadata_provider.extract(&base_images_with_image).expect("cannot extract hashes");

            // Save hashes to the repository
            trace!("saving {} hashes", &hashes.len());
            let _ = image_hash_metadata_repository.insert_many(&hashes).await.expect("could not save hashes");

            // Read basic metadata of the image data.
            trace!("read basic metadata for {} base_images", &base_images_with_image.len());
            let basic_metadata = basic_metadata_provider.extract(&base_images_with_image).expect("cannot extract basic metadata");

            // Save basic metadata to the repository
            trace!("saving {} hashes", &basic_metadata.len());
            basic_metadata_repository.insert_many(&basic_metadata).await.expect("could not save hashes");

            // Run face recognition on images.
            trace!("run face recognition on {} images", &base_images_with_image.len());
            let faces = face_recognition_metadata_provider.extract(&base_images_with_image).expect("cannot extract face recognition metadata");
            // Save face recognition metadata to the repository.
            trace!("saving {} face recognition metadata entries", &faces.len());
            let faces = face_recognition_metadata_repository.insert_many_face_in_picture(&faces).await.expect("cannot save discovered faces to database.");

            // Embed discovered faces.
            trace!("embedding {} faces that were found in the images.", &faces.len());
            let face_embeddings = face_recognition_metadata_provider.extract(&faces).expect("cannot embed faces");
            trace!("saving {} face embeddings", &face_embeddings.len());
            face_recognition_metadata_repository.insert_many_face_embeddings(&face_embeddings).await.expect("cannot save face embeddings");

            // Embed full images.
            trace!("embedding {} images", &base_images_with_image.len());
            let image_embeddings = image_embedding_metadata_provider.extract(&base_images_with_image).expect("cannot embed images");
            trace!("saving {} image embeddings", &image_embeddings.len());
            image_embedding_metadata_repository.insert_many_image_embeddings(&image_embeddings).await.expect("cannot save image embeddings");

        }
        info!("Finished indexing metadata for images in {}", path.to_str().unwrap_or("provided path"));
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_metadata_indexer() {


    }
}