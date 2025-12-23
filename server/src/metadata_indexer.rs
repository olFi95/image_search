use std::mem::take;
use std::path::PathBuf;
use log::trace;
use rayon::iter::IntoParallelRefIterator;
use surrealdb::engine::remote::ws::Client;
use surrealdb::Surreal;
use crate::clip::get_all_directories_in_dir;
use crate::metadata_provider::image_hash_metadata_provider::{ImageHashMetadataProvider, ImageHashMetadataRepository};
use crate::metadata_provider::metadata_provider::{BaseImage, BaseImageRepository, MetadataProvider};
use rayon::iter::ParallelIterator;
pub struct MetadataIndexer {
    db: Surreal<Client>
}

impl MetadataIndexer {
    pub fn new(db: Surreal<Client>) -> Self {
        MetadataIndexer { db }
    }

    pub async fn index_metadata(&self, path: PathBuf) -> anyhow::Result<()> {
        let image_hash_metadata_provider = ImageHashMetadataProvider;
        let image_hash_metadata_repository = ImageHashMetadataRepository::new(self.db.clone()).await;
        let base_image_repository = BaseImageRepository::new(self.db.clone()).await;
        let all_image_paths: Vec<PathBuf> = get_all_directories_in_dir(&path)
            .par_iter()
            .map(PathBuf::from)
            .collect();
        let chunk_size = 100;
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
            image_hash_metadata_repository.insert_many(hashes).await.expect("could not save hashes");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_metadata_indexer() {


    }
}