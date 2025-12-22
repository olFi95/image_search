use std::mem::take;
use std::path::PathBuf;
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

        for image_paths in all_image_paths.chunks(50) {
            let mut base_images: Vec<BaseImage> = image_paths.par_iter().map(|path| BaseImage::new(PathBuf::from(path))).collect();
            base_images = base_image_repository.insert_many(base_images).await.expect("Inserting base image failed");
            let base_images_with_image: Vec<_> = base_images.par_iter().cloned().map(|bi| bi.into()).collect();
            let hashes = image_hash_metadata_provider.extract(&base_images_with_image).expect("cannot extract hashes");
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