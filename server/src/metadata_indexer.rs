use crate::clip::get_all_directories_in_dir;
use crate::metadata_provider::age_and_gender_metadata_provider::{
    AgeAndGenderMetadataProvider, FaceAgeAndGenderMetadataRepository,
};
use crate::metadata_provider::basic_metadata_provider::{
    BasicMetadataProvider, BasicMetadataRepository,
};
use crate::metadata_provider::face_recognition_metadata_provider::{
    FaceRecognitionMetadataProvider, FaceRecognitionMetadataRepository,
};
use crate::metadata_provider::image_embedding_metadata_provider::{
    ImageEmbeddingMetadataProvider, ImageEmbeddingMetadataRepository,
};
use crate::metadata_provider::image_hash_metadata_provider::{
    ImageHashMetadataProvider, ImageHashMetadataRepository,
};
use crate::metadata_provider::metadata_provider::{
    BaseImage, BaseImageRepository, MetadataProvider,
};
use burn::tensor::Device;
use burn_wgpu::Wgpu;
use log::{info, trace};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::path::PathBuf;
use std::sync::Arc;
use surrealdb::{Connection, Surreal};

pub struct MetadataIndexer<C> where C: Connection {
    db: Surreal<C>,
    device: Arc<Box<Device<Wgpu>>>,
    face_detector: String,
    face_embedder: String,
    face_age_and_gender: String,
    image_embedder: String,
}

impl <C>MetadataIndexer<C> where C: Connection {
    pub fn new(
        db: Surreal<C>,
        device: Arc<Box<Device<Wgpu>>>,
        face_embedder: String,
        face_detector: String,
        image_embedder: String,
        face_age_and_gender: String,
    ) -> Self {
        MetadataIndexer {
            db,
            device,
            face_embedder,
            face_detector,
            image_embedder,
            face_age_and_gender,
        }
    }

    pub async fn index_metadata(&self, path: PathBuf) -> anyhow::Result<()> {
        // Metadata Provider
        let image_hash_metadata_provider = ImageHashMetadataProvider;
        let basic_metadata_provider = BasicMetadataProvider;
        let face_recognition_metadata_provider = FaceRecognitionMetadataProvider::new(
            self.device.clone(),
            self.face_detector.as_str(),
            self.face_embedder.as_str(),
        );
        let face_age_and_gender_metadata_provider = AgeAndGenderMetadataProvider::new(
            self.device.clone(),
            self.face_age_and_gender.as_str(),
        );
        let image_embedding_metadata_provider =
            ImageEmbeddingMetadataProvider::new(self.device.clone(), self.image_embedder.as_str());
        // Metadata Repositories
        let image_hash_metadata_repository =
            ImageHashMetadataRepository::new(self.db.clone()).await;
        let basic_metadata_repository = BasicMetadataRepository::new(self.db.clone()).await;
        let face_recognition_metadata_repository =
            FaceRecognitionMetadataRepository::new(self.db.clone()).await;
        let face_age_and_gender_metadata_repository =
            FaceAgeAndGenderMetadataRepository::new(self.db.clone()).await;
        let image_embedding_metadata_repository =
            ImageEmbeddingMetadataRepository::new(self.db.clone()).await;

        let base_image_repository = BaseImageRepository::new(self.db.clone()).await;
        let all_image_paths: Vec<PathBuf> = get_all_directories_in_dir(&path)
            .par_iter()
            .map(PathBuf::from)
            .collect();
        let chunk_size = 25;
        for image_paths in all_image_paths.chunks(chunk_size) {
            // Convert Path Strings into PathBufs and then into BaseImages
            trace!("converting {chunk_size} paths to base_images");
            let mut base_images: Vec<BaseImage> = image_paths
                .par_iter()
                .map(|path| BaseImage::new(PathBuf::from(path)))
                .collect();
            trace!("Loading inserting {} base_images", &base_images.len());

            // Save BaseImages to the repository
            base_images = base_image_repository
                .insert_many(base_images)
                .await
                .expect("Inserting base image failed");

            // Now actually load the images. Only continue if the image could be loaded.
            trace!("loading images into {} base_images", &base_images.len());
            let base_images_with_image: Vec<_> = base_images
                .par_iter()
                .cloned()
                .map(|bi| bi.try_into())
                .filter(|biwi| biwi.is_ok())
                .map(|biwi| biwi.unwrap())
                .collect();

            // Calculate hashes of the image data.
            trace!(
                "calculating hashes for {} base_images",
                &base_images_with_image.len()
            );
            let hashes = image_hash_metadata_provider
                .extract(&base_images_with_image)
                .expect("cannot extract hashes");

            // Save hashes to the repository
            trace!("saving {} hashes", &hashes.len());
            let _ = image_hash_metadata_repository
                .insert_many(&hashes)
                .await
                .expect("could not save hashes");

            // Read basic metadata of the image data.
            trace!(
                "read basic metadata for {} base_images",
                &base_images_with_image.len()
            );
            let basic_metadata = basic_metadata_provider
                .extract(&base_images_with_image)
                .expect("cannot extract basic metadata");

            // Save basic metadata to the repository
            trace!("saving {} hashes", &basic_metadata.len());
            basic_metadata_repository
                .insert_many(&basic_metadata)
                .await
                .expect("could not save hashes");

            // Run face recognition on images.
            trace!(
                "run face recognition on {} images",
                &base_images_with_image.len()
            );
            let faces = face_recognition_metadata_provider
                .extract(&base_images_with_image)
                .expect("cannot extract face recognition metadata");
            // Save face recognition metadata to the repository.
            trace!("saving {} face recognition metadata entries", &faces.len());
            let faces = face_recognition_metadata_repository
                .insert_many_face_in_picture(&faces)
                .await
                .expect("cannot save discovered faces to database.");

            // Run Age and Gender estimation on discovered faces.
            trace!(
                "running age and gender estimation on {} faces",
                &faces.len()
            );
            let res = face_age_and_gender_metadata_provider
                .extract(&faces)
                .expect("cannot extract age and gender");
            trace!("saving age and gender of {} faces.", &res.len());
            face_age_and_gender_metadata_repository
                .insert_many_age_and_gender(&res)
                .await
                .expect("could not save age and gender metadata");

            // Embed discovered faces.
            trace!(
                "embedding {} faces that were found in the images.",
                &faces.len()
            );
            let face_embeddings = face_recognition_metadata_provider
                .extract(&faces)
                .expect("cannot embed faces");
            trace!("saving {} face embeddings", &face_embeddings.len());
            face_recognition_metadata_repository
                .insert_many_face_embeddings(&face_embeddings)
                .await
                .expect("cannot save face embeddings");

            // Embed full images.
            trace!("embedding {} images", &base_images_with_image.len());
            let image_embeddings = image_embedding_metadata_provider
                .extract(&base_images_with_image)
                .expect("cannot embed images");
            trace!("saving {} image embeddings", &image_embeddings.len());
            image_embedding_metadata_repository
                .insert_many_image_embeddings(&image_embeddings)
                .await
                .expect("cannot save image embeddings");
        }
        info!(
            "Finished indexing metadata for images in {}. Rebuilding indexes now.",
            path.to_str().unwrap_or("provided path")
        );
        image_embedding_metadata_repository
            .rebuild_index()
            .await
            .expect("cannot rebuild image embedding metadata index");
        info!(
            "Finished rebuilding indexes of metadata in {}.",
            path.to_str().unwrap_or("provided path")
        );
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::metadata_indexer::MetadataIndexer;
    use crate::metadata_provider::metadata_provider::BaseImageRepository;
    use crate::metadata_provider::metadata_query_engine::MetadataQueryEngine;
    use burn_wgpu::WgpuDevice;
    use std::path::PathBuf;
    use std::sync::Arc;
    use surrealdb::engine::local::Mem;
    use surrealdb::engine::remote::ws::Ws;
    use surrealdb::opt::auth::Root;
    use surrealdb::Surreal;

    #[test]
    fn embed_test_images() {
        std::thread::Builder::new()
            .name("embed-test".into())
            .stack_size(64 * 1024 * 1024) // 64 MB Stack
            .spawn(|| {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .thread_stack_size(32 * 1024 * 1024)
                    .build()
                    .unwrap();

                rt.block_on(async {
                    use std::path::PathBuf;
                    use std::sync::Arc;
                    use burn_wgpu::WgpuDevice;
                    use surrealdb::engine::local::Mem;
                    use surrealdb::Surreal;

                    let db = Surreal::new::<Mem>(()).await.unwrap();
                    db.use_ns("test").use_db("test").await.unwrap();

                    let metadata_indexer = MetadataIndexer::new(
                        db.clone(),
                        Arc::new(Box::new(WgpuDevice::DefaultDevice)),
                        "../models/arcface_model.bpk".to_string(),
                        "../models/yolo.bpk".to_string(),
                        "../models/vision_model.bpk".to_string(),
                        "../models/age_gender.bpk".to_string(),
                    );

                    metadata_indexer
                        .index_metadata(PathBuf::from("../test_pictures"))
                        .await
                        .expect("cannot use db");

                    let base_image_repository = BaseImageRepository::new(db.clone()).await;
                    let image_hash_metadata_repository =
                        super::ImageHashMetadataRepository::new(db.clone()).await;
                    let basic_metadata_repository =
                        super::BasicMetadataRepository::new(db.clone()).await;
                    let face_recognition_metadata_repository =
                        super::FaceRecognitionMetadataRepository::new(db.clone()).await;
                    let image_embedding_metadata_repository =
                        super::ImageEmbeddingMetadataRepository::new(db.clone()).await;
                    let age_and_gender_metadata_repository =
                        super::FaceAgeAndGenderMetadataRepository::new(db.clone()).await;
                    let metadata_query_engine = MetadataQueryEngine::new(db.clone());

                    // Image 0_1.jpg -> 0 People in there.
                    {
                        let base_image_0_1 = base_image_repository.get_base_image_by_path("../test_pictures/0_1.jpg").await;
                        assert!(base_image_0_1.is_some());
                        let base_image_0_1 = base_image_0_1.unwrap();

                        let base_image_0_1_metadata = metadata_query_engine.get_all_metadata_attached_to_base_image(&base_image_0_1).await.expect("cannot get metadata");

                        assert_eq!(base_image_0_1_metadata.path, "../test_pictures/0_1.jpg");

                        assert_eq!(hex::encode(base_image_0_1_metadata.image_hash[0].hash), "d78f6226b8b5bab6ba377b9de4f2d7172336a82688e288fbfa85533d73dcd3c6");
                        assert_eq!(base_image_0_1_metadata.image_hash[0].hash_type, "SHA256");

                        assert_eq!(base_image_0_1_metadata.faces.len(), 0);

                        assert_eq!(base_image_0_1_metadata.image_embedding[0].embedding.len(), 768);

                        assert_eq!(base_image_0_1_metadata.basic_metadata[0].height, 882);
                        assert_eq!(base_image_0_1_metadata.basic_metadata[0].width, 1280);
                        assert_eq!(base_image_0_1_metadata.basic_metadata[0].size_in_bytes, 138108);
                        assert_eq!(base_image_0_1_metadata.basic_metadata[0].file_extension, Some("jpg".to_string()));

                    }
                    // Image 0_2.jpg -> 0 People in there.
                    {
                        let base_image_0_2 = base_image_repository.get_base_image_by_path("../test_pictures/0_2.jpg").await;
                        assert!(base_image_0_2.is_some());
                        let base_image_0_2 = base_image_0_2.unwrap();

                        let base_image_0_2_metadata = metadata_query_engine.get_all_metadata_attached_to_base_image(&base_image_0_2).await.expect("cannot get metadata");

                        assert_eq!(base_image_0_2_metadata.path, "../test_pictures/0_2.jpg");

                        assert_eq!(hex::encode(base_image_0_2_metadata.image_hash[0].hash), "5bd29a53940be3567570757683ea71493b81a94089a79986f79f7d2db19e4976");
                        assert_eq!(base_image_0_2_metadata.image_hash[0].hash_type, "SHA256");

                        assert_eq!(base_image_0_2_metadata.faces.len(), 0);

                        assert_eq!(base_image_0_2_metadata.image_embedding[0].embedding.len(), 768);

                        assert_eq!(base_image_0_2_metadata.basic_metadata[0].height, 801);
                        assert_eq!(base_image_0_2_metadata.basic_metadata[0].width, 1200);
                        assert_eq!(base_image_0_2_metadata.basic_metadata[0].size_in_bytes, 150070);
                        assert_eq!(base_image_0_2_metadata.basic_metadata[0].file_extension, Some("jpg".to_string()));
                    }
                    // Image 0_3.jpg -> 0 People in there.
                    {
                        let base_image_0_3 = base_image_repository.get_base_image_by_path("../test_pictures/0_3.jpg").await;
                        assert!(base_image_0_3.is_some());
                        let base_image_0_3 = base_image_0_3.unwrap();

                        let base_image_0_3_metadata = metadata_query_engine.get_all_metadata_attached_to_base_image(&base_image_0_3).await.expect("cannot get metadata");

                        assert_eq!(base_image_0_3_metadata.path, "../test_pictures/0_3.jpg");

                        assert_eq!(hex::encode(base_image_0_3_metadata.image_hash[0].hash), "58722cabb0a7ab17685eb3bda6ae9f356bcae3996130169eda8a0b03d0258065");
                        assert_eq!(base_image_0_3_metadata.image_hash[0].hash_type, "SHA256");

                        assert_eq!(base_image_0_3_metadata.faces.len(), 0);

                        assert_eq!(base_image_0_3_metadata.image_embedding[0].embedding.len(), 768);

                        assert_eq!(base_image_0_3_metadata.basic_metadata[0].height, 798);
                        assert_eq!(base_image_0_3_metadata.basic_metadata[0].width, 1200);
                        assert_eq!(base_image_0_3_metadata.basic_metadata[0].size_in_bytes, 108473);
                        assert_eq!(base_image_0_3_metadata.basic_metadata[0].file_extension, Some("jpg".to_string()));
                    }
                    // Image 1_1.jpg -> 1 Person in there.
                    {
                        let base_image_1_1 = base_image_repository.get_base_image_by_path("../test_pictures/1_1.jpg").await;
                        assert!(base_image_1_1.is_some());
                        let base_image_1_1 = base_image_1_1.unwrap();

                        let base_image_1_1_metadata = metadata_query_engine.get_all_metadata_attached_to_base_image(&base_image_1_1).await.expect("cannot get metadata");

                        assert_eq!(base_image_1_1_metadata.path, "../test_pictures/1_1.jpg");

                        assert_eq!(hex::encode(base_image_1_1_metadata.image_hash[0].hash), "c57fc6e6e7a6922eeb2815baee3d3405768968b1b98205be3713ec399f0a09ee");
                        assert_eq!(base_image_1_1_metadata.image_hash[0].hash_type, "SHA256");

                        assert_eq!(base_image_1_1_metadata.faces.len(), 1);

                        assert_eq!(base_image_1_1_metadata.image_embedding[0].embedding.len(), 768);

                        assert_eq!(base_image_1_1_metadata.basic_metadata[0].height, 1280);
                        assert_eq!(base_image_1_1_metadata.basic_metadata[0].width, 853);
                        assert_eq!(base_image_1_1_metadata.basic_metadata[0].size_in_bytes, 76361);
                        assert_eq!(base_image_1_1_metadata.basic_metadata[0].file_extension, Some("jpg".to_string()));
                    }
                    // Image 3_1.jpg -> 3 Persons in there.
                    {
                        let base_image_3_1 = base_image_repository.get_base_image_by_path("../test_pictures/3_1.jpg").await;
                        assert!(base_image_3_1.is_some());
                        let base_image_3_1 = base_image_3_1.unwrap();

                        let base_image_3_1_metadata = metadata_query_engine.get_all_metadata_attached_to_base_image(&base_image_3_1).await.expect("cannot get metadata");

                        assert_eq!(base_image_3_1_metadata.path, "../test_pictures/3_1.jpg");

                        assert_eq!(hex::encode(base_image_3_1_metadata.image_hash[0].hash), "5b3b05a8484dbfe7b483251e087f84a2e31a95867d509a9ea034a055509195a6");
                        assert_eq!(base_image_3_1_metadata.image_hash[0].hash_type, "SHA256");

                        assert_eq!(base_image_3_1_metadata.faces.len(), 3);

                        assert_eq!(base_image_3_1_metadata.image_embedding[0].embedding.len(), 768);

                        assert_eq!(base_image_3_1_metadata.basic_metadata[0].height, 853);
                        assert_eq!(base_image_3_1_metadata.basic_metadata[0].width, 1280);
                        assert_eq!(base_image_3_1_metadata.basic_metadata[0].size_in_bytes, 247712);
                        assert_eq!(base_image_3_1_metadata.basic_metadata[0].file_extension, Some("jpg".to_string()));
                    }
                    // Image 7_1.jpg -> 7 Persons in there.
                    {
                        let base_image_7_1 = base_image_repository.get_base_image_by_path("../test_pictures/7_1.jpg").await;
                        assert!(base_image_7_1.is_some());
                        let base_image_7_1 = base_image_7_1.unwrap();

                        let base_image_7_1_metadata = metadata_query_engine.get_all_metadata_attached_to_base_image(&base_image_7_1).await.expect("cannot get metadata");

                        assert_eq!(base_image_7_1_metadata.path, "../test_pictures/7_1.jpg");

                        assert_eq!(hex::encode(base_image_7_1_metadata.image_hash[0].hash), "836513f25131ef5497e8cb9ee0d696b5d9597bab0126cd6abaa7a8590fbda00e");
                        assert_eq!(base_image_7_1_metadata.image_hash[0].hash_type, "SHA256");

                        assert_eq!(base_image_7_1_metadata.faces.len(), 7);

                        assert_eq!(base_image_7_1_metadata.image_embedding[0].embedding.len(), 768);

                        assert_eq!(base_image_7_1_metadata.basic_metadata[0].height, 3887);
                        assert_eq!(base_image_7_1_metadata.basic_metadata[0].width, 6000);
                        assert_eq!(base_image_7_1_metadata.basic_metadata[0].size_in_bytes, 1496498);
                        assert_eq!(base_image_7_1_metadata.basic_metadata[0].file_extension, Some("jpg".to_string()));
                    }
                });
            })
            .unwrap()
            .join()
            .unwrap();
    }
}