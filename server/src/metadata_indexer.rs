use crate::clip::get_all_directories_in_dir;
use crate::metadata_provider::age_and_gender_metadata_provider::{
    AgeAndGenderMetadataProvider, FaceAgeAndGenderMetadataRepository,
};
use crate::metadata_provider::basic_metadata_provider::{BasicMetadata, BasicMetadataProvider, BasicMetadataRepository};
use crate::metadata_provider::face_recognition_metadata_provider::{
    FaceRecognitionMetadataProvider, FaceRecognitionMetadataRepository,
};
use crate::metadata_provider::image_embedding_metadata_provider::{
    ImageEmbeddingMetadataProvider, ImageEmbeddingMetadataRepository,
};
use crate::metadata_provider::image_hash_metadata_provider::{ImageHashMetadata, ImageHashMetadataProvider, ImageHashMetadataRepository};
use crate::metadata_provider::model::{BaseImage, BaseImageRepository, BaseImageWithImage, Metadata, MetadataProvider};
use burn::tensor::Device;
use log::{info, trace};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::path::PathBuf;
use std::thread;
use std::time::Instant;
use burn::prelude::Backend;
use crossbeam_channel::{bounded, Receiver};
use surrealdb::{Connection, Surreal};
use tokio::join;

pub struct MetadataIndexer<C, B>
where
    C: Connection,
    B: Backend,
{
    db: Surreal<C>,
    device: Device<B>,
    face_detector: String,
    face_embedder: String,
    face_age_and_gender: String,
    image_embedder: String,
}
impl <C, B>MetadataIndexer<C, B>
where
    C: Connection,
    B: Backend,
{
    pub fn new(
        db: Surreal<C>,
        device: Device<B>,
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
        let total_start = Instant::now();

        // Metadata Provider
        let image_hash_metadata_provider = ImageHashMetadataProvider;
        let basic_metadata_provider = BasicMetadataProvider;
        let face_recognition_metadata_provider: FaceRecognitionMetadataProvider<B> = FaceRecognitionMetadataProvider::new(
            self.device.clone(),
            self.face_detector.as_str(),
            self.face_embedder.as_str(),
        );
        let face_age_and_gender_metadata_provider: AgeAndGenderMetadataProvider<B> = AgeAndGenderMetadataProvider::new(
            self.device.clone(),
            self.face_age_and_gender.as_str(),
        );
        let image_embedding_metadata_provider: ImageEmbeddingMetadataProvider<B> =
            ImageEmbeddingMetadataProvider::new(self.device.clone(), self.image_embedder.as_str());
        // Metadata Repositories
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

        let total_images = all_image_paths.len();
        info!("Starting indexing of {} images in {}", total_images, path.to_str().unwrap_or("provided path"));

        let chunk_size = 25;
        let total_chunks = (total_images.div_ceil(chunk_size)) / chunk_size;

        const BUFFER: usize = 100;
        const BATCH: usize = 25;

        // Thread to load the images.
        let (tx_base_image_with_image, rx_base_image_with_image) = bounded::<BaseImageWithImage>(BUFFER);
        let (tx_base_image, rx_base_image) = bounded::<BaseImage>(BUFFER);
        let image_loader_handle = thread::spawn(move || {
            all_image_paths.par_iter().for_each(|path| {
                trace!("Loading image from path: {}", path.to_str().unwrap_or("cannot convert path to string"));
                let base = BaseImage::new(path.to_path_buf());
                tx_base_image.send(base.clone()).expect("cannot send base image.");

                if let Ok(image) = base.clone().try_into() {
                    tx_base_image_with_image.send(image).expect("cannot send image.");
                }
            });
        });
        let (tx_base_image_with_image_and_id, rx_base_image_with_image_and_id) = bounded::<BaseImageWithImage>(BUFFER);
        let base_image_with_image_saver_handler = {
            let base_image_repository = BaseImageRepository::new(self.db.clone()).await;
            tokio::spawn(async move {
                loop {
                    let batch = collect_batch(&rx_base_image, BATCH);
                    if batch.is_empty() { break; }
                    trace!("Saving hash metadata for batch of {} images", batch.len());
                    let inserted_batch = base_image_repository.insert_many(batch).await.expect("cannot save hash metadata");
                    for base_image in inserted_batch {
                        tx_base_image_with_image_and_id.send(base_image.into()).expect("cannot send image.");
                    }
                }
            })
        };

        // Thread to extract Basic and Hash Metadata
        let (tx_hash_metadata, rx_hash_metadata) = bounded::<Metadata<ImageHashMetadata>>(BUFFER);
        let (tx_basic_metadata, rx_basic_metadata) = bounded::<Metadata<BasicMetadata>>(BUFFER);
        let hash_and_basic_metadata_handle = thread::spawn(move || {
            loop {
                let batch = collect_batch(&rx_base_image_with_image_and_id, BATCH);
                if batch.is_empty() { break; }
                trace!("Extracting hash and basic metadata for batch of {} images", batch.len());

                let bash_metadata = image_hash_metadata_provider.extract(&batch).unwrap();
                for hash in bash_metadata {
                    if tx_hash_metadata.send(hash).is_err() {
                        break; // downstream channel closed, exit thread
                    }
                }

                let basic_metadata = basic_metadata_provider.extract(&batch).unwrap();
                for basic in basic_metadata {
                    if tx_basic_metadata.send(basic).is_err() {
                        break; // downstream channel closed, exit thread
                    }
                }
            }
        });

        let hash_metadata_saving_handle = {
            let image_hash_metadata_repository =
                ImageHashMetadataRepository::new(self.db.clone()).await;
            tokio::spawn(async move {
                loop {
                    let batch = collect_batch(&rx_hash_metadata, BATCH);
                    if batch.is_empty() { break; }
                    trace!("Saving hash metadata for batch of {} images", batch.len());
                    image_hash_metadata_repository.insert_many(&batch).await.expect("cannot save hash metadata");
                }
            })
        };

        let basic_metadata_saving_handle = {
            let basic_metadata_repository = BasicMetadataRepository::new(self.db.clone()).await;
            tokio::spawn(async move {
                loop {
                    let batch = collect_batch(&rx_basic_metadata, BATCH);
                    if batch.is_empty() { break; }
                    trace!("Saving basic metadata for batch of {} images", batch.len());
                    basic_metadata_repository.insert_many(&batch).await.expect("cannot save basic metadata");
                }
            })
        };

        image_loader_handle.join().unwrap();
        hash_and_basic_metadata_handle.join().unwrap();
        join!(hash_metadata_saving_handle, basic_metadata_saving_handle, base_image_with_image_saver_handler);

        info!(
            "Finished indexing metadata for {} images in {:?}. Rebuilding indexes now.",
            total_images, total_start.elapsed()
        );
        image_embedding_metadata_repository
            .rebuild_index()
            .await
            .expect("cannot rebuild image embedding metadata index");
        face_recognition_metadata_repository
            .rebuild_index()
            .await
            .expect("cannot rebuild face embedding metadata index");
        info!(
            "Finished rebuilding indexes. Total time: {:?}",
            total_start.elapsed()
        );



        // for (chunk_idx, image_paths) in all_image_paths.chunks(chunk_size).enumerate() {
        //     let chunk_start = Instant::now();
        //     info!("Processing chunk {}/{} ({} images)", chunk_idx + 1, total_chunks, image_paths.len());
        //
        //     // Convert Path Strings into PathBufs and then into BaseImages
        //     let mut base_images: Vec<BaseImage> = image_paths
        //         .par_iter()
        //         .map(|path| BaseImage::new(PathBuf::from(path)))
        //         .collect();
        //
        //     // Save BaseImages to the repository
        //     base_images = base_image_repository
        //         .insert_many(base_images)
        //         .await
        //         .expect("Inserting base image failed");
        //
        //     // Now actually load the images in parallel on CPU. Drop the Images that were not able to load properly.
        //     let t = Instant::now();
        //     let base_images_with_image: Vec<_> = base_images
        //         .par_iter()
        //         .cloned()
        //         .map(|bi| bi.try_into())
        //         .filter_map(Result::ok)
        //         .collect();
        //     trace!("Image loading: {:?} ({} images)", t.elapsed(), base_images_with_image.len());
        //
        //     // --- CPU-only work: hashes + basic metadata (uses rayon, no GPU) ---
        //     let t = Instant::now();
        //     let hashes = image_hash_metadata_provider
        //         .extract(&base_images_with_image)
        //         .expect("cannot extract hashes");
        //     let basic_metadata = basic_metadata_provider
        //         .extract(&base_images_with_image)
        //         .expect("cannot extract basic metadata");
        //     trace!("CPU metadata (hashes + basic): {:?}", t.elapsed());
        //
        //     // --- GPU work: face detection (batched) ---
        //     let t = Instant::now();
        //     let faces = face_recognition_metadata_provider
        //         .extract(&base_images_with_image)
        //         .expect("cannot extract face recognition metadata");
        //     trace!("Face detection (GPU, batched): {:?} ({} faces found)", t.elapsed(), faces.len());
        //
        //     // --- GPU work: image embedding (batched) ---
        //     let t = Instant::now();
        //     let image_embeddings = image_embedding_metadata_provider
        //         .extract(&base_images_with_image)
        //         .expect("cannot embed images");
        //     trace!("Image embedding (GPU, batched): {:?}", t.elapsed());
        //
        //     // Save CPU-only metadata to DB
        //     let t = Instant::now();
        //     let _ = image_hash_metadata_repository
        //         .insert_many(&hashes)
        //         .await
        //         .expect("could not save hashes");
        //     basic_metadata_repository
        //         .insert_many(&basic_metadata)
        //         .await
        //         .expect("could not save basic metadata");
        //     trace!("DB insert (hashes + basic metadata): {:?}", t.elapsed());
        //
        //     // Save face recognition metadata to the repository.
        //     let t = Instant::now();
        //     let faces = face_recognition_metadata_repository
        //         .insert_many_face_in_picture(&faces)
        //         .await
        //         .expect("cannot save discovered faces to database.");
        //     trace!("DB insert (face positions): {:?}", t.elapsed());
        //
        //     // --- GPU work: age/gender estimation + face embedding (batched) ---
        //     let t = Instant::now();
        //     let age_gender_results = face_age_and_gender_metadata_provider
        //         .extract(&faces)
        //         .expect("cannot extract age and gender");
        //     trace!("Age/gender estimation (GPU, batched): {:?} ({} faces)", t.elapsed(), age_gender_results.len());
        //
        //     let t = Instant::now();
        //     let face_embeddings = face_recognition_metadata_provider
        //         .extract(&faces)
        //         .expect("cannot embed faces");
        //     trace!("Face embedding (GPU, batched): {:?} ({} faces)", t.elapsed(), face_embeddings.len());
        //
        //     // Save remaining metadata to DB
        //     let t = Instant::now();
        //     face_age_and_gender_metadata_repository
        //         .insert_many_age_and_gender(&age_gender_results)
        //         .await
        //         .expect("could not save age and gender metadata");
        //     face_recognition_metadata_repository
        //         .insert_many_face_embeddings(&face_embeddings)
        //         .await
        //         .expect("cannot save face embeddings");
        //     image_embedding_metadata_repository
        //         .insert_many_image_embeddings(&image_embeddings)
        //         .await
        //         .expect("cannot save image embeddings");
        //     trace!("DB insert (age/gender + face embeddings + image embeddings): {:?}", t.elapsed());
        //
        //     info!("Chunk {}/{} done in {:?}", chunk_idx + 1, total_chunks, chunk_start.elapsed());
        // }
        // info!(
        //     "Finished indexing metadata for {} images in {:?}. Rebuilding indexes now.",
        //     total_images, total_start.elapsed()
        // );
        // image_embedding_metadata_repository
        //     .rebuild_index()
        //     .await
        //     .expect("cannot rebuild image embedding metadata index");
        // face_recognition_metadata_repository
        //     .rebuild_index()
        //     .await
        //     .expect("cannot rebuild face embedding metadata index");
        // info!(
        //     "Finished rebuilding indexes. Total time: {:?}",
        //     total_start.elapsed()
        // );
        Ok(())
    }
}

fn collect_batch<T>(rx: &Receiver<T>, max: usize) -> Vec<T> {
    let mut items = Vec::with_capacity(max);

    match rx.recv() {
        Ok(item) => items.push(item),
        Err(_) => return items,
    }

    // Danach non-blocking sammeln
    while items.len() < max {
        match rx.try_recv() {
            Ok(item) => items.push(item),
            Err(_) => break,
        }
    }

    items
}

#[cfg(test)]
mod test {
    use crate::metadata_indexer::MetadataIndexer;
    use crate::metadata_provider::model::BaseImageRepository;
    use crate::metadata_provider::metadata_query_engine::MetadataQueryEngine;

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
                    use surrealdb::engine::local::Mem;
                    use surrealdb::Surreal;
                    use burn_ndarray::{NdArrayDevice, NdArray};

                    let db = Surreal::new::<Mem>(()).await.unwrap();
                    db.use_ns("test").use_db("test").await.unwrap();

                    let metadata_indexer = MetadataIndexer::<_, NdArray>::new(
                        db.clone(),
                        NdArrayDevice::default(),
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
