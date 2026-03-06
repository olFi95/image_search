use crate::clip::get_all_directories_in_dir;
use crate::metadata_provider::age_and_gender_metadata_provider::{AgeAndGenderMetadataProvider, FaceAgeAndGender, FaceAgeAndGenderMetadataRepository};
use crate::metadata_provider::basic_metadata_provider::{BasicMetadata, BasicMetadataProvider, BasicMetadataRepository};
use crate::metadata_provider::face_recognition_metadata_provider::{FaceInPicture, FaceInPictureVector, FaceRecognitionMetadataProvider, FaceRecognitionMetadataRepository};
use crate::metadata_provider::image_embedding_metadata_provider::{ImageEmbedding, ImageEmbeddingMetadataProvider, ImageEmbeddingMetadataRepository};
use crate::metadata_provider::image_hash_metadata_provider::{ImageHashMetadata, ImageHashMetadataProvider, ImageHashMetadataRepository};
use crate::metadata_provider::model::{BaseImage, BaseImageRepository, BaseImageWithImage, Metadata, MetadataProvider};
use burn::prelude::Backend;
use burn::tensor::Device;
use crossbeam_channel::{bounded, Receiver};
use log::{info, trace};
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::path::PathBuf;
use std::thread;
use std::time::Instant;
use surrealdb::{Connection, Surreal};

const BUFFER: usize = 100;
const BATCH: usize = 25;

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

        let all_image_paths: Vec<PathBuf> = get_all_directories_in_dir(&path)
            .par_iter()
            .map(PathBuf::from)
            .collect();

        let total_images = all_image_paths.len();
        info!("Starting indexing of {} images in {}", total_images, path.to_str().unwrap_or("provided path"));

        let (tx_base_image, rx_base_image) = bounded::<BaseImage>(BUFFER);
        let producer = thread::spawn(move || {
            for path in all_image_paths {
                let base = BaseImage::new(path);
                if tx_base_image.send(base).is_err() {
                    break;
                }
            }
        });

        let (tx_base_with_id, rx_base_with_id) = bounded::<BaseImage>(BUFFER);
        let base_image_saver = {
            let repo = BaseImageRepository::new(self.db.clone()).await;
            tokio::spawn(async move {
                loop {
                    let batch = collect_batch_async(&rx_base_image, BATCH).await;
                    if batch.is_empty() { break; }
                    let inserted = repo.insert_many(batch).await.unwrap();
                    for base in inserted {
                        if tx_base_with_id.send(base).is_err() { break; }
                    }
                }
            })
        };

        let (tx_loaded, rx_loaded) = bounded::<BaseImageWithImage>(BUFFER);
        let (tx_for_embedding, rx_for_embedding) = bounded::<BaseImageWithImage>(BUFFER);
        let (tx_for_basic_metadata, rx_for_basic_metadata) = bounded::<BaseImageWithImage>(BUFFER);
        let (tx_for_face, rx_for_face) = bounded::<BaseImageWithImage>(BUFFER);

        let image_loader = thread::spawn(move || {
            loop {
                let batch = collect_batch(&rx_base_with_id, BATCH);
                trace!("loading batch of {} images, {} in queue", batch.len(), rx_base_with_id.len());
                if batch.is_empty() { break; }

                batch
                    .into_par_iter()
                    .filter_map(|base| base.clone().try_into().ok())
                    .for_each(|img: BaseImageWithImage| {
                        let _ = tx_loaded.send(img);
                    });
            }
        });

        let image_dispatcher = thread::spawn(move || {
            for img in rx_loaded {
                let _ = tx_for_embedding.send(img.clone());
                let _ = tx_for_basic_metadata.send(img.clone());
                let _ = tx_for_face.send(img);
            }
        });

        let (tx_hash, rx_hash) = bounded::<Metadata<ImageHashMetadata>>(BUFFER);
        let (tx_basic, rx_basic) = bounded::<Metadata<BasicMetadata>>(BUFFER);
        let basic_extractor = thread::spawn(move || {
            let image_hash_provider = ImageHashMetadataProvider;
            let basic_provider = BasicMetadataProvider;
            loop {
                let batch = collect_batch(&rx_for_basic_metadata, BATCH);
                trace!("extracting basic+hash metadata for {} images, {} in queue", batch.len(), rx_for_basic_metadata.len());
                if batch.is_empty() { break; }

                let hashes = image_hash_provider.extract(&batch).unwrap();
                let basics = basic_provider.extract(&batch).unwrap();
                for h in hashes { if tx_hash.send(h).is_err() { break; } }
                for b in basics { if tx_basic.send(b).is_err() { break; } }
            }
        });

        let image_embedder_device = self.device.clone();
        let image_embedder_model = self.image_embedder.clone();
        let (tx_image_embedding, rx_image_embedding) = bounded::<Metadata<ImageEmbedding>>(BUFFER);
        let image_embedder = thread::Builder::new()
            .name("image_embedder".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                let provider: ImageEmbeddingMetadataProvider<B> =
                    ImageEmbeddingMetadataProvider::new(image_embedder_device, image_embedder_model.as_str());
                loop {
                    let batch = collect_batch(&rx_for_embedding, BATCH);
                    trace!("embedding {} images, {} in queue", batch.len(), rx_for_embedding.len());
                    if batch.is_empty() { break; }
                    let embeddings = provider.extract(&batch).expect("cannot embed images");
                    for e in embeddings { if tx_image_embedding.send(e).is_err() { break; } }
                }
            }).expect("cannot spawn image_embedder thread");

        let face_recognition_device = self.device.clone();
        let face_detection_model = self.face_detector.clone();
        let face_embedding_model = self.face_embedder.clone();
        let (tx_face_for_db, rx_face_for_db) = bounded::<Metadata<FaceInPicture>>(BUFFER);
        let (tx_face_for_age_gender, rx_face_for_age_gender) = bounded::<Metadata<FaceInPicture>>(BUFFER);
        let (tx_face_embedding, rx_face_embedding) = bounded::<Metadata<FaceInPictureVector>>(BUFFER);

        let face_embedder = thread::Builder::new()
            .name("face_metadata".into())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                let provider: FaceRecognitionMetadataProvider<B> = FaceRecognitionMetadataProvider::new(
                    face_recognition_device,
                    face_detection_model.as_str(),
                    face_embedding_model.as_str(),
                );
                loop {
                    let batch = collect_batch(&rx_for_face, BATCH);
                    trace!("detecting faces in {} images, {} in queue", batch.len(), rx_for_face.len());
                    if batch.is_empty() { break; }

                    let faces = provider.extract(&batch).expect("cannot detect faces");
                    for face in &faces {
                        let _ = tx_face_for_db.send(face.clone());
                    }
                    let face_embeddings = provider.extract(&faces).expect("cannot embed faces");
                    for fe in face_embeddings { if tx_face_embedding.send(fe).is_err() { break; } }
                }
            }).expect("cannot spawn face_metadata thread");

        let face_age_gender_device = self.device.clone();
        let face_age_gender_model = self.face_age_and_gender.clone();
        let (tx_age_gender, rx_age_gender) = bounded::<Metadata<FaceAgeAndGender>>(BUFFER);
        let age_gender_estimator = thread::Builder::new()
            .name("face_age_and_gender".into())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                let provider: AgeAndGenderMetadataProvider<B> =
                    AgeAndGenderMetadataProvider::new(face_age_gender_device, face_age_gender_model.as_str());
                loop {
                    let batch = collect_batch(&rx_face_for_age_gender, BATCH);
                    trace!("estimating age+gender for {} faces, {} in queue", batch.len(), rx_face_for_age_gender.len());
                    if batch.is_empty() { break; }
                    let results = provider.extract(&batch).expect("cannot estimate age and gender");
                    for r in results {
                        if tx_age_gender.send(r).is_err() { break; }
                    }
                }
            }).expect("cannot spawn face_age_and_gender thread");

        let hash_saver = {
            let repo = ImageHashMetadataRepository::new(self.db.clone()).await;
            tokio::spawn(async move {
                loop {
                    let batch = collect_batch_async(&rx_hash, BATCH).await;
                    trace!("saving {} image hashes, {} in queue", batch.len(), rx_hash.len());
                    if batch.is_empty() { break; }
                    repo.insert_many(&batch).await.unwrap();
                }
            })
        };
        let basic_saver = {
            let repo = BasicMetadataRepository::new(self.db.clone()).await;
            tokio::spawn(async move {
                loop {
                    let batch = collect_batch_async(&rx_basic, BATCH).await;
                    trace!("saving {} basic metadata entries, {} in queue", batch.len(), rx_basic.len());
                    if batch.is_empty() { break; }
                    repo.insert_many(&batch).await.unwrap();
                }
            })
        };
        let embedding_saver = {
            let repo = ImageEmbeddingMetadataRepository::new(self.db.clone()).await;
            tokio::spawn(async move {
                loop {
                    let batch = collect_batch_async(&rx_image_embedding, BATCH).await;
                    trace!("saving {} image embeddings, {} in queue", batch.len(), rx_image_embedding.len());
                    if batch.is_empty() { break; }
                    repo.insert_many_image_embeddings(&batch).await.unwrap();
                }
            })
        };
        let face_embedding_saver = {
            let repo = FaceRecognitionMetadataRepository::new(self.db.clone()).await;
            tokio::spawn(async move {
                loop {
                    let batch = collect_batch_async(&rx_face_embedding, BATCH).await;
                    trace!("saving {} face embeddings, {} in queue", batch.len(), rx_face_embedding.len());
                    if batch.is_empty() { break; }
                    repo.insert_many_face_embeddings(&batch).await.unwrap();
                }
            })
        };
        let face_in_picture_saver = {
            let repo = FaceRecognitionMetadataRepository::new(self.db.clone()).await;
            let rt = tokio::runtime::Handle::current();
            thread::spawn(move || {
                loop {
                    let batch = collect_batch(&rx_face_for_db, BATCH);
                    trace!("saving {} faces in pictures, {} in queue", batch.len(), rx_face_for_db.len());
                    if batch.is_empty() { break; }
                    let saved = rt.block_on(repo.insert_many_face_in_picture(&batch)).unwrap();
                    for face in saved {
                        if tx_face_for_age_gender.send(face).is_err() { break; }
                    }
                }
            })
        };
        let age_gender_saver = {
            let repo = FaceAgeAndGenderMetadataRepository::new(self.db.clone()).await;
            tokio::spawn(async move {
                loop {
                    let batch = collect_batch_async(&rx_age_gender, BATCH).await;
                    trace!("saving {} age+gender estimations, {} in queue", batch.len(), rx_age_gender.len());
                    if batch.is_empty() { break; }
                    repo.insert_many_age_and_gender(&batch).await.unwrap();
                }
            })
        };

        let (
            r_producer,
            r_base_image_saver,
            r_image_loader,
            r_image_dispatcher,
            r_basic_extractor,
            r_image_embedder,
            r_face_embedder,
            r_face_in_picture_saver,
            r_age_gender_estimator,
            r_hash_saver,
            r_basic_saver,
            r_embedding_saver,
            r_face_embedding_saver,
            r_age_gender_saver,
        ) = tokio::join!(
            tokio::task::spawn_blocking(move || producer.join()),
            base_image_saver,
            tokio::task::spawn_blocking(move || image_loader.join()),
            tokio::task::spawn_blocking(move || image_dispatcher.join()),
            tokio::task::spawn_blocking(move || basic_extractor.join()),
            tokio::task::spawn_blocking(move || image_embedder.join()),
            tokio::task::spawn_blocking(move || face_embedder.join()),
            tokio::task::spawn_blocking(move || face_in_picture_saver.join()),
            tokio::task::spawn_blocking(move || age_gender_estimator.join()),
            hash_saver,
            basic_saver,
            embedding_saver,
            face_embedding_saver,
            age_gender_saver,
        );

        r_producer.expect("producer spawn_blocking panicked").expect("producer thread panicked");
        r_base_image_saver.expect("base_image_saver panicked");
        r_image_loader.expect("image_loader spawn_blocking panicked").expect("image_loader thread panicked");
        r_image_dispatcher.expect("image_dispatcher spawn_blocking panicked").expect("image_dispatcher thread panicked");
        r_basic_extractor.expect("basic_extractor spawn_blocking panicked").expect("basic_extractor thread panicked");
        r_image_embedder.expect("image_embedder spawn_blocking panicked").expect("image_embedder thread panicked");
        r_face_embedder.expect("face_embedder spawn_blocking panicked").expect("face_embedder thread panicked");
        r_face_in_picture_saver.expect("face_in_picture_saver spawn_blocking panicked").expect("face_in_picture_saver thread panicked");
        r_age_gender_estimator.expect("age_gender_estimator spawn_blocking panicked").expect("age_gender_estimator thread panicked");
        r_hash_saver.expect("hash_saver panicked");
        r_basic_saver.expect("basic_saver panicked");
        r_embedding_saver.expect("embedding_saver panicked");
        r_face_embedding_saver.expect("face_embedding_saver panicked");
        r_age_gender_saver.expect("age_gender_saver panicked");

        info!(
            "Finished indexing metadata for {} images in {:?}.",
            total_images, total_start.elapsed()
        );

        Ok(())
    }
}

async fn collect_batch_async<T: Send + 'static>(rx: &Receiver<T>, max: usize) -> Vec<T> {
    let rx_clone = rx.clone();
    let first = tokio::task::spawn_blocking(move || rx_clone.recv().ok()).await.unwrap();

    let Some(first_item) = first else {
        return Vec::new(); // channel closed
    };

    let mut items = Vec::with_capacity(max);
    items.push(first_item);

    while items.len() < max {
        match rx.try_recv() {
            Ok(item) => items.push(item),
            Err(_) => break,
        }
    }

    items
}

fn collect_batch<T>(rx: &Receiver<T>, max: usize) -> Vec<T> {
    let mut items = Vec::with_capacity(max);

    match rx.recv() {
        Ok(item) => items.push(item),
        Err(_) => return items,
    }

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
    use crate::metadata_provider::metadata_query_engine::MetadataQueryEngine;
    use crate::metadata_provider::model::BaseImageRepository;

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
                    use burn_ndarray::{NdArray, NdArrayDevice};

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
