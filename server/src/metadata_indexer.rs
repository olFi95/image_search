use crate::clip::get_all_directories_in_dir;
use crate::metadata_provider::age_and_gender_metadata_provider::{
    AgeAndGenderMetadataProvider, FaceAgeAndGender, FaceAgeAndGenderMetadataRepository,
};
use crate::metadata_provider::basic_metadata_provider::{
    BasicMetadata, BasicMetadataProvider, BasicMetadataRepository,
};
use crate::metadata_provider::face_recognition_metadata_provider::{
    FaceInPicture, FaceInPictureVector, FaceRecognitionMetadataProvider,
    FaceRecognitionMetadataRepository,
};
use crate::metadata_provider::image_embedding_metadata_provider::{
    ImageEmbedding, ImageEmbeddingMetadataProvider, ImageEmbeddingMetadataRepository,
};
use crate::metadata_provider::image_hash_metadata_provider::{
    ImageHashMetadata, ImageHashMetadataProvider, ImageHashMetadataRepository,
};
use crate::metadata_provider::model::{
    BaseImage, BaseImageRepository, BaseImageWithImage, Metadata, MetadataProvider,
};
use crate::search::IndexingStatus;
use burn::prelude::Backend;
use burn::tensor::Device;
use log::{info, trace};
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use surrealdb::{Connection, Surreal};
use tokio::sync::mpsc::Receiver;
use tokio::sync::Mutex;
use tracing::{debug, error};

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
    clip_vision: String,
    clip_text: String,
}
impl<C, B> MetadataIndexer<C, B>
where
    C: Connection,
    B: Backend,
{
    pub fn new(
        db: Surreal<C>,
        device: Device<B>,
        face_embedder: String,
        face_detector: String,
        clip_vision: String,
        clip_text: String,
        face_age_and_gender: String,
    ) -> Self {
        MetadataIndexer {
            db,
            device,
            face_embedder,
            face_detector,
            clip_vision,
            clip_text,
            face_age_and_gender,
        }
    }

    pub async fn index_metadata(
        &self,
        path: PathBuf,
        index_state: Arc<Mutex<IndexingStatus>>,
    ) -> anyhow::Result<()> {
        let total_start = Instant::now();

        let all_image_paths: Vec<PathBuf> = get_all_directories_in_dir(&path)
            .par_iter()
            .map(PathBuf::from)
            .collect();

        let total_images = all_image_paths.len();
        info!(
            "Starting indexing of {} images in {}",
            total_images,
            path.to_str().unwrap_or("provided path")
        );
        *index_state.clone().lock_owned().await =
            IndexingStatus::InProgress(crate::search::IndexState {
                total: total_images as u32,
                already_indexed: 0,
                indexed: 0,
            });
        let (tx_base_image, mut rx_base_image) = tokio::sync::mpsc::channel(BUFFER);
        let producer = {
            tokio::spawn(async move {
                for path in all_image_paths {
                    let base = BaseImage::new(path);
                    tx_base_image
                        .send(base.clone())
                        .await
                        .expect("cannot send base_image");
                }
                trace!("drop(tx_base_image)");
                drop(tx_base_image);
            })
        };
        let (tx_base_with_id, mut rx_base_with_id) = tokio::sync::mpsc::channel(BUFFER);
        let base_image_saver = {
            let repo = BaseImageRepository::new(self.db.clone()).await;
            let index_state = index_state.clone();
            tokio::spawn(async move {
                let mut skipped = 0usize;
                loop {
                    trace!("base_image_saver waiting for entries");
                    let batch = collect_batch_async(&mut rx_base_image, BATCH).await;
                    if batch.is_empty() {
                        break;
                    }
                    let inserted = repo.insert_many(batch).await.unwrap();

                    // Skip images that have already been fully indexed.
                    let already_indexed = repo.already_indexed(&inserted).await.unwrap_or_default();
                    let new_images: Vec<BaseImage> = inserted
                        .into_iter()
                        .filter(|b| !already_indexed.contains(&b.path))
                        .collect();

                    skipped += already_indexed.len();
                    match *index_state.clone().lock_owned().await {
                        IndexingStatus::InProgress(ref mut state) => {
                            state.already_indexed += already_indexed.len() as u32;
                        }
                        _ => {
                            error!("Indexing status was expected to be InProgress, but was not.");
                        }
                    }

                    for base in new_images {
                        tx_base_with_id
                            .send(base)
                            .await
                            .expect("cannot send base_image_with_id");
                    }
                }
                trace!("drop(tx_base_with_id)");
                debug!("base_image_saver finished. Skipped {skipped} already-indexed image(s).");
                drop(tx_base_with_id)
            })
        };

        let (tx_loaded, mut rx_loaded) = tokio::sync::mpsc::channel(BUFFER);
        let (tx_for_embedding, mut rx_for_embedding) = tokio::sync::mpsc::channel(BUFFER);
        let (tx_for_basic_metadata, mut rx_for_basic_metadata) = tokio::sync::mpsc::channel(BUFFER);
        let (tx_for_face, mut rx_for_face) = tokio::sync::mpsc::channel(BUFFER);

        let image_loader = {
            tokio::spawn(async move {
                loop {
                    trace!("image_loader waiting for entries");
                    let batch = collect_batch_async(&mut rx_base_with_id, BATCH).await;
                    trace!(
                        "loading batch of {} images, {} in queue",
                        batch.len(),
                        rx_base_with_id.len()
                    );

                    if batch.is_empty() {
                        error!("image_loader error: no more entries to load");
                        break;
                    }

                    // Load images in parallel (CPU-bound); each result is wrapped in Arc
                    // so all downstream workers share the same pixel buffer.
                    let loaded_images: Vec<Arc<BaseImageWithImage>> = batch
                        .into_par_iter()
                        .filter_map(|base| base.try_into().ok())
                        .collect();

                    for img in loaded_images {
                        tx_loaded.send(Arc::clone(&img)).await.expect("")
                    }
                }
                trace!("drop(tx_loaded)");
                drop(tx_loaded);
            })
        };

        let image_dispatcher = {
            tokio::spawn(async move {
                loop {
                    trace!("image_dispatcher waiting for entries");
                    let batch = collect_batch_async(&mut rx_loaded, BATCH).await;
                    trace!(
                        "image_dispatcher distributing batch of {} images, {} in queue",
                        batch.len(),
                        rx_loaded.len()
                    );

                    if batch.is_empty() {
                        error!("image_dispatcher error while receiving");
                        break;
                    }

                    for img in batch {
                        tx_for_embedding
                            .send(Arc::clone(&img))
                            .await
                            .expect("cannot send image to embedding processor");
                        tx_for_basic_metadata
                            .send(Arc::clone(&img))
                            .await
                            .expect("cannot send image to basic metadata processor");
                        tx_for_face
                            .send(Arc::clone(&img))
                            .await
                            .expect("cannot send image to face processor");
                    }
                }
                trace!("drop(tx_for_embedding, tx_for_basic_metadata, tx_for_face)");

                drop(tx_for_embedding);
                drop(tx_for_basic_metadata);
                drop(tx_for_face);
            })
        };

        let (tx_hash, mut rx_hash) = tokio::sync::mpsc::channel(BUFFER);
        let (tx_basic, mut rx_basic) = tokio::sync::mpsc::channel(BUFFER);
        let basic_extractor = {
            tokio::spawn(async move {
                let image_hash_provider = ImageHashMetadataProvider;
                let basic_provider = BasicMetadataProvider;

                loop {
                    trace!("basic_extractor waiting for entries");
                    let batch = collect_batch_async(&mut rx_for_basic_metadata, BATCH).await;

                    if batch.is_empty() {
                        break;
                    }

                    let hashes = image_hash_provider.extract(&batch).unwrap();
                    let basics = basic_provider.extract(&batch).unwrap();

                    for h in hashes {
                        tx_hash
                            .send(h.clone())
                            .await
                            .expect("cannot send hash metadata");
                    }

                    for b in basics {
                        tx_basic
                            .send(b.clone())
                            .await
                            .expect("cannot send basic metadata");
                    }
                }
                trace!("drop(tx_hash, tx_basic)");
                drop(tx_hash);
                drop(tx_basic);
            })
        };
        let image_embedder_device = self.device.clone();
        let clip_vision_model = self.clip_vision.clone();
        let clip_text_model = self.clip_text.clone();
        let (tx_image_embedding, mut rx_image_embedding) = tokio::sync::mpsc::channel(BUFFER);
        let image_embedder = {
            tokio::spawn(async move {
                let provider: ImageEmbeddingMetadataProvider<B> =
                    ImageEmbeddingMetadataProvider::new(
                        image_embedder_device,
                        clip_vision_model.as_str(),
                        clip_text_model.as_str(),
                    );

                loop {
                    trace!("image_embedder waiting for entries");
                    let batch = collect_batch_async(&mut rx_for_embedding, BATCH).await;
                    trace!(
                        "embedding {} images, {} in queue",
                        batch.len(),
                        rx_for_embedding.len()
                    );

                    if batch.is_empty() {
                        break;
                    }
                    let embeddings = provider.extract(&batch).expect("cannot embed images");

                    for e in embeddings {
                        tx_image_embedding
                            .send(e.clone())
                            .await
                            .expect("cannot send embedding");
                    }
                }
                trace!("drop(tx_image_embedding)");
                drop(tx_image_embedding);
            })
        };

        let face_recognition_device = self.device.clone();
        let face_detection_model = self.face_detector.clone();
        let face_embedding_model = self.face_embedder.clone();
        let (tx_face_for_db, mut rx_face_for_db) = tokio::sync::mpsc::channel(BUFFER);
        let (tx_face_embedding, mut rx_face_embedding) = tokio::sync::mpsc::channel(BUFFER);

        let face_embedder = {
            let provider: FaceRecognitionMetadataProvider<B> = FaceRecognitionMetadataProvider::new(
                face_recognition_device,
                face_detection_model.as_str(),
                face_embedding_model.as_str(),
            );

            tokio::spawn(async move {
                loop {
                    trace!("face_embedder waiting for entries");
                    let batch = collect_batch_async(&mut rx_for_face, BATCH).await;
                    trace!(
                        "detecting faces in {} images, {} in queue",
                        batch.len(),
                        rx_for_face.len()
                    );

                    if batch.is_empty() {
                        break;
                    }

                    let faces = provider.extract(&batch).expect("cannot detect faces");

                    for face in faces.iter() {
                        tx_face_for_db
                            .send(face.clone())
                            .await
                            .expect("cannot send face");
                    }

                    let face_embeddings = provider.extract(&faces).expect("cannot embed faces");
                    for fe in face_embeddings {
                        tx_face_embedding
                            .send(fe.clone())
                            .await
                            .expect("cannot send face_embedding");
                    }
                }
                trace!("drop(tx_face_for_db, tx_face_embedding)");

                drop(tx_face_for_db);
                drop(tx_face_embedding);
            })
        };

        let (tx_face_for_age_gender_with_id, mut rx_face_for_age_gender_with_id) =
            tokio::sync::mpsc::channel(BUFFER);
        let face_in_picture_saver = {
            let repo = FaceRecognitionMetadataRepository::new(self.db.clone()).await;
            tokio::spawn(async move {
                loop {
                    let batch = collect_batch_async(&mut rx_face_for_db, BATCH).await;
                    if batch.is_empty() {
                        break;
                    }
                    let saved = repo.insert_many_face_in_picture(&batch).await.unwrap();
                    for face in saved {
                        tx_face_for_age_gender_with_id
                            .send(face.clone())
                            .await
                            .expect("cannot send face with id to");
                    }
                }
                trace!("drop(tx_face_for_age_gender_with_id)");

                drop(tx_face_for_age_gender_with_id);
            })
        };
        let face_age_gender_device = self.device.clone();
        let face_age_gender_model = self.face_age_and_gender.clone();
        let (tx_age_gender, mut rx_age_gender) = tokio::sync::mpsc::channel(BUFFER);
        let age_gender_estimator = {
            tokio::spawn(async move {
                let provider: AgeAndGenderMetadataProvider<B> = AgeAndGenderMetadataProvider::new(
                    face_age_gender_device,
                    face_age_gender_model.as_str(),
                );

                loop {
                    trace!("age_gender_estimator waiting for entries");
                    let batch =
                        collect_batch_async(&mut rx_face_for_age_gender_with_id, BATCH).await;
                    trace!(
                        "estimating age+gender for {} faces, {} in queue",
                        batch.len(),
                        rx_face_for_age_gender_with_id.len()
                    );

                    if batch.is_empty() {
                        break;
                    }

                    let results = provider
                        .extract(&batch)
                        .expect("cannot estimate age and gender");

                    for r in results {
                        tx_age_gender
                            .send(r.clone())
                            .await
                            .expect("cannot send age and gender estimation");
                    }
                }
                trace!("drop(tx_age_gender)");

                drop(tx_age_gender);
            })
        };

        let hash_saver = {
            let repo = ImageHashMetadataRepository::new(self.db.clone()).await;
            tokio::spawn(async move {
                loop {
                    trace!("hash_saver waiting for entries");
                    let batch = collect_batch_async(&mut rx_hash, BATCH).await;
                    trace!(
                        "saving {} image hashes, {} in queue",
                        batch.len(),
                        rx_hash.len()
                    );
                    if batch.is_empty() {
                        break;
                    }
                    repo.insert_many(&batch).await.unwrap();
                }
            })
        };
        let basic_saver = {
            let repo = BasicMetadataRepository::new(self.db.clone()).await;
            tokio::spawn(async move {
                loop {
                    trace!("basic_saver waiting for entries");
                    let batch = collect_batch_async(&mut rx_basic, BATCH).await;
                    trace!(
                        "saving {} basic metadata entries, {} in queue",
                        batch.len(),
                        rx_basic.len()
                    );
                    if batch.is_empty() {
                        break;
                    }
                    repo.insert_many(&batch).await.unwrap();
                }
            })
        };

        let embedding_saver = {
            let index_state = index_state.clone();
            let repo = ImageEmbeddingMetadataRepository::new(self.db.clone()).await;
            tokio::spawn(async move {
                loop {
                    trace!("embedding_saver waiting for entries");
                    let batch = collect_batch_async(&mut rx_image_embedding, BATCH).await;
                    trace!(
                        "saving {} image embeddings, {} in queue",
                        batch.len(),
                        rx_image_embedding.len()
                    );
                    if batch.is_empty() {
                        break;
                    }
                    repo.insert_many_image_embeddings(&batch).await.unwrap();
                    match *index_state.clone().lock_owned().await {
                        IndexingStatus::InProgress(ref mut state) => {
                            debug!(
                                "Progress update: +{} embeddings. Previous progress: {}/{}",
                                batch.len(),
                                state.indexed,
                                state.total
                            );
                            state.indexed += batch.len() as u32;
                        }
                        _ => {
                            error!("Indexing status was expected to be InProgress, but was not.");
                        }
                    }
                }
            })
        };
        let face_embedding_saver = {
            let repo = FaceRecognitionMetadataRepository::new(self.db.clone()).await;
            tokio::spawn(async move {
                loop {
                    trace!("face_embedding_saver waiting for entries");
                    let batch = collect_batch_async(&mut rx_face_embedding, BATCH).await;
                    trace!(
                        "saving {} face embeddings, {} in queue",
                        batch.len(),
                        rx_face_embedding.len()
                    );
                    if batch.is_empty() {
                        break;
                    }
                    repo.insert_many_face_embeddings(&batch).await.unwrap();
                }
            })
        };
        let age_gender_saver = {
            let repo = FaceAgeAndGenderMetadataRepository::new(self.db.clone()).await;
            tokio::spawn(async move {
                loop {
                    trace!("age_gender_saver waiting for entries");
                    let batch = collect_batch_async(&mut rx_age_gender, BATCH).await;
                    trace!(
                        "saving {} age+gender estimations, {} in queue",
                        batch.len(),
                        rx_age_gender.len()
                    );
                    if batch.is_empty() {
                        break;
                    }
                    repo.insert_many_age_and_gender(&batch).await.unwrap();
                }
            })
        };

        producer.await.expect("producer panicked");
        face_in_picture_saver
            .await
            .expect("face_in_picture_saver panicked");
        face_embedder.await.expect("face_embedder panicked");
        image_loader.await.expect("image_loader panicked");
        image_dispatcher
            .await
            .expect("image_dispatcher spawn_blocking panicked");
        basic_extractor
            .await
            .expect("basic_extractor spawn_blocking panicked");
        image_embedder
            .await
            .expect("image_embedder spawn_blocking panicked");

        age_gender_estimator
            .await
            .expect("age_gender_estimator spawn_blocking panicked");

        let (
            r_base_image_saver,
            r_hash_saver,
            r_basic_saver,
            r_embedding_saver,
            r_face_embedding_saver,
            r_age_gender_saver,
        ) = tokio::join!(
            base_image_saver,
            hash_saver,
            basic_saver,
            embedding_saver,
            face_embedding_saver,
            age_gender_saver,
        );
        r_base_image_saver.expect("base_image_saver panicked");
        r_hash_saver.expect("hash_saver panicked");
        r_basic_saver.expect("basic_saver panicked");
        r_embedding_saver.expect("embedding_saver panicked");
        r_face_embedding_saver.expect("face_embedding_saver panicked");
        r_age_gender_saver.expect("age_gender_saver panicked");

        info!(
            "Finished indexing metadata for {} images in {:?}.",
            total_images,
            total_start.elapsed()
        );

        Ok(())
    }
}

async fn collect_batch_async<T>(rx: &mut tokio::sync::mpsc::Receiver<T>, max: usize) -> Vec<T> {
    let mut items = Vec::with_capacity(max);

    let Some(first_item) = rx.recv().await else {
        return Vec::new();
    };

    items.push(first_item);

    while items.len() < max {
        match rx.try_recv() {
            Ok(item) => items.push(item),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
        }
    }

    items
}
#[cfg(test)]
mod test {
    use crate::metadata_indexer::MetadataIndexer;
    use crate::metadata_provider::metadata_query_engine::MetadataQueryEngine;
    use crate::metadata_provider::model::BaseImageRepository;
    use std::sync::Arc;
    use tokio::sync::Mutex;

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
                    use burn_ndarray::{NdArray, NdArrayDevice};
                    use std::path::PathBuf;
                    use surrealdb::Surreal;
                    use surrealdb::engine::local::Mem;

                    let db = Surreal::new::<Mem>(()).await.unwrap();
                    db.use_ns("test").use_db("test").await.unwrap();
                    let index_state = Arc::new(Mutex::new(crate::search::IndexingStatus::Idle));
                    let metadata_indexer = MetadataIndexer::<_, NdArray>::new(
                        db.clone(),
                        NdArrayDevice::default(),
                        "../models/arcface_model.bpk".to_string(),
                        "../models/yolo.bpk".to_string(),
                        "../models/vision_model.bpk".to_string(),
                        "../models/text_model.bpk".to_string(),
                        "../models/age_gender.bpk".to_string(),
                    );

                    metadata_indexer
                        .index_metadata(PathBuf::from("../test_pictures"), index_state)
                        .await
                        .expect("cannot use db");

                    let base_image_repository = BaseImageRepository::new(db.clone()).await;
                    let metadata_query_engine = MetadataQueryEngine::new(db.clone());

                    // Image 0_1.jpg -> 0 People in there.
                    {
                        let base_image_0_1 = base_image_repository
                            .get_base_image_by_path("../test_pictures/0_1.jpg")
                            .await;
                        assert!(base_image_0_1.is_some());
                        let base_image_0_1 = base_image_0_1.unwrap();

                        let base_image_0_1_metadata = metadata_query_engine
                            .get_all_metadata_attached_to_base_image(&base_image_0_1)
                            .await
                            .expect("cannot get metadata");

                        assert_eq!(base_image_0_1_metadata.path, "../test_pictures/0_1.jpg");

                        assert_eq!(
                            hex::encode(base_image_0_1_metadata.image_hash[0].hash),
                            "d78f6226b8b5bab6ba377b9de4f2d7172336a82688e288fbfa85533d73dcd3c6"
                        );
                        assert_eq!(base_image_0_1_metadata.image_hash[0].hash_type, "SHA256");

                        assert_eq!(base_image_0_1_metadata.faces.len(), 0);

                        assert_eq!(
                            base_image_0_1_metadata.image_embedding[0].embedding.len(),
                            768
                        );

                        assert_eq!(base_image_0_1_metadata.basic_metadata[0].height, 882);
                        assert_eq!(base_image_0_1_metadata.basic_metadata[0].width, 1280);
                        assert_eq!(
                            base_image_0_1_metadata.basic_metadata[0].size_in_bytes,
                            138108
                        );
                        assert_eq!(
                            base_image_0_1_metadata.basic_metadata[0].file_extension,
                            Some("jpg".to_string())
                        );
                    }
                    // Image 0_2.jpg -> 0 People in there.
                    {
                        let base_image_0_2 = base_image_repository
                            .get_base_image_by_path("../test_pictures/0_2.jpg")
                            .await;
                        assert!(base_image_0_2.is_some());
                        let base_image_0_2 = base_image_0_2.unwrap();

                        let base_image_0_2_metadata = metadata_query_engine
                            .get_all_metadata_attached_to_base_image(&base_image_0_2)
                            .await
                            .expect("cannot get metadata");

                        assert_eq!(base_image_0_2_metadata.path, "../test_pictures/0_2.jpg");

                        assert_eq!(
                            hex::encode(base_image_0_2_metadata.image_hash[0].hash),
                            "5bd29a53940be3567570757683ea71493b81a94089a79986f79f7d2db19e4976"
                        );
                        assert_eq!(base_image_0_2_metadata.image_hash[0].hash_type, "SHA256");

                        assert_eq!(base_image_0_2_metadata.faces.len(), 0);

                        assert_eq!(
                            base_image_0_2_metadata.image_embedding[0].embedding.len(),
                            768
                        );

                        assert_eq!(base_image_0_2_metadata.basic_metadata[0].height, 801);
                        assert_eq!(base_image_0_2_metadata.basic_metadata[0].width, 1200);
                        assert_eq!(
                            base_image_0_2_metadata.basic_metadata[0].size_in_bytes,
                            150070
                        );
                        assert_eq!(
                            base_image_0_2_metadata.basic_metadata[0].file_extension,
                            Some("jpg".to_string())
                        );
                    }
                    // Image 0_3.jpg -> 0 People in there.
                    {
                        let base_image_0_3 = base_image_repository
                            .get_base_image_by_path("../test_pictures/0_3.jpg")
                            .await;
                        assert!(base_image_0_3.is_some());
                        let base_image_0_3 = base_image_0_3.unwrap();

                        let base_image_0_3_metadata = metadata_query_engine
                            .get_all_metadata_attached_to_base_image(&base_image_0_3)
                            .await
                            .expect("cannot get metadata");

                        assert_eq!(base_image_0_3_metadata.path, "../test_pictures/0_3.jpg");

                        assert_eq!(
                            hex::encode(base_image_0_3_metadata.image_hash[0].hash),
                            "58722cabb0a7ab17685eb3bda6ae9f356bcae3996130169eda8a0b03d0258065"
                        );
                        assert_eq!(base_image_0_3_metadata.image_hash[0].hash_type, "SHA256");

                        assert_eq!(base_image_0_3_metadata.faces.len(), 0);

                        assert_eq!(
                            base_image_0_3_metadata.image_embedding[0].embedding.len(),
                            768
                        );

                        assert_eq!(base_image_0_3_metadata.basic_metadata[0].height, 798);
                        assert_eq!(base_image_0_3_metadata.basic_metadata[0].width, 1200);
                        assert_eq!(
                            base_image_0_3_metadata.basic_metadata[0].size_in_bytes,
                            108473
                        );
                        assert_eq!(
                            base_image_0_3_metadata.basic_metadata[0].file_extension,
                            Some("jpg".to_string())
                        );
                    }
                    // Image 1_1.jpg -> 1 Person in there.
                    {
                        let base_image_1_1 = base_image_repository
                            .get_base_image_by_path("../test_pictures/1_1.jpg")
                            .await;
                        assert!(base_image_1_1.is_some());
                        let base_image_1_1 = base_image_1_1.unwrap();

                        let base_image_1_1_metadata = metadata_query_engine
                            .get_all_metadata_attached_to_base_image(&base_image_1_1)
                            .await
                            .expect("cannot get metadata");

                        assert_eq!(base_image_1_1_metadata.path, "../test_pictures/1_1.jpg");

                        assert_eq!(
                            hex::encode(base_image_1_1_metadata.image_hash[0].hash),
                            "c57fc6e6e7a6922eeb2815baee3d3405768968b1b98205be3713ec399f0a09ee"
                        );
                        assert_eq!(base_image_1_1_metadata.image_hash[0].hash_type, "SHA256");

                        assert_eq!(base_image_1_1_metadata.faces.len(), 1);

                        assert_eq!(
                            base_image_1_1_metadata.image_embedding[0].embedding.len(),
                            768
                        );

                        assert_eq!(base_image_1_1_metadata.basic_metadata[0].height, 1280);
                        assert_eq!(base_image_1_1_metadata.basic_metadata[0].width, 853);
                        assert_eq!(
                            base_image_1_1_metadata.basic_metadata[0].size_in_bytes,
                            76361
                        );
                        assert_eq!(
                            base_image_1_1_metadata.basic_metadata[0].file_extension,
                            Some("jpg".to_string())
                        );
                    }
                    // Image 3_1.jpg -> 3 Persons in there.
                    {
                        let base_image_3_1 = base_image_repository
                            .get_base_image_by_path("../test_pictures/3_1.jpg")
                            .await;
                        assert!(base_image_3_1.is_some());
                        let base_image_3_1 = base_image_3_1.unwrap();

                        let base_image_3_1_metadata = metadata_query_engine
                            .get_all_metadata_attached_to_base_image(&base_image_3_1)
                            .await
                            .expect("cannot get metadata");

                        assert_eq!(base_image_3_1_metadata.path, "../test_pictures/3_1.jpg");

                        assert_eq!(
                            hex::encode(base_image_3_1_metadata.image_hash[0].hash),
                            "5b3b05a8484dbfe7b483251e087f84a2e31a95867d509a9ea034a055509195a6"
                        );
                        assert_eq!(base_image_3_1_metadata.image_hash[0].hash_type, "SHA256");

                        assert_eq!(base_image_3_1_metadata.faces.len(), 3);

                        assert_eq!(
                            base_image_3_1_metadata.image_embedding[0].embedding.len(),
                            768
                        );

                        assert_eq!(base_image_3_1_metadata.basic_metadata[0].height, 853);
                        assert_eq!(base_image_3_1_metadata.basic_metadata[0].width, 1280);
                        assert_eq!(
                            base_image_3_1_metadata.basic_metadata[0].size_in_bytes,
                            247712
                        );
                        assert_eq!(
                            base_image_3_1_metadata.basic_metadata[0].file_extension,
                            Some("jpg".to_string())
                        );
                    }
                    // Image 7_1.jpg -> 7 Persons in there.
                    {
                        let base_image_7_1 = base_image_repository
                            .get_base_image_by_path("../test_pictures/7_1.jpg")
                            .await;
                        assert!(base_image_7_1.is_some());
                        let base_image_7_1 = base_image_7_1.unwrap();

                        let base_image_7_1_metadata = metadata_query_engine
                            .get_all_metadata_attached_to_base_image(&base_image_7_1)
                            .await
                            .expect("cannot get metadata");

                        assert_eq!(base_image_7_1_metadata.path, "../test_pictures/7_1.jpg");

                        assert_eq!(
                            hex::encode(base_image_7_1_metadata.image_hash[0].hash),
                            "836513f25131ef5497e8cb9ee0d696b5d9597bab0126cd6abaa7a8590fbda00e"
                        );
                        assert_eq!(base_image_7_1_metadata.image_hash[0].hash_type, "SHA256");

                        assert_eq!(base_image_7_1_metadata.faces.len(), 7);

                        assert_eq!(
                            base_image_7_1_metadata.image_embedding[0].embedding.len(),
                            768
                        );

                        assert_eq!(base_image_7_1_metadata.basic_metadata[0].height, 3887);
                        assert_eq!(base_image_7_1_metadata.basic_metadata[0].width, 6000);
                        assert_eq!(
                            base_image_7_1_metadata.basic_metadata[0].size_in_bytes,
                            1496498
                        );
                        assert_eq!(
                            base_image_7_1_metadata.basic_metadata[0].file_extension,
                            Some("jpg".to_string())
                        );
                    }
                });
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
