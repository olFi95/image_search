#[cfg(feature = "age_gender")]
pub mod face_age_and_gender_estimator;
#[cfg(feature = "yolo")]
pub mod face_detector;
#[cfg(feature = "arcface")]
pub mod face_embedder;
#[cfg(feature = "clip")]
pub mod image_embedder;
pub mod utils;

extern crate alloc;

#[cfg(feature = "arcface")]
#[allow(warnings)]
pub mod arcface {
    include!(concat!(env!("OUT_DIR"), "/arcface/arc.rs"));
}
#[cfg(feature = "yolo")]
#[allow(warnings)]
pub mod yolo {
    include!(concat!(env!("OUT_DIR"), "/yolo/yolo.rs"));
}
#[cfg(feature = "age_gender")]
// It's not great but the warning is annoying and not directly related to the code in this repository
#[allow(clippy::eq_op)]
#[allow(warnings)]
pub mod age_gender {
    include!(concat!(env!("OUT_DIR"), "/age_gender/age_gender.rs"));
}

#[cfg(feature = "clip")]
#[allow(warnings)]
pub mod clip {
    include!(concat!(
    env!("OUT_DIR"),
    "/clip_vit_large_patch14/vision_model.rs"
    ));
}
