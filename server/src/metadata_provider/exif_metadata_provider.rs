use crate::metadata_provider::model::{
    BaseImageWithImage, Metadata, MetadataProvider,
};
use log::{error, warn};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use anyhow::Context;
use chrono::Datelike;
use image::metadata::Orientation;
use surrealdb::{Connection, Surreal};

pub struct ExifMetadataProvider;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum NomDatetime {
    Time(chrono::DateTime<chrono::FixedOffset>),
    NaiveDateTime(chrono::NaiveDateTime),
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ExifMetadata {
    pub date_time_original: Option<NomDatetime>,
    pub year: Option<i32>,
    pub height: Option<u32>,
    pub width: Option<u32>,
    pub orientation: Option<image::metadata::Orientation>,
    pub make: Option<String>,
    pub model: Option<String>,
    pub copyright: Option<String>,

}

impl NomDatetime {
    fn year(&self) -> i32 {
        match self {
            NomDatetime::Time(time) => time.year(),
            NomDatetime::NaiveDateTime(date) => date.year()
        }
    }
}
fn convert_nom_date_entry(value: &nom_exif::EntryValue) -> Option<NomDatetime> {
    match value {
        nom_exif::EntryValue::Time(time) => {
            Some(NomDatetime::Time(*time))
        }
        nom_exif::EntryValue::NaiveDateTime(naive) => {
            Some(NomDatetime::NaiveDateTime(*naive))
        }
        _ => {
            None
        }
    }
}
fn convert_nom_unsigned_integer_entry(value: &nom_exif::EntryValue) -> Option<u32> {
    match value {
        nom_exif::EntryValue::U8(u8) => {
            Some(*u8 as u32)
        }
        nom_exif::EntryValue::U16(u16) => {
            Some(*u16 as u32)
        }
        nom_exif::EntryValue::U32(u32) => {
            Some(*u32)
        },
        &_ => None
    }
}



fn read_selected_nom_exif(path: &Path) -> anyhow::Result<ExifMetadata> {
    let mut parser = nom_exif::MediaParser::new();
    let ms = nom_exif::MediaSource::file_path(path)
        .context(format!("Failed reading image metadata from file {}", path.display()))?;
    let iter: nom_exif::ExifIter = parser.parse(ms)
        .context(format!("Could not parse exif from file {}", path.display()))?;
    let exif: nom_exif::Exif = iter.clone().into();
    let date_time_original = exif.get(nom_exif::ExifTag::DateTimeOriginal)
        .and_then(convert_nom_date_entry);
    let height = exif.get(nom_exif::ExifTag::ExifImageHeight)
        .and_then(convert_nom_unsigned_integer_entry);
    let width = exif.get(nom_exif::ExifTag::ExifImageWidth)
        .and_then(convert_nom_unsigned_integer_entry);
    let orientation = exif.get(nom_exif::ExifTag::Orientation)
        .and_then(|orientation| orientation.as_u16())
        .and_then(|orientation| Orientation::from_exif(orientation as u8));
    let copyright = exif.get(nom_exif::ExifTag::Copyright)
        .and_then(|value| value.as_str())
        .map(|s| s.to_string());
    let make = exif.get(nom_exif::ExifTag::Make)
        .and_then(|value| value.as_str())
        .map(|s| s.to_string());
    let model = exif.get(nom_exif::ExifTag::Model)
        .and_then(|value| value.as_str())
        .map(|s| s.to_string());

    let year = date_time_original.clone().map(|date| date.year());
    Ok(ExifMetadata {
        date_time_original,
        year,
        height,
        width,
        orientation,
        make,
        model,
        copyright,
    })
}

impl MetadataProvider<BaseImageWithImage, ExifMetadata> for ExifMetadataProvider {
    fn extract(
        &self,
        base_images: &[BaseImageWithImage],
    ) -> anyhow::Result<Vec<Metadata<ExifMetadata>>> {
        let results: Vec<Metadata<ExifMetadata>> = base_images
            .par_iter()
            .map(|base_image| {
                let path = PathBuf::from(base_image.base_image.path.clone());
                let exif_result = read_selected_nom_exif(&path);
                match exif_result {
                    Ok(exif) => {
                        Ok(Metadata {
                            id: None,
                            metadata: Some(exif),
                            base: base_image.base_image.id.clone(),
                        })
                    }
                    Err(error) => {
                        warn!("Unable to get file EXIF metadata for image {} due to error {}", &base_image.base_image.path, error);
                        Err(error)
                    }
                }
            })
            .flat_map(|metadata_result| metadata_result.ok())
            .collect();

        Ok(results)
    }
}

static EXIF_METADATA_DATA_NAME: &str = "exif_metadata";
static EXIF_METADATA_RELATION_NAME: &str = "has_exif_metadata";

pub struct ExifMetadataRepository<C: Connection> {
    db: Surreal<C>,
}

impl <C: Connection>ExifMetadataRepository<C> {
    pub async fn new(db: Surreal<C>) -> Self {
        Self::prepare_repository(&db)
            .await
            .expect("cannot prepare repository with indexes");
        Self { db }
    }
    async fn prepare_repository(db: &Surreal<C>) -> anyhow::Result<()> {
        db.query(format!(
            r#"
            DEFINE INDEX IF NOT EXISTS {EXIF_METADATA_DATA_NAME}_base_unique
            ON {EXIF_METADATA_DATA_NAME}
            FIELDS base
            UNIQUE;
            "#
        ))
        .await?;

        Ok(())
    }

    pub async fn insert_many(
        &self,
        items: &Vec<Metadata<ExifMetadata>>,
    ) -> anyhow::Result<Vec<Metadata<ExifMetadata>>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let mut inserted = Vec::new();

        for item in items {
            let base_id = item
                .base
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Base ID missing"))?;

            let metadata = match item.metadata.clone() {
                Some(metadata) => metadata,
                None => {
                    error!("ExifMetadata is missing for ID {:?}", item.id);
                    continue;
                }
            };
            let mut response = self
                .db
                .query(format!(
                    r#"
                LET $tmp = (
                    UPSERT {EXIF_METADATA_DATA_NAME}
                    SET date_time_original = $date_time_original,
                        year = $year,
                        height = $height,
                        width = $width,
                        orientation = $orientation,
                        make = $make,
                        model = $model,
                        copyright = $copyright
                );
                LET $id = $tmp[0].id;
                RELATE $base -> {EXIF_METADATA_RELATION_NAME} -> $id;
                "#
                ))
                // reference to base image
                .bind(("base", base_id.clone()))
                // Exif metadata
                .bind(("date_time_original", metadata.date_time_original))
                .bind(("year", metadata.year))
                .bind(("height", metadata.height))
                .bind(("width", metadata.width))
                .bind(("orientation", metadata.orientation))
                .bind(("make", metadata.make))
                .bind(("model", metadata.model))
                .bind(("copyright", metadata.copyright))
                .await
                .context("ExifMetadataRepository::insert_many Database insert failed")?;

            if let Ok(mut rows) = response.take::<Vec<Metadata<ExifMetadata>>>(0) {
                inserted.append(&mut rows);
            }
        }

        Ok(inserted)
    }
}
