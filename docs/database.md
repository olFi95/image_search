
All data is stored in a Surrealdb.
The database can be in memory `mem://`, on disk `rocksdb://my_database` or a remote surrealdb 3.0.0+ instance `ws://surrealdb_server:8000`.

The database schema is as follows:
```mermaid
erDiagram
    base_image {
        id RecordId
        path String
    }
    basic_metadata {
        id RecordId
        created DateTime
        file_extension String
        height u32
        width u32
        size_in_bytes u64
    }
    image_embedding_vector {
        id RecordId
        embedding f32[768]
    }
    image_hash_metadata {
        id RecordId
        hash u8[32]
        hash_type String
    }
    face_in_picture {
        top_left_x f32
        top_left_y f32
        bottom_right_x f32
        bottom_right_y f32
        confidence f32
    }
    face_age_and_gender_estimation {
        age f32
        gender f32
    }
    face_in_picture_vector {
        embedding f32[512]
    }

    base_image      ||--o| basic_metadata : has_basic_metadata
    base_image      ||--o| image_embedding_vector : has_image_embedding_vector
    base_image      ||--o| image_hash_metadata : has_image_hash_metadata
    face_in_picture ||--o| face_age_and_gender_estimation : has_face_age_and_gender_estimation
    base_image      ||--o{ face_in_picture : has_face_in_picture
    face_in_picture ||--o| face_in_picture_vector : has_face_in_picture_vector

```