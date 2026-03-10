# Metadata Indexer - Data Flow Overview

This documentation describes the data flow of the Metadata Indexer, which processes images and extracts various metadata.

## Architecture

The Metadata Indexer uses a pipeline architecture with multiple parallel components communicating via channels (`crossbeam_channel`).

## Legend

| Symbol | Meaning |
|--------|---------|
| 🤖 | **AI Model** - Neural network inference node |
| 💾 | Database operation (save/load) |
| 📁 | File system operation |
| 📤 | Data distribution/routing |

## Data Flow Diagram

```mermaid
flowchart TD
    subgraph Input
        A[📁 Image Paths<br/>get_all_directories_in_dir]
    end

    subgraph Producer
        B[📁 Producer<br/>Create BaseImage]
    end

    subgraph BaseImageProcessing
        C[💾 Base Image Saver<br/>BaseImageRepository]
    end

    subgraph ImageLoading
        D[📁 Image Loader<br/>Load images with Rayon]
    end

    subgraph Dispatcher
        E[📤 Image Dispatcher<br/>Distributes to 3 Pipelines]
    end

    subgraph EmbeddingPipeline["Embedding Pipeline"]
        F[🤖 Image Embedder<br/>CLIP Vision Model<br/>ImageEmbeddingMetadataProvider]
        G[💾 Embedding Saver<br/>ImageEmbeddingMetadataRepository]
    end

    subgraph BasicMetadataPipeline["Basic Metadata Pipeline"]
        H[📊 Basic Extractor<br/>BasicMetadataProvider<br/>ImageHashMetadataProvider]
        I[💾 Hash Saver<br/>ImageHashMetadataRepository]
        J[💾 Basic Saver<br/>BasicMetadataRepository]
    end

    subgraph FacePipeline["Face Recognition Pipeline"]
        K[🤖 Face Detector<br/>YOLO<br/>FaceRecognitionMetadataProvider]
        K2[🤖 Face Embedder<br/>ArcFace<br/>FaceRecognitionMetadataProvider]
        L[💾 Face In Picture Saver<br/>FaceRecognitionMetadataRepository]
        M[💾 Face Embedding Saver<br/>FaceRecognitionMetadataRepository]
        N[🤖 Age & Gender Estimator<br/>Age/Gender CNN<br/>AgeAndGenderMetadataProvider]
        O[💾 Age & Gender Saver<br/>FaceAgeAndGenderMetadataRepository]
    end

    subgraph Database["🗄️ SurrealDB"]
        DB[(Database)]
    end

    A --> B
    B -->|tx_base_image<br/>BUFFER: 100| C
    C -->|tx_base_with_id<br/>BUFFER: 100| D
    D -->|tx_loaded<br/>BUFFER: 100| E

    E -->|tx_for_embedding<br/>BUFFER: 100| F
    E -->|tx_for_basic_metadata<br/>BUFFER: 100| H
    E -->|tx_for_face<br/>BUFFER: 100| K

    F -->|tx_image_embedding<br/>BUFFER: 100| G

    H -->|tx_hash<br/>BUFFER: 100| I
    H -->|tx_basic<br/>BUFFER: 100| J

    K -->|tx_face_for_db<br/>BUFFER: 100| L
    K --> K2
    K2 -->|tx_face_embedding<br/>BUFFER: 100| M
    L -->|tx_face_for_age_gender_with_id<br/>BUFFER: 100| N
    N -->|tx_age_gender<br/>BUFFER: 100| O

    G --> DB
    I --> DB
    J --> DB
    L --> DB
    M --> DB
    O --> DB
    C --> DB

    style F fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px,color:#fff
    style K fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px,color:#fff
    style K2 fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px,color:#fff
    style N fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px,color:#fff
```

## Component Description

### 1. Producer
- **Task**: Creates `BaseImage` objects from file paths
- **Input**: All image paths from directory
- **Output**: `BaseImage` objects

### 2. Base Image Saver
- **Task**: Persists base image information to database
- **Repository**: `BaseImageRepository`
- **Output**: `BaseImage` with database ID

### 3. Image Loader
- **Task**: Loads images in parallel using Rayon (CPU-bound)
- **Input**: `BaseImage`
- **Output**: `BaseImageWithImage` (contains loaded image data)

### 4. Image Dispatcher
- **Task**: Distributes loaded images to three parallel pipelines
- **Output Channels**:
  - `tx_for_embedding` → Embedding Pipeline
  - `tx_for_basic_metadata` → Basic Metadata Pipeline
  - `tx_for_face` → Face Recognition Pipeline

### 5. Embedding Pipeline
| Component | Description | AI Model |
|-----------|-------------|----------|
| 🤖 Image Embedder | Creates 768-dimensional image embeddings | **CLIP Vision Model** |
| Embedding Saver | Persists embeddings to database | - |

### 6. Basic Metadata Pipeline
| Component | Description | AI Model |
|-----------|-------------|----------|
| Basic Extractor | Extracts hash (SHA256) and basic metadata (size, dimensions, extension) | - |
| Hash Saver | Persists image hashes | - |
| Basic Saver | Persists basic metadata | - |

### 7. Face Recognition Pipeline
| Component | Description | AI Model |
|-----------|-------------|----------|
| 🤖 Face Detector | Detects faces in images | **YOLO** |
| 🤖 Face Embedder | Creates face embeddings from detected faces | **ArcFace** |
| Face In Picture Saver | Persists detected faces | - |
| Face Embedding Saver | Persists face embeddings | - |
| 🤖 Age & Gender Estimator | Estimates age and gender for each face | **Age/Gender CNN** |
| Age & Gender Saver | Persists age/gender estimates | - |

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BUFFER` | 100 | Channel buffer size |
| `BATCH` | 25 | Batch size for processing |

## AI Models Used

| Model | File | Purpose | Architecture |
|-------|------|---------|--------------|
| 🤖 YOLO | `yolo.bpk` | Face Detection | YOLOv8 |
| 🤖 ArcFace | `arcface_model.bpk` | Face Embedding (512-dim) | ResNet-based |
| 🤖 CLIP Vision | `vision_model.bpk` | Image Embedding (768-dim) | Vision Transformer |
| 🤖 Age/Gender | `age_gender.bpk` | Age & Gender Estimation | CNN |

## Data Types

```mermaid
classDiagram
    class BaseImage {
        +PathBuf path
        +Option~String~ id
    }

    class BaseImageWithImage {
        +BaseImage base
        +DynamicImage image
    }

    class ImageEmbedding {
        +Vec~f32~ embedding
    }

    class BasicMetadata {
        +u32 width
        +u32 height
        +u64 size_in_bytes
        +Option~String~ file_extension
    }

    class ImageHashMetadata {
        +Vec~u8~ hash
        +String hash_type
    }

    class FaceInPicture {
        +BoundingBox bbox
        +f32 confidence
    }

    class FaceEmbedding {
        +Vec~f32~ embedding
    }

    class FaceAgeAndGender {
        +f32 age
        +String gender
    }

    BaseImage <|-- BaseImageWithImage
    BaseImageWithImage --> ImageEmbedding
    BaseImageWithImage --> BasicMetadata
    BaseImageWithImage --> ImageHashMetadata
    BaseImageWithImage --> FaceInPicture
    FaceInPicture --> FaceEmbedding
    FaceInPicture --> FaceAgeAndGender
```

## Parallelization

- **Tokio Tasks**: Asynchronous task processing for I/O-bound operations
- **Rayon**: Parallel processing for CPU-bound operations (Image Loading)
- **Crossbeam Channels**: Thread-safe communication between tasks

## Error Handling

- On full channel: Retry with 1ms sleep
- Logging of stall times for performance monitoring
- Graceful shutdown through channel drop propagation

