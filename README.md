# Image Search
This is a toy project of mine where i tried if my on image search works or not.  
The idea was, that all images are embedded with a AI-Model like `clip-vit-large-patch14`.  
Then one can search by a text query and get the images that match the query the most.  
After that i want to select images where the style, mood, pose of the person, you name it matches whatever i search for.  
Then the next round starts, the Vectors of the search query and the average vector of all selected images get averaged. We search Images with the resulting vector.  

As it turns out, this somehow worked out and one can get pritty precise searches with just a few rounds of selecting images.

# Used models

The model `clip-vit-large-patch14` is used in this project altough be it in two variants.

- In the Image processor the Model run on accelerator hardware using Burn with the webgpu backend.
- To Embed the text prompts EmbedAnything is used which runs only on CPU.
# Build requirements
- installed wasm toolchain `rustup target add wasm32-unknown-unknown`
- installed trunk  `cargo install trunk --locked`
- installed python version.  


## Build the project

### Local
```shell
pushd clip
  trunk build --release -d ../target/client/dist
popd
cargo build --release --bin server
```

### Docker
```shell
docker build -t image-search .
docker run --rm -p 3000:3000 \                                                                                                                                                                                                                                    8m 27.538s
    -e RUST_LOG=info \
    --device /dev/dri:/dev/dri \ # GPU Passthrough
    -v ~/Pictures:/pictures \
    --name embedded-server \
    localhost/embedder-server
```
### Podman
For podman we need to build and run the image as root to get access to the GPU devices.
```shell
sudo podman build -t localhost/embedder-server .
```

# Run requirements
- running surrealdb instance. For testing one can use `docker run --rm --pull always --name surrealdb -p 8000:8000 surrealdb/surrealdb:latest start --user root --pass root memory`.
- set the `model-weights` parameter where the model-weights are stored. They get exported on build to `models/vision_model.mpk`

# Benchmarks
Start the benchmarks with.
```shell
cargo bench --workspace
```
Get the report from [target/criterion/report/index.html](target/criterion/report/index.html).

## My results
Setup: AMD Ryzen 7 5800X, 64GB RAM, AMD Radeon 7900 XTX, Manjaro Linux  

| Benchmark                            | Time (Mean) | Time (Median) | Std Dev   |
|--------------------------------------|-------------|---------------|-----------|
| Embedding of a single string         | 175.99 ms   | 174.89 ms     | 11.483 ms |
| Embedding of a single image          | 37.496 ms   | 31.678 ms     | 8.6860 ms |
| Embedding of 10 images in a batch    | 346.90 ms   | 258.21 ms     | 120.07 ms |
| Embedding of 100 images in a batch   | 3.3392 s    | 2.4832 s      | 1.2344 s  |
| Preprocessing of a picture 1920x1080 | 2.5358 ms   | 2.5364 ms     | 39.870 µs | 
| Preprocessing of a picture 3840x2160 | 5.9815 ms   | 5.9764 ms     | 144.31 µs | 
