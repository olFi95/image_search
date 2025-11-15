# --- Stage 1: Build ---
FROM rust:1.91.1 as builder
WORKDIR /app
RUN cargo install trunk
RUN rustup target add wasm32-unknown-unknown
RUN apt-get update  \
    && apt-get install -y  \
        python3  \
        python3-pip  \
        python3-venv  \
    && rm -rf /var/lib/apt/lists/*
COPY . .
RUN mkdir -p target/client/dist
RUN cd client && trunk build --release -d ../target/client/dist && cd ..
RUN cargo build --release --bin server

# --- Stage 2: Runtime ---
FROM debian:trixie-slim
WORKDIR /app
RUN mkdir /pictures
# Copy server binary and model file
COPY --from=builder /app/target/release/server /app/server
COPY --from=builder /app/models/vision_model.mpk /app/models/vision_model.mpk
RUN chmod +x /app/server
EXPOSE 3000
ENTRYPOINT ["server", "-w", "/app/models/vision_model.mpk", "-m", "/pictures"]


