struct Params {
    width: u32,
    height: u32,
};

@group(0) @binding(0) var<storage, read> inputBytes: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputCHW: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = (y * params.width + x);

    // ---- Read 3 bytes (RGB) ----
    // Each u32 holds 4 bytes in little-endian order.
    let byteIndex = idx * 3u;

    let wordIndex = byteIndex / 4u;
    let byteInWord = byteIndex % 4u;

    var r: u32;
    var g: u32;
    var b: u32;

    // We need to read 3 consecutive bytes starting at byteIndex
    // They might span across 2 u32 words

    if (byteInWord == 0u) {
        // RGB all in same word: [R, G, B, X]
        let word = inputBytes[wordIndex];
        r = (word >> 0u) & 0xFFu;
        g = (word >> 8u) & 0xFFu;
        b = (word >> 16u) & 0xFFu;
    } else if (byteInWord == 1u) {
        // RGB spans: [X, R, G, B]
        let word = inputBytes[wordIndex];
        r = (word >> 8u) & 0xFFu;
        g = (word >> 16u) & 0xFFu;
        b = (word >> 24u) & 0xFFu;
    } else if (byteInWord == 2u) {
        // RGB spans 2 words: [X, X, R, G] [B, ...]
        let word0 = inputBytes[wordIndex];
        let word1 = inputBytes[wordIndex + 1u];
        r = (word0 >> 16u) & 0xFFu;
        g = (word0 >> 24u) & 0xFFu;
        b = (word1 >> 0u) & 0xFFu;
    } else { // byteInWord == 3u
        // RGB spans 2 words: [X, X, X, R] [G, B, ...]
        let word0 = inputBytes[wordIndex];
        let word1 = inputBytes[wordIndex + 1u];
        r = (word0 >> 24u) & 0xFFu;
        g = (word1 >> 0u) & 0xFFu;
        b = (word1 >> 8u) & 0xFFu;
    }

    // ---- Convert to float (normalized 0.0–1.0) ----
    let rf = f32(r) / 255.0;
    let gf = f32(g) / 255.0;
    let bf = f32(b) / 255.0;

    // ---- Write to CHW ----
    let w = params.width;
    let h = params.height;

    let offsetR = 0u * w * h + idx;
    let offsetG = 1u * w * h + idx;
    let offsetB = 2u * w * h + idx;

    outputCHW[offsetR] = rf;
    outputCHW[offsetG] = gf;
    outputCHW[offsetB] = bf;
}