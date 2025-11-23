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
    // Each u32 holds 4 bytes. We find which u32 and which byte.
    let byteIndex = idx * 3u;

    let wordIndex = byteIndex / 4u;
    let byteInWord = byteIndex % 4u;

    let word = inputBytes[wordIndex];

    // Extract bytes safely
    let b0 = (word >> (8u * byteInWord)) & 0xFFu;

    // For bytes 2 and 3 we may spill into next word
    var r: u32;
    var g: u32;
    var b: u32;

    // Read R
    r = b0;

    // Read G
    if (byteInWord == 3u) {
        g = inputBytes[wordIndex + 1u] & 0xFFu;
    } else {
        g = (word >> (8u * (byteInWord + 1u))) & 0xFFu;
    }

    // Read B
    if (byteInWord >= 2u) {
        let nextWord = inputBytes[wordIndex + 1u];
        b = (nextWord >> (8u * (byteInWord - 2u))) & 0xFFu;
    } else {
        b = (word >> (8u * (byteInWord + 2u))) & 0xFFu;
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