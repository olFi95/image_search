use image::DynamicImage;
use wgpu::util::DeviceExt;
use wgpu::wgt::PollType;

pub struct Preprocessor {
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    shader_module: wgpu::ShaderModule,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
}
impl Preprocessor {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                // input buffer RGB stuffed into u32 array.
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // output buffer f32 CHW
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params (image width, height)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform { },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Preprocessing shader (RGB to CHW)"),
            source: wgpu::ShaderSource::Wgsl(include_str!("preprocess.wgsl").into()),
        });



        Self {instance, device, queue, bind_group_layout, shader_module}
    }

    pub async fn preprocess(&self, data: &DynamicImage) -> Vec<f32> {
        let input = data.resize_exact(224, 224, image::imageops::FilterType::Nearest).to_rgb8().into_raw();
        let input: &[u8]= bytemuck::cast_slice(input.as_slice());
        let output: &[f32] = &[0.0; 3 * 224 * 224];
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input buffer RGB"),
            contents: input,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output buffer CHW"),
            size: (3 * 224 * 224 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params buffer"),
            contents: bytemuck::bytes_of(&Params { width: 224, height: 224 }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });



        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
            label: Some("preprocess Bind Group"),
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preprocess Pipeline Layout"),
            bind_group_layouts: &[&self.bind_group_layout],
            push_constant_ranges: &[],
        });
        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("preprocess Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &self.shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut compute_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut compute_pass = compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("preprocess pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 16u32;
            let dispatch_x = (224 + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (224 + workgroup_size - 1) / workgroup_size;

            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);        }

        let command_buffer = compute_encoder.finish();
        let preprocess_index = self.queue.submit(Some(command_buffer));
        // GPU flush
        self.device.poll(PollType::Wait {submission_index: Some(preprocess_index), timeout: None}).expect("cannot poll device");


        let mut readback_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback Encoder"),
            });
        let buffer_size = (std::mem::size_of::<f32>() * 3*224*224) as u64;
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Readback Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        readback_encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &readback_buffer,
            0,
            buffer_size,
        );
        let command_buffer = readback_encoder.finish();
        let readback_index = self.queue.submit(Some(command_buffer));

        let (sender, receiver) = futures_channel::oneshot::channel();
        readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| {
                let _ = sender.send(result);
            });
        self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(readback_index),
            timeout: None,
        }).expect("cannot poll device");
        receiver
            .await
            .expect("communication failed")
            .expect("buffer reading failed");
        let slice: &[u8] = &readback_buffer.slice(..).get_mapped_range();
        let float_slice: &[f32] = bytemuck::cast_slice(slice);
        float_slice.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};

    #[tokio::test]
    async fn test_preprocessor() {
        let preprocessor = Preprocessor::new().await;
        let img = image::open("../testdata/pictures/cat.jpg").unwrap();
        let result = preprocessor.preprocess(&img).await;
        assert_eq!(result.len(), 3 * 224 * 224);
    }

    #[tokio::test]
    async fn test_shader_logic_and_pixel_values() {
        // Create a test image with known values
        // We'll create a 224x224 image so the resize doesn't alter our test pixels
        let mut img = RgbImage::new(224, 224);

        // Fill with a base color (black)
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([0, 0, 0]);
        }

        // Set specific test pixels with known values
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));     // Red at top-left
        img.put_pixel(223, 0, image::Rgb([0, 255, 0]));   // Green at top-right
        img.put_pixel(0, 223, image::Rgb([0, 0, 255]));   // Blue at bottom-left
        img.put_pixel(223, 223, image::Rgb([128, 128, 128])); // Gray at bottom-right
        img.put_pixel(112, 112, image::Rgb([255, 255, 255])); // White at center

        let dynamic_img = DynamicImage::ImageRgb8(img);

        let preprocessor = Preprocessor::new().await;
        let result = preprocessor.preprocess(&dynamic_img).await;

        // Assert correct output size
        let expected_size = 3 * 224 * 224;
        assert_eq!(result.len(), expected_size, "Output size mismatch: expected {} (3 * 224 * 224), got {}", expected_size, result.len());

        // Assert all values are in valid range [0.0, 1.0]
        let min_val = result.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        assert!(min_val >= 0.0, "Minimum value {} is below 0.0", min_val);
        assert!(max_val <= 1.0, "Maximum value {} is above 1.0", max_val);

        // Assert CHW layout is correct
        let pixels_per_channel = 224 * 224;

        // Check that we have exactly 3 channels
        assert_eq!(result.len(), 3 * pixels_per_channel, "CHW layout incorrect: expected {} values for 3 channels", 3 * pixels_per_channel);

        // Helper function to get pixel value from CHW layout
        let get_pixel = |x: u32, y: u32| -> (f32, f32, f32) {
            let idx = (y * 224 + x) as usize;
            let r = result[idx];
            let g = result[pixels_per_channel + idx];
            let b = result[2 * pixels_per_channel + idx];
            (r, g, b)
        };

        // Test pixel (0, 0) - should be red (255, 0, 0) = (1.0, 0.0, 0.0)
        let (r, g, b) = get_pixel(0, 0);
        assert!((r - 1.0).abs() < 0.01, "Red pixel: R channel should be ~1.0, got {}", r);
        assert!(g.abs() < 0.01, "Red pixel: G channel should be ~0.0, got {}", g);
        assert!(b.abs() < 0.01, "Red pixel: B channel should be ~0.0, got {}", b);

        // Test pixel (223, 0) - should be green (0, 255, 0) = (0.0, 1.0, 0.0)
        let (r, g, b) = get_pixel(223, 0);
        assert!(r.abs() < 0.01, "Green pixel: R channel should be ~0.0, got {}", r);
        assert!((g - 1.0).abs() < 0.01, "Green pixel: G channel should be ~1.0, got {}", g);
        assert!(b.abs() < 0.01, "Green pixel: B channel should be ~0.0, got {}", b);

        // Test pixel (0, 223) - should be blue (0, 0, 255) = (0.0, 0.0, 1.0)
        let (r, g, b) = get_pixel(0, 223);
        assert!(r.abs() < 0.01, "Blue pixel: R channel should be ~0.0, got {}", r);
        assert!(g.abs() < 0.01, "Blue pixel: G channel should be ~0.0, got {}", g);
        assert!((b - 1.0).abs() < 0.01, "Blue pixel: B channel should be ~1.0, got {}", b);

        // Test pixel (223, 223) - should be gray (128, 128, 128) = (0.502, 0.502, 0.502)
        let (r, g, b) = get_pixel(223, 223);
        let expected_gray = 128.0 / 255.0; // ~0.502
        assert!((r - expected_gray).abs() < 0.01, "Gray pixel: R channel should be ~{:.3}, got {}", expected_gray, r);
        assert!((g - expected_gray).abs() < 0.01, "Gray pixel: G channel should be ~{:.3}, got {}", expected_gray, g);
        assert!((b - expected_gray).abs() < 0.01, "Gray pixel: B channel should be ~{:.3}, got {}", expected_gray, b);

        // Test pixel (112, 112) - should be white (255, 255, 255) = (1.0, 1.0, 1.0)
        let (r, g, b) = get_pixel(112, 112);
        assert!((r - 1.0).abs() < 0.01, "White pixel: R channel should be ~1.0, got {}", r);
        assert!((g - 1.0).abs() < 0.01, "White pixel: G channel should be ~1.0, got {}", g);
        assert!((b - 1.0).abs() < 0.01, "White pixel: B channel should be ~1.0, got {}", b);

        // Test a black pixel (should be most of the image)
        let (r, g, b) = get_pixel(50, 50);
        assert!(r.abs() < 0.01, "Black pixel: R channel should be ~0.0, got {}", r);
        assert!(g.abs() < 0.01, "Black pixel: G channel should be ~0.0, got {}", g);
        assert!(b.abs() < 0.01, "Black pixel: B channel should be ~0.0, got {}", b);
    }
}