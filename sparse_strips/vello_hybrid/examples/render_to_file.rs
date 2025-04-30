// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SVG rendering example for headless rendering
//!
//! This example demonstrates rendering an SVG file without a window or display.
//! It takes an input SVG file and renders it to a PNG file using the hybrid CPU/GPU renderer.

use std::io::BufWriter;
use vello_common::color::palette::css::{BLACK, DARK_BLUE, DARK_GREEN, REBECCA_PURPLE, RED};
use vello_common::kurbo::{Affine, BezPath, Circle, Point, Rect, Shape, Stroke, Vec2};
use vello_common::paint::Paint;
use vello_common::pico_svg::{Item, PicoSvg};
use vello_common::pixmap::Pixmap;
use vello_hybrid::{DimensionConstraints, Scene};

/// Main entry point for the headless rendering example.
/// Takes two command line arguments:
/// - Input SVG filename to render
/// - Output PNG filename to save the rendered result
///
/// Renders the SVG using the hybrid CPU/GPU renderer and saves the output as a PNG file.
fn main() {
    pollster::block_on(run());
}

enum SceneType {
    Rect,
    Star,
    NestedRect,
}

const SCENE_TYPE: SceneType = SceneType::NestedRect;
//const SCENE_TYPE: SceneType = SceneType::Rect;

/// Draws a simple scene with shapes
pub fn render(ctx: &mut Scene) {
    match SCENE_TYPE {
        SceneType::Rect => {
            // Create first clipping region - a rectangle on the left side
            let clip_rect = Rect::new(10.0, 30.0, 50.0, 70.0);

            // Then a filled rectangle that covers most of the canvas
            let large_rect = Rect::new(0.0, 0.0, 100.0, 100.0);

            let stroke = Stroke::new(10.0);
            ctx.set_paint(DARK_BLUE.into());
            ctx.set_stroke(stroke);
            ctx.stroke_rect(&clip_rect);

            ctx.push_clip_layer(&clip_rect.to_path(0.1));
            ctx.set_paint(RED.into());
            ctx.fill_rect(&large_rect);
            ctx.pop_layer();
        }
        SceneType::Star => {
            fn circular_star(center: Point, n: usize, inner: f64, outer: f64) -> BezPath {
                let mut path = BezPath::new();
                let start_angle = -std::f64::consts::FRAC_PI_2;
                path.move_to(center + outer * Vec2::from_angle(start_angle));
                for i in 1..n * 2 {
                    let th = start_angle + i as f64 * std::f64::consts::PI / n as f64;
                    let r = if i % 2 == 0 { outer } else { inner };
                    path.line_to(center + r * Vec2::from_angle(th));
                }
                path.close_path();
                path
            }
            let mut triangle_path = BezPath::new();
            triangle_path.move_to((10.0, 10.0));
            triangle_path.line_to((90.0, 20.0));
            triangle_path.line_to((20.0, 90.0));
            triangle_path.close_path();

            let stroke = Stroke::new(1.0);
            ctx.set_paint(DARK_BLUE.into());
            ctx.set_stroke(stroke);
            ctx.stroke_path(&triangle_path);

            let star_path = circular_star(Point::new(50., 50.), 13, 25., 45.);

            ctx.push_clip_layer(&star_path);
            ctx.set_paint(REBECCA_PURPLE.into());
            ctx.fill_path(&triangle_path);
            ctx.pop_layer();
        }
        SceneType::NestedRect => {
            // Create a series of 10 nested rectangles with clipping
            let colors: [Paint; 10] = [
                RED.into(),
                DARK_BLUE.into(),
                DARK_GREEN.into(),
                REBECCA_PURPLE.into(),
                BLACK.into(),
                RED.into(),
                DARK_BLUE.into(),
                DARK_GREEN.into(),
                REBECCA_PURPLE.into(),
                BLACK.into(),
            ];

            // Start with a large rectangle (100x100) and decrease by 10 each time
            let mut size = 100.0;
            let mut offset = 0.0;

            const COUNT: usize = 2;

            for i in 0..COUNT {
                // Create a rectangle for clipping
                let clip_rect = Rect::new(offset, offset, size, size);
                ctx.set_paint(BLACK.into());
                // visualise clip
                ctx.stroke_rect(&clip_rect);
                // Draw a filled rectangle with current color
                ctx.set_paint(colors[i].clone());

                // Push clip layer
                ctx.push_clip_layer(&clip_rect.to_path(0.1));

                ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));

                // Decrease size and increase offset for next iteration
                size -= 10.0;
                offset += 5.0;
            }

            // Pop all clip layers
            for _ in 0..COUNT {
                ctx.pop_layer();
            }
        }
    }
}

async fn run() {
    let mut args = std::env::args().skip(1);
    let svg_filename: String = args.next().expect("svg filename is first arg");
    let output_filename: String = args.next().expect("output filename is second arg");
    let svg = std::fs::read_to_string(svg_filename).expect("error reading file");
    let render_scale = 5.0;
    let parsed = PicoSvg::load(&svg, 1.0).expect("error parsing SVG");

    let constraints = DimensionConstraints::default();
    let svg_width = parsed.size.width * render_scale;
    let svg_height = parsed.size.height * render_scale;
    let (width, height) = constraints.calculate_dimensions(svg_width, svg_height);

    let width = 100 as u16;
    let height = 100 as u16;

    let mut scene = Scene::new(width, height);
    render(&mut scene);
    //render_svg(&mut scene, &parsed.items, Affine::scale(render_scale));

    // Initialize wgpu device and queue for GPU rendering
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("Failed to find an appropriate adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .expect("Failed to create device");

    // Create a render target texture
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Render Target"),
        size: wgpu::Extent3d {
            width: width.into(),
            height: height.into(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Create renderer and render the scene to the texture
    let mut renderer = vello_hybrid::Renderer::new(
        &device,
        &vello_hybrid::RenderTargetConfig {
            format: texture.format(),
            width: width.into(),
            height: height.into(),
        },
    );
    let render_size = vello_hybrid::RenderSize {
        width: width.into(),
        height: height.into(),
    };
    //let render_data = scene.prepare_render_data();
    //renderer.prepare(&device, &queue, &render_data, &render_size);
    // Copy texture to buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Vello Render To Buffer"),
    });

    // Create a vector to hold debug buffers from render passes
    let mut debug_buffers = Vec::new();

    {
        renderer.render2(
            &scene,
            &device,
            &queue,
            &mut encoder,
            &render_size,
            &texture_view,
            Some(&mut debug_buffers),
        );
    }

    // Get clip texture dimensions
    let clip_texture_width = vello_common::coarse::WideTile::WIDTH as u32;
    let clip_texture_height =
        vello_common::tile::Tile::HEIGHT as u32 * vello_hybrid::Renderer::N_SLOTS as u32;

    // Create buffer for the main texture
    let bytes_per_row = (u32::from(width) * 4).next_multiple_of(256);
    let texture_copy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: u64::from(bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Copy main texture to buffer
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &texture_copy_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: width.into(),
            height: height.into(),
            depth_or_array_layers: 1,
        },
    );

    queue.submit([encoder.finish()]);

    // Map the main buffer for reading
    texture_copy_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            if result.is_err() {
                panic!("Failed to map texture for reading");
            }
        });

    // Map all debug buffers for reading
    for (_, buffer) in &debug_buffers {
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_err() {
                    panic!("Failed to map debug texture for reading");
                }
            });
    }

    device.poll(wgpu::Maintain::Wait);

    // Calculate the total width needed for the full debug output
    // Main texture + all debug buffers
    let debug_buffer_count = debug_buffers.len();
    let total_width = width as u32 + (clip_texture_width * debug_buffer_count as u32);

    // Create a new pixmap that can fit all textures side by side
    let mut pixmap = Pixmap::new(total_width as u16, height.max(clip_texture_height as u16));

    // Fill the pixmap with black pixels
    for i in 0..pixmap.buf.len() {
        pixmap.buf[i] = 0;
    }

    // Read the main texture data
    let main_data = texture_copy_buffer.slice(..).get_mapped_range();

    // Copy the main texture data to the pixmap
    let main_width = width as usize;
    let main_height = height as usize;
    for y in 0..main_height {
        let src_offset = y * bytes_per_row as usize;
        let dst_offset = y * (total_width as usize * 4);
        for x in 0..main_width {
            let src_pixel_offset = src_offset + x * 4;
            let dst_pixel_offset = dst_offset + x * 4;
            if src_pixel_offset + 3 < main_data.len() && dst_pixel_offset + 3 < pixmap.buf.len() {
                pixmap.buf[dst_pixel_offset] = main_data[src_pixel_offset];
                pixmap.buf[dst_pixel_offset + 1] = main_data[src_pixel_offset + 1];
                pixmap.buf[dst_pixel_offset + 2] = main_data[src_pixel_offset + 2];
                pixmap.buf[dst_pixel_offset + 3] = main_data[src_pixel_offset + 3];
            }
        }
    }

    // Copy all debug buffers to the pixmap
    let clip_texture_bytes_per_row = 256 * 4;
    for (i, (label, buffer)) in debug_buffers.iter().enumerate() {
        let debug_data = buffer.slice(..).get_mapped_range();
        println!("Adding debug texture: {}", label);

        let clip_width = clip_texture_width as usize;
        let clip_height = clip_texture_height.min(pixmap.height() as u32) as usize;

        let x_offset = main_width + (i * clip_width);

        for y in 0..clip_height {
            let src_offset = y * clip_texture_bytes_per_row as usize;
            let dst_offset = y * (total_width as usize * 4) + x_offset * 4;

            for x in 0..clip_width {
                let src_pixel_offset = src_offset + x * 4;
                let dst_pixel_offset = dst_offset + x * 4;

                if src_pixel_offset + 3 < debug_data.len()
                    && dst_pixel_offset + 3 < pixmap.buf.len()
                {
                    pixmap.buf[dst_pixel_offset] = debug_data[src_pixel_offset];
                    pixmap.buf[dst_pixel_offset + 1] = debug_data[src_pixel_offset + 1];
                    pixmap.buf[dst_pixel_offset + 2] = debug_data[src_pixel_offset + 2];
                    pixmap.buf[dst_pixel_offset + 3] = debug_data[src_pixel_offset + 3];
                }
            }
        }
    }

    // Unmap the buffers
    drop(main_data);
    texture_copy_buffer.unmap();
    for (_, buffer) in &debug_buffers {
        drop(buffer.slice(..).get_mapped_range());
    }

    pixmap.unpremultiply();

    // Write the pixmap to a file
    let file = std::fs::File::create(output_filename).unwrap();
    let w = BufWriter::new(file);
    let mut png_encoder = png::Encoder::new(w, total_width, pixmap.height().into());
    png_encoder.set_color(png::ColorType::Rgba);
    let mut writer = png_encoder.write_header().unwrap();
    writer.write_image_data(&pixmap.buf).unwrap();
}

fn render_svg(ctx: &mut Scene, items: &[Item], transform: Affine) {
    ctx.set_transform(transform);
    for item in items {
        match item {
            Item::Fill(fill_item) => {
                ctx.set_paint(fill_item.color.into());
                ctx.fill_path(&fill_item.path);
            }
            Item::Stroke(stroke_item) => {
                let style = Stroke::new(stroke_item.width);
                ctx.set_stroke(style);
                ctx.set_paint(stroke_item.color.into());
                ctx.stroke_path(&stroke_item.path);
            }
            Item::Group(group_item) => {
                render_svg(ctx, &group_item.children, transform * group_item.affine);
                ctx.set_transform(transform);
            }
        }
    }
}
