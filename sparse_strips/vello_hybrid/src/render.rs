// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// TODO:
// - Keep a limit on buffer pools.
// - Refactor and re-read

//! GPU rendering module for the sparse strips CPU/GPU rendering engine.
//!
//! This module provides the GPU-side implementation of the hybrid rendering system.
//! It handles:
//! - GPU resource management (buffers, textures, pipelines)
//! - Surface/window management and presentation
//! - Shader execution and rendering
//!
//! The hybrid approach combines CPU-side path processing with efficient GPU rendering
//! to balance flexibility and performance.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;

use bytemuck::{Pod, Zeroable};
use vello_common::{coarse::WideTile, tile::Tile};
use wgpu::{
    BindGroup, BindGroupLayout, BlendState, Buffer, ColorTargetState, ColorWrites, CommandEncoder,
    Device, PipelineCompilationOptions, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, Texture, TextureView, util::DeviceExt,
};

use crate::{scene::Scene, schedule::Scheduler};

/// Dimensions of the rendering target
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RenderSize {
    /// Width of the rendering target
    pub width: u32,
    /// Height of the rendering target
    pub height: u32,
}

/// Options for the renderer
#[derive(Debug)]
pub struct RenderTargetConfig {
    /// Format of the rendering target
    pub format: wgpu::TextureFormat,
    /// Width of the rendering target
    pub width: u32,
    /// Height of the rendering target
    pub height: u32,
}

/// GPU renderer for the hybrid rendering system
///
/// This struct contains GPU resources that are created once at startup and
/// are never reallocated or rebuilt.
#[derive(Debug)]
pub struct Renderer {
    programs: Programs,
    scheduler: Scheduler,
}

impl Renderer {
    /// Creates a new renderer
    ///
    /// The target parameter determines if we render to a window or headless
    pub fn new(device: &Device, render_target_config: &RenderTargetConfig) -> Self {
        let slot_count = (device.limits().max_texture_dimension_2d / Tile::HEIGHT as u32) as usize;

        Self {
            programs: Programs::new(device, render_target_config, slot_count),
            scheduler: Scheduler::new(slot_count),
        }
    }

    /// Render `scene` into the provided command encoder.
    ///
    /// This method creates GPU resources as needed, and schedules potentially multiple
    /// render passes.
    pub fn render(
        &mut self,
        scene: &Scene,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        render_size: &RenderSize,
        view: &TextureView,
    ) {
        // For the time being, we upload the entire alpha buffer as one big chunk. As a future
        // refinement, we could have a bounded alpha buffer, and break draws when the alpha
        // buffer fills.
        self.programs
            .prepare_alphas(device, queue, &scene.alphas, render_size);

        // Reset buffer offsets at the start of each frame
        self.programs.reset_buffer_offsets();

        let mut junk = RendererJunk {
            programs: &mut self.programs,
            device,
            queue,
            encoder,
            view,
        };
        self.scheduler.do_scene(&mut junk, scene);
    }
}

/// Defines the GPU resources and pipelines for rendering.
#[derive(Debug)]
struct Programs {
    /// Pipeline for rendering wide tile commands.
    clip_pipeline: RenderPipeline,
    /// Bind group layout for clip draws
    clip_bind_group_layout: BindGroupLayout,
    /// Clip texture views
    clip_texture_views: [TextureView; 2],
    /// Clip config buffer
    clip_config_buffer: Buffer,
    /// GPU resources for rendering (created during prepare)
    resources: GpuResources,
    /// Scratch buffer for staging alpha texture data.
    alpha_data: Vec<u8>,
    /// Dimensions of the rendering target
    render_size: RenderSize,

    /// Pipeline for clearing slots in clip textures
    clear_pipeline: RenderPipeline,
    /// Bind group for clear slots operation
    clear_bind_group: BindGroup,
}

/// Contains all GPU resources needed for rendering
///
/// This struct contains the GPU resources that may be reallocated depending
/// on the scene. Resources that are created once at startup are simply in
/// `Renderer`.
#[derive(Debug)]
struct GpuResources {
    /// Current main buffer for strip data
    strips_buffer: Buffer,
    /// Current offset in the strips buffer for appending new strips
    strips_buffer_offset: u64,
    /// Pool of strips buffers for reuse
    strips_buffer_pool: Vec<Buffer>,
    /// Texture for alpha values
    alphas_texture: Texture,
    /// Buffer for config data
    config_buffer: Buffer,
    // Bind groups for rendering with clip buffers
    clip_bind_groups: [BindGroup; 3],
    /// Slot indices buffer with current offset
    slot_indices_offset: u64,
    /// Buffer for slot indices used in clear_slots
    slot_indices_buffer: Buffer,
    /// Pool of slot index buffers for reuse
    slot_indices_buffer_pool: Vec<Buffer>,
}

/// Configuration for the GPU renderer
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct Config {
    /// Width of the rendering target
    pub width: u32,
    /// Height of the rendering target
    pub height: u32,
    /// Height of a strip in the rendering
    pub strip_height: u32,
    /// Number of trailing zeros in `alphas_tex_width` (log2 of width).
    /// Pre-calculated on CPU since downlevel targets do not support `firstTrailingBit`.
    pub alphas_tex_width_bits: u32,
}

/// Represents a GPU strip for rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct GpuStrip {
    /// X coordinate of the strip
    pub x: u16,
    /// Y coordinate of the strip
    pub y: u16,
    /// Width of the strip
    pub width: u16,
    /// Width of the portion where alpha blending should be applied.
    pub dense_width: u16,
    /// Column-index into the alpha texture where this strip's alpha values begin.
    ///
    /// There are [`Config::strip_height`] alpha values per column.
    pub col: u32,
    /// RGBA color value
    pub rgba: u32,
}

/// Config for the clear slots pipeline
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct ClearSlotsConfig {
    /// Width of a slot
    pub slot_width: u32,
    /// Height of a slot
    pub slot_height: u32,
    /// Total height of the texture
    pub texture_height: u32,
    /// Padding for 16-byte alignment
    pub _padding: u32,
}

/// A struct containing references to the many objects needed to get work
/// scheduled onto the GPU.
pub(crate) struct RendererJunk<'a> {
    programs: &'a mut Programs,
    device: &'a Device,
    queue: &'a Queue,
    encoder: &'a mut CommandEncoder,
    view: &'a TextureView,
}

impl GpuStrip {
    /// Vertex attributes for the strip
    pub fn vertex_attributes() -> [wgpu::VertexAttribute; 4] {
        wgpu::vertex_attr_array![
            0 => Uint32,
            1 => Uint32,
            2 => Uint32,
            3 => Uint32,
        ]
    }
}

impl Programs {
    fn new(device: &Device, render_target_config: &RenderTargetConfig, slot_count: usize) -> Self {
        let clip_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Clip Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // Create bind group layout for clearing slots
        let clear_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Clear Slots Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let clip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Clip Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sparse_strip_clip.wgsl").into(),
            ),
        });

        // Create shader module for clearing slots
        let clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Clear Slots Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sparse_strip_clear_slots.wgsl").into(),
            ),
        });

        let clip_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Clip Pipeline Layout"),
            bind_group_layouts: &[&clip_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create pipeline layout for clearing slots
        let clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Clear Slots Pipeline Layout"),
                bind_group_layouts: &[&clear_bind_group_layout],
                push_constant_ranges: &[],
            });

        let clip_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Clip Pipeline"),
            layout: Some(&clip_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &clip_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<GpuStrip>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &GpuStrip::vertex_attributes(),
                }],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &clip_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: render_target_config.format,
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create pipeline for clearing slots
        let clear_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Clear Slots Pipeline"),
            layout: Some(&clear_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &clear_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<u32>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Uint32,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &clear_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: render_target_config.format,
                    // No blending needed for clearing
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let clip_texture_views: [TextureView; 2] = core::array::from_fn(|_| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("clip temp texture"),
                    size: wgpu::Extent3d {
                        width: WideTile::WIDTH as u32,
                        height: Tile::HEIGHT as u32 * slot_count as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    // TODO: Is this correct or need it be RGBA8Unorm?
                    format: render_target_config.format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        });

        // Create clear config buffer
        let clear_config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Clear Slots Config Buffer"),
            contents: bytemuck::bytes_of(&ClearSlotsConfig {
                slot_width: WideTile::WIDTH as u32,
                slot_height: Tile::HEIGHT as u32,
                texture_height: Tile::HEIGHT as u32 * slot_count as u32,
                _padding: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create clear bind group
        let clear_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Clear Slots Bind Group"),
            layout: &clear_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: clear_config_buffer.as_entire_binding(),
            }],
        });

        // Create slot indices buffer with initial capacity for 64 slots
        // This can store up to 64 slot indices (adjust size if needed)
        let slot_indices_buffer =
            Self::make_slot_indices_buffer(device, slot_count as u64 * size_of::<u32>() as u64);

        let clip_config_buffer = Self::make_config_buffer(
            device,
            &RenderSize {
                width: WideTile::WIDTH as u32,
                height: Tile::HEIGHT as u32 * slot_count as u32,
            },
            device.limits().max_texture_dimension_2d,
        );

        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        let alpha_texture_height = 2;
        let alphas_texture =
            Self::make_alphas_texture(device, max_texture_dimension_2d, alpha_texture_height);
        let alpha_data = vec![0; (max_texture_dimension_2d * alpha_texture_height * 16) as usize];
        let config_buffer = Self::make_config_buffer(
            device,
            &RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },
            max_texture_dimension_2d,
        );

        let clip_bind_groups = Self::make_clip_bind_groups(
            device,
            &clip_bind_group_layout,
            &alphas_texture,
            &clip_config_buffer,
            &config_buffer,
            &clip_texture_views,
        );

        let resources = GpuResources {
            strips_buffer: Self::make_strips_buffer(device, 0),
            strips_buffer_offset: 0,
            strips_buffer_pool: Vec::new(),
            slot_indices_buffer,
            slot_indices_buffer_pool: Vec::new(),
            slot_indices_offset: 0,
            alphas_texture,
            clip_bind_groups,
            config_buffer,
        };

        Self {
            clip_pipeline,
            clip_bind_group_layout,
            clip_texture_views,
            clip_config_buffer,
            resources,
            alpha_data,
            render_size: RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },

            clear_pipeline,
            clear_bind_group,
        }
    }

    fn make_strips_buffer(device: &Device, required_strips_size: u64) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Strips Buffer"),
            size: required_strips_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_slot_indices_buffer(device: &Device, required_size: u64) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Slot Indices Buffer"),
            size: required_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_config_buffer(
        device: &Device,
        render_size: &RenderSize,
        max_texture_dimension_2d: u32,
    ) -> Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::bytes_of(&Config {
                width: render_size.width,
                height: render_size.height,
                strip_height: Tile::HEIGHT.into(),
                alphas_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn make_alphas_texture(
        device: &Device,
        max_texture_dimension_2d: u32,
        alpha_texture_height: u32,
    ) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Alpha Texture"),
            size: wgpu::Extent3d {
                width: max_texture_dimension_2d,
                height: alpha_texture_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn make_clip_bind_groups(
        device: &Device,
        clip_bind_group_layout: &BindGroupLayout,
        alphas_texture: &Texture,
        clip_config_buffer: &Buffer,
        config_buffer: &Buffer,
        clip_texture_views: &[TextureView],
    ) -> [BindGroup; 3] {
        [
            Self::make_clip_bind_group(
                device,
                clip_bind_group_layout,
                alphas_texture,
                clip_config_buffer,
                &clip_texture_views[1],
            ),
            Self::make_clip_bind_group(
                device,
                clip_bind_group_layout,
                alphas_texture,
                clip_config_buffer,
                &clip_texture_views[0],
            ),
            Self::make_clip_bind_group(
                device,
                clip_bind_group_layout,
                alphas_texture,
                config_buffer,
                &clip_texture_views[1],
            ),
        ]
    }

    fn make_clip_bind_group(
        device: &Device,
        clip_bind_group_layout: &BindGroupLayout,
        alphas_texture: &Texture,
        config_buffer: &Buffer,
        clip_texture_view: &TextureView,
    ) -> BindGroup {
        let alphas_texture_view =
            alphas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Clip Bind Group"),
            layout: clip_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&alphas_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(clip_texture_view),
                },
            ],
        })
    }

    /// Prepare the GPU buffers for rendering, given alphas
    ///
    /// Does not guarantee that the strip buffer is big enough
    fn prepare_alphas(
        &mut self,
        device: &Device,
        queue: &Queue,
        alphas: &[u8],
        new_render_size: &RenderSize,
    ) {
        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        // Update existing resources as needed
        {
            let alpha_len = alphas.len();
            // There are 16 1-byte alpha values per texel.
            let required_alpha_height = u32::try_from(alpha_len)
                .unwrap()
                .div_ceil(max_texture_dimension_2d * 16);
            let width = max_texture_dimension_2d;
            let required_alpha_size = width * required_alpha_height * 16;

            let current_alpha_size = {
                let alphas_texture = &self.resources.alphas_texture;
                alphas_texture.width() * alphas_texture.height() * 16
            };
            if required_alpha_size > current_alpha_size {
                assert!(
                    required_alpha_height <= max_texture_dimension_2d,
                    "Alpha texture height exceeds max texture dimensions"
                );

                // Resize the alpha texture staging buffer.
                self.alpha_data.resize(
                    (max_texture_dimension_2d * required_alpha_height * 16) as usize,
                    0,
                );
                // The alpha texture encodes 16 1-byte alpha values per texel, with 4 alpha values packed in each channel
                let alphas_texture = Self::make_alphas_texture(
                    device,
                    max_texture_dimension_2d,
                    required_alpha_height,
                );
                self.resources.alphas_texture = alphas_texture;

                // Since the alpha texture has changed, we need to update the clip bind groups.
                self.resources.clip_bind_groups = Self::make_clip_bind_groups(
                    device,
                    &self.clip_bind_group_layout,
                    &self.resources.alphas_texture,
                    &self.clip_config_buffer,
                    &self.resources.config_buffer,
                    &self.clip_texture_views,
                );
            }
        }

        // Update config buffer if dimensions changed.
        // We don't need to initialize a new config buffer because it's fixed size (uniform buffer).
        if self.render_size != *new_render_size {
            let config = Config {
                width: new_render_size.width,
                height: new_render_size.height,
                strip_height: Tile::HEIGHT.into(),
                alphas_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
            };
            queue.write_buffer(
                &self.resources.config_buffer,
                0,
                bytemuck::bytes_of(&config),
            );
            self.render_size = new_render_size.clone();
        }

        // Prepare alpha data for the texture with 16 1-byte alpha values per texel (4 per channel)
        let texture_width = self.resources.alphas_texture.width();
        let texture_height = self.resources.alphas_texture.height();
        assert!(
            alphas.len() <= (texture_width * texture_height * 16) as usize,
            "Alpha texture dimensions are too small to fit the alpha data"
        );
        // After this copy to `self.alpha_data`, there may be stale trailing alpha values. These
        // are not sampled, so can be left as-is.
        self.alpha_data[0..alphas.len()].copy_from_slice(alphas);

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.resources.alphas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.alpha_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // 16 bytes per RGBA32Uint texel (4 u32s × 4 bytes each)
                bytes_per_row: Some(texture_width * 16),
                rows_per_image: Some(texture_height),
            },
            wgpu::Extent3d {
                width: texture_width,
                height: texture_height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload the strip data by appending to the buffer
    /// Returns the buffer and the range of the uploaded strips
    fn upload_strips(&mut self, device: &Device, queue: &Queue, strips: &[GpuStrip]) -> (u64, u64) {
        if strips.is_empty() {
            return (0, 0);
        }

        let required_strips_size = size_of_val(strips) as u64;

        // Check if we need to switch to a new buffer
        let current_buffer_size = self.resources.strips_buffer.size();
        let current_offset = self.resources.strips_buffer_offset;

        // If this upload won't fit in the remaining space, try to find a buffer from the pool
        // or create a new one if necessary
        if current_offset + required_strips_size > current_buffer_size {
            // First try to find a suitable buffer in the pool
            let needed_size = required_strips_size.max(current_buffer_size * 2);
            let suitable_buffer_idx = self
                .resources
                .strips_buffer_pool
                .iter()
                .position(|buffer| buffer.size() >= needed_size);

            if let Some(idx) = suitable_buffer_idx {
                // Found a suitable buffer in the pool, swap it with the current one
                let new_buffer = self.resources.strips_buffer_pool.remove(idx);
                self.resources
                    .strips_buffer_pool
                    .push(self.resources.strips_buffer.clone());
                self.resources.strips_buffer = new_buffer;
            } else {
                // No suitable buffer found, create a new one and put the old one in the pool for re-use.
                let new_buffer = Self::make_strips_buffer(device, needed_size);
                self.resources
                    .strips_buffer_pool
                    .push(self.resources.strips_buffer.clone());
                self.resources.strips_buffer = new_buffer;
            }

            // Reset the offset for the new buffer
            self.resources.strips_buffer_offset = 0;
        }

        // Append strips to the buffer at the current offset
        let offset = self.resources.strips_buffer_offset;
        let mut buffer = queue
            .write_buffer_with(
                &self.resources.strips_buffer,
                offset,
                required_strips_size.try_into().unwrap(),
            )
            .expect("Capacity for strips per above");
        buffer.copy_from_slice(bytemuck::cast_slice(strips));

        // Update the offset for next upload
        self.resources.strips_buffer_offset += required_strips_size;

        (offset, required_strips_size)
    }

    /// Reset buffer offsets to start appending from the beginning again
    fn reset_buffer_offsets(&mut self) {
        self.resources.strips_buffer_offset = 0;
        self.resources.slot_indices_offset = 0;
    }
}

impl RendererJunk<'_> {
    pub(crate) fn do_clip_render_pass(
        &mut self,
        strips: &[GpuStrip],
        ix: usize,
        load: wgpu::LoadOp<wgpu::Color>,
    ) {
        // Upload strips and get the offset, size, and buffer reference
        let (offset, size) = self.programs.upload_strips(self.device, self.queue, strips);

        {
            let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Render to Texture Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: if ix == 2 {
                        self.view
                    } else {
                        &self.programs.clip_texture_views[ix]
                    },
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            render_pass.set_pipeline(&self.programs.clip_pipeline);
            render_pass.set_bind_group(0, &self.programs.resources.clip_bind_groups[ix], &[]);

            // Use the specific slice of the buffer for this draw call
            render_pass.set_vertex_buffer(
                0,
                self.programs
                    .resources
                    .strips_buffer
                    .slice(offset..(offset + size)),
            );

            let strips_to_draw = strips.len();
            render_pass.draw(0..4, 0..u32::try_from(strips_to_draw).unwrap());
        }
    }

    /// Clear specific slots in a clip texture
    pub(crate) fn clear_slots(&mut self, ix: usize, slot_indices: &[u32]) {
        if slot_indices.is_empty() {
            return;
        }

        let resources = &mut self.programs.resources;
        let required_size = (size_of::<u32>() * slot_indices.len()) as u64;
        let current_offset = resources.slot_indices_offset;

        // Check if we need to reset or find a new buffer
        if current_offset + required_size > resources.slot_indices_buffer.size() {
            // Try to find a suitable buffer in the pool
            let needed_size = required_size.max(resources.slot_indices_buffer.size() * 2);
            let suitable_buffer_idx = resources
                .slot_indices_buffer_pool
                .iter()
                .position(|buffer| buffer.size() >= needed_size);

            if let Some(idx) = suitable_buffer_idx {
                // Found a suitable buffer in the pool, swap it with the current one
                let new_buffer = resources.slot_indices_buffer_pool.remove(idx);
                resources
                    .slot_indices_buffer_pool
                    .push(resources.slot_indices_buffer.clone());
                resources.slot_indices_buffer = new_buffer;
            } else {
                // No suitable buffer found, create a new one and put the old one in the pool
                let new_buffer = Programs::make_slot_indices_buffer(self.device, needed_size);
                resources
                    .slot_indices_buffer_pool
                    .push(resources.slot_indices_buffer.clone());
                resources.slot_indices_buffer = new_buffer;
            }

            resources.slot_indices_offset = 0;
        }

        // Write slot indices to the buffer at the current offset
        let offset = resources.slot_indices_offset;
        self.queue.write_buffer(
            &resources.slot_indices_buffer,
            offset,
            bytemuck::cast_slice(slot_indices),
        );

        {
            let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Clear Slots Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &self.programs.clip_texture_views[ix],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Don't clear the entire texture, just specific slots
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.programs.clear_pipeline);
            render_pass.set_bind_group(0, &self.programs.clear_bind_group, &[]);

            // Use the specific slice of the buffer for this draw call
            render_pass.set_vertex_buffer(
                0,
                resources
                    .slot_indices_buffer
                    .slice(offset..(offset + required_size)),
            );

            render_pass.draw(0..4, 0..slot_indices.len() as u32);
        }

        // Update offset for next upload
        resources.slot_indices_offset += required_size;
    }
}
