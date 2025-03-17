// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

use crate::render::{GpuStrip, RenderData};
use kurbo::{Affine, BezPath, Cap, Join, Rect, Shape, Stroke};
use peniko::color::palette::css::BLACK;
use peniko::{BlendMode, Compose, Fill, Mix};
use vello_common::coarse::{WIDE_TILE_WIDTH, Wide};
use vello_common::flatten::Line;
use vello_common::paint::Paint;
use vello_common::strip::{STRIP_HEIGHT, Strip};
use vello_common::tile::Tiles;
use vello_common::{flatten, strip};

/// Default tolerance for curve flattening
pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

/// A render context for hybrid CPU/GPU rendering.
///
/// This context maintains the state for path rendering and manages the rendering
/// pipeline from paths to strips that can be rendered by the GPU.
#[derive(Debug)]
pub struct Scene {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) wide: Wide,
    pub(crate) alphas: Vec<u32>,
    pub(crate) line_buf: Vec<Line>,
    pub(crate) tiles: Tiles,
    pub(crate) strip_buf: Vec<Strip>,
    pub(crate) paint: Paint,
    pub(crate) stroke: Stroke,
    pub(crate) transform: Affine,
    pub(crate) fill_rule: Fill,
    pub(crate) blend_mode: BlendMode,
}

/// A render state which contains the style properties for path rendering and
/// the current transform.
#[derive(Debug)]
struct RenderState {
    pub(crate) paint: Paint,
    pub(crate) stroke: Stroke,
    pub(crate) transform: Affine,
    pub(crate) fill_rule: Fill,
    pub(crate) blend_mode: BlendMode,
}


impl Scene {
    /// Create a new render context with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        let render_state = Self::default_render_state();
        Self {
            width: width as usize,
            height: height as usize,
            wide: Wide::new(width as usize, height as usize),
            alphas: vec![],
            line_buf: vec![],
            tiles: Tiles::new(),
            strip_buf: vec![],
            paint: render_state.paint,
            stroke: render_state.stroke,
            transform: render_state.transform,
            fill_rule: render_state.fill_rule,
            blend_mode: render_state.blend_mode,
        }
    }

    /// Create default rendering state.
    fn default_render_state() -> RenderState {
        let transform = Affine::IDENTITY;
        let fill_rule = Fill::NonZero;
        let paint = BLACK.into();
        let stroke = Stroke {
            width: 1.0,
            join: Join::Bevel,
            start_cap: Cap::Butt,
            end_cap: Cap::Butt,
            ..Default::default()
        };
        let blend_mode = BlendMode::new(Mix::Normal, Compose::SrcOver);
        RenderState {
            transform,
            fill_rule,
            paint,
            stroke,
            blend_mode,
        }
    }

    /// Fill a path with the current paint and fill rule.
    pub fn fill_path(&mut self, path: &BezPath) {
        flatten::fill(path, self.transform, &mut self.line_buf);
        self.render_path(self.fill_rule, self.paint.clone());
    }

    /// Stroke a path with the current paint and stroke settings.
    pub fn stroke_path(&mut self, path: &BezPath) {
        flatten::stroke(path, &self.stroke, self.transform, &mut self.line_buf);
        self.render_path(Fill::NonZero, self.paint.clone());
    }

    /// Fill a rectangle with the current paint and fill rule.
    pub fn fill_rect(&mut self, rect: &Rect) {
        self.fill_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Stroke a rectangle with the current paint and stroke settings.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Set the blend mode for subsequent rendering operations.
    pub fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        self.blend_mode = blend_mode;
    }

    /// Set the stroke settings for subsequent stroke operations.
    pub fn set_stroke(&mut self, stroke: Stroke) {
        self.stroke = stroke;
    }

    /// Set the paint for subsequent rendering operations.
    pub fn set_paint(&mut self, paint: Paint) {
        self.paint = paint;
    }

    /// Set the fill rule for subsequent fill operations.
    pub fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.fill_rule = fill_rule;
    }

    /// Set the transform for subsequent rendering operations.
    pub fn set_transform(&mut self, transform: Affine) {
        self.transform = transform;
    }

    /// Reset the transform to identity.
    pub fn reset_transform(&mut self) {
        self.transform = Affine::IDENTITY;
    }

    pub fn reset(&mut self) {
        self.wide.reset();
        self.alphas.clear();
        self.line_buf.clear();
        self.tiles.reset();
        self.strip_buf.clear();

        let render_state = Self::default_render_state();
        self.transform = render_state.transform;
        self.fill_rule = render_state.fill_rule;
        self.paint = render_state.paint;
        self.stroke = render_state.stroke;
        self.blend_mode = render_state.blend_mode;
    }

    /// Get the width of the render context.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Width is expected to fit within u16 range"
    )]
    pub fn width(&self) -> u16 {
        self.width as u16
    }

    /// Get the height of the render context.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Height is expected to fit within u16 range"
    )]
    pub fn height(&self) -> u16 {
        self.height as u16
    }

    // Assumes that `line_buf` contains the flattened path.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Width and height are expected to fit within u16 range"
    )]
    fn render_path(&mut self, fill_rule: Fill, paint: Paint) {
        self.tiles
            .make_tiles(&self.line_buf, self.width as u16, self.height as u16);
        self.tiles.sort_tiles();

        strip::render(
            &self.tiles,
            &mut self.strip_buf,
            &mut self.alphas,
            fill_rule,
            &self.line_buf,
        );

        self.wide.generate(&self.strip_buf, fill_rule, paint);
    }
}

impl Scene {
    /// Prepares render data from the current context for GPU rendering
    ///
    /// This method converts the rendering context's state into a format
    /// suitable for GPU rendering, including strips and alpha values.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "GpuStrip fields use u16 and values are expected to fit within that range"
    )]
    pub fn prepare_render_data(&self) -> RenderData {
        let mut strips: Vec<GpuStrip> = Vec::new();
        let wide_tiles_per_row = (self.width).div_ceil(WIDE_TILE_WIDTH);
        let wide_tiles_per_col = (self.height).div_ceil(STRIP_HEIGHT);
        for wide_tile_row in 0..wide_tiles_per_col {
            for wide_tile_col in 0..wide_tiles_per_row {
                let wide_tile =
                    &self.wide.tiles[wide_tile_row * wide_tiles_per_row + wide_tile_col];
                let wide_tile_x = wide_tile_col * WIDE_TILE_WIDTH;
                let wide_tile_y = wide_tile_row * STRIP_HEIGHT;
                let bg = wide_tile.bg.premultiply().to_rgba8().to_u32();
                if bg != 0 {
                    strips.push(GpuStrip {
                        x: wide_tile_x as u16,
                        y: wide_tile_y as u16,
                        width: WIDE_TILE_WIDTH as u16,
                        dense_width: 0,
                        col: 0,
                        rgba: bg,
                    });
                }
                for cmd in &wide_tile.cmds {
                    match cmd {
                        vello_common::coarse::Cmd::Fill(fill) => {
                            let color: peniko::color::AlphaColor<peniko::color::Srgb> =
                                match fill.paint {
                                    Paint::Solid(color) => color,
                                    _ => peniko::color::AlphaColor::TRANSPARENT,
                                };
                            strips.push(GpuStrip {
                                x: (wide_tile_x as u32 + fill.x) as u16,
                                y: wide_tile_y as u16,
                                width: fill.width as u16,
                                dense_width: 0,
                                col: 0,
                                rgba: color.premultiply().to_rgba8().to_u32(),
                            });
                        }
                        vello_common::coarse::Cmd::AlphaFill(cmd_strip) => {
                            let color: peniko::color::AlphaColor<peniko::color::Srgb> =
                                match cmd_strip.paint {
                                    Paint::Solid(color) => color,
                                    _ => peniko::color::AlphaColor::TRANSPARENT,
                                };
                            strips.push(GpuStrip {
                                x: (wide_tile_x as u32 + cmd_strip.x) as u16,
                                y: wide_tile_y as u16,
                                width: cmd_strip.width as u16,
                                dense_width: cmd_strip.width as u16,
                                col: cmd_strip.alpha_ix as u32,
                                rgba: color.premultiply().to_rgba8().to_u32(),
                            });
                        }
                    }
                }
            }
        }
        RenderData {
            strips,
            alphas: self.alphas.clone(),
        }
    }
}
