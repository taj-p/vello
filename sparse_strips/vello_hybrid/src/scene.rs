// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

use alloc::vec;
use alloc::vec::Vec;
use vello_common::coarse::Wide;
use vello_common::flatten::Line;
use vello_common::glyph::{GlyphRenderer, GlyphRunBuilder, GlyphType, PreparedGlyph};
use vello_common::kurbo::{Affine, BezPath, Cap, Join, Rect, Shape, Stroke};
use vello_common::mask::Mask;
use vello_common::paint::{Paint, PaintType};
use vello_common::peniko::Font;
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Compose, Fill, Mix};
use vello_common::strip::Strip;
use vello_common::tile::Tiles;
use vello_common::{flatten, strip};

/// Default tolerance for curve flattening
pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

/// A rendering command that can be cached and executed later.
#[derive(Debug, Clone)]
pub enum RenderCommand {
    /// Fill a path with the current paint and fill rule.
    FillPath(BezPath),
    /// Stroke a path with the current paint and stroke settings.
    StrokePath(BezPath),
    /// Fill a rectangle with the current paint and fill rule.
    FillRect(Rect),
    /// Stroke a rectangle with the current paint and stroke settings.
    StrokeRect(Rect),
}

impl RenderCommand {
    /// Execute this command on the given scene.
    fn execute(&self, scene: &mut Scene) {
        match self {
            RenderCommand::FillPath(path) => scene.fill_path(path),
            RenderCommand::StrokePath(path) => scene.stroke_path(path),
            RenderCommand::FillRect(rect) => scene.fill_rect(rect),
            RenderCommand::StrokeRect(rect) => scene.stroke_rect(rect),
        }
    }
}

/// A render state which contains the style properties for path rendering and
/// the current transform.
#[derive(Debug)]
struct RenderState {
    pub(crate) paint: PaintType,
    pub(crate) stroke: Stroke,
    pub(crate) transform: Affine,
    pub(crate) fill_rule: Fill,
    pub(crate) blend_mode: BlendMode,
}

/// A render context for hybrid CPU/GPU rendering.
///
/// This context maintains the state for path rendering and manages the rendering
/// pipeline from paths to strips that can be rendered by the GPU.
#[derive(Debug)]
pub struct Scene {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) wide: Wide,
    pub(crate) alphas: Vec<u8>,
    pub(crate) line_buf: Vec<Line>,
    pub(crate) tiles: Tiles,
    pub(crate) strip_buf: Vec<Strip>,
    pub(crate) paint: PaintType,
    paint_visible: bool,
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
            width,
            height,
            wide: Wide::new(width, height),
            alphas: vec![],
            line_buf: vec![],
            tiles: Tiles::new(),
            strip_buf: vec![],
            paint: render_state.paint,
            paint_visible: true,
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

    fn encode_current_paint(&mut self) -> Paint {
        match self.paint.clone() {
            PaintType::Solid(s) => s.into(),
            PaintType::Gradient(_) => {
                unimplemented!("gradient not implemented")
            }
            PaintType::Image(_) => {
                unimplemented!("images not implemented")
            }
        }
    }

    /// Fill a path with the current paint and fill rule.
    pub fn fill_path(&mut self, path: &BezPath) {
        if !self.paint_visible {
            return;
        }
        flatten::fill(path, self.transform, &mut self.line_buf);
        let paint = self.encode_current_paint();
        self.render_path(self.fill_rule, paint);
    }

    /// Stroke a path with the current paint and stroke settings.
    pub fn stroke_path(&mut self, path: &BezPath) {
        if !self.paint_visible {
            return;
        }
        flatten::stroke(path, &self.stroke, self.transform, &mut self.line_buf);
        let paint = self.encode_current_paint();
        self.render_path(Fill::NonZero, paint);
    }

    /// Fill a rectangle with the current paint and fill rule.
    pub fn fill_rect(&mut self, rect: &Rect) {
        self.fill_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Stroke a rectangle with the current paint and stroke settings.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Creates a builder for drawing a run of glyphs that have the same attributes.
    pub fn glyph_run(&mut self, font: &Font) -> GlyphRunBuilder<'_, Self> {
        GlyphRunBuilder::new(font.clone(), self.transform, self)
    }

    /// Push a new layer with the given properties.
    ///
    /// Only `clip_path` is supported for now.
    pub fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
    ) {
        let clip = if let Some(c) = clip_path {
            flatten::fill(c, self.transform, &mut self.line_buf);
            self.make_strips(self.fill_rule);
            Some((self.strip_buf.as_slice(), self.fill_rule))
        } else {
            None
        };

        // Blend mode, opacity, and mask are not supported yet.
        if blend_mode.is_some() {
            unimplemented!()
        }
        if opacity.is_some() {
            unimplemented!()
        }
        if mask.is_some() {
            unimplemented!()
        }

        self.wide.push_layer(
            clip,
            BlendMode::new(Mix::Normal, Compose::SrcOver),
            None,
            1.0,
        );
    }

    /// Push a new clip layer.
    pub fn push_clip_layer(&mut self, path: &BezPath) {
        self.push_layer(Some(path), None, None, None);
    }

    /// Pop the last pushed layer.
    pub fn pop_layer(&mut self) {
        self.wide.pop_layer();
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
    // TODO: This API is not final. Supporting images from a pixmap is explicitly out of scope.
    //       Instead images should be passed via a backend-agnostic opaque id, and be hydrated at
    //       render time into a texture usable by the renderer backend.
    pub fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.paint = paint.into();
        self.paint_visible =
            matches!(&self.paint, PaintType::Solid(color) if color.components[3] != 0.0);
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

    /// Reset scene to default values.
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
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Get the height of the render context.
    pub fn height(&self) -> u16 {
        self.height
    }

    // Assumes that `line_buf` contains the flattened path.
    fn render_path(&mut self, fill_rule: Fill, paint: Paint) {
        self.make_strips(fill_rule);
        self.wide.generate(&self.strip_buf, fill_rule, paint);
    }

    fn make_strips(&mut self, fill_rule: Fill) {
        self.tiles
            .make_tiles(&self.line_buf, self.width, self.height);
        self.tiles.sort_tiles();

        strip::render(
            &self.tiles,
            &mut self.strip_buf,
            &mut self.alphas,
            fill_rule,
            &self.line_buf,
        );
    }

    // TODO: Probably want API for caching a single command...
    // TODO: There's definitely a better API and way to prevent some allocations.
    // TODO: There might be a way to use 1 vec for N commands by understanding where in the vec
    // some command points to (i.e. an arena-like approach).
    /// Cache multiple rendering commands executed serially.
    ///
    /// # Example
    /// ```rust,ignore
    /// let commands = vec![
    ///     RenderCommand::FillRect(rect1),
    ///     RenderCommand::StrokePath(path1),
    /// ];
    /// let cached = scene.cache_commands(&commands);
    /// scene.render_cached_strips(&cached);
    /// ```
    pub fn cache_commands(&mut self, commands: &[RenderCommand]) -> CachedStrips {
        let mut combined_strips = Vec::new();
        let mut combined_alphas = Vec::new();

        for command in commands {
            let initial_strips_len = self.strip_buf.len();
            let initial_alphas_len = self.alphas.len();
            command.execute(self);
            let cached = self.create_cached_strips(initial_strips_len, initial_alphas_len);

            // Adjust alpha indices for the combined buffer
            let alpha_offset = combined_alphas.len() as u32;
            let mut adjusted_strips = cached.strips;
            for strip in &mut adjusted_strips {
                strip.alpha_idx += alpha_offset;
            }

            combined_strips.extend(adjusted_strips);
            combined_alphas.extend(cached.alphas);
        }

        CachedStrips {
            strips: combined_strips,
            alphas: combined_alphas,
        }
    }

    fn create_cached_strips(
        &mut self,
        initial_strips_len: usize,
        initial_alphas_len: usize,
    ) -> CachedStrips {
        // Extract the new strips and alphas that were added
        let new_strips = if self.strip_buf.len() > initial_strips_len {
            let mut strips = self.strip_buf[initial_strips_len..].to_vec();
            // Normalize alpha indices to start from 0
            let alpha_offset = initial_alphas_len as u32;
            for strip in &mut strips {
                strip.alpha_idx -= alpha_offset;
            }
            strips
        } else {
            Vec::new()
        };

        let new_alphas = if self.alphas.len() > initial_alphas_len {
            self.alphas[initial_alphas_len..].to_vec()
        } else {
            Vec::new()
        };

        CachedStrips {
            strips: new_strips,
            alphas: new_alphas,
        }
    }

    /// Render previously cached strips with the current paint settings.
    pub fn render_cached_strips(&mut self, cached: &CachedStrips) {
        if cached.strips.is_empty() {
            return;
        }

        let start_alpha_idx = self.alphas.len() as u32;
        let mut strips = cached.strips.clone();
        for strip in &mut strips {
            strip.alpha_idx += start_alpha_idx;
        }

        self.alphas.extend_from_slice(&cached.alphas);
        let paint = self.encode_current_paint();
        self.wide.generate(&strips, self.fill_rule, paint);
    }
}

/// Cached strip data that can be rendered multiple times efficiently.
// TODO: Reuse this alloc
#[derive(Debug, Clone)]
pub struct CachedStrips {
    strips: Vec<Strip>,
    alphas: Vec<u8>,
}

// TODO: Allow translation of these strips.
impl CachedStrips {
    pub fn new() -> Self {
        Self {
            strips: Vec::new(),
            alphas: Vec::new(),
        }
    }
}

impl Default for CachedStrips {
    fn default() -> Self {
        Self::new()
    }
}

impl GlyphRenderer for Scene {
    fn fill_glyph(&mut self, prepared_glyph: PreparedGlyph<'_>) {
        match prepared_glyph.glyph_type {
            GlyphType::Outline(glyph) => {
                flatten::fill(glyph.path, prepared_glyph.transform, &mut self.line_buf);
                let paint = self.encode_current_paint();
                self.render_path(Fill::NonZero, paint);
            }
            GlyphType::Bitmap(_) => {}
            GlyphType::Colr(_) => {}
        }
    }

    fn stroke_glyph(&mut self, prepared_glyph: PreparedGlyph<'_>) {
        match prepared_glyph.glyph_type {
            GlyphType::Outline(glyph) => {
                flatten::stroke(
                    glyph.path,
                    &self.stroke,
                    prepared_glyph.transform,
                    &mut self.line_buf,
                );
                let paint = self.encode_current_paint();
                self.render_path(Fill::NonZero, paint);
            }
            GlyphType::Bitmap(_) => {}
            GlyphType::Colr(_) => {}
        }
    }
}
