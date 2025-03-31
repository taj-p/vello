// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Processing and drawing glyphs.

use skrifa::instance::Size;
use skrifa::outline::DrawSettings;
use skrifa::{
    GlyphId, MetadataProvider,
    outline::{HintingInstance, HintingOptions, OutlinePen},
};
use vello_api::kurbo::{Affine, BezPath, Vec2};

pub use vello_api::glyph::*;

/// A glyph prepared for rendering.
#[derive(Debug)]
pub enum PreparedGlyph {
    /// A contour glyph.
    Contour((BezPath, Affine)),
    // TODO: Image and Colr variants.
}

/// Returns an iterator over the renderable glyphs in the given run.
pub fn iter_renderable_glyphs(run: &GlyphRun) -> impl Iterator<Item = PreparedGlyph> + '_ {
    let font = skrifa::FontRef::from_index(run.font.data.as_ref(), run.font.index).unwrap();
    let outlines = font.outline_glyphs();
    let size = Size::new(run.font_size);
    let normalized_coords = run.normalized_coords.as_slice();
    let hinting_instance = if run.hint {
        // TODO: Cache hinting instance.
        Some(HintingInstance::new(&outlines, size, normalized_coords, HINTING_OPTIONS).unwrap())
    } else {
        None
    };
    run.glyphs.iter().filter_map(move |glyph| {
        let draw_settings = if let Some(hinting_instance) = &hinting_instance {
            DrawSettings::hinted(&hinting_instance, false)
        } else {
            DrawSettings::unhinted(size, normalized_coords)
        };
        let outline = outlines.get(GlyphId::new(glyph.id))?;
        let mut path = OutlinePath(BezPath::new());
        outline.draw(draw_settings, &mut path).ok()?;
        let transform = run
            .transform
            .then_translate(Vec2::new(glyph.x as f64, glyph.y as f64));
        Some(PreparedGlyph::Contour((path.0, transform)))
    })
}

struct OutlinePath(BezPath);

impl OutlinePen for OutlinePath {
    #[inline]
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to((x, -y));
    }

    #[inline]
    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to((x, -y));
    }

    #[inline]
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.0.curve_to((cx0, -cy0), (cx1, -cy1), (x, -y));
    }

    #[inline]
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.0.quad_to((cx, -cy), (x, -y));
    }

    #[inline]
    fn close(&mut self) {
        self.0.close_path();
    }
}

// TODO: Make this configurable.
const HINTING_OPTIONS: HintingOptions = HintingOptions {
    engine: skrifa::outline::Engine::AutoFallback,
    target: skrifa::outline::Target::Smooth {
        mode: skrifa::outline::SmoothMode::Lcd,
        symmetric_rendering: false,
        preserve_linear_metrics: true,
    },
};
