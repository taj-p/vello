// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Types for glyphs.

use peniko::Font;
use peniko::kurbo::Affine;
use skrifa::instance::NormalizedCoord;

/// Positioned glyph.
#[derive(Copy, Clone, Default, Debug)]
pub struct Glyph {
    /// The font-specific identifier for this glyph.
    ///
    /// This ID is specific to the font being used and corresponds to the
    /// glyph index within that font. It is *not* a Unicode code point.
    pub id: u32,
    /// X-offset in run, relative to transform.
    pub x: f32,
    /// Y-offset in run, relative to transform.
    pub y: f32,
}

/// A sequence of glyphs with shared rendering properties.
#[derive(Clone, Debug)]
pub struct GlyphRun {
    /// Glyphs in the run.
    pub glyphs: Vec<Glyph>,
    /// Font for all glyphs in the run.
    pub font: Font,
    /// Size of the font in pixels per em.
    pub font_size: f32,
    /// Global run transform.
    pub transform: Affine,
    /// Normalized variation coordinates for variable fonts.
    pub normalized_coords: Vec<NormalizedCoord>,
    /// Controls whether font hinting is enabled.
    pub hint: bool,
}
