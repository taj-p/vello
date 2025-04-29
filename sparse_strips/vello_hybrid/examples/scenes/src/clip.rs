// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Clip example scene with basic shapes.

use vello_common::color::palette::css::{
    BLACK, BLUE, CORNFLOWER_BLUE, DARK_BLUE, DARK_GREEN, REBECCA_PURPLE, RED,
};
use vello_common::kurbo::{Affine, BezPath, Circle, Point, Rect, Shape, Stroke, Vec2};
use vello_hybrid::Scene;

use crate::ExampleScene;

/// Clip scene state
#[derive(Debug)]
pub struct ClipScene {}

impl ExampleScene for ClipScene {
    fn render(&mut self, ctx: &mut Scene, root_transform: Affine) {
        render(ctx, root_transform);
    }
}

impl ClipScene {
    /// Create a new `ClipScene`
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for ClipScene {
    fn default() -> Self {
        Self::new()
    }
}

/// Draws a simple scene with shapes
pub fn render(ctx: &mut Scene, root_transform: Affine) {
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

    //let clip_rect = Rect::new(0.0, 0.0, 400.0, 400.0);

    //let stroke = Stroke::new(10.0);
    //ctx.set_paint(CORNFLOWER_BLUE.into());
    //ctx.set_stroke(stroke);
    //ctx.stroke_rect(&clip_rect);

    //ctx.push_clip_layer(&clip_rect.to_path(0.1));

    //// Make a checkerboard pattern of 10x10 squares
    ////ctx.set_paint(RED.into());
    ////ctx.fill_rect(&Rect::new(0.0, 0.0, 300.0, 100.0));

    //let mut red = false;
    ////for i in 0..100 {
    ////    println!("i: {}", i);
    ////    if red {
    ////        ctx.set_paint(RED.into());
    ////    } else {
    ////        ctx.set_paint(BLUE.into());
    ////    }
    ////    ctx.fill_rect(&Rect::new(
    ////        0.0,
    ////        i as f64 * 10.0,
    ////        100.0,
    ////        i as f64 * 10.0 + 100.0,
    ////    ));
    ////    red = !red;
    ////}

    //ctx.set_paint(REBECCA_PURPLE.into());
    //ctx.fill_rect(&Rect::new(10.0, 10.0, 500.0, 500.0));
    //ctx.pop_layer();

    //// ctx.set_transform(Affine::translate(Vec2::new(100.0, 0.0)));

    ////let mut triangle_path = BezPath::new();
    ////triangle_path.move_to((10.0, 10.0));
    ////triangle_path.line_to((90.0, 20.0));
    ////triangle_path.line_to((20.0, 90.0));
    ////triangle_path.close_path();

    ////let stroke = Stroke::new(3.0);
    ////ctx.set_paint(DARK_BLUE.into());
    ////ctx.set_stroke(stroke);
    ////ctx.stroke_path(&triangle_path);

    ////let star_path = circular_star(Point::new(50., 50.), 13, 25., 45.);

    ////ctx.push_clip_layer(&star_path);
    ////ctx.set_paint(REBECCA_PURPLE.into());
    ////ctx.fill_path(&triangle_path);
    ////ctx.pop_layer();
}

pub(crate) fn circular_star(center: Point, n: usize, inner: f64, outer: f64) -> BezPath {
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
