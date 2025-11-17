// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter example showing deeply nested clipping.

#![expect(
    clippy::cast_possible_truncation,
    reason = "We temporarily ignore those because the casts\
only break in edge cases, and some of them are also only related to conversions from f64 to f32."
)]

use crate::{ExampleScene, RenderingContext};
use vello_common::color::palette::css::{
    BLACK, BLUE, DARK_BLUE, DARK_GREEN, GREEN, REBECCA_PURPLE, RED,
};
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, BezPath, Circle, Point, Rect, Shape, Stroke};
use vello_common::peniko::Color;
use vello_cpu::color::AlphaColor;
use vello_cpu::color::palette::css::{PURPLE, ROYAL_BLUE, SEA_GREEN, TOMATO, VIOLET};
use vello_cpu::peniko::{BlendMode, Compose, Mix};

/// Filter scene state
#[derive(Debug)]
pub struct FilterScene {}

impl ExampleScene for FilterScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        render(ctx, root_transform);
    }
}

impl FilterScene {
    /// Create a new `FilterScene`
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for FilterScene {
    fn default() -> Self {
        Self::new()
    }
}

pub fn render(ctx: &mut impl RenderingContext, root_transform: Affine) {
    ctx.set_transform(root_transform);
    let filter_drop_shadow = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: 2.0,
        dy: 2.0,
        std_deviation: 4.0,
        color: AlphaColor::from_rgba8(0, 0, 0, 255),
        edge_mode: EdgeMode::None,
    });
    let filter_gaussian_blur = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 2.0,
        edge_mode: EdgeMode::None,
    });

    let spacing = 32.;
    let width = 10.;
    let overlap = 2.;
    let between = 6.;

    // Test 1
    let mut x = 4.;
    let mut y = 4.;
    let mut left = x;
    let mut top = y;
    {
        ctx.push_layer(None, None, None, None, None);
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_layer(None, None, None, None, None);
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(TOMATO);
                left = x + width - overlap;
                top = y + width - overlap;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_filter_layer(filter_gaussian_blur.clone());
                    ctx.set_paint(VIOLET);
                    left = x;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    ctx.pop_layer();
                }
                {
                    ctx.push_filter_layer(filter_drop_shadow.clone());
                    ctx.set_paint(SEA_GREEN);
                    left = x + width + between;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }

    // Test 2
    x += spacing;
    left = x;
    top = y;
    {
        ctx.push_filter_layer(filter_drop_shadow.clone());
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_filter_layer(filter_gaussian_blur.clone());
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(TOMATO);
                left = x + width - overlap;
                top = y + width - overlap;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        {
            ctx.push_layer(None, None, None, None, None);
            ctx.set_paint(VIOLET);
            left = x;
            top = y + width + between;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(SEA_GREEN);
                left = x + width + between;
                top = y + width + between;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }

    // Test 3
    x += spacing;
    left = x;
    top = y;
    {
        ctx.push_filter_layer(filter_gaussian_blur.clone());
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_layer(None, None, None, None, None);
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(TOMATO);
                left = x + width - overlap;
                top = y + width - overlap;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_layer(None, None, None, None, None);
                    ctx.set_paint(VIOLET);
                    left = x;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    {
                        ctx.push_layer(None, None, None, None, None);
                        ctx.set_paint(SEA_GREEN);
                        left = x + width + between;
                        top = y + width + between;
                        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                        ctx.pop_layer();
                    }
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }

    // Test 4
    x = 4.;
    y += spacing;
    left = x;
    top = y;
    let mut circle_path = Circle::new((x + 13., y + 13.), 13.).to_path(0.1);
    {
        ctx.push_layer(
            Some(&circle_path),
            None,
            None,
            None,
            Some(filter_drop_shadow.clone()),
        );
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_layer(None, None, None, None, None);
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(TOMATO);
                left = x + width - overlap;
                top = y + width - overlap;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_layer(None, None, None, None, None);
                    ctx.set_paint(VIOLET);
                    left = x;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    {
                        ctx.push_layer(None, None, None, None, None);
                        ctx.set_paint(SEA_GREEN);
                        left = x + width + between;
                        top = y + width + between;
                        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                        ctx.pop_layer();
                    }
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }

    // Test 5
    x += spacing;
    left = x;
    top = y;
    let mut quad_path = BezPath::new();
    quad_path.move_to((x, y));
    quad_path.line_to((x + 26., y + 5.));
    quad_path.line_to((x + 30., y + 21.));
    quad_path.line_to((x + 5., y + 30.));
    quad_path.close_path();
    {
        ctx.push_layer(
            Some(&quad_path),
            None,
            None,
            None,
            Some(filter_drop_shadow.clone()),
        );
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_layer(None, None, None, None, None);
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(TOMATO);
                left = x + width - overlap;
                top = y + width - overlap;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_layer(None, None, None, None, None);
                    ctx.set_paint(VIOLET);
                    left = x;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    {
                        ctx.push_layer(None, None, None, None, None);
                        ctx.set_paint(SEA_GREEN);
                        left = x + width + between;
                        top = y + width + between;
                        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                        ctx.pop_layer();
                    }
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }

    // Test 6
    x += spacing;
    left = x;
    top = y;
    circle_path = Circle::new((x + 13., y + 13.), 13.).to_path(0.1);
    {
        ctx.push_layer(
            Some(&circle_path),
            None,
            None,
            None,
            Some(filter_gaussian_blur.clone()),
        );
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_layer(None, None, None, None, None);
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(TOMATO);
                left = x + width - overlap;
                top = y + width - overlap;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_layer(None, None, None, None, None);
                    ctx.set_paint(VIOLET);
                    left = x;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    {
                        ctx.push_layer(None, None, None, None, None);
                        ctx.set_paint(SEA_GREEN);
                        left = x + width + between;
                        top = y + width + between;
                        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                        ctx.pop_layer();
                    }
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }

    // Test 7
    x = 4.;
    y += spacing;
    left = x;
    top = y;
    {
        ctx.push_layer(None, None, Some(0.5), None, None);
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            {
                ctx.push_filter_layer(filter_gaussian_blur.clone());
                ctx.set_paint(PURPLE);
                left = x + width + between;
                top = y;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                ctx.pop_layer();
            }
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(TOMATO);
                left = x + width - overlap;
                top = y + width - overlap;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_filter_layer(filter_gaussian_blur.clone());
                    ctx.set_paint(VIOLET);
                    left = x;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    {
                        ctx.push_layer(None, None, None, None, None);
                        ctx.set_paint(SEA_GREEN);
                        left = x + width + between;
                        top = y + width + between;
                        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                        ctx.pop_layer();
                    }
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
        }
        ctx.pop_layer();
    }

    // Test 8
    x += spacing;
    left = x;
    top = y;
    {
        ctx.push_layer(None, None, None, None, None);
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_layer(None, None, None, None, None);
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(VIOLET);
                left = x;
                top = y + width + between;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_layer(None, None, None, None, None);
                    ctx.set_paint(SEA_GREEN);
                    left = x + width + between;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        {
            ctx.push_layer(
                None,
                Some(BlendMode::new(Mix::Normal, Compose::DestOut)),
                None,
                None,
                Some(filter_gaussian_blur.clone()),
            );
            ctx.set_paint(TOMATO);
            left = x + width - overlap;
            top = y + width - overlap;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }

    // Test 9
    x += spacing;
    left = x;
    top = y;
    {
        ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
            std_deviation: 2.0,
            edge_mode: EdgeMode::None,
        }));
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
                std_deviation: 2.0,
                edge_mode: EdgeMode::None,
            }));
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
                    std_deviation: 2.0,
                    edge_mode: EdgeMode::None,
                }));
                ctx.set_paint(VIOLET);
                left = x;
                top = y + width + between;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
                        std_deviation: 2.0,
                        edge_mode: EdgeMode::None,
                    }));
                    ctx.set_paint(SEA_GREEN);
                    left = x + width + between;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    {
                        ctx.push_filter_layer(Filter::from_primitive(
                            FilterPrimitive::GaussianBlur {
                                std_deviation: 2.0,
                                edge_mode: EdgeMode::None,
                            },
                        ));
                        ctx.set_paint(TOMATO);
                        left = x + width - overlap;
                        top = y + width - overlap;
                        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                        ctx.pop_layer();
                    }
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }
}
