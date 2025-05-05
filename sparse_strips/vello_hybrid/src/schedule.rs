// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use vello_common::{
    coarse::{Cmd, WideTile},
    paint::Paint,
    tile::Tile,
};

use crate::{GpuStrip, Scene, render::RendererJunk};

#[derive(Debug)]
pub(crate) struct Scheduler {
    /// Index of the current round
    round: usize,
    total_slots: usize,
    free: [Vec<usize>; 2],
    /// Slots that require clearing before subsequent draws.
    clear: [Vec<u32>; 2],
    /// Rounds are enqueued on push clip commands and dequeued on flush.
    rounds_queue: VecDeque<Round>,
}

/// A "round" is a coarse scheduling quantum.
///
/// It represents draws in up to three render targets; two for intermediate
/// clip/blend buffers, and the third for the actual render target. The two
/// clip buffers are for even and odd clip depths.
#[derive(Debug, Default)]
struct Round {
    /// [even clip depth, odd depth, final target] draw calls
    draws: [Draw; 3],
    /// Slots that will be freed after the draws
    free: [Vec<usize>; 2],
}

/// State for a single tile.
///
/// Perhaps this should just be a field in the scheduler.
#[derive(Default)]
struct TileState {
    stack: Vec<TileEl>,
}

#[derive(Clone, Copy)]
struct TileEl {
    slot_ix: usize,
    round: usize,
}

#[derive(Debug, Default)]
struct Draw(Vec<GpuStrip>);

impl Scheduler {
    pub(crate) fn new(total_slots: usize) -> Self {
        let free0: Vec<_> = (0..total_slots).collect();
        let free1 = free0.clone();
        let free = [free0, free1];
        let clear = [Vec::new(), Vec::new()];
        let mut rounds_queue = VecDeque::new();
        rounds_queue.push_back(Round::default());
        Self {
            round: 0,
            total_slots,
            free,
            clear,
            rounds_queue,
        }
    }

    pub(crate) fn do_scene(&mut self, junk: &mut RendererJunk<'_>, scene: &Scene) {
        let mut state = TileState::default();
        let wide_tiles_per_row = (scene.width).div_ceil(WideTile::WIDTH);
        let wide_tiles_per_col = (scene.height).div_ceil(Tile::HEIGHT);

        // Left to right, top to bottom iteration over wide tiles.
        for wide_tile_row in 0..wide_tiles_per_col {
            for wide_tile_col in 0..wide_tiles_per_row {
                let wide_tile_idx = usize::from(wide_tile_row * wide_tiles_per_row + wide_tile_col);
                let wide_tile = &scene.wide.tiles[wide_tile_idx];
                let wide_tile_x = wide_tile_col * WideTile::WIDTH;
                let wide_tile_y = wide_tile_row * Tile::HEIGHT;
                self.do_tile(junk, wide_tile_x, wide_tile_y, wide_tile, &mut state);
            }
        }
        while !self.rounds_queue.is_empty() {
            self.flush(junk);
        }

        // When a scene ends, state should return to its initial state for reuse.
        self.round = 0;
        if cfg!(debug_assertions) {
            for i in 0..self.total_slots {
                debug_assert!(self.free[0].contains(&i), "free[0] is missing slot {}", i);
                debug_assert!(self.free[1].contains(&i), "free[1] is missing slot {}", i);
            }
        }
        debug_assert!(self.rounds_queue.is_empty());
        debug_assert!(self.clear[0].len() == 0);
        debug_assert!(self.clear[1].len() == 0);
    }

    /// Flush one round.
    ///
    /// The rounds queue must not be empty.
    fn flush(&mut self, junk: &mut RendererJunk<'_>) {
        let round = self.rounds_queue.pop_front().unwrap();
        for (i, draw) in round.draws.iter().enumerate() {
            if draw.0.is_empty() {
                continue;
            }

            let load = {
                if i == 2 {
                    // We're rendering to the view, don't clear.
                    wgpu::LoadOp::Load
                } else {
                    if self.clear[i].len() + self.free[i].len() == self.total_slots {
                        // All slots are either unoccupied or need to be cleared.
                        // Simply clear the slots via a load operation.
                        self.clear[i].clear();
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                    } else {
                        // Some slots need to be preserved, so clear only the dirty slots.
                        junk.do_clear_slots_render_pass(i, &self.clear[i].as_slice());
                        self.clear[i].clear();
                        wgpu::LoadOp::Load
                    }
                }
            };
            junk.do_strip_render_pass(&draw.0, i, load);
        }
        for i in 0..2 {
            self.free[i].extend(&round.free[i]);
        }
        self.round += 1;
    }

    // Find the appropriate draw call for rendering.
    fn draw_mut(&mut self, el_round: usize, clip_depth: usize) -> &mut Draw {
        let ix = if clip_depth == 1 {
            // We can draw to the final target
            2
        } else {
            // Clip depth even => draw to index 1
            // Clip depth odd => draw to index 0
            // Clip depth starts at 1, so even `clip_depth` represents an odd "real" clip depth.
            1 - clip_depth % 2
        };
        let rel_round = el_round.saturating_sub(self.round);
        if self.rounds_queue.len() == rel_round {
            self.rounds_queue.push_back(Round::default());
        }
        &mut self.rounds_queue[rel_round].draws[ix]
    }

    /// Iterates over wide tile commands and schedules them for rendering.
    #[allow(clippy::todo, reason = "still working on this")]
    fn do_tile(
        &mut self,
        junk: &mut RendererJunk<'_>,
        wide_tile_x: u16,
        wide_tile_y: u16,
        tile: &WideTile,
        state: &mut TileState,
    ) {
        state.stack.clear();
        // Sentinel `TileEl` to indicate the end of the stack where we draw all
        // commands to the final target.
        state.stack.push(TileEl {
            slot_ix: !0,
            round: self.round,
        });
        let bg = tile.bg.as_premul_rgba8().to_u32();
        // If the background has a non-zero alpha then we need to render it.
        if has_non_zero_alpha(bg) {
            let draw = self.draw_mut(self.round, 1);
            draw.0.push(GpuStrip {
                x: wide_tile_x,
                y: wide_tile_y,
                width: WideTile::WIDTH,
                dense_width: 0,
                col: 0,
                rgba: bg,
            });
        }
        for cmd in &tile.cmds {
            // Note: this starts at 1 (for the final target)
            // TODO: Maybe change this to be the "real" clip depth after we have this all working?
            // Since this is "real" clip depth + 1, it can be confusing.
            let clip_depth = state.stack.len();
            match cmd {
                Cmd::Fill(fill) => {
                    let el = state.stack.last().unwrap();
                    let draw = self.draw_mut(el.round, clip_depth);
                    let color = match fill.paint {
                        Paint::Solid(color) => color,
                        Paint::Indexed(_) => unimplemented!(),
                    };
                    let rgba = color.as_premul_rgba8().to_u32();
                    debug_assert!(
                        has_non_zero_alpha(rgba),
                        "Color fields with 0 alpha are reserved for clipping"
                    );
                    let (x, y) = if clip_depth == 1 {
                        (wide_tile_x + fill.x, wide_tile_y)
                    } else {
                        (fill.x, el.slot_ix as u16 * Tile::HEIGHT)
                    };
                    draw.0.push(GpuStrip {
                        x,
                        y,
                        width: fill.width,
                        dense_width: 0,
                        col: 0,
                        rgba,
                    });
                }
                Cmd::AlphaFill(alpha_fill) => {
                    let el = state.stack.last().unwrap();
                    let draw = self.draw_mut(el.round, clip_depth);
                    let color = match alpha_fill.paint {
                        Paint::Solid(color) => color,
                        Paint::Indexed(_) => unimplemented!(),
                    };
                    let rgba = color.as_premul_rgba8().to_u32();
                    debug_assert!(
                        has_non_zero_alpha(rgba),
                        "Color fields with 0 alpha are reserved for clipping"
                    );
                    let (x, y) = if clip_depth == 1 {
                        (wide_tile_x + alpha_fill.x, wide_tile_y)
                    } else {
                        (alpha_fill.x, el.slot_ix as u16 * Tile::HEIGHT)
                    };
                    draw.0.push(GpuStrip {
                        x,
                        y,
                        width: alpha_fill.width,
                        dense_width: alpha_fill.width,
                        col: (alpha_fill.alpha_idx / usize::from(Tile::HEIGHT))
                            .try_into()
                            .expect("Sparse strips are bound to u32 range"),
                        rgba,
                    });
                }
                Cmd::PushBuf => {
                    let ix = clip_depth % 2;
                    while self.free[ix].is_empty() {
                        if self.rounds_queue.is_empty() {
                            // TODO: Probably should return error here
                            panic!("failed to allocate slot");
                        }
                        self.flush(junk);
                    }
                    let slot_ix = self.free[ix].pop().unwrap();
                    self.clear[ix].push(slot_ix as u32);
                    state.stack.push(TileEl {
                        slot_ix,
                        round: self.round,
                    });
                }
                Cmd::PopBuf => {
                    let tos = state.stack.pop().unwrap();
                    let nos = state.stack.last_mut().unwrap();
                    // If the pixels for the slot we are sampling from won't be drawn until the next round,
                    // then we need to schedule these commands for the next round and preserve the slot's
                    // contents.
                    let next_round = clip_depth % 2 == 0 && clip_depth > 2;
                    let round = nos.round.max(tos.round + next_round as usize);
                    nos.round = round;
                    // free slot after draw
                    debug_assert!(round >= self.round);
                    debug_assert!(round - self.round < self.rounds_queue.len());
                    self.rounds_queue[round - self.round].free[1 - clip_depth % 2]
                        .push(tos.slot_ix);
                }
                Cmd::ClipFill(clip_fill) => {
                    let tos = &state.stack[clip_depth - 1];
                    let nos = &state.stack[clip_depth - 2];
                    // If the pixels for the slot we are sampling from won't be drawn until the next round,
                    // then we need to schedule these commands for the next round and preserve the slot's
                    // contents.
                    let next_round = clip_depth % 2 == 0 && clip_depth > 2;
                    let round = nos.round.max(tos.round + next_round as usize);
                    let draw = self.draw_mut(round, clip_depth - 1);
                    // At clip depth 2, we're drawing to the final target, so use the wide tile coords.
                    let (x, y) = if clip_depth <= 2 {
                        (wide_tile_x + clip_fill.x as u16, wide_tile_y)
                    } else {
                        (clip_fill.x as u16, nos.slot_ix as u16 * Tile::HEIGHT)
                    };
                    draw.0.push(GpuStrip {
                        x,
                        y,
                        width: clip_fill.width as u16,
                        dense_width: 0,
                        col: 0,
                        rgba: tos.slot_ix as u32,
                    });
                }
                Cmd::ClipStrip(clip_alpha_fill) => {
                    let tos = &state.stack[clip_depth - 1];
                    let nos = &state.stack[clip_depth - 2];
                    // If the pixels for the slot we are sampling from won't be drawn until the next round,
                    // then we need to schedule these commands for the next round and preserve the slot's
                    // contents.
                    let next_round = clip_depth % 2 == 0 && clip_depth > 2;
                    let round = nos.round.max(tos.round + next_round as usize);
                    let draw = self.draw_mut(round, clip_depth - 1);
                    let (x, y) = if clip_depth <= 2 {
                        (wide_tile_x + clip_alpha_fill.x as u16, wide_tile_y)
                    } else {
                        (clip_alpha_fill.x as u16, nos.slot_ix as u16 * Tile::HEIGHT)
                    };
                    draw.0.push(GpuStrip {
                        x,
                        y,
                        width: clip_alpha_fill.width as u16,
                        dense_width: clip_alpha_fill.width as u16,
                        col: (clip_alpha_fill.alpha_idx / usize::from(Tile::HEIGHT))
                            .try_into()
                            .expect("Sparse strips are bound to u32 range"),
                        rgba: tos.slot_ix as u32,
                    });
                }
                _ => unimplemented!(),
            }
        }
    }
}

#[inline(always)]
fn has_non_zero_alpha(rgba: u32) -> bool {
    rgba >= 0x1_00_00_00
}
