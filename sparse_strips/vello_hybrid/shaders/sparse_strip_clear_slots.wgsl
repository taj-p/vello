// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// This shader clears specific slots in clip textures with transparent pixels.

// Assumes this texture consists of a single column of slots of config.slot_height, numbering
// config.texture_height / config.slot_height where the topmost slot is 0.

struct Config {
    // Width of a slot (typically matching WideTile::WIDTH)
    slot_width: u32,
    // Height of a slot (typically matching Tile::HEIGHT)
    slot_height: u32,
    // Total height of the texture (slot_height * number_of_slots)
    texture_height: u32,
    // Padding for 16-byte alignment
    _padding: u32,
}

struct SlotIndex {
    // The index of the slot to clear
    @location(0) index: u32,
}

struct VertexOutput {
    // Normalized device coordinates (NDC) for the current vertex
    @builtin(position) position: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> config: Config;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    instance: SlotIndex,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Map vertex_index (0-3) to quad corners:
    // 0 → (0,0), 1 → (1,0), 2 → (0,1), 3 → (1,1)
    let x = f32(in_vertex_index & 1u);
    let y = f32(in_vertex_index >> 1u);
    
    // Calculate the y-position based on the slot index
    let slot_y_offset = f32(instance.index * config.slot_height);
    
    // Scale to match slot dimensions
    let pix_x = x * f32(config.slot_width);
    let pix_y = slot_y_offset + y * f32(config.slot_height);
    
    // Convert to normalized device coordinates
    // NDC ranges from -1 to 1
    let ndc_x = pix_x * 2.0 / f32(config.slot_width) - 1.0;
    let ndc_y = 1.0 - pix_y * 2.0 / f32(config.texture_height);
    
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Clear with transparent pixels
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
} 