#version 310 es

precision highp float;
precision highp int;

struct Config {
    uint width;
    uint height;
    uint strip_height;
    uint alphas_tex_width_bits;
};
struct StripInstance {
    uint xy;
    uint widths;
    uint col;
    uint rgba;
};
struct VertexOutput {
    vec2 tex_coord;
    uint dense_end;
    uint color;
    vec4 position;
};
uniform Config_block_0Fragment { Config _group_0_binding_1_fs; };

uniform highp usampler2D _group_0_binding_0_fs;

layout(location = 0) smooth in vec2 _vs2fs_location0;
layout(location = 1) flat in uint _vs2fs_location1;
layout(location = 2) flat in uint _vs2fs_location2;
layout(location = 0) out vec4 _fs2p_location0;

uint unpack_alphas_from_channel(uvec4 rgba, uint channel_index) {
    switch(channel_index) {
        case 0u: {
            return rgba.x;
        }
        case 1u: {
            return rgba.y;
        }
        case 2u: {
            return rgba.z;
        }
        case 3u: {
            return rgba.w;
        }
        default: {
            return rgba.x;
        }
    }
}

vec4 unpack4x8unorm(uint rgba_packed) {
    return vec4((float(((rgba_packed >> 0u) & 255u)) / 255.0), (float(((rgba_packed >> 8u) & 255u)) / 255.0), (float(((rgba_packed >> 16u) & 255u)) / 255.0), (float(((rgba_packed >> 24u) & 255u)) / 255.0));
}

void main() {
    VertexOutput in_ = VertexOutput(_vs2fs_location0, _vs2fs_location1, _vs2fs_location2, gl_FragCoord);
    float alpha = 1.0;
    uint alphas_index = uint(floor(in_.tex_coord.x));
    if ((alphas_index < in_.dense_end)) {
        uint y = uint(floor(in_.tex_coord.y));
        uvec2 tex_dimensions = uvec2(textureSize(_group_0_binding_0_fs, 0).xy);
        uint alphas_tex_width = tex_dimensions.x;
        uint texel_index = (alphas_index / 4u);
        uint channel_index_1 = (alphas_index % 4u);
        uint tex_x = (texel_index & (alphas_tex_width - 1u));
        uint _e25 = _group_0_binding_1_fs.alphas_tex_width_bits;
        uint tex_y = (texel_index >> _e25);
        uvec4 rgba_values = texelFetch(_group_0_binding_0_fs, ivec2(uvec2(tex_x, tex_y)), 0);
        uint _e31 = unpack_alphas_from_channel(rgba_values, channel_index_1);
        alpha = (float(((_e31 >> (y * 8u)) & 255u)) * 0.003921569);
    }
    float _e40 = alpha;
    vec4 _e42 = unpack4x8unorm(in_.color);
    _fs2p_location0 = (_e40 * _e42);
    return;
}

