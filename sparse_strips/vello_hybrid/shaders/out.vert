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
uniform Config_block_0Vertex { Config _group_0_binding_1_vs; };

layout(location = 0) in uint _p2vs_location0;
layout(location = 1) in uint _p2vs_location1;
layout(location = 2) in uint _p2vs_location2;
layout(location = 3) in uint _p2vs_location3;
layout(location = 0) smooth out vec2 _vs2fs_location0;
layout(location = 1) flat out uint _vs2fs_location1;
layout(location = 2) flat out uint _vs2fs_location2;

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
    uint in_vertex_index = uint(gl_VertexID);
    StripInstance instance = StripInstance(_p2vs_location0, _p2vs_location1, _p2vs_location2, _p2vs_location3);
    VertexOutput out_ = VertexOutput(vec2(0.0), 0u, 0u, vec4(0.0));
    float x = float((in_vertex_index & 1u));
    float y = float((in_vertex_index >> 1u));
    uint x0_ = (instance.xy & 65535u);
    uint y0_ = (instance.xy >> 16u);
    uint width = (instance.widths & 65535u);
    uint dense_width = (instance.widths >> 16u);
    out_.dense_end = (instance.col + dense_width);
    float pix_x = (float(x0_) + (float(width) * x));
    uint _e31 = _group_0_binding_1_vs.strip_height;
    float pix_y = (float(y0_) + (y * float(_e31)));
    uint _e39 = _group_0_binding_1_vs.width;
    float ndc_x = (((pix_x * 2.0) / float(_e39)) - 1.0);
    uint _e48 = _group_0_binding_1_vs.height;
    float ndc_y = (1.0 - ((pix_y * 2.0) / float(_e48)));
    out_.position = vec4(ndc_x, ndc_y, 0.0, 1.0);
    uint _e65 = _group_0_binding_1_vs.strip_height;
    out_.tex_coord = vec2((float(instance.col) + (x * float(width))), (y * float(_e65)));
    out_.color = instance.rgba;
    VertexOutput _e71 = out_;
    _vs2fs_location0 = _e71.tex_coord;
    _vs2fs_location1 = _e71.dense_end;
    _vs2fs_location2 = _e71.color;
    gl_Position = _e71.position;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

