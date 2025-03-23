/// This integration test ensures that `vello_hybrid` can be used in a browser environment 
/// targetting WebGL2.
///
/// In order to run this locally, you need to have [`wasm-pack`] installed. Install it using
/// the steps found in https://rustwasm.github.io/wasm-pack/installer/.
///
/// Thereafter, you can run this test with `wasm-pack test --chrome --headless`. This assumes
/// you have chrome installed on your system.
///
/// [`wasm-pack`]: https://rustwasm.github.io/wasm-pack/

#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);
    use vello_common::peniko::{
        color::palette,
        kurbo::BezPath,
    };

    #[wasm_bindgen]
    struct RendererWrapper {
        renderer: vello_hybrid::Renderer,
        device: wgpu::Device,
        queue: wgpu::Queue,
        scene: vello_hybrid::Scene,
        surface: wgpu::Surface<'static>,
    }

    #[wasm_bindgen]
    impl RendererWrapper {
        pub async fn new(canvas: web_sys::HtmlCanvasElement) -> Self {
            let width = canvas.width();
            let height = canvas.height();
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                // Target WebGL2
                backends: wgpu::Backends::GL,
                ..Default::default()
            });
            let surface = instance
                .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
                .expect("Canvas surface to be valid");

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    compatible_surface: Some(&surface),
                    ..Default::default()
                })
                .await
                .expect("Adapter to be valid");

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("Primary device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                        memory_hints: wgpu::MemoryHints::MemoryUsage,
                    },
                    None,
                )
                .await
                .expect("Device to be valid");

            // Configure the surface
            let surface_format = wgpu::TextureFormat::Rgba8Unorm;
            let surface_config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width,
                height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                desired_maximum_frame_latency: 2,
                view_formats: vec![],
            };
            surface.configure(&device, &surface_config);

            let scene = vello_hybrid::Scene::new(width as u16, height as u16);
            let renderer = vello_hybrid::Renderer::new(
                &device,
                &vello_hybrid::RendererOptions {
                    format: surface_format,
                },
            );

            Self {
                renderer,
                scene,
                device,
                queue,
                surface,
            }
        }
    }

    #[wasm_bindgen_test]
    async fn test_renders_triangle() {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Debug).unwrap();
        let canvas = web_sys::Window::document(&web_sys::window().unwrap())
            .unwrap()
            .create_element("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();

        canvas.set_width(200);
        canvas.set_height(200);

        // Add canvas to body
        web_sys::Window::document(&web_sys::window().unwrap())
            .unwrap()
            .body()
            .unwrap()
            .append_child(&canvas)
            .unwrap();

        let RendererWrapper {
            mut renderer,
            device,
            queue,
            mut scene,
            surface,
        } = RendererWrapper::new(canvas).await;

        draw_triangle(&mut scene);

        let params = vello_hybrid::RenderParams {
            width: 100,
            height: 100,
        };

        renderer.prepare(&device, &queue, &scene, &params);

        let surface_texture = surface
            .get_current_texture()
            .expect("Surface texture to be valid");
        let surface_texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            renderer.render(&scene, &mut pass, &params);
        }

        queue.submit([encoder.finish()]);
        surface_texture.present();
    }

    fn draw_triangle(ctx: &mut vello_hybrid::Scene) {
        let mut path = BezPath::new();
        path.move_to((30.0, 40.0));
        path.line_to((50.0, 20.0));
        path.line_to((70.0, 40.0));
        path.close_path();
        ctx.set_paint(palette::css::BLUE.into());
        ctx.fill_path(&path);
    }
}
