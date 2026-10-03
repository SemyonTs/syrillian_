use crate::cache::AssetCache;
use crate::passes::fsr::{FsrPass, FsrSettings};
use crate::passes::post_process::{
    BloomRenderPass, BloomSettings, FinalRenderPass, FxaaRenderPass, PostProcessPass,
    PostProcessPassContext, PostProcessRoute, PostProcessSharedViews,
    ScreenSpaceAmbientOcclusionRenderPass, ScreenSpaceReflectionRenderPass,
};
use crate::passes::ui_pass::UiRenderPass;
use crate::rendering::offscreen_surface::OffscreenSurface;
use crate::rendering::render_data::RenderUniformData;
use crate::rendering::renderer::RenderedFrame;
use crate::rendering::state::State;
use crate::rendering::viewport::{RenderViewport, ViewportId};
use crate::strobe::StrobeRenderer;
use syrillian_utils::{AntiAliasingMode, EngineArgs};
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Extent3d, Queue,
    SurfaceConfiguration, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureView, TextureViewDescriptor,
};
use winit::dpi::PhysicalSize;

const COLOR_ID_BASE: u32 = 0;
const COLOR_ID_POST_A: u32 = 1;
const COLOR_ID_POST_B: u32 = 2;
const COLOR_ID_FINAL_A: u32 = 3;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct PostProcessRouting {
    run_ssr: bool,
    run_ssao: bool,
    run_bloom: bool,
    // AA part — mutually exclusive
    aa_mode: AntiAliasingMode,
    run_fsr: bool,
    run_fxaa: bool,
}

impl PostProcessRouting {
    fn current() -> Self {
        let aa = EngineArgs::aa_mode();
        Self {
            run_ssr: !EngineArgs::get().no_ssr,
            run_ssao: !EngineArgs::get().no_ssao,
            run_bloom: !EngineArgs::get().no_bloom,
            aa_mode: aa,
            run_fsr: matches!(aa, AntiAliasingMode::Fsr),
            run_fxaa: matches!(aa, AntiAliasingMode::Fxaa),
        }
    }

    /// Completely disable post-processing (used with `disable_post_processing`).
    fn disabled() -> Self {
        Self {
            run_ssr: false,
            run_ssao: false,
            run_bloom: false,
            aa_mode: AntiAliasingMode::Off,
            run_fsr: false,
            run_fxaa: false,
        }
    }
}

struct ActivePostProcessRoutes {
    ssr: PostProcessRoute,
    ssao: PostProcessRoute,
    bloom: PostProcessRoute,
    fxaa: PostProcessRoute,
    // The `fsr` field was removed: FSR is called directly in `run_post_process_chain`,
    // and its route is only needed locally during compose to switch `current_view`
    // to `fsr_output` for the final pass.
    final_pass: PostProcessRoute,
}

pub struct FinalFrameContext<'a> {
    pub render_data: &'a RenderUniformData,
    pub target: ViewportId,
    pub size: PhysicalSize<u32>,
    pub format: TextureFormat,
    pub frame_count: usize,
    pub disable_post_processing: bool,
}

pub struct RenderPipeline {
    device: Device,
    pub depth_texture: Texture,
    pub offscreen_surface: OffscreenSurface,
    pub final_surfaces: [OffscreenSurface; 2],
    post_process_surfaces: [OffscreenSurface; 2],
    pub g_normal: Texture,
    pub g_material: Texture,
    pub g_velocity: Texture,
    shared_views: PostProcessSharedViews,

    // FSR-specific resources
    fsr_pass: FsrPass,
    fsr_output: Texture,
    dilated_depth: Texture,
    dilated_motion_vectors: Texture,
    reconstructed_previous_depth: Buffer,

    pub ssr_pass: ScreenSpaceReflectionRenderPass,
    pub ssao_pass: ScreenSpaceAmbientOcclusionRenderPass,
    pub fxaa_pass: FxaaRenderPass,
    pub bloom_pass: BloomRenderPass,
    pub final_pass: FinalRenderPass,

    route_key: PostProcessRouting,
    bloom_settings: BloomSettings,
    bloom_settings_dirty: bool,

    // Render size. `upscale_size` lives inside `FsrPass` and in `final_surfaces`,
    // so it is not needed as a separate field.
    render_size: Extent3d,
}

impl RenderPipeline {
    pub fn new(
        device: &Device,
        queue: &Queue,
        cache: &AssetCache,
        config: &SurfaceConfiguration,
    ) -> Self {
        let pp_bgl = cache.bgl_post_process();

        let aa_mode = EngineArgs::aa_mode();
        let scale = aa_mode.render_scale();

        let upscale_size = Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        };
        let render_size = Extent3d {
            width: ((upscale_size.width as f32) * scale).round().max(1.0) as u32,
            height: ((upscale_size.height as f32) * scale).round().max(1.0) as u32,
            depth_or_array_layers: 1,
        };
        tracing::trace!(
            "[FSR] upscale={}x{} render={}x{} scale={:.3}",
            upscale_size.width,
            upscale_size.height,
            render_size.width,
            render_size.height,
            scale
        );

        let normal_texture = Self::create_g_buffer(
            "GBuffer (Normals)",
            device,
            render_size.width,
            render_size.height,
        );
        let material_texture =
            Self::create_material_texture(device, render_size.width, render_size.height);
        let velocity_texture =
            Self::create_velocity_texture(device, render_size.width, render_size.height);
        let depth_texture =
            Self::create_depth_texture(device, render_size.width, render_size.height);
        let shared_views = PostProcessSharedViews {
            depth: depth_texture.create_view(&TextureViewDescriptor::default()),
            g_normal: normal_texture.create_view(&TextureViewDescriptor::default()),
            g_material: material_texture.create_view(&TextureViewDescriptor::default()),
            g_velocity: velocity_texture.create_view(&TextureViewDescriptor::default()),
        };

        let offscreen_surface = OffscreenSurface::new_sized_with(
            device,
            render_size.width,
            render_size.height,
            TextureFormat::Rgba8Unorm,
            TextureUsages::empty(),
        );

        let post_process_surfaces = [
            OffscreenSurface::new_sized_with(
                device,
                render_size.width,
                render_size.height,
                TextureFormat::Rgba8Unorm,
                TextureUsages::STORAGE_BINDING,
            ),
            OffscreenSurface::new_sized_with(
                device,
                render_size.width,
                render_size.height,
                TextureFormat::Rgba8Unorm,
                TextureUsages::STORAGE_BINDING,
            ),
        ];

        let final_surfaces = [
            OffscreenSurface::new_sized_with(
                device,
                upscale_size.width,
                upscale_size.height,
                config.format,
                TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            ),
            OffscreenSurface::new_sized_with(
                device,
                upscale_size.width,
                upscale_size.height,
                config.format,
                TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            ),
        ];

        // --- FSR resources ---
        // FSR output — upscale_size
        let fsr_output = Self::create_fsr_texture(
            device,
            upscale_size.width,
            upscale_size.height,
            "FSR Output",
            TextureFormat::Rgba16Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        );
        let dilated_depth = Self::create_fsr_texture(
            device,
            render_size.width,
            render_size.height,
            "Dilated Depth",
            TextureFormat::R32Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        );
        let dilated_motion_vectors = Self::create_fsr_texture(
            device,
            render_size.width,
            render_size.height,
            "Dilated Motion Vectors",
            TextureFormat::Rg16Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        );

        let reconstructed_previous_depth = device.create_buffer(&BufferDescriptor {
            label: Some("Reconstructed Previous Depth"),
            size: (render_size.width * render_size.height * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let fsr_pass = FsrPass::new(
            device,
            queue,
            [render_size.width, render_size.height],
            [upscale_size.width, upscale_size.height],
            FsrSettings::from_engine_args(),
        );
        // --- End FSR resources ---

        let bloom_settings = BloomSettings::from_engine_args();
        let routing = PostProcessRouting::current();

        let routes = Self::compose_routes(
            routing,
            offscreen_surface.view().clone(),
            post_process_surfaces[0].view().clone(),
            post_process_surfaces[1].view().clone(),
            fsr_output.create_view(&TextureViewDescriptor::default()),
            final_surfaces[0].view().clone(),
        );

        let ssr_pass = ScreenSpaceReflectionRenderPass::new(
            device,
            cache.bgl_post_process_compute(),
            &shared_views,
            &routes.ssr,
        );

        let ssao_pass = ScreenSpaceAmbientOcclusionRenderPass::new(
            device,
            render_size.width,
            render_size.height,
            cache.bgl_ssao_compute(),
            cache.bgl_ssao_apply_compute(),
            &shared_views,
            &routes.ssao,
        );

        let bloom_pass = BloomRenderPass::new(
            device,
            render_size.width,
            render_size.height,
            cache.bgl_bloom_compute(),
            &routes.bloom,
            &bloom_settings,
        );

        let fxaa_pass = FxaaRenderPass::new(device, &pp_bgl, &shared_views, &routes.fxaa);

        let final_pass = FinalRenderPass::new(device, &pp_bgl, &shared_views, &routes.final_pass);

        Self {
            device: device.clone(),
            depth_texture,
            offscreen_surface,
            final_surfaces,
            post_process_surfaces,
            g_normal: normal_texture,
            g_material: material_texture,
            g_velocity: velocity_texture,
            shared_views,
            fsr_pass,
            fsr_output,
            dilated_depth,
            dilated_motion_vectors,
            reconstructed_previous_depth,
            ssr_pass,
            ssao_pass,
            fxaa_pass,
            bloom_pass,
            final_pass,
            route_key: routing,
            bloom_settings,
            bloom_settings_dirty: false,
            render_size,
        }
    }

    pub fn recreate(
        &mut self,
        device: &Device,
        queue: &Queue,
        cache: &AssetCache,
        config: &SurfaceConfiguration,
    ) {
        let bloom_settings = self.bloom_settings;
        *self = Self::new(device, queue, cache, config);
        self.set_bloom_settings(bloom_settings);
    }

    #[inline]
    fn current_surface_index(frame_count: usize) -> usize {
        frame_count % 2
    }

    fn compose_routes(
        key: PostProcessRouting,
        base_view: TextureView,
        post_a_view: TextureView,
        post_b_view: TextureView,
        fsr_output_view: TextureView,
        final_a_view: TextureView,
    ) -> ActivePostProcessRoutes {
        let default_route = PostProcessRoute {
            input_id: COLOR_ID_BASE,
            output_id: COLOR_ID_POST_A,
            input_color: base_view.clone(),
            output_color: post_a_view.clone(),
        };

        let mut ssr = default_route.clone();
        let mut ssao = default_route.clone();
        let mut bloom = default_route.clone();
        let mut fxaa = default_route.clone();

        let mut current_id = COLOR_ID_BASE;
        let mut current_view = base_view;
        let mut write_to_a = true;

        let mut next_output = || {
            if write_to_a {
                write_to_a = false;
                (COLOR_ID_POST_A, post_a_view.clone())
            } else {
                write_to_a = true;
                (COLOR_ID_POST_B, post_b_view.clone())
            }
        };

        if key.run_ssr {
            let (output_id, output_view) = next_output();
            ssr = PostProcessRoute {
                input_id: current_id,
                output_id,
                input_color: current_view.clone(),
                output_color: output_view.clone(),
            };
            current_id = output_id;
            current_view = output_view;
        }

        if key.run_ssao {
            let (output_id, output_view) = next_output();
            ssao = PostProcessRoute {
                input_id: current_id,
                output_id,
                input_color: current_view.clone(),
                output_color: output_view.clone(),
            };
            current_id = output_id;
            current_view = output_view;
        }

        if key.run_bloom {
            let (output_id, output_view) = next_output();
            bloom = PostProcessRoute {
                input_id: current_id,
                output_id,
                input_color: current_view.clone(),
                output_color: output_view.clone(),
            };
            current_id = output_id;
            current_view = output_view;
        }

        // AA effects are mutually exclusive.
        // FXAA is a real route; it is called in the post-process chain.
        // FSR is not stored as a route because it is called directly from
        // `run_post_process_chain`, but it must switch `current_view` to
        // `fsr_output` so that `final_pass` reads the FSR result.
        match key.aa_mode {
            AntiAliasingMode::Fxaa => {
                let (output_id, output_view) = next_output();
                fxaa = PostProcessRoute {
                    input_id: current_id,
                    output_id,
                    input_color: current_view.clone(),
                    output_color: output_view.clone(),
                };
                current_id = output_id;
                current_view = output_view;
            }
            AntiAliasingMode::Fsr => {
                current_id = COLOR_ID_FINAL_A;
                current_view = fsr_output_view;
            }
            AntiAliasingMode::Off
            | AntiAliasingMode::Msaa(_)
            | AntiAliasingMode::Supersample(_) => {
                // MSAA/SSAA work at the render-target level, not as post-process
            }
        }

        let final_pass = PostProcessRoute {
            input_id: current_id,
            output_id: COLOR_ID_FINAL_A,
            input_color: current_view,
            output_color: final_a_view,
        };

        ActivePostProcessRoutes {
            ssr,
            ssao,
            bloom,
            fxaa,
            final_pass,
        }
    }

    fn create_depth_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&TextureDescriptor {
            label: Some("Depth Texture"),
            size: Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::COPY_SRC
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_g_buffer(which: &'static str, device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&TextureDescriptor {
            label: Some(which),
            size: Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg16Float,
            usage: TextureUsages::COPY_SRC
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_material_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&TextureDescriptor {
            label: Some("Material Property Texture"),
            size: Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8Unorm,
            usage: TextureUsages::COPY_SRC
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_velocity_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&TextureDescriptor {
            label: Some("GBuffer (Velocity)"),
            size: Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg16Float,
            usage: TextureUsages::COPY_SRC
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_fsr_texture(
        device: &Device,
        width: u32,
        height: u32,
        label: &str,
        format: TextureFormat,
        usage: TextureUsages,
    ) -> Texture {
        device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    fn rebuild_post_process_passes(&mut self, cache: &AssetCache, key: PostProcessRouting) {
        let routes = Self::compose_routes(
            key,
            self.offscreen_surface.view().clone(),
            self.post_process_surfaces[0].view().clone(),
            self.post_process_surfaces[1].view().clone(),
            self.fsr_output
                .create_view(&TextureViewDescriptor::default()),
            self.final_surfaces[0].view().clone(),
        );

        self.ssr_pass = ScreenSpaceReflectionRenderPass::new(
            &self.device,
            cache.bgl_post_process_compute(),
            &self.shared_views,
            &routes.ssr,
        );

        self.ssao_pass = ScreenSpaceAmbientOcclusionRenderPass::new(
            &self.device,
            self.render_size.width,
            self.render_size.height,
            cache.bgl_ssao_compute(),
            cache.bgl_ssao_apply_compute(),
            &self.shared_views,
            &routes.ssao,
        );

        self.bloom_pass = BloomRenderPass::new(
            &self.device,
            self.render_size.width,
            self.render_size.height,
            cache.bgl_bloom_compute(),
            &routes.bloom,
            &self.bloom_settings,
        );

        self.fxaa_pass = FxaaRenderPass::new(
            &self.device,
            &cache.bgl_post_process(),
            &self.shared_views,
            &routes.fxaa,
        );

        self.final_pass = FinalRenderPass::new(
            &self.device,
            &cache.bgl_post_process(),
            &self.shared_views,
            &routes.final_pass,
        );

        self.route_key = key;
        self.bloom_settings_dirty = false;
    }

    pub fn prepare_frame(
        &mut self,
        render_data: &mut RenderUniformData,
        queue: &Queue,
        frame_count: usize,
    ) {
        let aa = EngineArgs::aa_mode();
        let mut proj = render_data.camera_data.projection_mat;

        if aa.needs_jitter() {
            let render_size = self.render_size;

            let jitter = self.fsr_pass.compute_jitter(frame_count);
            // glam Mat4 — column-major. Jitter on x/y is applied to the third column
            // (z_axis) of the projection matrix — this is the projection center position on x/y.
            proj.z_axis.x += jitter[0] * 2.0 / render_size.width as f32;
            proj.z_axis.y += jitter[1] * 2.0 / render_size.height as f32;

            self.fsr_pass.set_jitter(jitter);
        }

        let base_view_proj = proj * render_data.camera_data.view_mat;
        render_data.camera_data.proj_view_mat = base_view_proj;
        render_data.camera_data.inv_proj_view_mat = base_view_proj.inverse();

        render_data.upload_camera_data(queue);

        if self.bloom_settings_dirty {
            let desired = PostProcessRouting::current();
            if desired == self.route_key {
                self.bloom_pass.update_settings(queue, &self.bloom_settings);
                self.bloom_settings_dirty = false;
            }
        }
    }

    pub fn set_bloom_settings(&mut self, settings: BloomSettings) {
        self.bloom_settings = settings.sanitized();
        self.bloom_settings_dirty = true;
    }

    pub fn bloom_settings(&self) -> &BloomSettings {
        &self.bloom_settings
    }

    pub fn render_ui_onto_final_frame(
        &self,
        encoder: &mut CommandEncoder,
        strobe: &mut StrobeRenderer,
        viewport: &RenderViewport,
        cache: &AssetCache,
        state: &State,
    ) {
        let final_color = &self.final_surfaces[Self::current_surface_index(viewport.frame_count())];

        UiRenderPass::render(encoder, strobe, final_color.view(), viewport, cache, state);
    }

    fn run_post_process_chain(
        &mut self,
        camera_render_data: &RenderUniformData,
        encoder: &mut CommandEncoder,
        cache: &AssetCache,
        final_output: TextureView,
    ) {
        // === Block 1: regular post-process passes ===
        // Scoped so `ctx` releases the `encoder` reborrow before the FSR call.
        {
            let mut ctx = PostProcessPassContext {
                camera_render_data,
                encoder: &mut *encoder,
                cache,
            };

            let mut ping_index = 0usize;

            if self.route_key.run_ssr {
                let output_color = self.post_process_surfaces[ping_index].view();
                self.ssr_pass.execute(&mut ctx, output_color);
                ping_index = 1 - ping_index;
            }

            if self.route_key.run_ssao {
                let output_color = self.post_process_surfaces[ping_index].view();
                self.ssao_pass.execute(&mut ctx, output_color);
                ping_index = 1 - ping_index;
            }

            if self.route_key.run_bloom {
                let output_color = self.post_process_surfaces[ping_index].view();
                self.bloom_pass.execute(&mut ctx, output_color);
                ping_index = 1 - ping_index;
            }

            if self.route_key.run_fxaa {
                let output_color = self.post_process_surfaces[ping_index].view();
                self.fxaa_pass.execute(&mut ctx, output_color);
            }
        }

        // === Block 2: FSR — works directly with the encoder ===
        if self.route_key.run_fsr {
            self.fsr_pass.execute(
                encoder,
                self.offscreen_surface.texture(), // was &self.fsr_color
                &self.depth_texture,
                &self.g_velocity,
                &self.fsr_output,
                &self.dilated_depth,
                &self.dilated_motion_vectors,
                &self.reconstructed_previous_depth,
                camera_render_data,
            );
        }

        // === Block 3: final pass — new ctx ===
        let mut ctx = PostProcessPassContext {
            camera_render_data,
            encoder: &mut *encoder,
            cache,
        };
        self.final_pass.execute(&mut ctx, &final_output);
    }

    pub fn finalize_frame(
        &mut self,
        encoder: &mut CommandEncoder,
        cache: &AssetCache,
        context: FinalFrameContext<'_>,
    ) -> RenderedFrame {
        let effective_key = if context.disable_post_processing {
            PostProcessRouting::disabled()
        } else {
            PostProcessRouting::current()
        };

        if effective_key != self.route_key {
            self.rebuild_post_process_passes(cache, effective_key);
        }

        let current_idx = Self::current_surface_index(context.frame_count);
        let final_color = &self.final_surfaces[current_idx];
        let final_frame_texture = final_color.texture().clone();

        self.run_post_process_chain(
            context.render_data,
            encoder,
            cache,
            final_color.view().clone(),
        );

        RenderedFrame {
            target: context.target,
            frame: final_frame_texture,
            size: context.size,
            format: context.format,
        }
    }
}
