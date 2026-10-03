use crate::rendering::render_data::RenderUniformData;
use syrillian_utils::EngineArgs;
use wgpu::{Buffer, CommandEncoder, Device, Queue, Texture};
use wgpu_ffx::{
    FsrContext, FsrContextFlags, FsrContextInfo, FsrDispatchFlags, FsrDispatchInfo, FsrView,
    get_jitter_offset, get_jitter_phase_count,
};

#[derive(Debug, Copy, Clone)]
pub struct FsrSettings {
    pub enabled: bool,
    pub sharpening: bool,
    pub sharpness: f32,
    pub high_dynamic_range: bool,
    pub depth_inverted: bool,
    pub depth_infinite: bool,
}

impl Default for FsrSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            sharpening: true,
            sharpness: 0.5,
            high_dynamic_range: false,
            depth_inverted: false, // requires integration with reversed z
            depth_infinite: false, // requires integration with reversed z
        }
    }
}

impl FsrSettings {
    pub fn from_engine_args() -> Self {
        Self {
            enabled: EngineArgs::fsr_enabled(),
            sharpening: EngineArgs::fsr_sharpening(),
            sharpness: EngineArgs::fsr_sharpness_value(),
            high_dynamic_range: false,
            depth_inverted: EngineArgs::depth_inverted(),
            depth_infinite: EngineArgs::depth_infinite(),
        }
    }
}

pub struct FsrPass {
    context: FsrContext,
    view: FsrView,
    settings: FsrSettings,
    /// Current render size (may be smaller than upscale_size when upscaling).
    render_size: [u32; 2],
    /// Current final image size.
    upscale_size: [u32; 2],
    current_jitter: [f32; 2],
    reset_next_frame: bool,
}

impl FsrPass {
    pub fn new(
        device: &Device,
        queue: &Queue,
        render_size: [u32; 2],
        upscale_size: [u32; 2],
        settings: FsrSettings,
    ) -> Self {
        let mut flags = FsrContextFlags::empty();
        if settings.high_dynamic_range {
            flags |= FsrContextFlags::HIGH_DYNAMIC_RANGE;
        }
        if settings.depth_inverted {
            flags |= FsrContextFlags::DEPTH_INVERTED;
        }
        if settings.depth_infinite {
            flags |= FsrContextFlags::DEPTH_INFINITE;
        }

        let context = FsrContext::new(FsrContextInfo {
            device: device.clone(),
            flags,
        });

        let view = context.create_view(queue, render_size, upscale_size);

        Self {
            context,
            view,
            settings,
            render_size,
            upscale_size,
            current_jitter: [0.0, 0.0],
            reset_next_frame: true,
        }
    }

    pub fn compute_jitter(&self, frame_count: usize) -> [f32; 2] {
        let phase_count =
            get_jitter_phase_count(self.render_size[0] as i32, self.upscale_size[0] as i32);
        let phase = (frame_count % phase_count as usize) as i32;
        get_jitter_offset(phase, phase_count)
    }

    pub fn set_jitter(&mut self, jitter: [f32; 2]) {
        self.current_jitter = jitter;
    }

    pub fn request_reset(&mut self) {
        self.reset_next_frame = true;
    }

    /// Full FSR state reset — recreates `FsrView`,
    /// clearing all temporal history.
    pub fn force_reset(&mut self, queue: &Queue) {
        self.view = self
            .context
            .create_view(queue, self.render_size, self.upscale_size);
        self.reset_next_frame = true;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        encoder: &mut CommandEncoder,
        color: &Texture,
        depth: &Texture,
        motion_vectors: &Texture,
        output: &Texture,
        dilated_depth: &Texture,
        dilated_motion_vectors: &Texture,
        reconstructed_previous_depth: &Buffer,
        render_data: &RenderUniformData,
    ) {
        // `delta_time` in `SystemData` is stored in seconds,
        // FSR expects milliseconds.
        let frame_time_delta_ms = (render_data.system_data.delta_time * 1000.0).max(1.0);

        // `CameraUniform.fov` is in degrees (see `CameraComponent::regenerate`).
        let camera_fov_y = render_data.camera_data.fov.to_radians();

        let info = FsrDispatchInfo {
            // Textures in/out
            color: color.clone(),
            depth: depth.clone(),
            motion_vectors: motion_vectors.clone(),
            output: output.clone(),
            dilated_depth: dilated_depth.clone(),
            dilated_motion_vectors: dilated_motion_vectors.clone(),
            reconstructed_previous_depth: reconstructed_previous_depth.clone(),

            // Optional textures
            exposure: None,
            reactive_mask: None,
            transparency_and_composition: None,

            // Sizes
            render_size: self.render_size,
            upscale_size: self.upscale_size,

            // Jitter and motion vectors
            jitter_offset: self.current_jitter,
            // X-flip: our velocity shader writes MVs with the opposite sign on X
            // relative to what FSR expects. Without this — ghosting on motion.
            motion_vector_scale: [-1.0, 1.0],

            // Camera parameters
            camera_near: render_data.camera_data.near,
            camera_far: render_data.camera_data.far,
            camera_fov_y,
            view_space_to_meters_factor: 1.0,

            // Timing and exposure
            frame_time_delta: frame_time_delta_ms,
            pre_exposure: 1.0,

            // History reset — set via `request_reset()` or `force_reset()`
            reset_history: self.reset_next_frame,

            // Sharpening
            enable_sharpening: self.settings.sharpening,
            sharpness: self.settings.sharpness,

            // Additional flags (debug view, etc.) — empty for now
            flags: FsrDispatchFlags::empty(),
        };

        self.reset_next_frame = false;

        if let Err(err) = self.context.dispatch(&mut self.view, encoder, &info) {
            tracing::error!("FSR dispatch failed: {err}");
        }
    }

    pub fn resize(&mut self, queue: &Queue, render_size: [u32; 2], upscale_size: [u32; 2]) {
        self.render_size = render_size;
        self.upscale_size = upscale_size;
        self.view.resize(queue, render_size, upscale_size);
        self.reset_next_frame = true;
    }
}
