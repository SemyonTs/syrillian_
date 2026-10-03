use argh::FromArgs;
use glamx::UVec2;
use std::cmp::Ordering;
use std::sync::LazyLock;
use std::time::Duration;

#[derive(Debug, Copy, Clone, Default)]
pub enum AntiAliasingMode {
    /// No anti-aliasing.
    Off,
    /// Fast Approximate Anti-Aliasing — a cheap post-process.
    #[default]
    Fxaa,
    /// FidelityFX Super Resolution 3.
    /// With `fsr_render_scale == 1.0` it works as TAA,
    /// otherwise as an upscaler.
    Fsr,
    /// Multi-Sample Anti-Aliasing with the given sample count (2/4/8).
    /// Requires changing `sample_count` on render targets.
    Msaa(u32),
    /// Supersampling — render at a higher resolution and downscale.
    /// `factor` — how many times larger to render (e.g. 1.5 or 2.0).
    Supersample(f32),
}

impl PartialEq for AntiAliasingMode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Off, Self::Off) => true,
            (Self::Fxaa, Self::Fxaa) => true,
            (Self::Fsr, Self::Fsr) => true,
            (Self::Msaa(a), Self::Msaa(b)) => a == b,
            (Self::Supersample(a), Self::Supersample(b)) => a.to_bits() == b.to_bits(),
            _ => false,
        }
    }
}

impl Eq for AntiAliasingMode {}

impl AntiAliasingMode {
    /// Whether the mode requires temporal accumulation (motion vectors, jitter).
    pub fn is_temporal(self) -> bool {
        matches!(self, Self::Fsr)
    }

    /// Whether the mode requires a post-process pass.
    pub fn is_post_process(self) -> bool {
        matches!(self, Self::Fxaa | Self::Fsr)
    }

    /// Whether motion vectors are needed by the scene.
    pub fn needs_motion_vectors(self) -> bool {
        matches!(self, Self::Fsr)
    }

    /// Whether projection jitter is needed.
    pub fn needs_jitter(self) -> bool {
        matches!(self, Self::Fsr)
    }

    /// Render resolution multiplier (>1.0 for SSAA, 1.0 for others).
    pub fn render_scale(self) -> f32 {
        match self {
            Self::Supersample(factor) => factor.clamp(1.0, 4.0),
            Self::Fsr => EngineArgs::fsr_render_scale(),
            _ => 1.0,
        }
    }

    /// Sample count for MSAA mode, otherwise 1.
    pub fn sample_count(self) -> u32 {
        match self {
            Self::Msaa(n) => n.clamp(1, 8),
            _ => 1,
        }
    }
}

fn present_mode(mode: &str) -> Result<Option<wgpu::PresentMode>, String> {
    let parsed = match mode {
        "vsync" => wgpu::PresentMode::AutoVsync,
        "no_vsync" => wgpu::PresentMode::AutoNoVsync,
        "fifo" => wgpu::PresentMode::Fifo,
        "fifo_relaxed" => wgpu::PresentMode::FifoRelaxed,
        "mailbox" => wgpu::PresentMode::Mailbox,
        "immediate" => wgpu::PresentMode::Immediate,
        _ => return Ok(None),
    };
    Ok(Some(parsed))
}

fn window_size(size: &str) -> Result<Option<UVec2>, String> {
    let char = if size.contains('x') { 'x' } else { ',' };

    let mut split = size.split(char);
    let x: Option<u32> = split.next().and_then(|x| x.parse().ok());
    let y: Option<u32> = split.next().and_then(|y| y.parse().ok());

    let size = match (x, y) {
        (Some(x), Some(y)) => UVec2::new(x, y),
        (Some(x), _) => UVec2::new(x, x),
        _ => return Ok(None),
    };

    Ok(Some(size))
}

fn force_backend(backend: &str) -> Result<Option<Vec<wgpu::Backends>>, String> {
    let backends: wgpu::Backends = wgpu::Backends::from_comma_list(backend);

    if backends.is_empty() {
        return Ok(None);
    }

    let mut backends: Vec<wgpu::Backends> = backends.into_iter().collect();

    // push vulkan back if it's a choice because all other backends are more stable
    backends.sort_by(|a, _| {
        if a.contains(wgpu::Backends::VULKAN) {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    });

    Ok(Some(backends))
}

fn aa_mode(mode: &str) -> Result<Option<AntiAliasingMode>, String> {
    let mode = match mode {
        "off" | "none" => AntiAliasingMode::Off,
        "fxaa" => AntiAliasingMode::Fxaa,
        "fsr" | "taa" => AntiAliasingMode::Fsr,
        "ssaa1.5" => AntiAliasingMode::Supersample(1.5),
        "ssaa2" => AntiAliasingMode::Supersample(2.0),
        _ => return Ok(None),
    };
    Ok(Some(mode))
}

/// Engine arguments
#[derive(Default, FromArgs)]
pub struct EngineArgs {
    #[argh(switch, hidden_help)]
    pub fullscreen: bool,
    #[argh(switch, hidden_help)]
    pub no_fullscreen: bool, // TODO: Implement
    #[argh(switch, hidden_help)]
    pub no_frustum_culling: bool,
    #[argh(switch, hidden_help)]
    pub no_shadows: bool,
    #[argh(switch, hidden_help)]
    pub no_ssr: bool,
    #[argh(switch, hidden_help)]
    pub no_ssao: bool,
    #[argh(switch, hidden_help)]
    pub no_bloom: bool,

    #[argh(option, hidden_help)]
    pub max_frames_in_flight: Option<u32>,
    #[argh(option, hidden_help)]
    pub max_fps: Option<u32>,
    #[argh(option, hidden_help)]
    pub physics_timestep: Option<f64>,

    #[argh(option, hidden_help, from_str_fn(present_mode))]
    pub present_mode: Option<Option<wgpu::PresentMode>>,
    #[argh(option, hidden_help, from_str_fn(window_size))]
    pub window_size: Option<Option<UVec2>>,
    #[argh(option, hidden_help, from_str_fn(force_backend))]
    pub force_backend: Option<Option<Vec<wgpu::Backends>>>,
    #[argh(option, hidden_help, from_str_fn(aa_mode))]
    pub aa_mode: Option<Option<AntiAliasingMode>>,
    #[argh(option, hidden_help)]
    pub bloom_threshold: Option<f32>,
    #[argh(option, hidden_help)]
    pub bloom_soft_knee: Option<f32>,
    #[argh(option, hidden_help)]
    pub bloom_intensity: Option<f32>,
    #[argh(option, hidden_help)]
    pub bloom_radius: Option<f32>,
    #[argh(option, hidden_help)]
    pub bloom_clamp_max: Option<f32>,
    #[argh(option, hidden_help)]
    pub bloom_blur_passes: Option<u32>,

    /// fsr sharpness in [0.0, 1.0]. 0.0 disables RCAS sharpening.
    #[argh(option, hidden_help)]
    pub fsr_sharpness: Option<f32>,

    /// render-scale for FSR upscaling. 1.0 = 1:1 (TAA only).
    #[argh(option, hidden_help)]
    pub fsr_render_scale: Option<f32>,

    /// treat depth buffer as inverted (reverse-Z).
    #[argh(switch, hidden_help)]
    pub depth_inverted: bool,

    /// treat depth buffer as infinite (reversed-Z with infinite far plane).
    #[argh(switch, hidden_help)]
    pub depth_infinite: bool,
}

impl EngineArgs {
    fn init() -> Option<EngineArgs> {
        let mut args = std::env::args();
        let cmd_name = args.next()?;
        let args: Vec<String> = args.collect();
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        EngineArgs::from_args(&[&cmd_name], &args).ok()
    }

    pub fn get() -> &'static EngineArgs {
        static INSTANCE: LazyLock<EngineArgs> =
            LazyLock::new(|| EngineArgs::init().unwrap_or_default());
        &INSTANCE
    }

    pub fn default_window_size() -> UVec2 {
        EngineArgs::get()
            .window_size
            .flatten()
            .unwrap_or(UVec2::new(800, 600))
    }

    pub fn max_fps() -> Option<u32> {
        EngineArgs::get().max_fps.filter(|&fps| fps > 0)
    }

    pub fn max_frame_interval() -> Option<Duration> {
        Self::max_fps().map(|fps| Duration::from_secs_f32(1.0 / fps as f32))
    }

    pub fn aa_mode() -> AntiAliasingMode {
        EngineArgs::get().aa_mode.flatten().unwrap_or_default()
    }

    /// Whether FSR is currently active based on the AA mode.
    pub fn fsr_enabled() -> bool {
        matches!(Self::aa_mode(), AntiAliasingMode::Fsr)
    }

    /// RCAS sharpness for FSR in [0.0, 1.0]. Returns `0.5` by default.
    pub fn fsr_sharpness_value() -> f32 {
        EngineArgs::get()
            .fsr_sharpness
            .unwrap_or(0.5)
            .clamp(0.0, 1.0)
    }

    /// Whether the RCAS sharpening pass should run.
    pub fn fsr_sharpening() -> bool {
        Self::fsr_sharpness_value() > 0.0
    }

    /// Render-scale factor for FSR upscaling. Clamped to `[0.1, 1.0]`.
    /// `1.0` means no upscaling (FSR acts as TAA).
    pub fn fsr_render_scale() -> f32 {
        EngineArgs::get()
            .fsr_render_scale
            .unwrap_or(1.0)
            .clamp(0.1, 1.0)
    }

    /// Whether the depth buffer uses reverse-Z (inverted depth).
    pub fn depth_inverted() -> bool {
        EngineArgs::get().depth_inverted
    }

    /// Whether the depth buffer has an infinite far plane.
    pub fn depth_infinite() -> bool {
        EngineArgs::get().depth_infinite
    }
}
