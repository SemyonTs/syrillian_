//! Abstraction over the GPU device and surface state.
//!
//! [`State`] is responsible for creating the GPU "device", swapchain and
//! depth textures. It also exposes methods to resize and recreate these
//! resources when the window changes.

use crate::utils::first_backend_to_str;
use futures::executor::block_on;
use snafu::{ResultExt, Snafu, ensure};
use std::mem;
use syrillian_utils::EngineArgs;
use tracing::{debug, info, trace, warn};
use wgpu::{
    Adapter, Backends, CreateSurfaceError, Device, DeviceDescriptor, ExperimentalFeatures,
    Features, Instance, InstanceDescriptor, Limits, MemoryHints, PowerPreference, Queue,
    RequestAdapterOptions, RequestDeviceError, Surface, SurfaceConfiguration, TextureFormat,
};
use winit::dpi::PhysicalSize;
use winit::event_loop::OwnedDisplayHandle;
use winit::window::Window;

const DEFAULT_BACKENDS: &[Backends] = &[
    Backends::DX12,
    Backends::METAL,
    Backends::VULKAN,
    Backends::GL,
];

type Result<T, E = StateError> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[snafu(context(suffix(Err)))]
pub enum StateError {
    #[snafu(display("Unable to get device: {source}"))]
    RequestDevice { source: RequestDeviceError },

    #[snafu(display(
        "Can only run on Bgra8UnormSrgb currently, but it's not supported by your GPU. Available: {formats:?}"
    ))]
    ColorFormatNotAvailable { formats: Vec<TextureFormat> },

    #[snafu(display("Unable to create surface: {source}"))]
    CreateSurface { source: CreateSurfaceError },
}

#[allow(unused)]
pub struct State {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl State {
    // will respect the order of backends passed instead of a plain `Backends`
    fn try_setup_instance_with<'a>(
        window: &'a Window,
        display_handle: OwnedDisplayHandle,
        backends: &[Backends],
    ) -> Result<(Instance, Surface<'a>)> {
        for backend in backends {
            let mut desc = InstanceDescriptor::new_with_display_handle_from_env(Box::new(
                display_handle.clone(),
            ));
            desc.backends = *backend;

            let instance = Instance::new(desc);
            let surface = instance.create_surface(window).context(CreateSurfaceErr);
            if let Ok(surface) = surface {
                info!("Selected backend: {}", first_backend_to_str(*backend));
                return Ok((instance, surface));
            } else {
                debug!(
                    "Failed to start on backend: {}",
                    first_backend_to_str(*backend)
                );
            }
        }

        warn!(
            "Couldn't start on any selected graphics backend. Retrying with all available backends"
        );

        Self::setup_instance(window, display_handle)
    }

    fn setup_instance<'a>(
        window: &'a Window,
        display_handle: OwnedDisplayHandle,
    ) -> Result<(Instance, Surface<'a>)> {
        let mut desc =
            InstanceDescriptor::new_with_display_handle_from_env(Box::new(display_handle));

        if !cfg!(target_os = "linux") {
            desc.backends ^= Backends::VULKAN;
        }

        let instance = Instance::new(desc);
        let surface = instance.create_surface(window).context(CreateSurfaceErr)?;
        Ok((instance, surface))
    }

    async fn setup_adapter(instance: &Instance, surface: Option<&Surface<'static>>) -> Adapter {
        instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: surface,
                ..RequestAdapterOptions::default()
            })
            .await
            .expect(
                "Couldn't find anything that supports rendering stuff. How are you reading this..?",
            )
    }

    // wgpu tracing is currently unavailable
    const fn trace_mode() -> wgpu::Trace {
        const _IS_DEBUG_ENABLED: bool = cfg!(debug_assertions);

        wgpu::Trace::Off
    }

    async fn get_device_and_queue(adapter: &Adapter) -> Result<(Device, Queue)> {
        let adapter_limits = adapter.limits();
        let max_storage_textures = adapter_limits.max_storage_textures_per_shader_stage;
        let required_storage_textures = if max_storage_textures >= 8 {
            8
        } else {
            warn!(
                "Adapter supports only {max_storage_textures} storage textures/stage; FSR needs 8"
            );
            max_storage_textures
        };

        let adapter_features = adapter.features();

        // FSR3 uses storage textures, which on some GPUs
        // require TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES.
        let mut required_features = Features::default()
            | Features::POLYGON_MODE_LINE
            | Features::IMMEDIATES
            | Features::ADDRESS_MODE_CLAMP_TO_BORDER
            | Features::TEXTURE_FORMAT_16BIT_NORM
            | Features::CLEAR_TEXTURE
            | Features::TEXTURE_COMPRESSION_BC;

        if adapter_features.contains(Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
            required_features |= Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        } else {
            warn!(
                "Adapter does not support TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES — FSR may fail"
            );
        }

        // In wgpu 29.0.4, naga mishandles @coherent/@volatile in FSR shaders.
        // PASSTHROUGH_SHADERS bypasses naga, passing SPIR-V directly to the driver.
        if adapter_features.contains(Features::PASSTHROUGH_SHADERS) {
            required_features |= Features::PASSTHROUGH_SHADERS;
            info!("Enabling PASSTHROUGH_SHADERS for FSR compatibility");
        } else {
            warn!(
                "Adapter does not support PASSTHROUGH_SHADERS — \
                FSR shaders may fail naga validation on this GPU"
            );
        }

        if adapter_features.contains(Features::FLOAT32_FILTERABLE) {
            required_features |= Features::FLOAT32_FILTERABLE;
        } else {
            warn!(
                "Adapter does not support FLOAT32_FILTERABLE — FSR internal Rg32Float textures will fail"
            );
        }

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("Renderer Hardware"),
                required_features,
                required_limits: Limits {
                    max_bind_groups: 6,
                    max_immediate_size: 128,
                    max_storage_textures_per_shader_stage: required_storage_textures,
                    ..Limits::default()
                },
                experimental_features: ExperimentalFeatures::disabled(),
                memory_hints: MemoryHints::default(),
                trace: Self::trace_mode(),
            })
            .await
            .context(RequestDeviceErr)?;

        Ok((device, queue))
    }

    fn preferred_surface_format(formats: &[TextureFormat]) -> Result<TextureFormat> {
        ensure!(
            formats.contains(&TextureFormat::Bgra8UnormSrgb),
            ColorFormatNotAvailableErr {
                formats: formats.to_vec()
            }
        );

        Ok(TextureFormat::Bgra8UnormSrgb)
    }

    fn clamp_size(size: PhysicalSize<u32>) -> PhysicalSize<u32> {
        PhysicalSize {
            width: size.width.max(1),
            height: size.height.max(1),
        }
    }

    pub fn surface_config<'a>(
        &self,
        surface: &Surface<'a>,
        size: PhysicalSize<u32>,
    ) -> Result<SurfaceConfiguration> {
        Self::_surface_config(&self.adapter, surface, size)
    }

    fn _surface_config<'a>(
        adapter: &Adapter,
        surface: &Surface<'a>,
        size: PhysicalSize<u32>,
    ) -> Result<SurfaceConfiguration> {
        let caps = surface.get_capabilities(adapter);
        let format = Self::preferred_surface_format(&caps.formats)?;
        let size = Self::clamp_size(size);

        let max_frame_latency = EngineArgs::get().max_frames_in_flight.unwrap_or(1);

        Ok(SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format,
            width: size.width,
            height: size.height,
            present_mode: caps
                .present_modes
                .first()
                .copied()
                .unwrap_or(wgpu::PresentMode::Fifo),
            alpha_mode: caps
                .alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: vec![],
            desired_maximum_frame_latency: max_frame_latency,
            color_space: wgpu::SurfaceColorSpace::Srgb,
        })
    }

    pub fn create_surface(&self, window: &Window) -> Result<Surface<'static>> {
        let surface = self
            .instance
            .create_surface(window)
            .context(CreateSurfaceErr)?;
        // SAFETY: The surface holds a boxed window handle, so extending the lifetime is safe as
        // long as the caller owns the window.
        Ok(unsafe { mem::transmute::<Surface<'_>, Surface<'static>>(surface) })
    }

    pub fn new(
        window: &Window,
        display_handle: OwnedDisplayHandle,
    ) -> Result<(Self, Surface<'static>, SurfaceConfiguration)> {
        let backends = EngineArgs::get()
            .force_backend
            .as_ref()
            .and_then(|o| o.as_deref())
            .unwrap_or(DEFAULT_BACKENDS);

        trace!("Starting with backends: {:?}", backends);

        let (instance, surface) = Self::try_setup_instance_with(window, display_handle, backends)?;
        // SAFETY: The surface stores the window handle internally and the caller owns the window.
        let surface = unsafe { mem::transmute::<Surface<'_>, Surface<'static>>(surface) };
        let adapter = block_on(Self::setup_adapter(&instance, Some(&surface)));
        let (device, queue) = block_on(Self::get_device_and_queue(&adapter))?;
        let size = Self::clamp_size(window.inner_size());
        let config = Self::_surface_config(&adapter, &surface, size)?;

        Ok((
            State {
                instance,
                adapter,
                device,
                queue,
            },
            surface,
            config,
        ))
    }
}
