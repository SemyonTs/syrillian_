use crate::rendering::state::StateError;
use snafu::Snafu;

pub type Result<T, E = RenderError> = std::result::Result<T, E>;

/// Failing outcomes of acquiring the next swapchain image.
/// wgpu 29's `CurrentSurfaceTexture` also carries the non-error
#[derive(Debug)]
pub enum SurfaceError {
    /// The swapchain is out of date and must be reconfigured.
    Outdated,
    /// The swapchain was lost (GPU reset, window moved off-screen, ...).
    Lost,
    /// Timed out while waiting for the next frame.
    Timeout,
    /// The window is not visible; no frame can be presented.
    Occluded,
    /// Validation error reported by the surface.
    Validation,
}

impl std::fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SurfaceError::Outdated => write!(f, "surface is outdated"),
            SurfaceError::Lost => write!(f, "surface was lost"),
            SurfaceError::Timeout => write!(f, "surface acquisition timed out"),
            SurfaceError::Occluded => write!(f, "surface is occluded"),
            SurfaceError::Validation => write!(f, "surface validation error"),
        }
    }
}

impl std::error::Error for SurfaceError {}

#[derive(Debug, Snafu)]
#[snafu(context(suffix(Err)), visibility(pub(crate)))]
pub enum RenderError {
    #[snafu(display("Render data should've been set"))]
    DataNotSet,

    #[snafu(display("Render pipeline is not set"))]
    NoRenderPipeline,

    #[snafu(display("Invalid Shader requested"))]
    InvalidShader,

    #[snafu(display("No camera set for rendering"))]
    NoCameraSet,

    #[snafu(display("Rendering camera doesn't have a camera component"))]
    NoCameraComponentSet,

    #[snafu(display("Light UBGL was not created"))]
    NoLightUBGL,

    #[snafu(display("Error with current render surface: {source}"))]
    Surface { source: SurfaceError },

    #[snafu(display("Failed to create render state: {source}"))]
    State { source: StateError },
}
