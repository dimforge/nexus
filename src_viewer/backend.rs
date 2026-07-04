#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackendType {
    /// GPU-accelerated physics using nexus + WebGPU.
    Gpu,
    /// CPU physics using nexus (same pipeline as GPU, executed on CPU).
    Cpu,
    /// GPU-accelerated physics using nexus + CUDA.
    #[cfg(feature = "cuda")]
    Cuda,
    /// GPU-accelerated physics using nexus + native Metal (macOS only).
    #[cfg(feature = "metal")]
    Metal,
}
