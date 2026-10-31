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
    /// Generated TileLang kernels on the CPU, with native CPU scene storage.
    #[cfg(all(feature = "tilelang", feature = "dim3", not(target_arch = "wasm32")))]
    TileLangC,
    /// Generated TileLang kernels through the CUDA driver and NVCC.
    #[cfg(all(feature = "tilelang", feature = "dim3", not(target_arch = "wasm32")))]
    TileLangCuda,
    /// Generated TileLang kernels through AscendCL and Bisheng (950 SIMT profile).
    #[cfg(all(feature = "tilelang", feature = "dim3", not(target_arch = "wasm32")))]
    TileLangAscend,
    /// Generated TileLang kernels on Metal, with native CPU scene storage.
    #[cfg(all(
        feature = "tilelang",
        feature = "dim3",
        feature = "metal",
        not(target_arch = "wasm32")
    ))]
    TileLangMetal,
}

impl BackendType {
    /// Whether the viewer, rather than `NexusPipeline`, advances the rigid solver.
    pub fn is_tilelang(self) -> bool {
        match self {
            #[cfg(all(feature = "tilelang", feature = "dim3", not(target_arch = "wasm32")))]
            Self::TileLangC | Self::TileLangCuda | Self::TileLangAscend => true,
            #[cfg(all(
                feature = "tilelang",
                feature = "dim3",
                feature = "metal",
                not(target_arch = "wasm32")
            ))]
            Self::TileLangMetal => true,
            _ => false,
        }
    }
}
