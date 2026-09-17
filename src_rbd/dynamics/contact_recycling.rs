//! Double-buffered recycling metadata with one storage binding.
use crate::shaders::dynamics::{ContactRecycleOffsets, ContactRecycleState};
use khal::BufferUsages;
use khal::backend::{GpuBackend, GpuBackendError};
use vortx::tensor::Tensor;

/// Previous and current contact metadata in disjoint halves of one allocation.
/// Constraint layouts are unchanged. Swapping frames only selects an immutable
/// offset uniform, so captured GPU work never sees a rewritten offset buffer.
pub struct ContactRecycleStates {
    storage: Tensor<ContactRecycleState>,
    offsets: [Tensor<ContactRecycleOffsets>; 2],
    phase: usize,
}

impl ContactRecycleStates {
    /// Allocate equally sized previous and current metadata regions.
    pub fn new(
        backend: &GpuBackend,
        capacity: u32,
        usage: BufferUsages,
    ) -> Result<Self, GpuBackendError> {
        Ok(Self {
            storage: Tensor::vector_uninit(
                backend,
                capacity
                    .checked_mul(2)
                    .expect("recycling capacity overflow"),
                usage,
            )?,
            offsets: [
                Tensor::scalar(
                    backend,
                    ContactRecycleOffsets {
                        old_base: 0,
                        new_base: capacity,
                        ..Default::default()
                    },
                    BufferUsages::UNIFORM,
                )?,
                Tensor::scalar(
                    backend,
                    ContactRecycleOffsets {
                        old_base: capacity,
                        new_base: 0,
                        ..Default::default()
                    },
                    BufferUsages::UNIFORM,
                )?,
            ],
            phase: 0,
        })
    }

    pub(crate) fn bindings(
        &mut self,
    ) -> (
        &mut Tensor<ContactRecycleState>,
        &Tensor<ContactRecycleOffsets>,
    ) {
        (&mut self.storage, &self.offsets[self.phase])
    }

    /// Advance alongside the old/new constraint-buffer swap after each step.
    pub fn swap_frames(&mut self) {
        self.phase ^= 1;
    }
}
