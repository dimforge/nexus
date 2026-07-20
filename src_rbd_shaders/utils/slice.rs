use core::ops::{Index, IndexMut};
use khal_std::index::MaybeIndexUnchecked;

// Actual rust slices &array[a..b] don’t compile with rust-gpu, so we
// simulated them manually with indices.
pub struct Slice<'a, T>(pub &'a [T], pub usize);

impl<'a, T: Copy> Slice<'a, T> {
    #[inline]
    pub fn at(&self, i: usize) -> &'a T {
        self.0.at(self.1 + i)
    }

    #[inline]
    pub fn read(&self, i: usize) -> T {
        self.0.read(self.1 + i)
    }

    #[inline]
    pub fn offset(self, offset: usize) -> Self {
        Slice(self.0, self.1 + offset)
    }
}

// Actual rust slices &mut array[a..b] don’t compile with rust-gpu, so we
// simulated them manually with indices.
pub struct SliceMut<'a, T>(pub &'a mut [T], pub usize);

impl<'a, T: Copy> SliceMut<'a, T> {
    #[inline]
    pub fn as_ref(&self) -> Slice<'_, T> {
        Slice(&*self.0, self.1)
    }

    #[inline]
    pub fn at(&self, i: usize) -> &T {
        self.0.at(self.1 + i)
    }

    #[inline]
    pub fn read(&self, i: usize) -> T {
        self.0.read(self.1 + i)
    }

    #[inline]
    pub fn at_mut(&mut self, i: usize) -> &mut T {
        self.0.at_mut(self.1 + i)
    }

    #[inline]
    pub fn write(&mut self, i: usize, value: T) {
        self.0.write(self.1 + i, value)
    }

    #[inline]
    pub fn offset(self, offset: usize) -> Self {
        SliceMut(self.0, self.1 + offset)
    }
}

impl<T: Copy> Index<usize> for Slice<'_, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        self.0.at(self.1 + i)
    }
}

impl<T: Copy> Index<usize> for SliceMut<'_, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        self.0.at(self.1 + i)
    }
}

impl<T: Copy> IndexMut<usize> for SliceMut<'_, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        self.0.at_mut(self.1 + i)
    }
}

// Batch-interleaved variants: element `i` of the view lives at
// `(base + i) · stride + shift` in the backing buffer. Used by the multibody
// dynamics buffers, which are laid out batch-minor (`intra · num_batches +
// batch_id`, Genesis-style) so that the flattened one-thread-per-(multibody,
// batch) kernels access memory coalesced across lanes. `stride = 1, shift =
// 0` degenerates to [`Slice`].
pub struct ISlice<'a, T> {
    pub buf: &'a [T],
    pub base: usize,
    pub stride: u32,
    pub shift: u32,
}

impl<'a, T: Copy> ISlice<'a, T> {
    #[inline(always)]
    fn flat(&self, i: usize) -> usize {
        (self.base + i) * self.stride as usize + self.shift as usize
    }

    #[inline]
    pub fn at(&self, i: usize) -> &'a T {
        self.buf.at(self.flat(i))
    }

    #[inline]
    pub fn read(&self, i: usize) -> T {
        self.buf.read(self.flat(i))
    }

    #[inline]
    pub fn offset(self, offset: usize) -> Self {
        ISlice {
            base: self.base + offset,
            ..self
        }
    }
}

pub struct ISliceMut<'a, T> {
    pub buf: &'a mut [T],
    pub base: usize,
    pub stride: u32,
    pub shift: u32,
}

impl<'a, T: Copy> ISliceMut<'a, T> {
    #[inline(always)]
    fn flat(&self, i: usize) -> usize {
        (self.base + i) * self.stride as usize + self.shift as usize
    }

    #[inline]
    pub fn as_ref(&self) -> ISlice<'_, T> {
        ISlice {
            buf: &*self.buf,
            base: self.base,
            stride: self.stride,
            shift: self.shift,
        }
    }

    #[inline]
    pub fn at(&self, i: usize) -> &T {
        self.buf.at(self.flat(i))
    }

    #[inline]
    pub fn read(&self, i: usize) -> T {
        self.buf.read(self.flat(i))
    }

    #[inline]
    pub fn at_mut(&mut self, i: usize) -> &mut T {
        let idx = self.flat(i);
        self.buf.at_mut(idx)
    }

    #[inline]
    pub fn write(&mut self, i: usize, value: T) {
        let idx = self.flat(i);
        self.buf.write(idx, value)
    }

    #[inline]
    pub fn offset(self, offset: usize) -> Self {
        ISliceMut {
            base: self.base + offset,
            ..self
        }
    }
}

impl<T: Copy> Index<usize> for ISlice<'_, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        self.buf.at(self.flat(i))
    }
}

impl<T: Copy> Index<usize> for ISliceMut<'_, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        self.buf.at(self.flat(i))
    }
}

impl<T: Copy> IndexMut<usize> for ISliceMut<'_, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        let idx = self.flat(i);
        self.buf.at_mut(idx)
    }
}
