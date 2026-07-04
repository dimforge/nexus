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
