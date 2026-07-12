// SPDX-License-Identifier: MIT OR Apache-2.0

//! Session-accounted batch result ownership.

use core::iter::FusedIterator;
use core::ops::{Deref, DerefMut};

#[cfg(feature = "cuda-runtime")]
use crate::{allocation::HostPhaseBudget, session::HostOwnerLease};
use crate::{CudaSession, Error};

/// Ordered CUDA JPEG batch results whose vector allocation remains charged to
/// the caller-owned session for the lifetime of the collection.
#[doc(hidden)]
#[derive(Debug)]
#[must_use]
pub struct CudaJpegBatch<T> {
    items: Vec<T>,
    what: &'static str,
    #[cfg(feature = "cuda-runtime")]
    _lease: HostOwnerLease,
}

impl<T> CudaJpegBatch<T> {
    pub(crate) fn try_with_capacity(
        session: &CudaSession,
        capacity: usize,
        what: &'static str,
    ) -> Result<Self, Error> {
        #[cfg(feature = "cuda-runtime")]
        {
            let (items, lease) = session.allocate_owned_host_owner(|external_live_bytes| {
                let mut budget = HostPhaseBudget::new(what);
                budget.account_bytes(external_live_bytes)?;
                let items = budget.try_vec_with_capacity(capacity)?;
                let actual_bytes = j2k_core::host_capacity_bytes::<T>(items.capacity());
                Ok((items, actual_bytes))
            })?;
            Ok(Self {
                items,
                what,
                _lease: lease,
            })
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            let _ = session;
            Ok(Self {
                items: crate::allocation::try_vec_with_capacity(capacity, what)?,
                what,
            })
        }
    }

    pub(crate) fn try_push(&mut self, item: T) -> Result<(), Error> {
        if self.items.len() >= self.items.capacity() {
            return Err(Error::BatchCapacityExceeded {
                capacity: self.items.capacity(),
                what: self.what,
            });
        }
        self.items.push(item);
        Ok(())
    }

    /// Number of completed results in input order.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the batch has no results.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Borrow the completed results in input order.
    pub fn as_slice(&self) -> &[T] {
        &self.items
    }
}

impl<T> AsRef<[T]> for CudaJpegBatch<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> Deref for CudaJpegBatch<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for CudaJpegBatch<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.items
    }
}

/// Owning iterator that keeps the batch-vector lease alive until iteration is
/// complete or the iterator is dropped.
#[doc(hidden)]
#[derive(Debug)]
pub struct CudaJpegBatchIntoIter<T> {
    inner: std::vec::IntoIter<T>,
    #[cfg(feature = "cuda-runtime")]
    _lease: HostOwnerLease,
}

impl<T> Iterator for CudaJpegBatchIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> DoubleEndedIterator for CudaJpegBatchIntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl<T> ExactSizeIterator for CudaJpegBatchIntoIter<T> {}
impl<T> FusedIterator for CudaJpegBatchIntoIter<T> {}

impl<T> IntoIterator for CudaJpegBatch<T> {
    type Item = T;
    type IntoIter = CudaJpegBatchIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let CudaJpegBatch {
            items,
            what: _,
            #[cfg(feature = "cuda-runtime")]
                _lease: lease,
        } = self;
        CudaJpegBatchIntoIter {
            inner: items.into_iter(),
            #[cfg(feature = "cuda-runtime")]
            _lease: lease,
        }
    }
}

impl<'a, T> IntoIterator for &'a CudaJpegBatch<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut CudaJpegBatch<T> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter_mut()
    }
}

#[cfg(test)]
mod tests;
