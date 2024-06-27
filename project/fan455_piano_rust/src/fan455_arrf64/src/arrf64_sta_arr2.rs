use super::arrf64_basic::*;
use std::ops::{Index, IndexMut};
use std::iter::IntoIterator;
use std::slice::{Iter, IterMut};
use num_complex::Complex;
use fan455_math_scalar::{General, Numeric, Float};


#[derive(Copy, Clone)]
pub struct SArr2<T: General, const M: usize, const N: usize, const S: usize>
{
    // Static M*N matrix, column major.
    pub data: [T; S],
}

impl<T: General, const M: usize, const N: usize, const S: usize> SArr2<T, M, N, S>
{
    #[inline]
    pub fn new() -> Self {
        Self { data: [T::default(); S] }
    }

    #[inline]
    pub fn new_set( val: T ) -> Self {
        Self { data: [val; S] }
    }

    #[inline]
    pub fn new_copy( x: &Self ) -> Self {
        Self { data: x.data }
    }

    #[inline]
    pub fn from_array( x: [T; S] ) -> Self {
        Self { data: x }
    }

    #[inline]
    pub fn copy<VT: GVec<T>>( &mut self, x: &VT ) {
        self.data.copy_from_slice(x.sl());
    }
}

impl<T: General, const M: usize, const N: usize, const S: usize> IntoIterator for SArr2<T, M, N, S>
{
    type Item = T;
    type IntoIter = std::array::IntoIter<T, S>;

    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T: General, const M: usize, const N: usize, const S: usize> IntoIterator for &'a SArr2<T, M, N, S>
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    
    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        (&self.data).into_iter()
    }
}

impl<'a, T: General, const M: usize, const N: usize, const S: usize> IntoIterator for &'a mut SArr2<T, M, N, S>
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    
    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        (&mut self.data).into_iter()
    }
}

impl<T: General, const M: usize, const N: usize, const S: usize> GMatAlloc<T> for SArr2<T, M, N, S>
{
    #[inline]
    fn alloc( _m: usize, _n: usize ) -> Self {
        Self::new()
    }

    #[inline]
    fn alloc_set( _m: usize, _n: usize, val: T ) -> Self {
        Self::new_set(val)
    }

    #[inline]
    fn alloc_copy<VT: GVec<T>>( x: &VT ) -> Self {
        Self { data: <[T; S]>::try_from(x.sl()).unwrap() }
    }
}

impl<T: General, const M: usize, const N: usize, const S: usize> GVec<T> for SArr2<T, M, N, S>
{
    #[inline]
    fn size ( &self ) -> usize {
        S
    }

    #[inline]
    fn ptr( &self ) -> *const T {
        self.data.as_ptr()
    }

    #[inline]
    fn sl( &self ) -> &[T] {
        self.data.as_slice()
    }

    #[inline]
    fn idx( &self, i: usize ) -> &T {
        self.data.index(i)
    }
}

impl<T: General, const M: usize, const N: usize, const S: usize> GVecMut<T> for SArr2<T, M, N, S>
{
    #[inline]
    fn ptrm( &mut self ) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline]
    fn slm( &mut self ) -> &mut [T] {
        self.data.as_mut_slice()
    }

    #[inline]
    fn idxm( &mut self, i: usize ) -> &mut T {
        self.data.index_mut(i)
    }
}

impl<T: General, const M: usize, const N: usize, const S: usize> GMat<T> for SArr2<T, M, N, S>
{
    #[inline]
    fn nrow( &self ) -> usize {
        M
    }

    #[inline]
    fn ncol( &self ) -> usize {
        N
    }

    #[inline]
    fn stride( &self ) -> usize {
        M
    }

    #[inline]
    fn idx2( &self, i: usize, j: usize ) -> &T {
        self.data.index(i+j*M)
    }
}

impl<T: General, const M: usize, const N: usize, const S: usize> GMatMut<T> for SArr2<T, M, N, S>
{
    #[inline]
    fn idxm2( &mut self, i: usize, j: usize ) -> &mut T {
        self.data.index_mut(i+j*M)
    }
}

impl<T: Numeric, const M: usize, const N: usize, const S: usize> NVec<T> for SArr2<T, M, N, S> {}
impl<T: Numeric, const M: usize, const N: usize, const S: usize> NVecMut<T> for SArr2<T, M, N, S> {}
impl<T: Numeric, const M: usize, const N: usize, const S: usize> NMat<T> for SArr2<T, M, N, S> {}
impl<T: Numeric, const M: usize, const N: usize, const S: usize> NMatMut<T> for SArr2<T, M, N, S> {}

impl<T: Float, const M: usize, const N: usize, const S: usize> RVec<T> for SArr2<T, M, N, S> {}
impl<T: Float, const M: usize, const N: usize, const S: usize> RVecMut<T> for SArr2<T, M, N, S> {}
impl<T: Float, const M: usize, const N: usize, const S: usize> RMat<T> for SArr2<T, M, N, S> {}
impl<T: Float, const M: usize, const N: usize, const S: usize> RMatMut<T> for SArr2<T, M, N, S> {}

impl<T: Float, const M: usize, const N: usize, const S: usize> CVec<T> for SArr2<Complex<T>, M, N, S> {}
impl<T: Float, const M: usize, const N: usize, const S: usize> CVecMut<T> for SArr2<Complex<T>, M, N, S> {}
impl<T: Float, const M: usize, const N: usize, const S: usize> CMat<T> for SArr2<Complex<T>, M, N, S> {}
impl<T: Float, const M: usize, const N: usize, const S: usize> CMatMut<T> for SArr2<Complex<T>, M, N, S> {}


impl<T: General, const M: usize, const N: usize, const S: usize> Index<(usize, usize)> for SArr2<T, M, N, S>
{
    type Output = T;

    #[inline]
    fn index( &self, index: (usize, usize) ) -> &T {
        self.data.index(index.0+index.1*M)
    }
}

impl<T: General, const M: usize, const N: usize, const S: usize> IndexMut<(usize, usize)> for SArr2<T, M, N, S>
{
    #[inline]
    fn index_mut( &mut self, index: (usize, usize) ) -> &mut T {
        self.data.index_mut(index.0+index.1*M)
    }
}
