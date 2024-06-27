use super::arrf64_basic::*;
use std::ops::{Index, IndexMut};
use std::iter::IntoIterator;
use std::slice::{Iter, IterMut};
use num_complex::Complex;
use fan455_math_scalar::{General, Numeric, Float};


#[derive(Copy, Clone)]
pub struct SArr1<T: General, const N: usize>
{
    pub data: [T; N],
}

impl<T: General, const N: usize> SArr1<T, N>
{
    #[inline]
    pub fn new() -> Self {
        Self { data: [T::default(); N] }
    }

    #[inline]
    pub fn new_set( val: T ) -> Self {
        Self { data: [val; N] }
    }

    #[inline]
    pub fn new_copy( x: &Self ) -> Self {
        Self { data: x.data }
    }

    #[inline]
    pub fn from_array( x: [T; N] ) -> Self {
        Self { data: x }
    }

    #[inline]
    pub fn copy<VT: GVec<T>>( &mut self, x: &VT ) {
        self.data.copy_from_slice( x.sl() );
    }

    #[inline]
    pub fn idx( &self, i: usize ) -> &T {
        self.data.index(i)
    }

    #[inline]
    pub fn idxm( &mut self, i: usize ) -> &mut T {
        self.data.index_mut(i)
    }

    #[inline]
    pub fn get( &self, i: usize ) -> &T {
        self.data.index(i)
    }

    #[inline]
    pub fn getm( &mut self, i: usize ) -> &mut T {
        self.data.index_mut(i)
    }

}


impl<T: General, const N: usize> IntoIterator for SArr1<T, N>
{
    type Item = T;
    type IntoIter = std::array::IntoIter<T, N>;

    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T: General, const N: usize> IntoIterator for &'a SArr1<T, N>
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    
    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        (&self.data).into_iter()
    }
}

impl<'a, T: General, const N: usize> IntoIterator for &'a mut SArr1<T, N>
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    
    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        (&mut self.data).into_iter()
    }
}


impl<T: General, const N: usize> GVecAlloc<T> for SArr1<T, N>
{
    #[inline]
    fn alloc( _n: usize ) -> Self {
        Self::new()
    }

    #[inline]
    fn alloc_set( _n: usize, val: T ) -> Self {
        Self::new_set(val)
    }

    #[inline]
    fn alloc_copy<VT: GVec<T>>( x: &VT ) -> Self {
        Self { data: <[T; N]>::try_from(x.sl()).unwrap() }
    }
}


impl<T: General, const N: usize> GVec<T> for SArr1<T, N>
{
    #[inline]
    fn size ( &self ) -> usize {
        self.data.len()
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

impl<T: General, const N: usize> GVecMut<T> for SArr1<T, N>
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

impl<T: General, const N: usize> GMat<T> for SArr1<T, N>
{
    #[inline]
    fn nrow( &self ) -> usize {
        N
    }

    #[inline]
    fn ncol( &self ) -> usize {
        1
    }

    #[inline]
    fn stride( &self ) -> usize {
        N
    }

    #[inline]
    fn idx2( &self, i: usize, _j: usize ) -> &T {
        self.data.index(i)
    }
}

impl<T: General, const N: usize> GMatMut<T> for SArr1<T, N>
{
    #[inline]
    fn idxm2( &mut self, i: usize, _j: usize ) -> &mut T {
        self.data.index_mut(i)
    }
}

impl<T: Numeric, const N: usize> NVec<T> for SArr1<T, N> {}
impl<T: Numeric, const N: usize> NVecMut<T> for SArr1<T, N> {}
impl<T: Numeric, const N: usize> NMat<T> for SArr1<T, N> {}
impl<T: Numeric, const N: usize> NMatMut<T> for SArr1<T, N> {}

impl<T: Float, const N: usize> RVec<T> for SArr1<T, N> {}
impl<T: Float, const N: usize> RVecMut<T> for SArr1<T, N> {}
impl<T: Float, const N: usize> RMat<T> for SArr1<T, N> {}
impl<T: Float, const N: usize> RMatMut<T> for SArr1<T, N> {}

impl<T: Float, const N: usize> CVec<T> for SArr1<Complex<T>, N> {}
impl<T: Float, const N: usize> CVecMut<T> for SArr1<Complex<T>, N> {}
impl<T: Float, const N: usize> CMat<T> for SArr1<Complex<T>, N> {}
impl<T: Float, const N: usize> CMatMut<T> for SArr1<Complex<T>, N> {}

impl<T: General, const N: usize> Index<usize> for SArr1<T, N>
{
    type Output = T;

    #[inline]
    fn index( &self, index: usize ) -> &T {
        self.data.index(index)
    }
}

impl<T: General, const N: usize> IndexMut<usize> for SArr1<T, N>
{
    #[inline]
    fn index_mut( &mut self, index: usize ) -> &mut T {
        self.data.index_mut(index)
    }
}
