use super::arrf64_basic::*;
use std::ops::{Index, IndexMut};
use std::iter::IntoIterator;
use std::slice::{Iter, IterMut};
use num_complex::Complex;
use fan455_math_scalar::{General, Numeric, Float};
use fan455_util::{NpyObject, NpyTrait, NpyDescrGetter, NpyVecLenGetter};


#[derive(Default, Debug)]
pub struct Arr2<T: General>
{
    pub dim0: usize,
    pub dim1: usize,
    pub data: Vec<T>,
}

impl<T: General> Arr2<T>
{
    #[inline]
    pub fn new( m: usize, n: usize ) -> Self {
        let data: Vec<T> = vec![T::default(); m*n];
        Self { dim0: n, dim1: m, data }
    }

    #[inline]
    pub fn new_empty() -> Self {
        Self { dim0: 0, dim1: 0, data: Vec::<T>::new() }
    }

    #[inline]
    pub fn new_set( m: usize, n: usize, val: T ) -> Self {
        let data: Vec<T> = vec![val; m*n];
        Self { dim0: n, dim1: m, data }
    }

    #[inline]
    pub fn new_copy<MT: GMat<T>>( x: &MT ) -> Self {
        let dim0 = x.ncol();
        let dim1 = x.nrow();
        let mut data: Vec<T> = Vec::with_capacity(x.size());
        data.extend_from_slice(x.sl());
        Self { dim0, dim1, data }
    }

    #[inline]
    pub fn from_vec( x: Vec<T>, m: usize, n: usize ) -> Self {
        Self { dim0: n, dim1: m, data: x }
    }

    #[inline]
    pub fn reshape( &mut self, m: usize, n: usize ) {
        self.dim0 = n;
        self.dim1 = m;
    }

    #[inline]
    pub fn clear( &mut self ) {
        self.data.resize(0, T::default());
        self.data.shrink_to_fit();
    }

    #[inline]
    pub fn resize( &mut self, m: usize, n: usize, val: T ) {
        self.reshape(m, n);
        self.data.resize(m*n, val);
    }

    #[inline]
    pub fn truncate( &mut self, m: usize, n: usize ) {
        self.reshape(m, n);
        self.data.truncate(m*n);
        self.data.shrink_to_fit();
    }
}


impl<T: General + NpyVecLenGetter> Arr2<T>
{
    #[inline]
    pub fn read_npy_tm( path: &String ) -> Self {
        let mut obj = NpyObject::<T>::new_reader(path);
        obj.read_header().unwrap();
        let data = unsafe { obj.read_tm() };
        let (dim0, dim1) = match obj.fortran_order {
            false => (obj.shape[0], obj.shape[1]),
            true => (obj.shape[1], obj.shape[0]),
        };
        Self { dim0, dim1, data }
    }
}



impl<T: General + NpyDescrGetter> Arr2<T>
{
    #[inline]
    pub fn write_npy_tm( &self, path: &String ) {
        let mut obj = NpyObject::<T>::new_writer(path, [1, 0], true, vec![self.nrow(), self.ncol()]);
        obj.write_header().unwrap();
        unsafe { obj.write_tm(&self.data); }
    }
}


impl Arr2<f64>
{
    #[inline]
    pub fn read_npy( path: &String ) -> Self {
        let mut obj = NpyObject::<f64>::new_reader(path);
        obj.read_header().unwrap();
        let data = obj.read();
        let (dim0, dim1) = match obj.fortran_order {
            false => (obj.shape[0], obj.shape[1]),
            true => (obj.shape[1], obj.shape[0]),
        };
        Self { dim0, dim1, data }
    }

    #[inline]
    pub fn write_npy( &self, path: &String ) {
        let mut obj = NpyObject::<f64>::new_writer(path, [1, 0], true, vec![self.nrow(), self.ncol()]);
        obj.write_header().unwrap();
        obj.write(&self.data);
    }
}


impl Arr2<f32>
{
    #[inline]
    pub fn read_npy( path: &String ) -> Self {
        let mut obj = NpyObject::<f32>::new_reader(path);
        obj.read_header().unwrap();
        let data = obj.read();
        let (dim0, dim1) = match obj.fortran_order {
            false => (obj.shape[0], obj.shape[1]),
            true => (obj.shape[1], obj.shape[0]),
        };
        Self { dim0, dim1, data }
    }

    #[inline]
    pub fn write_npy( &self, path: &String ) {
        let mut obj = NpyObject::<f32>::new_writer(path, [1, 0], true, vec![self.nrow(), self.ncol()]);
        obj.write_header().unwrap();
        obj.write(&self.data);
    }
}


impl Arr2<usize>
{
    #[inline]
    pub fn read_npy( path: &String ) -> Self {
        let mut obj = NpyObject::<usize>::new_reader(path);
        obj.read_header().unwrap();
        let data = obj.read();
        let (dim0, dim1) = match obj.fortran_order {
            false => (obj.shape[0], obj.shape[1]),
            true => (obj.shape[1], obj.shape[0]),
        };
        Self { dim0, dim1, data }
    }

    #[inline]
    pub fn write_npy( &self, path: &String ) {
        let mut obj = NpyObject::<usize>::new_writer(path, [1, 0], true, vec![self.nrow(), self.ncol()]);
        obj.write_header().unwrap();
        obj.write(&self.data);
    }
}


impl<T: General> IntoIterator for Arr2<T>
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T: General> IntoIterator for &'a Arr2<T>
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    
    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        (&self.data).into_iter()
    }
}

impl<'a, T: General> IntoIterator for &'a mut Arr2<T>
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    
    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        (&mut self.data).into_iter()
    }
}

impl<T: General> GMatAlloc<T> for Arr2<T>
{
    #[inline]
    fn alloc( m: usize, n: usize ) -> Self {
        Self::new(m, n)
    }

    #[inline]
    fn alloc_set( m: usize,n: usize, val: T ) -> Self {
        Self::new_set(m, n, val)
    }

    #[inline]
    fn alloc_copy<VT: GMat<T>>( x: &VT ) -> Self {
        Self::new_copy(x)
    }
}

impl<T: General> GVec<T> for Arr2<T>
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

impl<T: General> GVecMut<T> for Arr2<T>
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

impl<T: General> GMat<T> for Arr2<T>
{
    #[inline]
    fn nrow( &self ) -> usize {
        self.dim1
    }

    #[inline]
    fn ncol( &self ) -> usize {
        self.dim0
    }

    #[inline]
    fn stride( &self ) -> usize {
        self.dim1
    }

    #[inline]
    fn idx2( &self, i: usize, j: usize ) -> &T {
        self.data.index(i+j*self.dim1)
    }

    #[inline]
    fn idx2_unchecked( &self, i: usize, j: usize ) -> &T {
        unsafe { self.data.get_unchecked(i+j*self.dim1) }
    }
}

impl<T: General> GMatMut<T> for Arr2<T>
{
    #[inline]
    fn idxm2( &mut self, i: usize, j: usize ) -> &mut T {
        self.data.index_mut(i+j*self.dim1)
    }

    #[inline]
    fn idxm2_unchecked( &mut self, i: usize, j: usize ) -> &mut T {
        unsafe { self.data.get_unchecked_mut(i+j*self.dim1) }
    }
}

impl<T: Numeric> NVec<T> for Arr2<T> {}
impl<T: Numeric> NVecMut<T> for Arr2<T> {}
impl<T: Numeric> NMat<T> for Arr2<T> {}
impl<T: Numeric> NMatMut<T> for Arr2<T> {}

impl<T: Float> RVec<T> for Arr2<T> {}
impl<T: Float> RVecMut<T> for Arr2<T> {}
impl<T: Float> RMat<T> for Arr2<T> {}
impl<T: Float> RMatMut<T> for Arr2<T> {}

impl<T: Float> CVec<T> for Arr2<Complex<T>> {}
impl<T: Float> CVecMut<T> for Arr2<Complex<T>> {}
impl<T: Float> CMat<T> for Arr2<Complex<T>> {}
impl<T: Float> CMatMut<T> for Arr2<Complex<T>> {}


impl<T: General> Index<(usize, usize)> for Arr2<T>
{
    type Output = T;

    #[inline]
    fn index( &self, index: (usize, usize) ) -> &T {
        self.data.index(index.0+index.1*self.dim1)
    }
}

impl<T: General> IndexMut<(usize, usize)> for Arr2<T>
{
    #[inline]
    fn index_mut( &mut self, index: (usize, usize) ) -> &mut T {
        self.data.index_mut(index.0+index.1*self.dim1)
    }
}

impl<T: General> Clone for Arr2<T>
{
    fn clone(&self) -> Self {
        Self { dim0: self.dim0, dim1: self.dim1, data: self.data.clone() }
    }

    fn clone_from(&mut self, source: &Self) {
        self.data.clone_from(&source.data);
    }
}
