use std::ops::{Index, IndexMut};
use std::iter::IntoIterator;
use std::slice::{Iter, IterMut};
use fan455_math_scalar::General;
use fan455_util::{NpyObject, NpyDescrGetter, NpyVecLenGetter};


#[derive(Debug)]
pub struct PackMat<T: General>  
// Packed matrix storage for upper/lower triangular matrix or symmetric matrix.
// If it is lower matrix, use functions with "_lo" suffix. If it is upper matrix, use functions with "_up" suffix.
// Indexing the upper part of a lower matrix, or indexing the lower part of a upper matrix, is not supported.
{
    pub dim: usize,
    pub data: Vec<T>, // size (n*(n+1))/2
}


impl<T: General> PackMat<T>
{
    #[inline]
    pub fn new( n: usize ) -> Self {
        let data: Vec<T> = vec![T::default(); (n*(n+1))/2];
        Self { dim: n, data }
    }

    #[inline]
    pub fn new_set( n: usize, val: T ) -> Self {
        let data: Vec<T> = vec![val; (n*(n+1))/2];
        Self { dim: n, data }
    }

    #[inline]
    pub fn new_copy( x: &Self ) -> Self {
        let mut data: Vec<T> = Vec::with_capacity(x.data.len());
        data.extend_from_slice(x.data.as_slice());
        Self { dim: x.dim, data }
    }

    #[inline]
    pub fn size ( &self ) -> usize {
        self.dim * self.dim
    }

    #[inline]
    pub fn storage_size ( &self ) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn ptr( &self ) -> *const T {
        self.data.as_ptr()
    }

    #[inline]
    pub fn sl( &self ) -> &[T] {
        self.data.as_slice()
    }

    #[inline]
    pub fn idx( &self, i: usize ) -> &T {
        self.data.index(i)
    }

    #[inline]
    pub fn ptrm( &mut self ) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline]
    pub fn slm( &mut self ) -> &mut [T] {
        self.data.as_mut_slice()
    }

    #[inline]
    pub fn idxm( &mut self, i: usize ) -> &mut T {
        self.data.index_mut(i)
    }

    #[inline]
    pub fn nrow( &self ) -> usize {
        self.dim
    }

    #[inline]
    pub fn ncol( &self ) -> usize {
        self.dim
    }

    #[inline]
    pub fn idx2_lo( &self, i: usize, j: usize ) -> &T {
        self.data.index(i+(2*self.dim-j-1)*j/2)
    }

    #[inline]
    pub fn idx2_up( &self, i: usize, j: usize ) -> &T {
        self.data.index(i+j*(j+1)/2)
    }

    #[inline]
    pub fn idxm2_lo( &mut self, i: usize, j: usize ) -> &mut T {
        self.data.index_mut(i+(2*self.dim-j-1)*j/2)
    }

    #[inline]
    pub fn idxm2_up( &mut self, i: usize, j: usize ) -> &mut T {
        self.data.index_mut(i+j*(j+1)/2)
    }
}


impl<T: General + NpyVecLenGetter> PackMat<T>
{
    #[inline]
    pub fn read_npy_tm( path: &str ) -> Self {
        let mut obj = NpyObject::<T>::new_reader(path);
        obj.read_header().unwrap();
        let data = unsafe {obj.read_tm()};
        let s = data.len();
        let n = (0.5 * ((8.* s as f64 + 1.).sqrt() - 1.)) as usize;
        assert_eq!((n*(n+1))/2, s, "The size of Hermitian matrix is incorrect.");
        Self { dim: n, data }
    }
}


impl<T: General + NpyDescrGetter> PackMat<T>
{
    #[inline]
    pub fn write_npy_tm( &self, path: &str ) {
        let mut obj = NpyObject::<T>::new_writer(path, [1,0], true, vec![self.data.len()]);
        obj.write_header().unwrap();
        unsafe {obj.write_tm(&self.data);}
    }
}



impl<T: General> IntoIterator for PackMat<T>
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T: General> IntoIterator for &'a PackMat<T>
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    
    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        (&self.data).into_iter()
    }
}

impl<'a, T: General> IntoIterator for &'a mut PackMat<T>
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    
    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        (&mut self.data).into_iter()
    }
}


impl<T: General> Clone for PackMat<T>
{
    fn clone(&self) -> Self {
        Self { dim: self.dim, data: self.data.clone() }
    }

    fn clone_from(&mut self, source: &Self) {
        self.data.clone_from(&source.data);
    }
}
