use fan455_math_scalar::Numeric;
use fan455_util::{elem, mzip};
use std::ops::{Index, IndexMut};
use std::iter::Zip;
use std::slice::{Iter, IterMut};

use crate::GVecMut;


#[derive(Default, Debug)]
pub struct CsrMat<T: Numeric> {
    // Sparse matrix in CSR format, row-major, zero-based indexing.
    pub nrow: usize,
    pub ncol: usize,
    pub nnz: usize, // number of non-zero elements.
    pub data: Vec<T>, // (nnz,)
    pub row_pos: Vec<usize>, // (nrow+1,)
    pub row_idx: Vec<usize>, // (nnz,)
    pub col_idx: Vec<usize>, // (nnz,)
}


impl<T: Numeric> CsrMat<T>
{
    #[inline]
    pub fn new_alloc( nrow: usize, ncol: usize, nnz: usize ) -> Self {
        let data: Vec<T> = vec![T::zero(); nnz];
        let row_pos: Vec<usize> = vec![0; nrow+1];
        let col_idx: Vec<usize> = vec![0; nnz];
        let row_idx: Vec<usize> = vec![0; nnz];
        Self { nrow, ncol, nnz, data, row_pos, row_idx, col_idx }
    }

    #[inline]
    pub fn new_empty() -> Self {
        Self { ..Default::default() }
    }

    #[inline]
    pub fn idx( &self, i: usize ) -> &T {
        let i_row = i / self.ncol;
        let i_col = i % self.ncol;
        let beg = self.row_pos[i_row];
        let end = self.row_pos[i_row+1];
        let mut i_nz: usize = usize::MAX;
        for elem!(i_, curr_col) in mzip!(beg..end, self.col_idx[beg..end].iter()) {
            if i_col == *curr_col {
                i_nz = i_;
                break;
            }
        }
        self.data.index(i_nz)
    }

    #[inline]
    pub fn idxm( &mut self, i: usize ) -> &mut T {
        let i_row = i / self.ncol;
        let i_col = i % self.ncol;
        let beg = self.row_pos[i_row];
        let end = self.row_pos[i_row+1];
        let mut i_nz: usize = usize::MAX;
        for elem!(i_, curr_col) in mzip!(beg..end, self.col_idx[beg..end].iter()) {
            if i_col == *curr_col {
                i_nz = i_;
                break;
            }
        }
        self.data.index_mut(i_nz)
    }

    #[inline]
    pub fn idx2( &self, i: usize, j: usize ) -> &T {
        let beg = self.row_pos[i];
        let end = self.row_pos[i+1];
        let mut i_nz: usize = usize::MAX;
        for elem!(i_, curr_col) in mzip!(beg..end, self.col_idx[beg..end].iter()) {
            if j == *curr_col {
                i_nz = i_;
                break;
            }
        }
        self.data.index(i_nz)
    }

    #[inline]
    pub fn idxm2( &mut self, i: usize, j: usize ) -> &mut T {
        let beg = self.row_pos[i];
        let end = self.row_pos[i+1];
        let mut i_nz: usize = usize::MAX;
        for elem!(i_, curr_col) in mzip!(beg..end, self.col_idx[beg..end].iter()) {
            if j == *curr_col {
                i_nz = i_;
                break;
            }
        }
        self.data.index_mut(i_nz)
    }

    #[inline]
    pub fn it( &self ) -> Zip<Iter<'_, T>, Zip<Iter<'_, usize>, Iter<'_, usize>>> {
        mzip!(
            self.data.iter(),
            self.row_idx.iter(),
            self.col_idx.iter()
        )
    }

    #[inline]
    pub fn itm( &mut self ) -> Zip<IterMut<'_, T>, Zip<Iter<'_, usize>, Iter<'_, usize>>> {
        mzip!(
            self.data.iter_mut(),
            self.row_idx.iter(),
            self.col_idx.iter()
        )
    }

    #[inline]
    pub fn change_to_one_based_index( &mut self ) {
        // This is only intended for calling MKL, not working with the struct's methods!
        for s in self.row_pos.itm() {
            *s += 1;
        }
        for elem!(s1, s2) in mzip!(self.row_idx.itm(), self.col_idx.itm()) {
            *s1 += 1;
            *s2 += 1;
        }
    }

    #[inline]
    pub fn change_to_zero_based_index( &mut self ) {
        for s in self.row_pos.itm() {
            *s -= 1;
        }
        for elem!(s1, s2) in mzip!(self.row_idx.itm(), self.col_idx.itm()) {
            *s1 -= 1;
            *s2 -= 1;
        }
    }
}


