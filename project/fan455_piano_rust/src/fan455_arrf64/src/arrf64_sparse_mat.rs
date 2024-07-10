use fan455_math_scalar::Numeric;
use fan455_util::*;
use std::ops::{Index, IndexMut};
use std::iter::Zip;
use std::slice::{Iter, IterMut};

use crate::NVecMut;


#[derive(Default, Debug, Clone)]
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
where T: Numeric + Default + Clone + NpyVecLenGetter, NpyObject<T>: NpyTrait<T>,
{
    #[inline]
    fn _read_npy(
        row_pos_path: &str,
        row_idx_path: &str,
        col_idx_path: &str,
        data_path: Option<&str>, // If is None, data will be set to default values (zeros).
        ncol_: Option<usize>, // If is None, treat as square matrix.
    ) -> Self {
        let row_pos: Vec<usize> = read_npy_vec::<usize>(row_pos_path);
        let row_idx: Vec<usize> = read_npy_vec::<usize>(row_idx_path);
        let col_idx: Vec<usize> = read_npy_vec::<usize>(col_idx_path);

        let nrow = row_pos.len() - 1;
        let ncol: usize = ncol_.unwrap_or(nrow);

        let nnz = row_idx.len();
        assert_eq!(nnz, col_idx.len(), "The length of row_idx and col_idx may be incorrect.");

        let data: Vec<T>;
        if let Some(path) = data_path {
            data = read_npy_vec(path);
            assert_eq!(nnz, data.len(), "The length of row_idx and col_idx may be incorrect.");
        } else {
            data = vec![T::default(); nnz];
        }
        Self { nrow, ncol, nnz, data, row_pos, row_idx, col_idx }
    }

    #[inline]
    pub fn read_npy_square_default(
        row_pos_path: &str,
        row_idx_path: &str,
        col_idx_path: &str,
    ) -> Self {
        Self::_read_npy(row_pos_path, row_idx_path, col_idx_path, None, None)
    }

    #[inline]
    pub fn read_npy_square(
        row_pos_path: &str,
        row_idx_path: &str,
        col_idx_path: &str,
        data_path: &str,
    ) -> Self {
        Self::_read_npy(row_pos_path, row_idx_path, col_idx_path, Some(data_path), None)
    }
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
    pub fn idx2( &self, i: usize, j: usize ) -> &T {
        let beg = self.row_pos[i];
        let end = self.row_pos[i+1];
        let mut i_nz: usize = usize::MAX;
        'outer: for elem!(i_, curr_col) in mzip!(beg..end, self.col_idx[beg..end].iter()) {
            if j == *curr_col {
                i_nz = i_;
                break 'outer;
            }
        }
        self.data.index(i_nz)
    }

    #[inline]
    pub fn idxm2( &mut self, i: usize, j: usize ) -> &mut T {
        let beg = self.row_pos[i];
        let end = self.row_pos[i+1];
        let mut i_nz: usize = usize::MAX;
        'outer: for elem!(i_, curr_col) in mzip!(beg..end, self.col_idx[beg..end].iter()) {
            if j == *curr_col {
                i_nz = i_;
                break 'outer;
            }
        }
        self.data.index_mut(i_nz)
    }

    #[inline]
    pub fn idxm2_option( &mut self, i: usize, j: usize ) -> Option<&mut T> {
        let beg = self.row_pos[i];
        let end = self.row_pos[i+1];
        let mut i_nz: usize = usize::MAX;
        'outer: for elem!(i_, curr_col) in mzip!(beg..end, self.col_idx[beg..end].iter()) {
            if j == *curr_col {
                i_nz = i_;
                break 'outer;
            }
        }
        self.data.get_mut(i_nz)
    }

    #[inline]
    pub fn it( &self ) -> Zip<Iter<T>, Zip<Iter<usize>, Iter<usize>>> {
        mzip!(
            self.data.iter(),
            self.row_idx.iter(),
            self.col_idx.iter()
        )
    }

    #[inline]
    pub fn itm( &mut self ) -> Zip<IterMut<T>, Zip<Iter<usize>, Iter<usize>>> {
        mzip!(
            self.data.iter_mut(),
            self.row_idx.iter(),
            self.col_idx.iter()
        )
    }

    #[inline]
    pub fn change_to_one_based_index( &mut self ) {
        // This is only intended for calling MKL, not working with the struct's methods!
        self.row_pos.addassign_scalar(1);
        self.row_idx.addassign_scalar(1);
        self.col_idx.addassign_scalar(1);
    }

    #[inline]
    pub fn change_to_zero_based_index( &mut self ) {
        self.row_pos.subassign_scalar(1);
        self.row_idx.subassign_scalar(1);
        self.col_idx.subassign_scalar(1);
    }
}


