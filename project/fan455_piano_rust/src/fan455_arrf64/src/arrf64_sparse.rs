use fan455_math_scalar::*;
use fan455_util::*;
use super::arrf64_basic::*;
use std::iter::Zip;
use std::slice::{Iter, IterMut};
use std::collections::BTreeMap;
use std::cmp::Ordering;
use std::collections::btree_map::Entry;


#[derive(Default, Debug, Clone)]
pub struct SparseRowMat<T: Numeric> {
    // A row-major sparse matrix that is easy to add values, and can convert to CSR format when fixed.
    pub nrow: usize,
    pub ncol: usize,
    pub data: Vec<BTreeMap<usize, T>>, // key is (i, j) index.
}

impl<T: Numeric> SparseRowMat<T>
{
    #[inline]
    pub fn new( nrow: usize, ncol: usize ) -> Self {
        let data = vec![BTreeMap::<usize, T>::new(); nrow];
        Self { nrow, ncol, data }
    }

    #[inline]
    pub fn addassign_at( &mut self, i: usize, j: usize, val: T ) {
        match self.data[i].entry(j) {
            Entry::Occupied(mut s) => {
                *s.get_mut() += val;
            },
            Entry::Vacant(s) => {
                s.insert(val);
            },
        }
    }

    #[inline]
    pub fn addassign_at_option( &mut self, i: usize, j: usize, val: Option<T> ) {
        if let Some(val_) = val {
            self.addassign_at(i, j, val_);
        }
    }

    #[inline]
    pub fn idx2( &self, i: usize, j: usize ) -> Option<&T> {
        self.data[i].get(&j)
    }

    #[inline]
    pub fn idxm2( &mut self, i: usize, j: usize ) -> Option<&mut T> {
        self.data[i].get_mut(&j)
    }

    #[inline]
    pub fn nnz( &self ) -> usize {
        let mut s: usize = 0;
        for row in self.data.iter() {
            s += row.len();
        }
        s
    }

    #[inline]
    pub fn to_csr( &self ) -> CsrMat<T> {
        let nrow = self.nrow;
        let ncol = self.ncol;
        let nnz = self.nnz();
        let mut data = Vec::<T>::with_capacity(nnz);
        let mut row_pos = Vec::<usize>::with_capacity(nrow+1);
        let mut row_idx = Vec::<usize>::with_capacity(nnz);
        let mut col_idx = Vec::<usize>::with_capacity(nnz);

        let mut i_nnz: usize = 0;
        row_pos.push(i_nnz);
        for elem!(i, row) in mzip!(0..nrow, self.data.iter()) {
            i_nnz += row.len();
            row_pos.push(i_nnz);
            for (j, val) in row.iter() {
                row_idx.push(i);
                col_idx.push(*j);
                data.push(*val);
            }
        }
        assert_multi_eq!(nnz, data.len(), row_idx.len(), col_idx.len());
        assert_eq!(nrow+1, row_pos.len());

        CsrMat { nrow, ncol, nnz, data, row_pos, row_idx, col_idx }
    }
}


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
    pub fn idx2( &self, i: usize, j: usize ) -> Option<&T> {
        let beg = self.row_pos[i];
        let end = self.row_pos[i+1];
        let mut i_nz: usize = usize::MAX;
        'outer: for elem!(i_, curr_col) in mzip!(beg..end, self.col_idx[beg..end].iter()) {
            if j == *curr_col {
                i_nz = i_;
                break 'outer;
            }
        }
        self.data.get(i_nz)
    }

    #[inline]
    pub fn idxm2( &mut self, i: usize, j: usize ) -> Option<&mut T> {
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


#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowMatIdx {
    i: usize,
    j: usize,
}

impl PartialOrd for RowMatIdx
{
    #[inline]
    fn partial_cmp( &self, other: &Self ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RowMatIdx
{
    #[inline]
    fn cmp( &self, other: &Self ) -> Ordering {
        self.i.cmp(&other.i).then(self.j.cmp(&other.j))
    }
}


#[derive(Default, Debug, Clone)]
pub struct SparseVec<T: Numeric> {
    // A row-major sparse matrix that is easy to add values, and can convert to CSR format when fixed.
    pub size: usize,
    pub data: BTreeMap<usize, T>, // key is i) index.
}

impl<T: Numeric> SparseVec<T>
{
    #[inline]
    pub fn new( size: usize ) -> Self {
        let data = BTreeMap::<usize, T>::new();
        Self { size, data }
    }

    #[inline]
    pub fn addassign_at( &mut self, i: usize, val: T ) {
        match self.data.entry(i) {
            Entry::Occupied(mut s) => {
                *s.get_mut() += val;
            },
            Entry::Vacant(s) => {
                s.insert(val);
            },
        }
    }

    #[inline]
    pub fn addassign_at_option( &mut self, i: usize, val: Option<T> ) {
        if let Some(val_) = val {
            self.addassign_at(i, val_);
        }
    }

    #[inline]
    pub fn idx2( &self, i: usize ) -> Option<&T> {
        self.data.get(&i)
    }

    #[inline]
    pub fn idxm2( &mut self, i: usize ) -> Option<&mut T> {
        self.data.get_mut(&i)
    }

    #[inline]
    pub fn nnz( &self ) -> usize {
        self.data.len()
    }
}