use fan455_util::{elem, mzip};
use std::fmt::Display;
use std::iter::{zip, Iterator};
use std::ops::{Index, IndexMut};
use std::slice::{Iter, IterMut};
use num_complex::Complex;
use fan455_math_scalar::{General, Numeric, Float};

#[derive(Clone, Copy)]
pub struct VecView<'a, T: General> {
    pub data: &'a [T],
}

pub struct VecViewMut<'a, T: General> {
    pub data: &'a mut [T],
}

#[derive(Clone, Copy)]
pub struct MatView<'a, T: General> {
    pub dim0: usize,
    pub dim1: usize,
    pub data: &'a [T],
}

pub struct MatViewMut<'a, T: General> {
    pub dim0: usize,
    pub dim1: usize,
    pub data: &'a mut [T],
}

impl<'a, T: General> Index<usize> for VecView<'a, T>
{
    type Output = T;

    #[inline]
    fn index( &self, index: usize ) -> &T {
        self.data.index(index)
    }
}

impl<'a, T: General> Index<usize> for VecViewMut<'a, T>
{
    type Output = T;

    #[inline]
    fn index( &self, index: usize ) -> &T {
        self.data.index(index)
    }
}

impl<'a, T: General> IndexMut<usize> for VecViewMut<'a, T>
{
    #[inline]
    fn index_mut( &mut self, index: usize ) -> &mut T {
        self.data.index_mut(index)
    }
}

impl<'a, T: General> MatView<'a, T>
{
    #[inline(always)]
    pub fn reshape( &mut self, m: usize, n: usize ) {
        self.dim0 = n;
        self.dim1 = m;
    }
}

impl<'a, T: General> MatViewMut<'a, T>
{
    #[inline(always)]
    pub fn reshape( &mut self, m: usize, n: usize ) {
        self.dim0 = n;
        self.dim1 = m;
    }
}

pub trait GVecAlloc<T: General>
{
    fn alloc( n: usize ) -> Self;

    fn alloc_set( n: usize, val: T ) -> Self;

    fn alloc_copy<VT: GVec<T>>( x: &VT ) -> Self;
}

pub trait GMatAlloc<T: General>
{
    fn alloc( m: usize, n: usize ) -> Self;

    fn alloc_set( m: usize, n: usize, val: T ) -> Self;

    fn alloc_copy<MT: GMat<T>>( x: &MT ) -> Self;
}

pub trait GVec<T: General>
{
    fn size( &self ) -> usize;

    fn ptr( &self ) -> *const T;

    fn sl( &self ) -> &[T];

    #[inline(always)]
    fn idx( &self, i: usize ) -> &T {
        self.sl().index(i)
    }

    #[inline(always)]
    fn idx_unchecked( &self, i: usize ) -> &T {
        unsafe {self.sl().get_unchecked(i)}
    }

    #[inline(always)]
    fn it( &self ) -> Iter<T> {
        self.sl().iter()
    }
        
    #[inline(always)]
    fn subvec( &self, i1: usize, i2: usize ) -> VecView<T> {
        VecView { data: &self.sl()[i1..i2] }
    }
}

pub trait GVecMut<T: General>: GVec<T>
{
    fn ptrm( &mut self ) -> *mut T;

    fn slm( &mut self ) -> &mut [T];

    #[inline(always)]
    fn idxm( &mut self, i: usize ) -> &mut T {
        self.slm().index_mut(i)
    }

    #[inline(always)]
    fn idxm_unchecked( &mut self, i: usize ) -> &mut T {
        unsafe {self.slm().get_unchecked_mut(i)}
    }

    #[inline(always)]
    fn itm( &mut self ) -> IterMut<T> {
        self.slm().iter_mut()
    }

    #[inline(always)]
    fn copy_sl( &mut self, x: &[T] ) {
        self.slm().copy_from_slice(x);
    }

    #[inline(always)]
    fn copy<VT: GVec<T>>( &mut self, x: &VT ) {
        self.copy_sl(x.sl());
    }

    #[inline(always)]
    fn reset( &mut self ) {
        self.slm().fill(T::default());
    }

    #[inline(always)]
    fn set( &mut self, val: T ) {
        self.slm().fill(val);
    }

    #[inline(always)]
    fn subvec_mut( &mut self, i1: usize, i2: usize ) -> VecViewMut<T> {
        VecViewMut { data: &mut self.slm()[i1..i2] }
    }

    #[inline(always)]
    fn get_elements<VT: GVec<T>>( &mut self, x: &VT, sel: &[usize] ) {
        for (y_, sel_) in mzip!(self.itm(), sel) {
            *y_ = *x.idx(*sel_);
        }
    }
}

pub trait NVec<T: Numeric>: GVec<T>
{
    #[inline(always)]
    fn sum( &self ) -> T {
        let mut s = T::zero();
        for y_ in self.it() {
            s += y_;
        }
        s
    }
}

pub trait NVecMut<T: Numeric>: NVec<T> + GVecMut<T> 
{
    #[inline(always)]
    fn assign_add<VT1, VT2>( &mut self, x1: &VT1, x2: &VT2 ) 
    where VT1: NVec<T>, VT2: NVec<T>
    {
        for elem!(y_, x1_, x2_) in mzip!(self.itm(), x1.it(), x2.it()) {
            *y_ = *x1_ + x2_;
        }
    }

    #[inline(always)]
    fn assign_sub<VT1, VT2>( &mut self, x1: &VT1, x2: &VT2 ) 
    where VT1: NVec<T>, VT2: NVec<T>
    {
        for elem!(y_, x1_, x2_) in mzip!(self.itm(), x1.it(), x2.it()) {
            *y_ = *x1_ - x2_;
        }
    }

    #[inline(always)]
    fn assign_mul<VT1, VT2>( &mut self, x1: &VT1, x2: &VT2 ) 
    where VT1: NVec<T>, VT2: NVec<T>
    {
        for elem!(y_, x1_, x2_) in mzip!(self.itm(), x1.it(), x2.it()) {
            *y_ = *x1_ * x2_;
        }
    }

    #[inline(always)]
    fn assign_mul3<VT1, VT2, VT3>( &mut self, x1: &VT1, x2: &VT2, x3: &VT3 ) 
    where VT1: NVec<T>, VT2: NVec<T>, VT3: NVec<T>
    {
        for elem!(y_, x1_, x2_, x3_) in mzip!(self.itm(), x1.it(), x2.it(), x3.it()) {
            *y_ = *x1_ * x2_ * x3_;
        }
    }

    #[inline(always)]
    fn assign_div<VT1, VT2>( &mut self, x1: &VT1, x2: &VT2 ) 
    where VT1: NVec<T>, VT2: NVec<T>
    {
        for elem!(y_, x1_, x2_) in mzip!(self.itm(), x1.it(), x2.it()) {
            *y_ = *x1_ / x2_;
        }
    }

    #[inline(always)]
    fn addassign<VT>( &mut self, x: &VT ) 
    where VT: NVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ += x_;
        } 
    }

    #[inline(always)]
    fn subassign<VT>( &mut self, x: &VT ) 
    where VT: NVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ -= x_;
        } 
    }

    #[inline(always)]
    fn mulassign<VT>( &mut self, x: &VT ) 
    where VT: NVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ *= x_;
        } 
    }

    #[inline(always)]
    fn divassign<VT>( &mut self, x: &VT )
    where VT: NVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ /= x_;
        }
    }

    #[inline(always)]
    fn addassign_scalar( &mut self, s: T ) {
        for y_ in self.itm() {
            *y_ += s;
        } 
    }

    #[inline(always)]
    fn subassign_scalar( &mut self, s: T ) {
        for y_ in self.itm() {
            *y_ -= s;
        } 
    }

    #[inline(always)]
    fn mulassign_scalar( &mut self, s: T ) {
        for y_ in self.itm() {
            *y_ *= s;
        } 
    }

    #[inline(always)]
    fn divassign_scalar( &mut self, s: T ) {
        for y_ in self.itm() {
            *y_ /= s;
        } 
    }

    #[inline(always)]
    fn assign_muladd<VT1, VT2, VT3>( &mut self, x: &VT1, a: &VT2, b: &VT3 )
    where VT1: GVec<T>, VT2: GVec<T>, VT3: GVec<T>,
    {
        for elem!(y_, x_, a_, b_) in mzip!(self.itm(), x.it(), a.it(), b.it()) {
            *y_ = x_.mul_add(*a_, *b_);
        }
    }

    #[inline(always)]
    fn muladdassign<VT1, VT2>( &mut self, a: &VT1, b: &VT2 )
    where VT1: GVec<T>, VT2: GVec<T>,
    {
        for elem!(y_, a_, b_) in mzip!(self.itm(), a.it(), b.it()) {
            y_.mul_add_assign(*a_, *b_);
        }
    }
}

pub trait RVec<T: Float>: NVec<T>
{
    #[inline(always)]
    fn sumsquare( &self ) -> T {
        let mut sum: T = T::zero();
        for x_ in self.it() {
            sum += x_.powi(2);
        }
        sum
    }

    #[inline(always)]
    fn norm( &self ) -> T {
        self.sumsquare().sqrt()
    }

    #[inline(always)]
    fn max( &self ) -> T {
        *self.it().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    #[inline(always)]
    fn min( &self ) -> T {
        *self.it().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    #[inline(always)]
    fn absmax( &self ) -> T {
        *self.it().max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap()).unwrap()
    }

    #[inline(always)]
    fn absmin( &self ) -> T {
        *self.it().min_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap()).unwrap()
    }
}

pub trait RVecMut<T: Float>: RVec<T> + NVecMut<T>
{
    #[inline(always)]
    fn sort_ascend( &mut self ) {
        self.slm().sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    #[inline(always)]
    fn scale( &mut self, s: T ) {
        for y_ in self.itm() {
            *y_ *= s;
        } 
    }

    #[inline(always)]
    fn unscale( &mut self, s: T ) {
        for y_ in self.itm() {
            *y_ /= s;
        } 
    }

    #[inline(always)]
    fn assign_scale<VT>( &mut self, x: &VT, s: T )
    where VT: RVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = s * x_;
        } 
    }

    #[inline(always)]
    fn assign_powi<VT>( &mut self, x: &VT, n: i32 )
    where VT: RVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = x_.powi(n);
        } 
    }

    #[inline(always)]
    fn assign_add_scale<VT1, VT2>( &mut self, x1: &VT1, x2: &VT2, s: T )
    where VT1: RVec<T>, VT2: RVec<T>,
    {
        for elem!(y_, x1_, x2_) in mzip!(self.itm(), x1.it(), x2.it()) {
            *y_ = *x1_ + s * x2_;
        } 
    }

    #[inline(always)]
    fn assign_sub_scale<VT1, VT2>( &mut self, x1: &VT1, x2: &VT2, s: T )
    where VT1: RVec<T>, VT2: RVec<T>,
    {
        for elem!(y_, x1_, x2_) in mzip!(self.itm(), x1.it(), x2.it()) {
            *y_ = *x1_ - s * x2_;
        } 
    }

    #[inline(always)]
    fn addassign_scale<VT>( &mut self, x: &VT, s: T ) 
    where VT: RVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ += s * x_;
        } 
    }

    #[inline(always)]
    fn subassign_scale<VT>( &mut self, x: &VT, s: T )
    where VT: RVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ -= s * x_;
        } 
    }

    #[inline(always)]
    fn assign_unscale<VT>( &mut self, x: &VT, s: T )
    where VT: RVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = *x_ / s;
        } 
    }

    #[inline(always)]
    fn assign_absdiff<VT1, VT2>( &mut self, x1: &VT1, x2: &VT2 )
    where VT1: RVec<T>, VT2: RVec<T>,
    {
        for elem!(y_, x1_, x2_) in mzip!(self.itm(), x1.it(), x2.it()) {
            *y_ = (*x1_ - x2_).abs();
        }
    }

    #[inline(always)]
    fn get_complex_norm<VT>( &mut self, x: &VT )
    where VT: CVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = x_.norm();
        }
    }

    #[inline(always)]
    fn get_complex_lognorm<VT>( &mut self, x: &VT )
    where VT: CVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = x_.norm().ln();
        }
    }

    #[inline(always)]
    fn get_complex_norm2<VT>( &mut self, x: &VT )
    where VT: CVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = x_.norm_sqr();
        }
    }

    #[inline(always)]
    fn clip<VT1, VT2>( &mut self, lb: &VT1, ub: &VT2 )
    where VT1: RVec<T>, VT2: RVec<T>,
    {
        for elem!(y_, lb_, ub_) in mzip!(self.itm(), lb.it(), ub.it()) {
            if *y_ < *lb_ {
                *y_ = *lb_;
            }
            if *y_ > *ub_ {
                *y_ = *ub_;
            }
        }
    }
    
    #[inline(always)]
    fn clip_lb<VT>( &mut self, lb: &VT )
    where VT: RVec<T>,
    {
        for (y_, lb_) in zip(self.itm(), lb.it()) {
            if *y_ < *lb_ {
                *y_ = *lb_;
            }
        }
    }

    #[inline(always)]
    fn clip_ub<VT>( &mut self, ub: &VT )
    where VT: RVec<T>,
    {
        for (y_, ub_) in zip(self.itm(), ub.it()) {
            if *y_ > *ub_ {
                *y_ = *ub_;
            }
        }
    }
}

pub trait CVec<T: Float>: NVec<Complex<T>> {}

pub trait CVecMut<T: Float>: CVec<T> + NVecMut<Complex<T>> 
{
    #[inline(always)]
    fn scale( &mut self, s: T ) {
        for y_ in self.itm() {
            y_.re *= s;
            y_.im *= s;
        } 
    }

    #[inline(always)]
    fn unscale( &mut self, s: T ) {
        for y_ in self.itm() {
            y_.re /= s;
            y_.im /= s;
        } 
    }

    #[inline(always)]
    fn conj( &mut self ) {
        for y_ in self.itm() {
            y_.im = -y_.im;
        } 
    }

    #[inline(always)]
    fn assign_scale<VT>( &mut self, x: &VT, s: T )
    where VT: CVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = x_.scale(s);
        } 
    }

    #[inline(always)]
    fn assign_unscale<VT>( &mut self, x: &VT, s: T )
    where VT: CVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = x_.unscale(s);
        } 
    }

    #[inline(always)]
    fn assign_conj<VT>( &mut self, x: &VT )
    where VT: CVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = x_.conj();
        } 
    }
}

pub trait GMat<T: General>: GVec<T>
{
    fn nrow( &self ) -> usize;

    fn ncol( &self ) -> usize;

    fn stride( &self ) -> usize;

    #[inline(always)]
    fn idx2( &self, i: usize, j: usize ) -> &T {
        self.sl().index(i+j*self.nrow())
    }

    #[inline(always)]
    fn idx2_unchecked( &self, i: usize, j: usize ) -> &T {
        unsafe { self.sl().get_unchecked(i+j*self.nrow()) }
    }

    #[inline(always)]
    fn col( &self, j: usize ) -> VecView<T> {
        let m = self.nrow();
        VecView { data: &self.sl()[j*m..(j+1)*m] }
    }

    #[inline(always)]
    fn col2( &self, j: usize ) -> MatView<T> {
        let m = self.nrow();
        MatView { dim0: 1, dim1: m, data: &self.sl()[j*m..(j+1)*m] }
    }

    #[inline(always)]
    fn col_as_mat( &self, j: usize, m: usize, n: usize ) -> MatView<T> { // For reshape purpose.
        let nrow = self.nrow();
        MatView { dim0: n, dim1: m, data: &self.sl()[j*nrow..(j+1)*nrow] }
    }

    #[inline(always)]
    fn cols( &self, j1: usize, j2: usize ) -> MatView<T> {
        let m = self.nrow();
        MatView { dim0: j2-j1, dim1: m, data: &self.sl()[j1*m..j2*m] }
    }

    #[inline(always)]
    fn subvec2( &self, i1: usize, j1: usize, i2: usize, j2: usize ) -> VecView<T> {
        let m = self.nrow();
        VecView { data: &self.sl()[i1+j1*m..i2+j2*m] }
    }

    #[inline(always)]
    fn diag( &self ) -> Vec<T> {
        let mut x: Vec<T> = Vec::with_capacity(self.nrow());
        for i in 0..self.nrow() {
            x.push(*self.idx2(i, i));
        }
        x
    }
}

pub trait GMatMut<T: General>: GMat<T> + GVecMut<T>
{
    #[inline(always)]
    fn idxm2( &mut self, i: usize, j: usize ) -> &mut T {
        let m = self.nrow();
        self.slm().index_mut(i+j*m)
    }

    #[inline(always)]
    fn idxm2_unchecked( &mut self, i: usize, j: usize ) -> &mut T {
        let m = self.nrow();
        unsafe { self.slm().get_unchecked_mut(i+j*m) }
    }

    #[inline(always)]
    fn col_mut( &mut self, j: usize ) -> VecViewMut<T> {
        let m = self.nrow();
        VecViewMut { data: &mut self.slm()[j*m..(j+1)*m] }
    }

    #[inline(always)]
    fn col2_mut( &mut self, j: usize ) -> MatViewMut<T> { // For reshape purpose.
        let m = self.nrow();
        MatViewMut { dim0: 1, dim1: m, data: &mut self.slm()[j*m..(j+1)*m] }
    }

    #[inline(always)]
    fn col_as_mat_mut( &mut self, j: usize, m: usize, n: usize ) -> MatViewMut<T> { // For reshape purpose.
        let nrow = self.nrow();
        MatViewMut { dim0: n, dim1: m, data: &mut self.slm()[j*nrow..(j+1)*nrow] }
    }

    #[inline(always)]
    fn cols_mut( &mut self, j1: usize, j2: usize ) -> MatViewMut<T> {
        let m = self.nrow();
        MatViewMut { dim0: j2-j1, dim1: m, data: &mut self.slm()[j1*m..j2*m] }
    }

    #[inline(always)]
    fn subvec2_mut( &mut self, i1: usize, j1: usize, i2: usize, j2: usize ) -> VecViewMut<T> {
        let m = self.nrow();
        VecViewMut { data: &mut self.slm()[i1+j1*m..i2+j2*m] }
    }

    #[inline(always)]
    fn get_rows<MT: GMat<T>>( &mut self, x: &MT, rows: &[usize] ) {
        for j in 0..self.ncol() {
            for (y_, r_) in mzip!(self.col_mut(j).itm(), rows.iter()) {
                *y_ = *x.idx2(*r_, j);
            }
        }
    }

    #[inline(always)]
    fn get_rows_as_cols<MT: GMat<T>>( &mut self, x: &MT, rows: &[usize] ) {
        for (j, r_) in mzip!( 0..self.ncol(), rows.iter() ) {
            for (y_, jx) in mzip!(self.col_mut(j).itm(), 0..x.ncol()) {
                *y_ = *x.idx2(*r_, jx);
            }
        }
    }

    #[inline(always)]
    fn set_diag( &mut self, val: T ) {
        for i in 0..self.ncol() {
            *self.idxm2(i, i) = val;
        }
    }

    #[inline(always)]
    fn set_diag_to_vec<VT: GVec<T>>( &mut self, x: &VT ) {
        for (i, x_) in zip(0..self.ncol(), x.it()) {
            *self.idxm2(i, i) = *x_;
        }
    }

    #[inline(always)]
    fn set_lband( &mut self, offset: usize, val: T ) {
        for j in 0..self.ncol()-offset {
            *self.idxm2(j+offset, j) = val;
        }
    }

    #[inline(always)]
    fn set_uband( &mut self, offset: usize, val: T ) {
        for j in offset..self.ncol() {
            *self.idxm2(j-offset, j) = val;
        }
    }

    #[inline(always)]
    fn set_lower( &mut self, val: T ) {
        for j in 0..self.ncol()-1 {
            for i in j+1..self.nrow() {
                *self.idxm2(i, j) = val;
            }
        }
    }

    #[inline(always)]
    fn set_upper( &mut self, val: T ) {
        for j in 1..self.ncol() {
            for i in 0..j {
                *self.idxm2(i, j) = val;
            }
        }
    }

    #[inline(always)]
    fn copy_lower_to_upper( &mut self ) {
        for j in 1..self.ncol() {
            for i in 0..j {
                *self.idxm2(i, j) = *self.idx2(j, i);
            }
        }
    }

    #[inline(always)]
    fn copy_upper_to_lower( &mut self ) {
        for j in 0..self.ncol()-1 {
            for i in j+1..self.nrow() {
                *self.idxm2(i, j) = *self.idx2(j, i);
            }
        }
    }

    #[inline(always)]
    fn get_trans<MT: GMat<T>>( &mut self, x: &MT ) {
        let m = self.nrow();
        for j in 0..self.ncol() {
            for (y_, i) in zip(self.col_mut(j).itm(), 0..m) {
                *y_ = *x.idx2(j, i);
            }
        }
    }

    #[inline(always)]
    fn get_trans_unchecked<MT: GMat<T>>( &mut self, x: &MT ) {
        let m = self.nrow();
        for j in 0..self.ncol() {
            for (y_, i) in zip(self.col_mut(j).itm(), 0..m) {
                *y_ = *x.idx_unchecked(j+i*m);
            }
        }
    }
}

pub trait NMat<T: Numeric>: GMat<T> + NVec<T> {}

pub trait NMatMut<T: Numeric>: NMat<T> + GMatMut<T> + NVecMut<T>
{
    #[inline(always)]
    fn addassign_iden( &mut self, s: T ) {
        for i in 0..self.ncol() {
            *self.idxm2(i, i) += s;
        }
    }
    
    #[inline(always)]
    fn addassign_rowvec<VT: NVec<T>>( &mut self, x: &VT ) {
        for (j, x_) in zip(0..self.ncol(), x.it()) {
            for y_ in self.col_mut(j) {
                *y_ += x_;
            }
        }
    }

    #[inline(always)]
    fn mulassign_rowvec<VT: NVec<T>>( &mut self, x: &VT ) {
        for (j, x_) in zip(0..self.ncol(), x.it()) {
            for y_ in self.col_mut(j) {
                *y_ *= x_;
            }
        }
    }

    #[inline(always)]
    fn muladdassign_rowvec<VT1: NVec<T>, VT2: NVec<T>>( &mut self, a: &VT1, b: &VT2 ) {
        for elem!(j, a_, b_) in mzip!(0..self.ncol(), a.it(), b.it()) {
            for y_ in self.col_mut(j) {
                y_.mul_add_assign(*a_, *b_);
            }
        }
    }
}

pub trait RMat<T: Float>: NMat<T> + RVec<T>
{
    #[inline(always)]
    fn sumlogdiag( &self ) -> T {
        let mut sum: T = T::zero();
        for i in 0..self.ncol() {
            sum += self.idx2(i, i).ln();
        }
        sum
    }
}

pub trait RMatMut<T: Float>: RMat<T> + NMatMut<T> + RVecMut<T> {}

pub trait CMat<T: Float>: NMat<Complex<T>> + CVec<T> {}

pub trait CMatMut<T: Float>: CMat<T> + NMatMut<Complex<T>> + CVecMut<T> {}


impl<T: General, const N: usize> GVec<T> for [T; N]
{
    #[inline(always)]
    fn size( &self ) -> usize {
        N
    }

    #[inline(always)]
    fn ptr( &self ) -> *const T {
        self.as_ptr()
    }

    #[inline(always)]
    fn sl( &self ) -> &[T] {
        self.as_slice()
    }

    #[inline(always)]
    fn idx( &self, i: usize ) -> &T {
        self.index(i)
    }
}

impl<T: General, const N: usize> GVecMut<T> for [T; N]
{
    #[inline(always)]
    fn ptrm( &mut self ) -> *mut T {
        self.as_mut_ptr()
    }

    #[inline(always)]
    fn slm( &mut self ) -> &mut [T] {
        self.as_mut_slice()
    }

    #[inline(always)]
    fn idxm( &mut self, i: usize ) -> &mut T {
        self.index_mut(i)
    }
}

impl<T: General, const N: usize> GMat<T> for [T; N]
{
    #[inline(always)]
    fn nrow( &self ) -> usize {
        N
    }

    #[inline(always)]
    fn ncol( &self ) -> usize {
        1
    }

    #[inline(always)]
    fn stride( &self ) -> usize {
        N
    }

    #[inline(always)]
    fn idx2( &self, i: usize, _j: usize ) -> &T {
        self.index(i)
    }
}

impl<T: General, const N: usize> GMatMut<T> for [T; N]
{
    #[inline(always)]
    fn idxm2( &mut self, i: usize, _j: usize ) -> &mut T {
        self.index_mut(i)
    }
}

impl<T: Numeric, const N: usize> NVec<T> for [T; N] {}
impl<T: Numeric, const N: usize> NVecMut<T> for [T; N] {}
impl<T: Numeric, const N: usize> NMat<T> for [T; N] {}
impl<T: Numeric, const N: usize> NMatMut<T> for [T; N] {}

impl<T: Float, const N: usize> RVec<T> for [T; N] {}
impl<T: Float, const N: usize> RVecMut<T> for [T; N] {}
impl<T: Float, const N: usize> RMat<T> for [T; N] {}
impl<T: Float, const N: usize> RMatMut<T> for [T; N] {}

impl<T: Float, const N: usize> CVec<T> for [Complex<T>; N] {}
impl<T: Float, const N: usize> CVecMut<T> for [Complex<T>; N] {}
impl<T: Float, const N: usize> CMat<T> for [Complex<T>; N] {}
impl<T: Float, const N: usize> CMatMut<T> for [Complex<T>; N] {}


impl<T: General> GVec<T> for Vec<T>
{
    #[inline(always)]
    fn size( &self ) -> usize {
        self.len()
    }

    #[inline(always)]
    fn ptr( &self ) -> *const T {
        self.as_ptr()
    }

    #[inline(always)]
    fn sl( &self ) -> &[T] {
        self.as_slice()
    }

    #[inline(always)]
    fn idx( &self, i: usize ) -> &T {
        self.index(i)
    }
}

impl<T: General> GVecMut<T> for Vec<T>
{
    #[inline(always)]
    fn ptrm( &mut self ) -> *mut T {
        self.as_mut_ptr()
    }

    #[inline(always)]
    fn slm( &mut self ) -> &mut [T] {
        self.as_mut_slice()
    }

    #[inline(always)]
    fn idxm( &mut self, i: usize ) -> &mut T {
        self.index_mut(i)
    }
}

impl<T: General> GMat<T> for Vec<T>
{
    #[inline(always)]
    fn nrow( &self ) -> usize {
        self.len()
    }

    #[inline(always)]
    fn ncol( &self ) -> usize {
        1
    }

    #[inline(always)]
    fn stride( &self ) -> usize {
        self.len()
    }

    #[inline(always)]
    fn idx2( &self, i: usize, _j: usize ) -> &T {
        self.index(i)
    }
}

impl<T: General> GMatMut<T> for Vec<T>
{
    #[inline(always)]
    fn idxm2( &mut self, i: usize, _j: usize ) -> &mut T {
        self.index_mut(i)
    }
}

impl<T: Numeric> NVec<T> for Vec<T> {}
impl<T: Numeric> NVecMut<T> for Vec<T> {}
impl<T: Numeric> NMat<T> for Vec<T> {}
impl<T: Numeric> NMatMut<T> for Vec<T> {}

impl<T: Float> RVec<T> for Vec<T> where T: Float, {}
impl<T: Float> RVecMut<T> for Vec<T> where T: Float, {}
impl<T: Float> RMat<T> for Vec<T> where T: Float, {}
impl<T: Float> RMatMut<T> for Vec<T> where T: Float, {}

impl<T: Float> CVec<T> for Vec<Complex<T>> {}
impl<T: Float> CVecMut<T> for Vec<Complex<T>> {}
impl<T: Float> CMat<T> for Vec<Complex<T>> {}
impl<T: Float> CMatMut<T> for Vec<Complex<T>> {}


impl<'a, T: General> GVec<T> for VecView<'a, T>
{
    #[inline(always)]
    fn size( &self ) -> usize {
        self.data.len()
    }

    #[inline(always)]
    fn ptr( &self ) -> *const T {
        self.data.as_ptr()
    }

    #[inline(always)]
    fn sl( &self ) -> &[T] {
        self.data
    }

    #[inline(always)]
    fn idx( &self, i: usize ) -> &T {
        self.data.index(i)
    }
}

impl<'a, T: General> GVec<T> for VecViewMut<'a, T>
{
    #[inline(always)]
    fn size( &self ) -> usize {
        self.data.len()
    }

    #[inline(always)]
    fn ptr( &self ) -> *const T {
        self.data.as_ptr()
    }

    #[inline(always)]
    fn sl( &self ) -> &[T] {
        self.data
    }

    #[inline(always)]
    fn idx( &self, i: usize ) -> &T {
        self.data.index(i)
    }
}

impl<'a, T: General> GVecMut<T> for VecViewMut<'a, T>
{
    #[inline(always)]
    fn ptrm( &mut self ) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline(always)]
    fn slm( &mut self ) -> &mut [T] {
        self.data
    }

    #[inline(always)]
    fn idxm( &mut self, i: usize ) -> &mut T {
        self.data.index_mut(i)
    }
}

impl<'a, T: General> GMat<T> for VecView<'a, T>
{
    #[inline(always)]
    fn nrow( &self ) -> usize {
        self.data.len()
    }

    #[inline(always)]
    fn ncol( &self ) -> usize {
        1
    }

    #[inline(always)]
    fn stride( &self ) -> usize {
        self.data.len()
    }

    #[inline(always)]
    fn idx2( &self, i: usize, _j: usize ) -> &T {
        self.data.index(i)
    }
}

impl<'a, T: General> GMat<T> for VecViewMut<'a, T>
{
    #[inline(always)]
    fn nrow( &self ) -> usize {
        self.data.len()
    }

    #[inline(always)]
    fn ncol( &self ) -> usize {
        1
    }

    #[inline(always)]
    fn stride( &self ) -> usize {
        self.data.len()
    }

    #[inline(always)]
    fn idx2( &self, i: usize, _j: usize ) -> &T {
        self.data.index(i)
    }
}

impl<'a, T: General> GMatMut<T> for VecViewMut<'a, T>
{
    #[inline(always)]
    fn idxm2( &mut self, i: usize, _j: usize ) -> &mut T {
        self.data.index_mut(i)
    }
}

impl<'a, T: Numeric> NVec<T> for VecView<'a, T> {}
impl<'a, T: Numeric> NVec<T> for VecViewMut<'a, T> {}
impl<'a, T: Numeric> NVecMut<T> for VecViewMut<'a, T> {}
impl<'a, T: Numeric> NMat<T> for VecView<'a, T> {}
impl<'a, T: Numeric> NMat<T> for VecViewMut<'a, T> {}
impl<'a, T: Numeric> NMatMut<T> for VecViewMut<'a, T> {}

impl<'a, T: Float> RVec<T> for VecView<'a, T> {}
impl<'a, T: Float> RVec<T> for VecViewMut<'a, T> {}
impl<'a, T: Float> RVecMut<T> for VecViewMut<'a, T> {}
impl<'a, T: Float> RMat<T> for VecView<'a, T> {}
impl<'a, T: Float> RMat<T> for VecViewMut<'a, T> {}
impl<'a, T: Float> RMatMut<T> for VecViewMut<'a, T> {}

impl<'a, T: Float> CVec<T> for VecView<'a, Complex<T>> {}
impl<'a, T: Float> CVec<T> for VecViewMut<'a, Complex<T>> {}
impl<'a, T: Float> CVecMut<T> for VecViewMut<'a, Complex<T>> {}
impl<'a, T: Float> CMat<T> for VecView<'a, Complex<T>> {}
impl<'a, T: Float> CMat<T> for VecViewMut<'a, Complex<T>> {}
impl<'a, T: Float> CMatMut<T> for VecViewMut<'a, Complex<T>> {}


impl<'a, T: General> GVec<T> for MatView<'a, T>
{
    #[inline(always)]
    fn size( &self ) -> usize {
        self.data.len()
    }

    #[inline(always)]
    fn ptr( &self ) -> *const T {
        self.data.as_ptr()
    }

    #[inline(always)]
    fn sl( &self ) -> &[T] {
        self.data
    }

    #[inline(always)]
    fn idx( &self, i: usize ) -> &T {
        self.data.index(i)
    }
}

impl<'a, T: General> GVec<T> for MatViewMut<'a, T>
{
    #[inline(always)]
    fn size( &self ) -> usize {
        self.data.len()
    }

    #[inline(always)]
    fn ptr( &self ) -> *const T {
        self.data.as_ptr()
    }

    #[inline(always)]
    fn sl( &self ) -> &[T] {
        self.data
    }

    #[inline(always)]
    fn idx( &self, i: usize ) -> &T {
        self.data.index(i)
    }
}

impl<'a, T: General> GVecMut<T> for MatViewMut<'a, T>
{
    #[inline(always)]
    fn ptrm( &mut self ) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline(always)]
    fn slm( &mut self ) -> &mut [T] {
        self.data
    }

    #[inline(always)]
    fn idxm( &mut self, i: usize ) -> &mut T {
        self.data.index_mut(i)
    }
}

impl<'a, T: General> GMat<T> for MatView<'a, T>
{
    #[inline(always)]
    fn nrow( &self ) -> usize {
        self.dim1
    }

    #[inline(always)]
    fn ncol( &self ) -> usize {
        self.dim0
    }

    #[inline(always)]
    fn stride( &self ) -> usize {
        self.dim1
    }

    #[inline(always)]
    fn idx2( &self, i: usize, j: usize ) -> &T {
        self.data.index(i+j*self.dim1)
    }
}

impl<'a, T: General> GMat<T> for MatViewMut<'a, T>
{
    #[inline(always)]
    fn nrow( &self ) -> usize {
        self.dim1
    }

    #[inline(always)]
    fn ncol( &self ) -> usize {
        self.dim0
    }

    #[inline(always)]
    fn stride( &self ) -> usize {
        self.dim1
    }

    #[inline(always)]
    fn idx2( &self, i: usize, j: usize ) -> &T {
        self.data.index(i+j*self.dim1)
    }
}

impl<'a, T: General> GMatMut<T> for MatViewMut<'a, T>
{
    #[inline(always)]
    fn idxm2( &mut self, i: usize, j: usize ) -> &mut T {
        self.data.index_mut(i+j*self.dim1)
    }
}

impl<'a, T: Numeric> NVec<T> for MatView<'a, T> {}
impl<'a, T: Numeric> NVec<T> for MatViewMut<'a, T> {}
impl<'a, T: Numeric> NVecMut<T> for MatViewMut<'a, T> {}
impl<'a, T: Numeric> NMat<T> for MatView<'a, T> {}
impl<'a, T: Numeric> NMat<T> for MatViewMut<'a, T> {}
impl<'a, T: Numeric> NMatMut<T> for MatViewMut<'a, T> {}

impl<'a, T: Float> RVec<T> for MatView<'a, T> {}
impl<'a, T: Float> RVec<T> for MatViewMut<'a, T> {}
impl<'a, T: Float> RVecMut<T> for MatViewMut<'a, T> {}
impl<'a, T: Float> RMat<T> for MatView<'a, T> {}
impl<'a, T: Float> RMat<T> for MatViewMut<'a, T> {}
impl<'a, T: Float> RMatMut<T> for MatViewMut<'a, T> {}

impl<'a, T: Float> CVec<T> for MatView<'a, Complex<T>> {}
impl<'a, T: Float> CVec<T> for MatViewMut<'a, Complex<T>> {}
impl<'a, T: Float> CVecMut<T> for MatViewMut<'a, Complex<T>> {}
impl<'a, T: Float> CMat<T> for MatView<'a, Complex<T>> {}
impl<'a, T: Float> CMat<T> for MatViewMut<'a, Complex<T>> {}
impl<'a, T: Float> CMatMut<T> for MatViewMut<'a, Complex<T>> {}


impl<'a, T> IntoIterator for VecView<'a, T>
where T: Default + Copy + Clone,
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline(always)]
    fn into_iter( self ) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T> IntoIterator for VecViewMut<'a, T>
where T: Default + Copy + Clone,
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    
    #[inline(always)]
    fn into_iter( self ) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T> IntoIterator for MatView<'a, T>
where T: Default + Copy + Clone,
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline(always)]
    fn into_iter( self ) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T> IntoIterator for MatViewMut<'a, T>
where T: Default + Copy + Clone,
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    
    #[inline(always)]
    fn into_iter( self ) -> Self::IntoIter {
        self.data.into_iter()
    }
}


pub struct RVecPrinter {
    pub width: usize,
    pub prec: usize,
}

pub struct CVecPrinter {
    pub width: usize,
    pub prec: usize,
}

pub struct RMatPrinter {
    pub width: usize,
    pub prec: usize,
}

pub struct CMatPrinter {
    pub width: usize,
    pub prec: usize,
}

impl RVecPrinter
{
    #[inline(always)]
    pub fn new( width: usize, prec: usize ) -> Self {
        Self { width, prec }
    }

    #[inline(always)]
    pub fn print<T: Float + Display, VT: RVec<T>>( &self, name: &str, x: &VT ) {
        let width = self.width;
        let prec = self.prec;
        print!("{name} = \n(");
        for x_ in x.it() {
            print!("{x_:>width$.prec$},");
        }
        println!(")");
    }

    #[inline(always)]
    pub fn print_usize<VT: NVec<usize>>( &self, name: &str, x: &VT ) {
        let width = self.width;
        print!("{name} = \n(");
        for x_ in x.it() {
            print!("{x_:>width$},");
        }
        println!(")");
    }
}

impl CVecPrinter
{
    #[inline(always)]
    pub fn new( width: usize, prec: usize ) -> Self {
        Self { width, prec }
    }

    #[inline(always)]
    pub fn print<T: Float + Display, VT: CVec<T>>( &self, name: &str, x: &VT ) {
        let width = self.width;
        let width2 = width - 1;
        let prec = self.prec;
        print!("{name} =\n(");
        for x_ in x.it() {
            print!("{:>width$.prec$} {:>+width2$.prec$}j,", x_.re, x_.im);
        }
        println!(")");
    }
}

impl RMatPrinter
{
    #[inline(always)]
    pub fn new( width: usize, prec: usize ) -> Self {
        Self { width, prec }
    }

    #[inline(always)]
    pub fn print<T: Float + Display, MT: RMat<T>>( &self, name: &str, x: &MT ) {
        let width = self.width;
        let prec = self.prec;
        print!("{name} =");
        for i in 0..x.nrow() {
            print!("\n(");
            for j in 0..x.ncol() {
                print!("{:>width$.prec$},", x.idx2(i,j));
            }
            print!(")");
        }
        println!();
    }

    #[inline(always)]
    pub fn print_usize<MT: NMat<usize>>( &self, name: &str, x: &MT ) {
        let width = self.width;
        print!("{name} =");
        for i in 0..x.nrow() {
            print!("\n(");
            for j in 0..x.ncol() {
                print!("{:>width$},", x.idx2(i,j));
            }
            print!(")");
        }
        println!();
    }
}

impl CMatPrinter
{
    #[inline(always)]
    pub fn new( width: usize, prec: usize ) -> Self {
        Self { width, prec }
    }

    #[inline(always)]
    pub fn print<T: Float + Display, MT: CMat<T>>( &self, name: &str, x: &MT ) {
        let width = self.width;
        let width2 = width - 1;
        let prec = self.prec;
        print!("{name} =");
        for i in 0..x.nrow() {
            print!("\n(");
            for j in 0..x.ncol() {
                let x_ = x.idx2(i,j);
                print!("{:>width$.prec$} {:>+width2$.prec$}j,", x_.re, x_.im);
            }
            print!(")");
        }
        println!();
    }
}


pub trait NestedArrayReset {
    fn reset( &mut self );
}


impl<T: Default+Clone, const M: usize, const N: usize> NestedArrayReset for [[T; M]; N]
{
    #[inline(always)]
    fn reset( &mut self ) {
        for s in self.iter_mut() {
            s.fill(T::default());
        }
    }
}