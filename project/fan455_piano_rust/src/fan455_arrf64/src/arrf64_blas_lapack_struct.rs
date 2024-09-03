use fan455_math_scalar::*;
use super::arrf64_basic::*;
use super::arrf64_blas_lapack::*;
use super::arrf64_sparse::*;
use std::fmt::Display;



#[derive(Default)]
pub struct FeastSparse<T: Float> {
    pub params: Vec<BlasUint>,

    pub uplo: BlasChar,
    pub eig_n: BlasUint,
    pub loops_n: BlasUint,
    pub info: BlasInt,
    pub epsout: T,
    pub residual: Vec<T>,

    pub eigval_lb: T,
    pub eigval_ub: T,
    pub eig_n_guess: BlasUint,
}


impl<T: Float+Display> FeastSparse<T>
{
    #[inline]
    pub fn new() -> Self {
        let uplo = LOWER;
        let mut params: Vec<usize> = vec![0; 128];
        unsafe {
            feastinit(params.ptrm());
        }
        Self { params, uplo, ..Default::default() }
    }

    #[inline]
    pub fn init_guess( &mut self, emin: T, emax: T, n: BlasUint ) {
        self.eigval_lb = emin;
        self.eigval_ub = emax;
        self.eig_n_guess = n;
        self.residual.resize(n, T::default());
    }

    #[inline]
    pub fn report( &self ) {
        println!("Finished MKL FEAST eigensolver call with info: {}.", self.info);

        if self.info == 0 {
            println!("The eigensolver was successful.\neigenvalues range: {} to {}\nnumber of eigenvalues found: {}\nnumber of loops: {}\nrelative error on the trace: {}\n", self.eigval_lb, self.eigval_ub, self.eig_n, self.loops_n, self.epsout);
        } else {
            println!("The eigensolver failed. Please check the MKL reference for more details.");
        }
    }

    #[inline]
    pub fn set_runtime_print( &mut self, val: bool ) {
        // 0: print runtime status; 1: not. Default is 0.
        self.params[0] = BlasUint::from(val);
    }

    #[inline]
    pub fn set_num_contour_points( &mut self, val: BlasUint ) {
        // Default is 8.
        self.params[1] = val;
    }

    #[inline]
    pub fn set_max_loops( &mut self, val: usize ) {
        // Default is 20.
        self.params[3] = val;
    }

    #[inline]
    pub fn set_stop_type( &mut self, val: BlasUint ) {
        // Default is 0.
        self.params[5] = val;
    }

    #[inline]
    pub fn set_sparse_mat_check( &mut self, val: bool ) {
        // Default is 0.
        self.params[26] = BlasUint::from(val);
    }

    #[inline]
    pub fn set_positive_mat_check( &mut self, val: bool ) {
        // Default is 0.
        self.params[27] = BlasUint::from(val);
    }
}


impl FeastSparse<f64>
{
    #[inline]
    pub fn set_tol( &mut self, val: BlasUint ) {
        // Default is 12.
        self.params[2] = val;
    }

    #[inline]
    pub fn solve<VT: RVecMut<f64>, MT: RMatMut<f64>>( 
        &mut self, 
        a: &CsrMat<f64>, 
        eigval: &mut VT,
        eigvec: &mut MT,
    ) {
        unsafe {
            dfeast_scsrev(
                &self.uplo as *const BlasChar, 
                &a.nrow as *const BlasUint, 
                a.data.ptr(), 
                a.row_pos.ptr(), 
                a.col_idx.ptr(), 
                self.params.ptrm(),
                &mut self.epsout as *mut f64, 
                &mut self.loops_n as *mut BlasUint, 
                &self.eigval_lb as *const f64, 
                &self.eigval_ub as *const f64, 
                &self.eig_n_guess as *const BlasUint, 
                eigval.ptrm(), 
                eigvec.ptrm(), 
                &mut self.eig_n as *mut BlasUint, 
                self.residual.ptrm(), 
                &mut self.info as *mut BlasInt
            );
        }
    }

    #[inline]
    pub fn solve_generalized<VT: RVecMut<f64>, MT: RMatMut<f64>>( 
        &mut self, 
        a: &CsrMat<f64>, 
        b: &CsrMat<f64>, 
        eigval: &mut VT,
        eigvec: &mut MT,
    ) {
        unsafe {
            dfeast_scsrgv(
                &self.uplo as *const BlasChar, 
                &a.nrow as *const BlasUint, 
                a.data.ptr(), 
                a.row_pos.ptr(), 
                a.col_idx.ptr(), 
                b.data.ptr(), 
                b.row_pos.ptr(), 
                b.col_idx.ptr(), 
                self.params.ptrm(),
                &mut self.epsout as *mut f64, 
                &mut self.loops_n as *mut BlasUint, 
                &self.eigval_lb as *const f64, 
                &self.eigval_ub as *const f64, 
                &self.eig_n_guess as *const BlasUint, 
                eigval.ptrm(), 
                eigvec.ptrm(), 
                &mut self.eig_n as *mut BlasUint, 
                self.residual.ptrm(), 
                &mut self.info as *mut BlasInt
            );
        }
    }
}
