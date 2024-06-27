use fan455_math_scalar::*;
use super::arrf64_basic::*;
use super::arrf64_blas_lapack::*;
use super::arrf64_sparse_mat::*;
use std::fmt::Display;



#[derive(Default)]
pub struct FeastSparse<T: Float> {
    pub params: Vec<BlasInt>,

    pub eig_n: BlasInt,
    pub loops_n: BlasInt,
    pub info: BlasInt,
    pub epsout: T,
    pub residual: Vec<T>,

    pub eigval_lb: T,
    pub eigval_ub: T,
    pub eig_n_guess: BlasInt,
}


impl<T: Float+Display> FeastSparse<T>
{
    #[inline]
    pub fn new() -> Self {
        let mut params: Vec<usize> = vec![0; 128];
        unsafe {
            feastinit(params.ptrm());
        }
        Self { params, ..Default::default() }
    }

    #[inline]
    pub fn init_guess( &mut self, emin: T, emax: T, n: BlasInt ) {
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
        self.params[0] = BlasInt::from(val);
    }

    #[inline]
    pub fn set_num_contour_points( &mut self, val: BlasInt ) {
        // Default is 8.
        self.params[1] = val;
    }

    #[inline]
    pub fn set_max_loops( &mut self, val: usize ) {
        // Default is 20.
        self.params[3] = val;
    }

    #[inline]
    pub fn set_stop_type( &mut self, val: BlasInt ) {
        // Default is 0.
        self.params[5] = val;
    }

    #[inline]
    pub fn set_sparse_mat_check( &mut self, val: bool ) {
        // Default is 0.
        self.params[26] = BlasInt::from(val);
    }

    #[inline]
    pub fn set_positive_mat_check( &mut self, val: bool ) {
        // Default is 0.
        self.params[27] = BlasInt::from(val);
    }
}


impl FeastSparse<f64>
{
    #[inline]
    pub fn set_tol( &mut self, val: BlasInt ) {
        // Default is 12.
        self.params[2] = val;
    }

    #[inline]
    pub fn call<VT: RVecMut<f64>, MT: RMatMut<f64>>( 
        &mut self, 
        matrix: &CsrMat<f64>, 
        eigval: &mut VT,
        eigvec: &mut MT,
    ) {
        unsafe {
            dfeast_scsrev(
                &LOWER as *const BlasChar, 
                &matrix.nrow as *const BlasInt, 
                matrix.data.ptr(), 
                matrix.row_pos.ptr(), 
                matrix.col_idx.ptr(), 
                self.params.ptrm(),
                &mut self.epsout as *mut f64, 
                &mut self.loops_n as *mut BlasInt, 
                &self.eigval_lb as *const f64, 
                &self.eigval_ub as *const f64, 
                &self.eig_n_guess as *const BlasInt, 
                eigval.ptrm(), 
                eigvec.ptrm(), 
                &mut self.eig_n as *mut BlasInt, 
                self.residual.ptrm(), 
                &mut self.info as *mut BlasInt
            );
        }
    }
}


impl FeastSparse<f32>
{
    #[inline]
    pub fn set_tol( &mut self, val: BlasInt ) {
        // Default is 5.
        self.params[6] = val;
    }

    #[inline]
    pub fn call<VT: RVecMut<f32>, MT: RMatMut<f32>>( 
        &mut self, 
        matrix: &CsrMat<f32>, 
        eigval: &mut VT,
        eigvec: &mut MT,
    ) {
        unsafe {
            sfeast_scsrev(
                &LOWER as *const BlasChar, 
                &matrix.nrow as *const BlasInt, 
                matrix.data.ptr(), 
                matrix.row_pos.ptr(), 
                matrix.col_idx.ptr(), 
                self.params.ptrm(),
                &mut self.epsout as *mut f32, 
                &mut self.loops_n as *mut BlasInt, 
                &self.eigval_lb as *const f32, 
                &self.eigval_ub as *const f32, 
                &self.eig_n_guess as *const BlasInt, 
                eigval.ptrm(), 
                eigvec.ptrm(), 
                &mut self.eig_n as *mut BlasInt, 
                self.residual.ptrm(), 
                &mut self.info as *mut BlasInt
            );
        }
    }
}