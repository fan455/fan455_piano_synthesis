use super::arrf64_basic::*;
use super::arrf64_dyn_special::PackMat;
use fan455_math_scalar::{c64, c128};

pub type BlasInt = isize; // or i64
pub type BlasUint = usize; // or u64, should not be negative, and should not exceed isize::MAX
pub type BlasChar = i8;

pub const NO_TRANS: i8 = 'N' as i8;
pub const TRANS: i8 = 'T' as i8;
pub const CONJ_TRANS: i8 = 'C' as i8;
pub const FULL: i8 = 'F' as i8;
pub const UPPER: i8 = 'U' as i8;
pub const LOWER: i8 = 'L' as i8;
pub const UNIT: i8 = 'U' as i8;
pub const NON_UNIT: i8 = 'N' as i8;
pub const LEFT: i8 = 'L' as i8;
pub const RIGHT: i8 = 'R' as i8;
pub const ROW_MAJ: BlasUint = 101;
pub const COL_MAJ: BlasUint = 102;


extern "C" {

pub fn dnrm2_( n: *const BlasUint, x: *const f64, incx: *const BlasUint ) -> f64;

pub fn ddot_( n: *const BlasUint, x: *const f64, incx: *const BlasUint, y: *const f64, incy: *const BlasUint ) -> f64;

pub fn daxpy_( n: *const BlasUint, a: *const f64, x: *const f64, incx: *const BlasUint, y: *mut f64, incy: *const BlasUint ); 

pub fn daxpby_( n: *const BlasUint, a: *const f64, x: *const f64, incx: *const BlasUint, b: *const f64, y: *mut f64, incy: *const BlasUint );

pub fn dgemv_( trans: *const BlasChar, m: *const BlasUint, n: *const BlasUint, alpha: *const f64, a: *const f64, lda: *const BlasUint, x: *const f64, incx: *const BlasUint, beta: *const f64, y: *mut f64, incy: *const BlasUint );

pub fn dsymv_( uplo: *const BlasChar, n: *const BlasUint, alpha: *const f64, a: *const f64, lda: *const BlasUint, x: *const f64, incx: *const BlasUint, beta: *const f64, y: *mut f64, incy: *const BlasUint );

pub fn dtrmv_( uplo: *const BlasChar, trans: *const BlasChar, diag: *const BlasChar, n: *const BlasUint, a: *const f64, lda: *const BlasUint, x: *mut f64, incx: *const BlasUint );

pub fn dger_( m: *const BlasUint, n: *const BlasUint, alpha: *const f64, x: *const f64, incx: *const BlasUint, y: *const f64, incy: *const BlasUint, a: *mut f64, lda: *const BlasUint );

pub fn dsyr_( uplo: *const BlasChar, n: *const BlasUint, alpha: *const f64, x: *const f64, incx: *const BlasUint, a: *mut f64, lda: *const BlasUint );

pub fn dsymm_( side: *const BlasChar, uplo: *const BlasChar, m: *const BlasUint, n: *const BlasUint, alpha: *const f64, a: *const f64, lda: *const BlasUint, b: *const f64, ldb: *const BlasUint, beta: *const f64, c: *mut f64, ldc: *const BlasUint );

pub fn dgemm_( transa: *const BlasChar, transb: *const BlasChar, m: *const BlasUint, n: *const BlasUint, k: *const BlasUint, alpha: *const f64, a: *const f64, lda: *const BlasUint, b: *const f64, ldb: *const BlasUint, beta: *const f64, c: *mut f64, ldc: *const BlasUint );

pub fn dsyrk_( uplo: *const BlasChar, trans: *const BlasChar, n: *const BlasUint, k: *const BlasUint, alpha: *const f64, a: *const f64, lda: *const BlasUint, beta: *const f64, c: *mut f64, ldc: *const BlasUint );

pub fn dtrmm_( side: *const BlasChar, uplo: *const BlasChar, transa: *const BlasChar, diag: *const BlasChar, m: *const BlasUint, n: *const BlasUint, alpha: *const f64, a: *const f64, lda: *const BlasUint, b: *mut f64, ldb: *const BlasUint );

pub fn dtrsm_( side: *const BlasChar, uplo: *const BlasChar, transa: *const BlasChar, diag: *const BlasChar, m: *const BlasUint, n: *const BlasUint, alpha: *const f64, a: *const f64, lda: *const BlasUint, b: *mut f64, ldb: *const BlasUint );

pub fn dpotrf_( uplo: *const BlasChar, n: *const BlasUint, a: *mut f64, lda: *const BlasUint, info: *mut BlasInt );

pub fn dpotri_( uplo: *const BlasChar, n: *const BlasUint, a: *mut f64, lda: *const BlasUint, info: *mut BlasInt );

pub fn LAPACKE_sgetrf( matrix_layout: BlasUint, m: BlasUint, n: BlasUint, a: *mut f32, lda: BlasUint, ipiv: *mut BlasUint ) -> BlasUint;
pub fn LAPACKE_dgetrf( matrix_layout: BlasUint, m: BlasUint, n: BlasUint, a: *mut f64, lda: BlasUint, ipiv: *mut BlasUint ) -> BlasUint;

pub fn LAPACKE_sgetri( matrix_layout: BlasUint, n: BlasUint, a: *mut f32, lda: BlasUint, ipiv: *const BlasUint ) -> BlasUint;
pub fn LAPACKE_dgetri( matrix_layout: BlasUint, n: BlasUint, a: *mut f64, lda: BlasUint, ipiv: *const BlasUint ) -> BlasUint;

pub fn dposv_( uplo: *const BlasChar, n: *const BlasUint, nrhs: *const BlasUint, a: *mut f64, lda: *const BlasUint, b: *mut f64, ldb: *const BlasUint, info: *mut BlasInt );

pub fn dgesv_( n: *const BlasUint, nrhs: *const BlasUint, a: *mut f64, lda: *const BlasUint, ipiv: *mut BlasUint, b: *mut f64, ldb: *const BlasUint, info: *mut BlasInt );

pub fn idamax_( n: *const BlasUint, x: *const f64, incx: *const BlasUint ) -> BlasUint;

pub fn LAPACKE_ssygvd( matrix_layout: BlasUint, itype: BlasUint, jobz: BlasChar, uplo: BlasChar, n: BlasUint, a: *mut f32, lda: BlasUint, b: *mut f32, ldb: BlasUint, w: *mut f32 ) -> BlasUint;

pub fn LAPACKE_dsygvd( matrix_layout: BlasUint, itype: BlasUint, jobz: BlasChar, uplo: BlasChar, n: BlasUint, a: *mut f64, lda: BlasUint, b: *mut f64, ldb: BlasUint, w: *mut f64 ) -> BlasUint;

// Below are routines for complex type.
pub fn cgemv_( trans: *const BlasChar, m: *const BlasUint, n: *const BlasUint, alpha: *const c64, a: *const c64, lda: *const BlasUint, x: *const c64, incx: *const BlasUint, beta: *const c64, y: *mut c64, incy: *const BlasUint );
pub fn zgemv_( trans: *const BlasChar, m: *const BlasUint, n: *const BlasUint, alpha: *const c128, a: *const c128, lda: *const BlasUint, x: *const c128, incx: *const BlasUint, beta: *const c128, y: *mut c128, incy: *const BlasUint );

pub fn LAPACKE_chegvd( matrix_layout: BlasUint, itype: BlasUint, jobz: BlasChar, uplo: BlasChar, n: BlasUint, a: *mut c64, lda: BlasUint, b: *mut c64, ldb: BlasUint, w: *mut f32 ) -> BlasUint;
pub fn LAPACKE_zhegvd( matrix_layout: BlasUint, itype: BlasUint, jobz: BlasChar, uplo: BlasChar, n: BlasUint, a: *mut c128, lda: BlasUint, b: *mut c128, ldb: BlasUint, w: *mut f64 ) -> BlasUint;

pub fn LAPACKE_chpgvd( matrix_layout: BlasUint, itype: BlasUint, jobz: BlasChar, uplo: BlasChar, n: BlasUint, ap: *mut c64, bp: *mut c64, w: *mut f32, z: *mut c64, ldz: BlasUint ) -> BlasUint;
pub fn LAPACKE_zhpgvd( matrix_layout: BlasUint, itype: BlasUint, jobz: BlasChar, uplo: BlasChar, n: BlasUint, ap: *mut c128, bp: *mut c128, w: *mut f64, z: *mut c128, ldz: BlasUint ) -> BlasUint;

pub fn LAPACKE_zhpgvx( matrix_layout: BlasUint, itype: BlasUint, jobz: BlasChar, range: BlasChar, uplo: BlasChar, n: BlasUint, ap: *mut c128, bp: *mut c128, vl: f64, vu: f64, il: BlasUint, iu: BlasUint, abstol: f64, m: *mut BlasUint, w: *mut f64, z: *mut c128, ldz: BlasUint, ifail: *mut BlasUint ) -> BlasUint;

pub fn LAPACKE_cgehrd( matrix_layout: BlasUint, n: BlasUint, ilo: BlasUint, ihi: BlasUint, a: *mut c64, lda: BlasUint, tau: *mut c64 ) -> BlasUint;
pub fn LAPACKE_zgehrd( matrix_layout: BlasUint, n: BlasUint, ilo: BlasUint, ihi: BlasUint, a: *mut c128, lda: BlasUint, tau: *mut c128 ) -> BlasUint;

pub fn LAPACKE_zheevr( matrix_layout: BlasUint, jobz: BlasChar, range: BlasChar, uplo: BlasChar, n: BlasUint, a: *mut c128, lda: BlasUint, vl: f64, vu: f64, il: BlasUint, iu: BlasUint, abstol: f64, m: *mut BlasUint, w: *mut f64, z: *mut c128, ldz: BlasUint, isuppz: *mut BlasUint ) -> BlasUint;

pub fn cpotrf_( uplo: *const BlasChar, n: *const BlasUint, a: *mut c64, lda: *const BlasUint, info: *mut BlasInt );
pub fn zpotrf_( uplo: *const BlasChar, n: *const BlasUint, a: *mut c128, lda: *const BlasUint, info: *mut BlasInt );

pub fn cpotri_( uplo: *const BlasChar, n: *const BlasUint, a: *mut c64, lda: *const BlasUint, info: *mut BlasInt );
pub fn zpotri_( uplo: *const BlasChar, n: *const BlasUint, a: *mut c128, lda: *const BlasUint, info: *mut BlasInt );

pub fn cpptrf_( uplo: *const BlasChar, n: *const BlasUint, ap: *mut c64, info: *mut BlasInt );
pub fn zpptrf_( uplo: *const BlasChar, n: *const BlasUint, ap: *mut c128, info: *mut BlasInt );

pub fn ctptri_( uplo: *const BlasChar, diag: *const BlasChar, n: *const BlasUint, ap: *mut c64, info: *mut BlasInt );
pub fn ztptri_( uplo: *const BlasChar, diag: *const BlasChar, n: *const BlasUint, ap: *mut c128, info: *mut BlasInt );

pub fn chpgst_( itype: *const BlasUint, uplo: *const BlasChar, n: *const BlasUint, ap: *mut c64, bp: *const c64, info: *mut BlasInt );
pub fn zhpgst_( itype: *const BlasUint, uplo: *const BlasChar, n: *const BlasUint, ap: *mut c128, bp: *const c128, info: *mut BlasInt );

pub fn chptrd_( uplo: *const BlasChar, n: *const BlasUint, ap: *mut c64, d: *mut f32, e: *mut f32, tau: *mut c64, info: *mut BlasInt );
pub fn zhptrd_( uplo: *const BlasChar, n: *const BlasUint, ap: *mut c128, d: *mut f64, e: *mut f64, tau: *mut c128, info: *mut BlasInt );

// MKL eigensolver
pub fn pardisoinit( pt: *mut BlasUint, mtype: *const BlasUint, iparm: *mut BlasUint );
pub fn feastinit(fpm: *mut BlasUint);

pub fn dfeast_scsrev( uplo: *const BlasChar, n: *const BlasUint, a: *const f64, ia: *const BlasUint, ja: *const BlasUint, fpm: *mut BlasUint, epsout: *mut f64, loop_: *mut BlasUint, emin: *const f64, emax: *const f64, m0: *const BlasUint, e: *mut f64, x: *mut f64, m: *mut BlasUint, res: *mut f64, info: *mut BlasInt);

pub fn dfeast_scsrgv( uplo: *const BlasChar, n: *const BlasUint, a: *const f64, ia: *const BlasUint, ja: *const BlasUint, b: *const f64, ib: *const BlasUint, jb: *const BlasUint, fpm: *mut BlasUint, epsout: *mut f64, loop_: *mut BlasUint, emin: *const f64, emax: *const f64, m0: *const BlasUint, e: *mut f64, x: *mut f64, m: *mut BlasUint, res: *mut f64, info: *mut BlasInt);

}


#[inline]
pub fn ssygvd<VT: RVecMut<f32>, MT1: RMatMut<f32>, MT2: RMatMut<f32>>(
    a: &mut MT1,
    b: &mut MT2,
    eigval: &mut VT,
    uplo: BlasChar
) {
    unsafe { LAPACKE_ssygvd(
        COL_MAJ, 1, 'V' as i8, uplo, a.nrow(), a.ptrm(), a.stride(), b.ptrm(), 
        b.stride(), eigval.ptrm()
    ); }
}


#[inline]
pub fn dsygvd<VT: RVecMut<f64>, MT1: RMatMut<f64>, MT2: RMatMut<f64>>(
    a: &mut MT1,
    b: &mut MT2,
    eigval: &mut VT,
    uplo: BlasChar
) {
    unsafe { LAPACKE_dsygvd(
        COL_MAJ, 1, 'V' as i8, uplo, a.nrow(), a.ptrm(), a.stride(), b.ptrm(), 
        b.stride(), eigval.ptrm()
    ); }
}


#[inline]
pub fn cgemv<VT1: CVec<f32>, VT2: CVecMut<f32>, MT: CMat<f32>>( 
    alpha: c64, 
    a: &MT, 
    x: &VT1, 
    beta: c64, 
    y: &mut VT2, 
    trans: BlasChar 
) {
    let m: BlasUint = a.nrow();
    let n: BlasUint = a.ncol();
    let lda: BlasUint = a.stride();
    let incx: BlasUint = 1;
    let incy: BlasUint = 1;
    unsafe { cgemv_(
        &trans as *const BlasChar,
        &m as *const BlasUint,
        &n as *const BlasUint,
        &alpha as *const c64,
        a.ptr(),
        &lda as *const BlasUint,
        x.ptr(),
        &incx as *const BlasUint,
        &beta as *const c64,
        y.ptrm(),
        &incy as *const BlasUint
    ); }
}


#[inline]
pub fn zgemv<VT1: CVec<f64>, VT2: CVecMut<f64>, MT: CMat<f64>>( 
    alpha: c128, 
    a: &MT, 
    x: &VT1, 
    beta: c128, 
    y: &mut VT2, 
    trans: BlasChar 
) {
    let m: BlasUint = a.nrow();
    let n: BlasUint = a.ncol();
    let lda: BlasUint = a.stride();
    let incx: BlasUint = 1;
    let incy: BlasUint = 1;
    unsafe { zgemv_(
        &trans as *const BlasChar,
        &m as *const BlasUint,
        &n as *const BlasUint,
        &alpha as *const c128,
        a.ptr(),
        &lda as *const BlasUint,
        x.ptr(),
        &incx as *const BlasUint,
        &beta as *const c128,
        y.ptrm(),
        &incy as *const BlasUint
    ); }
}


#[inline]
pub fn chegvd<VT: RVecMut<f32>, MT1: CMatMut<f32>, MT2: CMatMut<f32>>(
    a: &mut MT1,
    b: &mut MT2,
    eigval: &mut VT,
    uplo: BlasChar
) {
    unsafe { LAPACKE_chegvd(
        COL_MAJ, 1, 'V' as i8, uplo, a.nrow(), a.ptrm(), a.stride(), b.ptrm(), 
        b.stride(), eigval.ptrm()
    ); }
}


#[inline]
pub fn zhegvd<VT: RVecMut<f64>, MT1: CMatMut<f64>, MT2: CMatMut<f64>>(
    a: &mut MT1,
    b: &mut MT2,
    eigval: &mut VT,
    uplo: BlasChar
) {
    unsafe { LAPACKE_zhegvd(
        COL_MAJ, 1, 'V' as i8, uplo, a.nrow(), a.ptrm(), a.stride(), b.ptrm(), 
        b.stride(), eigval.ptrm()
    ); }
}


#[inline]
pub fn chpgvd<VT: RVecMut<f32>, MT: CMatMut<f32>>(
    a: &mut PackMat<c64>,
    b: &mut PackMat<c64>,
    eigval: &mut VT,
    eigvec: &mut MT,
    uplo: BlasChar
) {
    unsafe { LAPACKE_chpgvd(
        COL_MAJ, 1, 'V' as i8, uplo, a.nrow(), a.ptrm(), b.ptrm(), 
        eigval.ptrm(), eigvec.ptrm(), eigvec.stride()
    ); }
}


#[inline]
pub fn zhpgvd<VT: RVecMut<f64>, MT: CMatMut<f64>>(
    a: &mut PackMat<c128>,
    b: &mut PackMat<c128>,
    eigval: &mut VT,
    eigvec: &mut MT,
    uplo: BlasChar
) {
    unsafe { LAPACKE_zhpgvd(
        COL_MAJ, 1, 'V' as i8, uplo, a.nrow(), a.ptrm(), b.ptrm(), 
        eigval.ptrm(), eigvec.ptrm(), eigvec.stride()
    ); }
}


#[inline]
pub fn cgehrd<MT1: CMatMut<f32>, MT2: CMatMut<f32>>(
    a: &mut MT1,
    tau: &mut MT2
) {
    let n: BlasUint = a.nrow();
    let lda: BlasUint = a.stride();
    unsafe { LAPACKE_cgehrd(
        COL_MAJ, n, 1, n, a.ptrm(), lda, tau.ptrm()
    ); }
}


#[inline]
pub fn zgehrd<MT1: CMatMut<f64>, MT2: CMatMut<f64>>(
    a: &mut MT1,
    tau: &mut MT2
) {
    let n: BlasUint = a.nrow();
    let lda: BlasUint = a.stride();
    unsafe { LAPACKE_zgehrd(
        COL_MAJ, n, 1, n, a.ptrm(), lda, tau.ptrm()
    ); }
}


#[inline]
pub fn cpotri<MT: CMatMut<f32>>(
    a: &mut MT,
    uplo: BlasChar
) {
    let n: BlasUint = a.nrow();
    let lda: BlasUint = a.stride();
    let mut info: BlasInt = 0;
    unsafe { cpotri_(
        &uplo as *const BlasChar,
        &n as *const BlasUint,
        a.ptrm(),
        &lda as *const BlasUint,
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn zpotri<MT: CMatMut<f64>>(
    a: &mut MT,
    uplo: BlasChar
) {
    let n: BlasUint = a.nrow();
    let lda: BlasUint = a.stride();
    let mut info: BlasInt = 0;
    unsafe { zpotri_(
        &uplo as *const BlasChar,
        &n as *const BlasUint,
        a.ptrm(),
        &lda as *const BlasUint,
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn cpotrf<MT: CMatMut<f32>>(
    a: &mut MT,
    uplo: BlasChar
) {
    let n: BlasUint = a.nrow();
    let lda: BlasUint = a.stride();
    let mut info: BlasInt = 0;
    unsafe { cpotrf_(
        &uplo as *const BlasChar,
        &n as *const BlasUint,
        a.ptrm(),
        &lda as *const BlasUint,
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn zpotrf<MT: CMatMut<f64>>(
    a: &mut MT,
    uplo: BlasChar
) {
    let n: BlasUint = a.nrow();
    let lda: BlasUint = a.stride();
    let mut info: BlasInt = 0;
    unsafe { zpotrf_(
        &uplo as *const BlasChar,
        &n as *const BlasUint,
        a.ptrm(),
        &lda as *const BlasUint,
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn chptrd<VT1: RVecMut<f32>, VT2: RVecMut<f32>, VT3: CVecMut<f32>>(
    a: &mut PackMat<c64>,
    d: &mut VT1,
    e: &mut VT2,
    tau: &mut VT3,
    uplo: BlasChar,
) {
    let n: BlasUint = a.nrow();
    let mut info: BlasInt = 0;
    unsafe { chptrd_(
        &uplo as *const BlasChar,
        &n as *const BlasUint,
        a.ptrm(),
        d.ptrm(),
        e.ptrm(),
        tau.ptrm(),
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn zhptrd<VT1: RVecMut<f64>, VT2: RVecMut<f64>, VT3: CVecMut<f64>>(
    a: &mut PackMat<c128>,
    d: &mut VT1,
    e: &mut VT2,
    tau: &mut VT3,
    uplo: BlasChar,
) {
    let n: BlasUint = a.nrow();
    let mut info: BlasInt = 0;
    unsafe { zhptrd_(
        &uplo as *const BlasChar,
        &n as *const BlasUint,
        a.ptrm(),
        d.ptrm(),
        e.ptrm(),
        tau.ptrm(),
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn chpgst(
    a: &mut PackMat<c64>,
    b: &PackMat<c64>,
    uplo: BlasChar,
    itype: BlasUint
) {
    let n: BlasUint = a.nrow();
    let mut info: BlasInt = 0;
    unsafe { chpgst_(
        &itype as *const BlasUint,
        &uplo as *const BlasChar,
        &n as *const BlasUint,
        a.ptrm(),
        b.ptr(),
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn zhpgst(
    a: &mut PackMat<c128>,
    b: &PackMat<c128>,
    uplo: BlasChar,
    itype: BlasUint
) {
    let n: BlasUint = a.nrow();
    let mut info: BlasInt = 0;
    unsafe { zhpgst_(
        &itype as *const BlasUint,
        &uplo as *const BlasChar,
        &n as *const BlasUint,
        a.ptrm(),
        b.ptr(),
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn ctptri(
    a: &mut PackMat<c64>,
    uplo: BlasChar,
    diag: BlasChar
) {
    let n: BlasUint = a.nrow();
    let mut info: BlasInt = 0;
    unsafe { ctptri_(
        &uplo as *const BlasChar,
        &diag as *const BlasChar,
        &n as *const BlasUint,
        a.ptrm(),
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn ztptri(
    a: &mut PackMat<c128>,
    uplo: BlasChar,
    diag: BlasChar
) {
    let n: BlasUint = a.nrow();
    let mut info: BlasInt = 0;
    unsafe { ztptri_(
        &uplo as *const BlasChar,
        &diag as *const BlasChar,
        &n as *const BlasUint,
        a.ptrm(),
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn cpptrf(
    a: &mut PackMat<c64>,
    uplo: BlasChar
) {
    let n: BlasUint = a.nrow();
    let mut info: BlasInt = 0;
    unsafe { cpptrf_(
        &uplo as *const BlasChar,
        &n as *const BlasUint,
        a.ptrm(),
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn zpptrf(
    a: &mut PackMat<c128>,
    uplo: BlasChar
) {
    let n: BlasUint = a.nrow();
    let mut info: BlasInt = 0;
    unsafe { zpptrf_(
        &uplo as *const BlasChar,
        &n as *const BlasUint,
        a.ptrm(),
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn idamax<VT: RVec<f64>>(
    x: &VT
) -> usize {
    let n: BlasUint = x.size() as BlasUint;
    let incx: BlasUint = 1;
    unsafe { (idamax_(
        &n as *const BlasUint,
        x.ptr(),
        &incx as *const BlasUint
    ) - 1) as usize }
}


#[inline]
pub fn dgesv<MT1: RMatMut<f64>, MT2: RMatMut<f64>>(
    a: &mut MT1, b: &mut MT2
) {
    let n: BlasUint = a.nrow();
    let nrhs: BlasUint = b.ncol();
    let lda: BlasUint = a.stride();
    let ldb: BlasUint = b.stride();
    let mut info: BlasInt = 0;
    let mut ipiv: Vec<BlasUint> = vec![0; a.nrow()];
    unsafe { dgesv_(
        &n as *const BlasUint,
        &nrhs as *const BlasUint,
        a.ptrm(),
        &lda as *const BlasUint,
        ipiv.as_mut_ptr(),
        b.ptrm(),
        &ldb as *const BlasUint,
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn dposv<MT1: RMatMut<f64>, MT2: RMatMut<f64>>(
    a: &mut MT1,
    b: &mut MT2,
    uplo: BlasChar
) {
    let n: BlasUint = a.nrow();
    let nrhs: BlasUint = b.ncol();
    let lda: BlasUint = a.stride();
    let ldb: BlasUint = b.stride();
    let mut info: BlasInt = 0;
    unsafe { dposv_(
        &uplo as *const BlasChar,
        &n as *const BlasUint,
        &nrhs as *const BlasUint,
        a.ptrm(),
        &lda as *const BlasUint,
        b.ptrm(),
        &ldb as *const BlasUint,
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn dpotri<MT: RMatMut<f64>>(
    a: &mut MT,
    uplo: BlasChar
) {
    let n: BlasUint = a.nrow();
    let lda: BlasUint = a.stride();
    let mut info: BlasInt = 0;
    unsafe { dpotri_(
        &uplo as *const BlasChar,
        &n as *const BlasUint,
        a.ptrm(),
        &lda as *const BlasUint,
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn dpotrf<MT: RMatMut<f64>>(
    a: &mut MT,
    uplo: BlasChar
) {
    let n: BlasUint = a.nrow();
    let lda: BlasUint = a.stride();
    let mut info: BlasInt = 0;
    unsafe { dpotrf_(
        &uplo as *const BlasChar,
        &n as *const BlasUint,
        a.ptrm(),
        &lda as *const BlasUint,
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn dtrmm<MT1: RMat<f64>, MT2: RMatMut<f64>>( 
    alpha: f64,   
    a: &MT1,
    b: &mut MT2,
    side: BlasChar,
    trans: BlasChar,
    uplo: BlasChar,
    diag: BlasChar
) {
    let m: BlasUint = b.nrow();
    let n: BlasUint = b.ncol();
    let lda: BlasUint = a.stride();
    let ldb: BlasUint = b.stride();
    unsafe { dtrmm_(
        &side as *const BlasChar,
        &uplo as *const BlasChar,
        &trans as *const BlasChar,
        &diag as *const BlasChar,
        &m as *const BlasUint,
        &n as *const BlasUint,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasUint,
        b.ptrm(),
        &ldb as *const BlasUint,
    ); }
}


#[inline]
pub fn dtrsm<MT1: RMat<f64>, MT2: RMatMut<f64>>( 
    alpha: f64,   
    a: &MT1,
    b: &mut MT2,
    side: BlasChar,
    trans: BlasChar,
    uplo: BlasChar,
    diag: BlasChar
) {
    let m: BlasUint = b.nrow();
    let n: BlasUint = b.ncol();
    let lda: BlasUint = a.stride();
    let ldb: BlasUint = b.stride();
    unsafe { dtrsm_(
        &side as *const BlasChar,
        &uplo as *const BlasChar,
        &trans as *const BlasChar,
        &diag as *const BlasChar,
        &m as *const BlasUint,
        &n as *const BlasUint,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasUint,
        b.ptrm(),
        &ldb as *const BlasUint,
    ); }
}


#[inline]
pub fn dsyrk<MT1: RMat<f64>, MT2: RMatMut<f64>>( 
    alpha: f64,   
    a: &MT1,
    beta: f64, 
    c: &mut MT2,
    trans: BlasChar,
    uplo: BlasChar
) {
    let n: BlasUint = c.nrow();
    let k: BlasUint = match trans {
        NO_TRANS => a.ncol(),
        TRANS => a.nrow(),
        _ => panic!("In DSYRK, trans is incorrect"),
    }; // difference here
    let lda: BlasUint = a.stride();
    let ldc: BlasUint = c.stride();
    unsafe { dsyrk_(
        &uplo as *const BlasChar,
        &trans as *const BlasChar,
        &n as *const BlasUint,
        &k as *const BlasUint,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasUint,
        &beta as *const f64,
        c.ptrm(),
        &ldc as *const BlasUint,
    ); }
}


#[inline]
pub fn dgemm<MT1: RMat<f64>, MT2: RMat<f64>, MT3: RMatMut<f64>>( 
    alpha: f64,   
    a: &MT1,
    b: &MT2,
    beta: f64, 
    c: &mut MT3,
    transa: BlasChar,
    transb: BlasChar
) {
    let m: BlasUint = c.nrow();
    let n: BlasUint = c.ncol();
    let k: BlasUint = match transa {
        NO_TRANS => a.ncol(),
        TRANS => a.nrow(),
        _ => panic!("In GEMM, transa is incorrect"),
    }; // difference here
    let lda: BlasUint = a.stride();
    let ldb: BlasUint = b.stride();
    let ldc: BlasUint = c.stride();
    unsafe { dgemm_(
        &transa as *const BlasChar,
        &transb as *const BlasChar,
        &m as *const BlasUint,
        &n as *const BlasUint,
        &k as *const BlasUint,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasUint,
        b.ptr(),
        &ldb as *const BlasUint,
        &beta as *const f64,
        c.ptrm(),
        &ldc as *const BlasUint,
    ); }
}


#[inline]
pub fn dsymm<MT1: RMat<f64>, MT2: RMat<f64>, MT3: RMatMut<f64>>( 
    alpha: f64,   
    a: &MT1,
    b: &MT2,
    beta: f64, 
    c: &mut MT3,
    side: BlasChar,
    uplo: BlasChar
) {
    let m: BlasUint = c.nrow();
    let n: BlasUint = c.ncol();
    let lda: BlasUint = a.stride();
    let ldb: BlasUint = b.stride();
    let ldc: BlasUint = c.stride();
    unsafe { dsymm_(
        &side as *const BlasChar,
        &uplo as *const BlasChar,
        &m as *const BlasUint,
        &n as *const BlasUint,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasUint,
        b.ptr(),
        &ldb as *const BlasUint,
        &beta as *const f64,
        c.ptrm(),
        &ldc as *const BlasUint,
    ); }
}


#[inline]
pub fn dsyr<VT: RVec<f64>, MT: RMatMut<f64>>( 
    alpha: f64, 
    x: &VT,  
    a: &mut MT,
    uplo: BlasChar
) {
    let n: BlasUint = a.nrow();
    let lda: BlasUint = a.stride();
    let incx: BlasUint = 1;
    unsafe { dsyr_(
        &uplo as *const BlasChar,
        &n as *const BlasUint,
        &alpha as *const f64,
        x.ptr(),
        &incx as *const BlasUint,
        a.ptrm(),
        &lda as *const BlasUint
    ); }
}


#[inline]
pub fn dger<VT1: RVec<f64>, VT2: RVec<f64>, MT: RMatMut<f64>>( 
    alpha: f64, 
    x: &VT1,  
    y: &VT2,
    a: &mut MT
) {
    let m: BlasUint = a.nrow();
    let n: BlasUint = a.ncol();
    let lda: BlasUint = a.stride();
    let incx: BlasUint = 1;
    let incy: BlasUint = 1;
    unsafe { dger_(
        &m as *const BlasUint,
        &n as *const BlasUint,
        &alpha as *const f64,
        x.ptr(),
        &incx as *const BlasUint,
        y.ptr(),
        &incy as *const BlasUint,
        a.ptrm(),
        &lda as *const BlasUint
    ); }
}


#[inline]
pub fn dtrmv<VT: RVecMut<f64>, MT: RMat<f64>>( 
    a: &MT, 
    x: &mut VT, 
    trans: BlasChar,
    uplo: BlasChar,
    diag: BlasChar
) {
    let n: BlasUint = a.nrow();
    let lda: BlasUint = a.stride();
    let incx: BlasUint = 1;
    unsafe { dtrmv_(
        &uplo as *const BlasChar,
        &trans as *const BlasChar,
        &diag as *const BlasChar,
        &n as *const BlasUint,
        a.ptr(),
        &lda as *const BlasUint,
        x.ptrm(),
        &incx as *const BlasUint
    ); }
}


#[inline]
pub fn ddot<VT1: RVec<f64>, VT2: RVec<f64>>( 
    x: &VT1, 
    y: &VT2 
) -> f64 {
    let n: BlasUint = x.size() as BlasUint;
    let incx: BlasUint = 1;
    let incy: BlasUint = 1;
    unsafe { ddot_(
        &n as *const BlasUint, 
        x.ptr(),
        &incx as *const BlasUint,
        y.ptr(),
        &incy as *const BlasUint
    ) }
}

#[inline]
pub fn dnrm2<VT: RVec<f64>>( 
    x: &VT, 
) -> f64 {
    let n: BlasUint = x.size() as BlasUint;
    let incx: BlasUint = 1;
    unsafe { dnrm2_(
        &n as *const BlasUint, 
        x.ptr(),
        &incx as *const BlasUint,
    ) }
}

#[inline]
pub fn daxpy<VT1: RVec<f64>, VT2: RVecMut<f64>>( 
    a: f64, 
    x: &VT1, 
    y: &mut VT2
) {
    let n: BlasUint = x.size() as BlasUint;
    let incx: BlasUint = 1;
    let incy: BlasUint = 1;
    unsafe { daxpy_(
        &n as *const BlasUint, 
        &a as *const f64,
        x.ptr(),
        &incx as *const BlasUint,
        y.ptrm(),
        &incy as *const BlasUint
    ); }
}

#[inline]
pub fn daxpby<VT1: RVec<f64>, VT2: RVecMut<f64>>( 
    a: f64, 
    x: &VT1,
    b: f64,
    y: &mut VT2
) {
    let n: BlasUint = x.size() as BlasUint;
    let incx: BlasUint = 1;
    let incy: BlasUint = 1;
    unsafe { daxpby_(
        &n as *const BlasUint, 
        &a as *const f64,
        x.ptr(),
        &incx as *const BlasUint,
        &b as *const f64,
        y.ptrm(),
        &incy as *const BlasUint
    ); }
}

#[inline]
pub fn dgemv<VT1: RVec<f64>, VT2: RVecMut<f64>, MT: RMat<f64>>( 
    alpha: f64, 
    a: &MT, 
    x: &VT1, 
    beta: f64, 
    y: &mut VT2, 
    trans: BlasChar 
) {
    let m: BlasUint = a.nrow();
    let n: BlasUint = a.ncol();
    let lda: BlasUint = a.stride();
    let incx: BlasUint = 1;
    let incy: BlasUint = 1;
    unsafe { dgemv_(
        &trans as *const BlasChar,
        &m as *const BlasUint,
        &n as *const BlasUint,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasUint,
        x.ptr(),
        &incx as *const BlasUint,
        &beta as *const f64,
        y.ptrm(),
        &incy as *const BlasUint
    ); }
}


#[inline]
pub fn dsymv<VT1: RVec<f64>, VT2: RVecMut<f64>, MT: RMat<f64>>( 
    alpha: f64, 
    a: &MT, 
    x: &VT1, 
    beta: f64, 
    y: &mut VT2,
    uplo: BlasChar
) {
    let n: BlasUint = a.nrow();
    let lda: BlasUint = a.stride();
    let incx: BlasUint = 1;
    let incy: BlasUint = 1;
    unsafe { dsymv_(
        &uplo as *const BlasChar,
        &n as *const BlasUint,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasUint,
        x.ptr(),
        &incx as *const BlasUint,
        &beta as *const f64,
        y.ptrm(),
        &incy as *const BlasUint
    ); }
}

