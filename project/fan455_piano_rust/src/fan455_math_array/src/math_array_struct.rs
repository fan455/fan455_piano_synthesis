use fan455_math_scalar::*;
use fan455_arrf64::*;
use fan455_util::{elem, mzip, assert_multi_eq};
use std::iter::zip;

use crate::{eval_poly2_tr_fxy_size, eval_poly2_tr_fxx_size, eval_poly2_tr_fx_size, eval_poly2_tr_size};

use super::math_array_func::*;


#[derive(Default)]
pub struct Poly2 {
    pub kind: u8,
    pub order: usize,

    pub n: usize,
    pub n_fx: usize,
    pub n_fy: usize,
    pub n_fxx: usize,
    pub n_fyy: usize,
    pub n_fxy: usize,

    pub idx_fx: Vec<usize>,
    pub idx_fy: Vec<usize>,
    pub idx_fxx: Vec<usize>,
    pub idx_fyy: Vec<usize>,
    pub idx_fxy: Vec<usize>,

    pub pow: Vec<[i32; 2]>,
    pub pow_fx: Vec<[i32; 2]>,
    pub pow_fy: Vec<[i32; 2]>,
    pub pow_fxx: Vec<[i32; 2]>,
    pub pow_fyy: Vec<[i32; 2]>,
    pub pow_fxy: Vec<[i32; 2]>,

    pub fct_fx: Vec<f64>,
    pub fct_fy: Vec<f64>,
    pub fct_fxx: Vec<f64>,
    pub fct_fyy: Vec<f64>,
    pub fct_fxy: Vec<f64>,
}

impl Poly2
{
    pub const TRIANGLE: u8 = 0;

    #[inline]
    pub fn new_tr( order: usize ) -> Self {
        let n = eval_poly2_tr_size!(order);
        let n_fx = eval_poly2_tr_fx_size!(order, n);
        let n_fy = n_fx;
        let n_fxx = eval_poly2_tr_fxx_size!(order, n);
        let n_fyy = n_fxx;
        let n_fxy = eval_poly2_tr_fxy_size!(order, n);

        let pow = {
            let mut p =  Vec::<[i32; 2]>::with_capacity(n);
            let nx = (order + 1) as i32;
            for px in 0..nx {
                for py in 0..nx {
                    if px + py < nx {
                        p.push([px, py]);
                    }
                }
            }
            p
        };
        let (idx_fx, pow_fx, fct_fx) = {
            let mut s = Vec::<usize>::with_capacity(n_fx);
            let mut p = Vec::<[i32; 2]>::with_capacity(n_fx);
            let mut c = Vec::<f64>::with_capacity(n_fx);
            for elem!(i, pow_) in mzip!(0..n, pow.iter()) {
                let [px, py] = *pow_;
                if px > 0 {
                    s.push(i);
                    c.push(px as f64);
                    p.push([px-1, py]);
                }
            }
            assert_multi_eq!(n_fx, s.len(), c.len(), p.len());
            (s, p, c)
        };
        let (idx_fy, pow_fy, fct_fy) = {
            let mut s = Vec::<usize>::with_capacity(n_fy);
            let mut p = Vec::<[i32; 2]>::with_capacity(n_fy);
            let mut c = Vec::<f64>::with_capacity(n_fy);
            for elem!(i, pow_) in mzip!(0..n, pow.iter()) {
                let [px, py] = *pow_;
                if py > 0 {
                    s.push(i);
                    c.push(py as f64);
                    p.push([px, py-1]);
                }
            }
            assert_multi_eq!(n_fy, s.len(), c.len(), p.len());
            (s, p, c)
        };
        let (idx_fxx, pow_fxx, fct_fxx) = {
            let mut s = Vec::<usize>::with_capacity(n_fxx);
            let mut p = Vec::<[i32; 2]>::with_capacity(n_fxx);
            let mut c = Vec::<f64>::with_capacity(n_fxx);
            for elem!(i, pow_) in mzip!(0..n, pow.iter()) {
                let [px, py] = *pow_;
                if px > 1 {
                    s.push(i);
                    c.push((px*(px-1)) as f64);
                    p.push([px-2, py]);
                }
            }
            assert_multi_eq!(n_fxx, s.len(), c.len(), p.len());
            (s, p, c)
        };
        let (idx_fyy, pow_fyy, fct_fyy) = {
            let mut s = Vec::<usize>::with_capacity(n_fyy);
            let mut p = Vec::<[i32; 2]>::with_capacity(n_fyy);
            let mut c = Vec::<f64>::with_capacity(n_fyy);
            for elem!(i, pow_) in mzip!(0..n, pow.iter()) {
                let [px, py] = *pow_;
                if py > 1 {
                    s.push(i);
                    c.push((py*(py-1)) as f64);
                    p.push([px, py-2]);
                }
            }
            assert_multi_eq!(n_fyy, s.len(), c.len(), p.len());
            (s, p, c)
        };
        let (idx_fxy, pow_fxy, fct_fxy) = {
            let mut s = Vec::<usize>::with_capacity(n_fxy);
            let mut p = Vec::<[i32; 2]>::with_capacity(n_fxy);
            let mut c = Vec::<f64>::with_capacity(n_fxy);
            for elem!(i, pow_) in mzip!(0..n, pow.iter()) {
                let [px, py] = *pow_;
                if px > 0 && py > 0 {
                    s.push(i);
                    c.push((px*py) as f64);
                    p.push([px-1, py-1]);
                }
            }
            assert_multi_eq!(n_fxy, s.len(), c.len(), p.len());
            (s, p, c)
        };
        Self { kind: Self::TRIANGLE, order, n, n_fx, n_fxx, n_fxy, n_fy, n_fyy, idx_fx, idx_fxx, idx_fxy, idx_fy, idx_fyy, pow, pow_fx, pow_fxx, pow_fxy, pow_fy, pow_fyy, fct_fx, fct_fxx, fct_fxy, fct_fy, fct_fyy }
    }

    #[inline]
    pub fn compute_vandermonde<MT: RMatMut<f64>>( &self, xy: &[[f64; 2]], co: &mut MT ) {
        assert_multi_eq!(self.n, xy.len(), co.nrow(), co.ncol());
        for elem!(i, [px, py]) in mzip!(0..self.n, self.pow.iter()) {
            for elem!(co_, [x, y]) in mzip!(co.col_mut(i).itm(), xy.iter()) {
                *co_ = x.powi(*px) * y.powi(*py);
            }
        }
    }

    #[inline]
    pub fn solve_vandermonde<MT: RMatMut<f64>>( co: &mut MT ) {
        let n = co.nrow();
        assert_eq!(n, co.ncol());
        let n_ = n as BlasInt;
        let mut ipiv: Vec<BlasInt> = vec![0; n];
        unsafe {
            #[cfg(feature="use_32bit_float")] {
                LAPACKE_sgetrf(COL_MAJ, n_, n_, co.ptrm(), n_, ipiv.as_mut_ptr());
                LAPACKE_sgetri(COL_MAJ, n_, co.ptrm(), n_, ipiv.as_ptr());
            }
            #[cfg(not(feature="use_32bit_float"))] {
                LAPACKE_dgetrf(COL_MAJ, n_, n_, co.ptrm(), n_, ipiv.as_mut_ptr());
                LAPACKE_dgetri(COL_MAJ, n_, co.ptrm(), n_, ipiv.as_ptr());
            }
        }
    }

    #[inline]
    pub fn fit<MT: RMatMut<f64>>( &self, xy: &[[f64; 2]], co: &mut MT ) {
        self.compute_vandermonde(xy, co);
        Self::solve_vandermonde(co);
    }

    #[inline]
    pub fn eval( xy: &[[f64; 2]], pow: &[[i32; 2]], co: &[f64], f: &mut [f64] ) {
        f.fill(0.);
        for elem!(s, [px, py]) in mzip!(co.iter(), pow.iter()) {
            for elem!(f_, [x, y]) in mzip!(f.iter_mut(), xy.iter()) {
                *f_ += s * x.powi(*px) * y.powi(*py);
            }
        }
    }

    #[inline]
    pub fn eval_single( xy: &[f64; 2], pow: &[[i32; 2]], co: &[f64] ) -> f64 {
        let mut f: f64 = 0.;
        let [x, y] = *xy;
        for elem!(s, [px, py]) in mzip!(co.iter(), pow.iter()) {
            f += s * x.powi(*px) * y.powi(*py);
        }
        f
    }

    #[inline]
    pub fn eval_with( xy: &[[f64; 2]], idx: &[usize], fct: &[f64], pow: &[[i32; 2]], co: &[f64], f: &mut [f64] ) {
        f.fill(0.);
        for elem!(i, c, [px, py]) in mzip!(idx.iter(), fct.iter(), pow.iter()) {
            let s = co[*i];
            for elem!(f_, [x, y]) in mzip!(f.iter_mut(), xy.iter()) {
                *f_ += s * c * x.powi(*px) * y.powi(*py);
            }
        }
    }

    #[inline]
    pub fn eval_co( idx: &[usize], fct: &[f64], co: &[f64] ) -> Vec<f64> {
        let mut s = Vec::<f64>::with_capacity(idx.len());
        for elem!(i, c) in mzip!(idx.iter(), fct.iter()) {
            s.push(co[*i]*c);
        }
        s
    }

    #[inline]
    pub fn f( &self, xy: &[[f64; 2]], co: &[f64], f: &mut [f64] ) {
        Self::eval(xy, &self.pow, co, f);
    }

    #[inline]
    pub fn f_single( &self, xy: &[f64; 2], co: &[f64] ) -> f64 {
        Self::eval_single(xy, &self.pow, co)
    }

    // Methods with "_pre" means using pre-computed poly coefficients.
    #[inline]
    pub fn fx_pre( &self, xy: &[[f64; 2]], co: &[f64], f: &mut [f64] ) {
        Self::eval(xy, &self.pow_fx, co, f);
    }

    #[inline]
    pub fn fy_pre( &self, xy: &[[f64; 2]], co: &[f64], f: &mut [f64] ) {
        Self::eval(xy, &self.pow_fy, co, f);
    }

    #[inline]
    pub fn fxx_pre( &self, xy: &[[f64; 2]], co: &[f64], f: &mut [f64] ) {
        Self::eval(xy, &self.pow_fxx, co, f);
    }

    #[inline]
    pub fn fyy_pre( &self, xy: &[[f64; 2]], co: &[f64], f: &mut [f64] ) {
        Self::eval(xy, &self.pow_fyy, co, f);
    }

    #[inline]
    pub fn fxy_pre( &self, xy: &[[f64; 2]], co: &[f64], f: &mut [f64] ) {
        Self::eval(xy, &self.pow_fxy, co, f);
    }

    // Methods without "_pre" means to compute poly coefficients of derivatives at each call.
    #[inline]
    pub fn fx( &self, xy: &[[f64; 2]], co: &[f64], f: &mut [f64] ) {
        Self::eval_with(xy, &self.idx_fx, &self.fct_fx, &self.pow_fx, co, f);
    }

    #[inline]
    pub fn fy( &self, xy: &[[f64; 2]], co: &[f64], f: &mut [f64] ) {
        Self::eval_with(xy, &self.idx_fy, &self.fct_fy, &self.pow_fy, co, f);
    }

    #[inline]
    pub fn fxx( &self, xy: &[[f64; 2]], co: &[f64], f: &mut [f64] ) {
        Self::eval_with(xy, &self.idx_fxx, &self.fct_fxx, &self.pow_fxx, co, f);
    }

    #[inline]
    pub fn fyy( &self, xy: &[[f64; 2]], co: &[f64], f: &mut [f64] ) {
        Self::eval_with(xy, &self.idx_fyy, &self.fct_fyy, &self.pow_fyy, co, f);
    }

    #[inline]
    pub fn fxy( &self, xy: &[[f64; 2]], co: &[f64], f: &mut [f64] ) {
        Self::eval_with(xy, &self.idx_fxy, &self.fct_fxy, &self.pow_fxy, co, f);
    }
}


pub struct EqSolver {
    pub max_iter: usize,
    pub tol: f64,
}

impl EqSolver
{
    #[inline]
    pub fn new_default() -> Self {
        Self { max_iter: 1000, tol: 1e-4 }
    }

    #[inline]
    pub fn new( max_iter: usize, tol: f64 ) -> Self {
        Self { max_iter, tol }
    }

    #[inline(always)]
    pub fn solve_1d<F: Fn(f64)->[f64; 2]>( 
        &self, call: F, x0: f64 
    ) -> Result<(f64, usize), (f64, f64)> {
        // "call" computes zero order and first order.
        let mut x: f64 = x0;
        let [mut f, mut fx] = call(x);
        let mut total_iter: usize = usize::MAX;
        for i in 1..self.max_iter+1 {
            x -= f / fx;
            [f, fx] = call(x);
            if f.abs() < self.tol {
                total_iter = i;
                break;
            }
        }
        match total_iter == usize::MAX {
            false => Ok((x, total_iter)),
            true => Err((x, f)),
        }
    }

    #[inline(always)]
    pub fn solve_2d<F: Fn([f64; 2])->([f64; 2], [f64; 4])>( 
        &self, call: F, p0: [f64; 2] 
    ) -> Result<([f64; 2], usize), ([f64; 2], [f64; 2])> {
        // "call" computes zero order and first order.
        let mut p = p0;
        let (mut f, mut jac) = call(p);
        let mut total_iter: usize = usize::MAX;
        for i in 1..self.max_iter+1 {
            p.subassign(&mat_vec_2(&mat_inv_2(&jac), &f));
            (f, jac) = call(p);
            if f.norm() < self.tol {
                total_iter = i;
                break;
            }
        }
        match total_iter == usize::MAX {
            false => Ok((p, total_iter)),
            true => Err((p, f)),
        }
    }
}



#[derive(Default, Clone, Copy)]
pub struct CoordSysShift {
    pub shift_x: f64, // + is right, - is left
    pub shift_y: f64, // + is up, - is low
}


impl CoordSysShift
{
    #[inline]
    pub fn new_default() -> Self {
        Self { ..Default::default() }
    }


    #[inline]
    pub fn new( shift_x: f64, shift_y: f64 ) -> Self {
        Self { shift_x, shift_y }
    }

    #[inline]
    pub fn set( &mut self, shift_x: f64, shift_y: f64 ) {
        self.shift_x = shift_x;
        self.shift_y = shift_y;
    }


    #[inline]
    pub fn shift( &self, x: &mut f64, y: &mut f64 ) {
        *x -= self.shift_x;
        *y -= self.shift_y;
    }


    #[inline]
    pub fn batch_shift( &self, xy_arr: &mut [[f64; 2]] )
    {
        for [x, y] in xy_arr {
            *x -= self.shift_x;
            *y -= self.shift_y;
        }
    }
}


#[derive(Default, Clone, Copy)]
pub struct CoordSysRotation {
    pub angle: f64, // rad
    pub direction: usize, // 0 is clockwise, 1 is anticlock
    theta: f64, // Treated as clockwise
    sin_theta: f64,
    cos_theta: f64,
}


impl CoordSysRotation
{
    pub const CLOCKWISE: usize = 0;
    pub const COUNTERCLOCK: usize = 1;

    #[inline]
    pub fn new_default() -> Self {
        Self { ..Default::default() }
    }


    #[inline]
    pub fn new( angle: f64, direction: usize ) -> Self {
        let theta = match direction {
            0 => angle,
            1 => 2.*PI - angle,
            _ => panic!("Parameter {{direction}} should be 0 or 1."),
        };
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        Self { angle, direction, theta, sin_theta, cos_theta }
    }

    #[inline]
    pub fn set( &mut self, angle: f64, direction: usize ) {
        self.angle = angle;
        self.direction = direction;
        self.theta  = match direction {
            0 => angle,
            1 => 2.*PI - angle,
            _ => panic!("Parameter {{direction}} should be 0 or 1."),
        };
        self.sin_theta = self.theta.sin();
        self.cos_theta = self.theta.cos();
    }


    #[inline]
    pub fn rotate( &self, x: f64, y: f64 ) -> [f64; 2] {
        [x * self.cos_theta - y * self.sin_theta, x * self.sin_theta + y * self.cos_theta]
    }


    #[inline]
    pub fn rot_mat( &self ) -> [f64; 4] {
        // 2*2 linear transform matrix in column major.
        [self.cos_theta, self.sin_theta, -self.sin_theta, self.cos_theta]
    }

    #[inline]
    pub fn rot_jac( &self ) -> [f64; 4] {
        // jacobian matrix of rotation.
        self.rot_mat()
    }

    #[inline]
    pub fn rot_jac_inv( &self ) -> [f64; 4] {
        // inverse jacobian matrix of rotation.
        mat_inv_2(&self.rot_jac())
    }

    #[inline]
    pub fn batch_rotate( &self, xy_arr: &mut [[f64; 2]] )
    {
        for [x, y] in xy_arr {
            let x_old = *x;
            let y_old = *y;
            *x = x_old * self.cos_theta - y_old * self.sin_theta;
            *y = x_old * self.sin_theta + y_old * self.cos_theta;
        }
    }


    #[inline]
    pub fn rotate_vec<VT1, VT2>( &self, x: &mut VT1, y: &mut VT2 )
    where VT1: RVecMut<f64>, VT2: RVecMut<f64>
    {
        for (x_, y_) in zip(x.itm(), y.itm()) {
            let x_old = *x_;
            let y_old = *y_;
            *x_ = x_old * self.cos_theta - y_old * self.sin_theta;
            *y_ = x_old * self.sin_theta + y_old * self.cos_theta;
        }
    }
}











