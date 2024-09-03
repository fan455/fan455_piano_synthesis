use fan455_math_scalar::*;
use fan455_arrf64::*;
use fan455_util::{elem, mzip, assert_multi_eq};
use std::iter::zip;
use super::math_array_func::*;


pub trait PolyStaticCall<const N: usize>
{
    fn f( x: &[f64], a: &[f64; N], f: &mut [f64] );
    fn fx( x: &[f64], a: &[f64; N], f: &mut [f64] );
}


#[derive(Default)]
pub struct PolyStatic<const ORDER: usize> {}


impl PolyStaticCall<2> for PolyStatic<1>
{
    #[inline]
    fn f( x: &[f64], a: &[f64; 2], f: &mut [f64] ) {
        f.fill(0.);
        for elem!(x_, f_) in mzip!(x.iter(), f.iter_mut()) {
            *f_ += a[0] + a[1]*x_;
        }
    }

    #[inline]
    fn fx( _x: &[f64], a: &[f64; 2], f: &mut [f64] ) {
        f.fill(0.);
        for f_ in f.iter_mut() {
            *f_ += a[1];
        }
    }
}

impl PolyStaticCall<3> for PolyStatic<2>
{
    #[inline]
    fn f( x: &[f64], a: &[f64; 3], f: &mut [f64] ) {
        f.fill(0.);
        for elem!(x_, f_) in mzip!(x.iter(), f.iter_mut()) {
            *f_ += a[0] + x_ * (a[1] + a[2]*x_);
        }
    }

    #[inline]
    fn fx( x: &[f64], a: &[f64; 3], f: &mut [f64] ) {
        f.fill(0.);
        for elem!(x_, f_) in mzip!(x.iter(), f.iter_mut()) {
            *f_ += a[1] + 2.*a[2]*x_;
        }
    }
}

impl PolyStaticCall<4> for PolyStatic<3>
{
    #[inline]
    fn f( x: &[f64], a: &[f64; 4], f: &mut [f64] ) {
        f.fill(0.);
        for elem!(x_, f_) in mzip!(x.iter(), f.iter_mut()) {
            *f_ += a[0] + x_ * (a[1] + x_ *(a[2] + a[3]*x_));
        }
    }

    #[inline]
    fn fx( x: &[f64], a: &[f64; 4], f: &mut [f64] ) {
        f.fill(0.);
        for elem!(x_, f_) in mzip!(x.iter(), f.iter_mut()) {
            *f_ += a[1] + x_ * (2.*a[2] + 3.*a[3]*x_);
        }
    }
}


#[derive(Copy, Clone)]
pub enum Poly3Kind {
    Tetrahedron(usize),
    TriPrismZ([usize; 2]),
}


pub struct Poly3 {
    pub kind: Poly3Kind,

    pub n: usize,
    pub n_fx: usize,
    pub n_fy: usize,
    pub n_fz: usize,

    pub pow: Vec<[i32; 3]>, // (n,), power
    pub p_fx: Vec<(usize, f64, [i32; 3])>, // (n_fx,), index, factor, power
    pub p_fy: Vec<(usize, f64, [i32; 3])>, // (n_fy,), index, factor, power
    pub p_fz: Vec<(usize, f64, [i32; 3])>, // (n_fz,), index, factor, power
}

impl Poly3
{
    #[inline]
    pub fn new( kind: Poly3Kind ) -> Self {
        let mut pow =  Vec::<[i32; 3]>::new();
        let mut p_fx =  Vec::<(usize, f64, [i32; 3])>::new();
        let mut p_fy =  Vec::<(usize, f64, [i32; 3])>::new();
        let mut p_fz =  Vec::<(usize, f64, [i32; 3])>::new();
        {
            let mut i: usize = 0;
            macro_rules! do_in_loop {
                ($px: ident, $py: ident, $pz: ident) => {
                    pow.push([$px, $py, $pz]);
                    if $px > 0 {
                        p_fx.push( (i, $px as f64, [$px-1, $py, $pz]) );
                    }
                    if $py > 0 {
                        p_fy.push( (i, $py as f64, [$px, $py-1, $pz]) );
                    }
                    if $pz > 0 {
                        p_fz.push( (i, $pz as f64, [$px, $py, $pz-1]) );
                    }
                    i += 1;
                };
            }
            match kind {
                Poly3Kind::Tetrahedron(order) => {
                    let n1 = (order + 1) as i32;
    
                    for px in 0..n1 {
                        for py in 0..n1 {
                            for pz in 0..n1 {
                                if px + py + pz < n1 {
                                    do_in_loop!(px, py, pz);
                                }
                            }
                        }
                    }
                },
                Poly3Kind::TriPrismZ([order_xy, order_z]) => {
                    let n1 = (order_xy + 1) as i32;
                    let n2 = (order_z + 1) as i32;
        
                    for px in 0..n1 {
                        for py in 0..n1 {
                            if px + py < n1 {
                                for pz in 0..n2 {
                                    do_in_loop!(px, py, pz);
                                }
                            }
                        }
                    }
                }
            }
        }
        pow.shrink_to_fit();
        p_fx.shrink_to_fit();
        p_fy.shrink_to_fit();
        p_fz.shrink_to_fit();

        let n = pow.len();
        let n_fx = p_fx.len();
        let n_fy = p_fy.len();
        let n_fz = p_fz.len();
        
        Self { kind, n, n_fx, n_fy, n_fz, pow, p_fx, p_fy, p_fz }
    }

    #[inline]
    pub fn compute_vandermonde<MT: RMatMut<f64>>( &self, xyz: &[[f64; 3]], co: &mut MT ) {
        assert_multi_eq!(self.n, xyz.len(), co.nrow(), co.ncol());
        for elem!(i, [px, py, pz]) in mzip!(0..self.n, self.pow.iter()) {
            for elem!(co_, [x, y, z]) in mzip!(co.col_mut(i).itm(), xyz.iter()) {
                *co_ = x.powi(*px) * y.powi(*py) * z.powi(*pz);
            }
        }
    }

    #[inline]
    pub fn solve_vandermonde<MT: RMatMut<f64>>( co: &mut MT ) {
        let n = co.nrow();
        assert_eq!(n, co.ncol());
        let n_ = n as BlasUint;
        let mut ipiv: Vec<BlasUint> = vec![0; n];
        unsafe {
            LAPACKE_dgetrf(COL_MAJ, n_, n_, co.ptrm(), n_, ipiv.as_mut_ptr());
            LAPACKE_dgetri(COL_MAJ, n_, co.ptrm(), n_, ipiv.as_ptr());
        }
    }

    #[inline]
    pub fn fit<MT: RMatMut<f64>>( &self, xyz: &[[f64; 3]], co: &mut MT ) {
        self.compute_vandermonde(xyz, co);
        Self::solve_vandermonde(co);
    }

    #[inline]
    pub fn eval( xyz: &[[f64; 3]], pow: &[[i32; 3]], co: &[f64], f: &mut [f64] ) {
        f.fill(0.);
        for elem!(s, [px, py, pz]) in mzip!(co.iter(), pow.iter()) {
            for elem!(f_, [x, y, z]) in mzip!(f.iter_mut(), xyz.iter()) {
                *f_ += s * x.powi(*px) * y.powi(*py) * z.powi(*pz);
            }
        }
    }

    #[inline]
    pub fn eval_single( xyz: &[f64; 3], pow: &[[i32; 3]], co: &[f64] ) -> f64 {
        let mut f: f64 = 0.;
        let [x, y, z] = *xyz;
        for elem!(co_, [px, py, pz]) in mzip!(co.iter(), pow.iter()) {
            f += co_ * x.powi(*px) * y.powi(*py) * z.powi(*pz);
        }
        f
    }

    #[inline]
    pub fn eval_derivative( xyz: &[[f64; 3]], p: &[(usize, f64, [i32; 3])], co: &[f64], f: &mut [f64] ) {
        f.fill(0.);
        for (i, a, [px, py, pz]) in p.iter() {
            let co_ = co[*i];
            for elem!(f_, [x, y, z]) in mzip!(f.iter_mut(), xyz.iter()) {
                *f_ += co_ * a * x.powi(*px) * y.powi(*py) * z.powi(*pz);
            }
        }
    }

    #[inline]
    pub fn eval_derivative_single( xyz: &[f64; 3], p: &[(usize, f64, [i32; 3])], co: &[f64] ) -> f64 {
        let mut f: f64 = 0.;
        let [x, y, z] = *xyz;
        for (i, a, [px, py, pz]) in p.iter() {
            f += co[*i] * a * x.powi(*px) * y.powi(*py) * z.powi(*pz);
        }
        f
    }

    #[inline]
    pub fn f( &self, xyz: &[[f64; 3]], co: &[f64], f: &mut [f64] ) {
        Self::eval(xyz, &self.pow, co, f);
    }

    #[inline]
    pub fn fx( &self, xyz: &[[f64; 3]], co: &[f64], f: &mut [f64] ) {
        Self::eval_derivative(xyz, &self.p_fx, co, f);
    }

    #[inline]
    pub fn fy( &self, xyz: &[[f64; 3]], co: &[f64], f: &mut [f64] ) {
        Self::eval_derivative(xyz, &self.p_fy, co, f);
    }

    #[inline]
    pub fn fz( &self, xyz: &[[f64; 3]], co: &[f64], f: &mut [f64] ) {
        Self::eval_derivative(xyz, &self.p_fz, co, f);
    }

    #[inline]
    pub fn f_single( &self, xyz: &[f64; 3], co: &[f64] ) -> f64 {
        Self::eval_single(xyz, &self.pow, co)
    }

    #[inline]
    pub fn fx_single( &self, xyz: &[f64; 3], co: &[f64] ) -> f64 {
        Self::eval_derivative_single(xyz, &self.p_fx, co)
    }

    #[inline]
    pub fn fy_single( &self, xyz: &[f64; 3], co: &[f64] ) -> f64 {
        Self::eval_derivative_single(xyz, &self.p_fy, co)
    }

    #[inline]
    pub fn fz_single( &self, xyz: &[f64; 3], co: &[f64] ) -> f64 {
        Self::eval_derivative_single(xyz, &self.p_fz, co)
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











