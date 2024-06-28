use fan455_math_scalar::*;
use fan455_arrf64::*;
use fan455_util::{elem, mzip};
use std::iter::zip;


#[macro_export]
macro_rules! eval_poly2_tr_size {
    ($order:expr) => {
        {
            let mut a: usize = $order;
            let mut b: usize = a + 1;
            while a != 0 {
                b += a;
                a -= 1;
            }
            b
        }
    };
}

#[macro_export]
macro_rules! eval_poly2_tr_fx_size {
    ($order:expr, $n:expr) => {
        $n - $order - 1
    };
}

#[macro_export]
macro_rules! eval_poly2_tr_fxx_size {
    ($order:expr, $n:expr) => {
        $n - 2*$order - 1
    };
}

#[macro_export]
macro_rules! eval_poly2_tr_fxy_size {
    ($order:expr, $n:expr) => {
        $n - 2*$order - 1
    };
}

#[macro_export]
macro_rules! eval_poly2_sq_size {
    ($order:expr) => {
        ($order+1)*($order+1)
    };
}

#[macro_export]
macro_rules! eval_poly2_sq_fx_size {
    ($order:expr, $n:expr) => {
        $n - $order - 1
    };
}

#[macro_export]
macro_rules! eval_poly2_sq_fxx_size {
    ($order:expr, $n:expr) => {
        $n - 2*$order - 2
    };
}

#[macro_export]
macro_rules! eval_poly2_sq_fxy_size {
    ($order:expr, $n:expr) => {
        $n - 2*$order - 1
    };
}

#[inline(always)]
pub fn poly2( coef: &[fsize], pow_x: &[i32], pow_y: &[i32], x: fsize, y: fsize ) -> fsize {
    let mut f: fsize = 0.;
    for elem!(a, px, py) in mzip!(coef.iter(), pow_x.iter(), pow_y.iter()) {
        f += a * x.powi(*px) * y.powi(*py);
    }
    f
}

#[inline(always)]
pub fn poly2_batch( 
    coef: &[fsize], pow_x: &[i32], pow_y: &[i32], 
    x_vec: &[fsize], y_vec: &[fsize], f_vec: &mut [fsize] 
) {
    f_vec.fill(0.);
    for elem!(a, px, py) in mzip!(coef.iter(), pow_x.iter(), pow_y.iter()) {
        for elem!(x, y, f) in mzip!(x_vec.iter(), y_vec.iter(), f_vec.iter_mut()) {
            *f += a * x.powi(*px) * y.powi(*py);
        } 
    }
}

/*#[inline(always)]
pub fn poly2_fx_coef_pow(
    f_coef: &[fsize], f_pow_x: &[i32], f_pow_y: &[i32],
    fx_coef: &mut [fsize], fx_pow_x: &mut [i32], fx_pow_y: &mut [i32],
) {
    let n = f_coef.len();
    let mut i: usize = 0;
    for elem!(j, px) in mzip!(0..n, f_pow_x.iter()) {
        if *px > 0 {
            let px_ = *px as fsize;
            fx_coef[i] = f_coef[j] * px_;
            fx_pow_x[i] = f_pow_x[j] - 1;
            fx_pow_y[i] = f_pow_y[j];
            i += 1;
        }
    }
}

#[inline(always)]
pub fn poly2_fy_coef_pow(
    f_coef: &[fsize], f_pow_x: &[i32], f_pow_y: &[i32],
    fy_coef: &mut [fsize], fy_pow_x: &mut [i32], fy_pow_y: &mut [i32],
) {
    let n = f_coef.len();
    let mut i: usize = 0;
    for elem!(j, py) in mzip!(0..n, f_pow_y.iter()) {
        if *py > 0 {
            let py_ = *py as fsize;
            fy_coef[i] = f_coef[j] * py_;
            fy_pow_x[i] = f_pow_x[j];
            fy_pow_y[i] = f_pow_y[j] - 1;
            i += 1;
        }
    }
}*/

#[inline(always)]
pub fn poly2_fx_idx(
    f_pow_x: &[i32], fx_idx: &mut [usize],
) {
    let n = f_pow_x.len();
    let mut it = fx_idx.iter_mut();
    for elem!(j, px) in mzip!(0..n, f_pow_x.iter()) {
        if *px > 0 {
            *it.next().unwrap() = j;
        }
    }
}

#[inline(always)]
pub fn poly2_fy_idx(
    f_pow_y: &[i32], fy_idx: &mut [usize],
) {
    poly2_fx_idx(f_pow_y, fy_idx);
}

#[inline(always)]
pub fn poly2_fx_pow(
    f_pow_x: &[i32], f_pow_y: &[i32], 
    fx_pow_x: &mut [i32], fx_pow_y: &mut [i32], fx_idx: &[usize],
) {
    for elem!(qx, qy, i) in mzip!(fx_pow_x.iter_mut(), fx_pow_y.iter_mut(), fx_idx.iter()) {
        *qx = f_pow_x[*i] - 1;
        *qy = f_pow_y[*i];
    }
}

#[inline(always)]
pub fn poly2_fy_pow(
    f_pow_x: &[i32], f_pow_y: &[i32], 
    fy_pow_x: &mut [i32], fy_pow_y: &mut [i32], fy_idx: &[usize],
) {
    poly2_fx_pow(f_pow_y, f_pow_x, fy_pow_y, fy_pow_x, fy_idx);
}

#[inline(always)]
pub fn poly2_fx_coef(
    f_coef: &[fsize], fx_coef: &mut [fsize], fx_pow_x: &[i32], fx_idx: &[usize]
) {
    // fx_pow_x should have been computed.
    for elem!(fx_coef_, px, i) in mzip!(fx_coef.iter_mut(), fx_pow_x.iter(), fx_idx.iter()) {
        *fx_coef_ = f_coef[*i] * (1. + *px as fsize);
    }
}

#[inline(always)]
pub fn poly2_fy_coef(
    f_coef: &[fsize], fy_coef: &mut [fsize], fy_pow_y: &[i32], fy_idx: &[usize]
) {
    poly2_fx_coef(f_coef, fy_coef, fy_pow_y, fy_idx);
}

#[inline(always)]
pub fn reorder_vec<T: Copy+Clone>( vec: &mut [T], idx: &[usize], buf: &mut [T] ) {
    // idx is the reordering of old indices.
    buf.copy_from_slice(vec);
    for elem!(s, i) in mzip!(vec.iter_mut(), idx.iter()) {
        *s = buf[*i];
    }
}

#[inline(always)]
pub fn index_vec<T: Copy+Clone>( src: &[T], idx: &[usize], dst: &mut [T] ) {
    // idx is the reordering of old indices.
    for elem!(s, i) in mzip!(dst.iter_mut(), idx.iter()) {
        *s = src[*i];
    }
}

#[inline(always)]
pub fn index_vec2_unbind<T: Copy+Clone>( src: &[[T; 2]], idx: &[usize], dst_0: &mut [T], dst_1: &mut [T] ) {
    // idx is the reordering of old indices.
    for elem!(i, dst_0_, dst_1_) in mzip!(idx.iter(), dst_0.iter_mut(), dst_1.iter_mut()) {
        let [src_0_, src_1_] = src[*i];
        *dst_0_ = src_0_;
        *dst_1_ = src_1_;
    }
}

#[inline(always)]
pub fn index_vec_unchecked<T: Copy+Clone>( src: &[T], idx: &[usize], dst: &mut [T] ) {
    // idx is the reordering of old indices.
    for elem!(s, i) in mzip!(dst.iter_mut(), idx.iter()) {
        *s = unsafe {*src.get_unchecked(*i)};
    }
}

#[inline(always)]
pub fn index_vec2_unbind_unchecked<T: Copy+Clone>( src: &[[T; 2]], idx: &[usize], dst_0: &mut [T], dst_1: &mut [T] ) {
    // idx is the reordering of old indices.
    for elem!(i, dst_0_, dst_1_) in mzip!(idx.iter(), dst_0.iter_mut(), dst_1.iter_mut()) {
        let [src_0_, src_1_] = unsafe {*src.get_unchecked(*i)};
        *dst_0_ = src_0_;
        *dst_1_ = src_1_;
    }
}

#[inline(always)]
pub fn equal_points_between_two(
    beg_x: fsize, beg_y: fsize, end_x: fsize, end_y: fsize, 
    mid_x: &mut [fsize], mid_y: &mut [fsize], r: fsize
) {
    // In x direction.
    let i_ub = mid_x.len()+1;
    for elem!(i, x, y) in mzip!(1..i_ub, mid_x.iter_mut(), mid_y.iter_mut()) {
        [*x, *y] = point_between_two(beg_x, beg_y, end_x, end_y, r*i as fsize);
    }
}

/*#[inline(always)]
pub fn matmul_2by2( a: &[fsize; 4], b: &[fsize; 4], c: &mut [fsize; 4] ) {

}*/


#[derive(Default)]
pub struct CoordSysShift {
    pub shift_x: fsize, // + is right, - is left
    pub shift_y: fsize, // + is up, - is low
}


impl CoordSysShift
{
    #[inline]
    pub fn new_default() -> Self {
        Self { ..Default::default() }
    }


    #[inline]
    pub fn new( shift_x: fsize, shift_y: fsize ) -> Self {
        Self { shift_x, shift_y }
    }

    #[inline]
    pub fn set( &mut self, shift_x: fsize, shift_y: fsize ) {
        self.shift_x = shift_x;
        self.shift_y = shift_y;
    }


    #[inline]
    pub fn shift( &self, x: &mut fsize, y: &mut fsize ) {
        *x -= self.shift_x;
        *y -= self.shift_y;
    }


    #[inline]
    pub fn batch_shift( &self, xy_arr: &mut [[fsize; 2]] )
    {
        for [x, y] in xy_arr {
            *x -= self.shift_x;
            *y -= self.shift_y;
        }
    }
}


#[derive(Default)]
pub struct CoordSysRotation {
    pub angle: fsize, // rad
    pub direction: usize, // 0 is clockwise, 1 is anticlock
    theta: fsize, // Treated as clockwise
    sin_theta: fsize,
    cos_theta: fsize,
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
    pub fn new( angle: fsize, direction: usize ) -> Self {
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
    pub fn set( &mut self, angle: fsize, direction: usize ) {
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
    pub fn rotate( &self, x: &mut fsize, y: &mut fsize ) {
        let x_old = *x;
        let y_old = *y;
        *x = x_old * self.cos_theta - y_old * self.sin_theta;
        *y = x_old * self.sin_theta + y_old * self.cos_theta;
    }


    #[inline]
    pub fn get_trans_mat( &self ) -> [fsize; 4] {
        // 2*2 linear transform matrix in column major.
        [self.cos_theta, self.sin_theta, -self.sin_theta, self.cos_theta]
    }


    #[inline]
    pub fn batch_rotate( &self, xy_arr: &mut [[fsize; 2]] )
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
    where VT1: RVecMut<fsize>, VT2: RVecMut<fsize>
    {
        for (x_, y_) in zip(x.itm(), y.itm()) {
            let x_old = *x_;
            let y_old = *y_;
            *x_ = x_old * self.cos_theta - y_old * self.sin_theta;
            *y_ = x_old * self.sin_theta + y_old * self.cos_theta;
        }
    }
}











