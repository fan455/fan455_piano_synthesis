use fan455_math_scalar::*;
//use fan455_arrf64::*;
use fan455_util::{elem, mzip};


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
pub fn poly2( coef: &[f64], pow_x: &[i32], pow_y: &[i32], x: f64, y: f64 ) -> f64 {
    let mut f: f64 = 0.;
    for elem!(a, px, py) in mzip!(coef.iter(), pow_x.iter(), pow_y.iter()) {
        f += a * x.powi(*px) * y.powi(*py);
    }
    f
}

#[inline(always)]
pub fn poly2_batch( 
    coef: &[f64], pow_x: &[i32], pow_y: &[i32], 
    x_vec: &[f64], y_vec: &[f64], f_vec: &mut [f64] 
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
    f_coef: &[f64], f_pow_x: &[i32], f_pow_y: &[i32],
    fx_coef: &mut [f64], fx_pow_x: &mut [i32], fx_pow_y: &mut [i32],
) {
    let n = f_coef.len();
    let mut i: usize = 0;
    for elem!(j, px) in mzip!(0..n, f_pow_x.iter()) {
        if *px > 0 {
            let px_ = *px as f64;
            fx_coef[i] = f_coef[j] * px_;
            fx_pow_x[i] = f_pow_x[j] - 1;
            fx_pow_y[i] = f_pow_y[j];
            i += 1;
        }
    }
}

#[inline(always)]
pub fn poly2_fy_coef_pow(
    f_coef: &[f64], f_pow_x: &[i32], f_pow_y: &[i32],
    fy_coef: &mut [f64], fy_pow_x: &mut [i32], fy_pow_y: &mut [i32],
) {
    let n = f_coef.len();
    let mut i: usize = 0;
    for elem!(j, py) in mzip!(0..n, f_pow_y.iter()) {
        if *py > 0 {
            let py_ = *py as f64;
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
    f_coef: &[f64], fx_coef: &mut [f64], fx_pow_x: &[i32], fx_idx: &[usize]
) {
    // fx_pow_x should have been computed.
    for elem!(fx_coef_, px, i) in mzip!(fx_coef.iter_mut(), fx_pow_x.iter(), fx_idx.iter()) {
        *fx_coef_ = f_coef[*i] * (1. + *px as f64);
    }
}

#[inline(always)]
pub fn poly2_fy_coef(
    f_coef: &[f64], fy_coef: &mut [f64], fy_pow_y: &[i32], fy_idx: &[usize]
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
pub fn src_to_dst_idx<T: Copy+Clone>( src: &[T], dst: &mut [T], idx: &[usize] ) {
    for elem!(s, i) in mzip!(src.iter(), idx.iter()) {
        dst[*i] = *s;
    }
}

#[inline(always)]
pub fn src_idx_to_dst<T: Copy+Clone>( src: &[T], dst: &mut [T], idx: &[usize] ) {
    for elem!(s, i) in mzip!(dst.iter_mut(), idx.iter()) {
        *s = src[*i];
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
    beg_x: f64, beg_y: f64, end_x: f64, end_y: f64, 
    mid_x: &mut [f64], mid_y: &mut [f64], r: f64
) {
    // In x direction.
    let i_ub = mid_x.len()+1;
    for elem!(i, x, y) in mzip!(1..i_ub, mid_x.iter_mut(), mid_y.iter_mut()) {
        [*x, *y] = point_between_two(beg_x, beg_y, end_x, end_y, r*i as f64);
    }
}

#[inline(always)]
pub fn mat_vec_2( a: &[f64; 4], x: &[f64; 2] ) -> [f64; 2] {
    [ x[0]*a[0]+x[1]*a[2], x[0]*a[1]+x[1]*a[3] ]
}

#[inline(always)]
pub fn mat_mat_2( a: &[f64; 4], b: &[f64; 4] ) -> [f64; 4] {
    [ b[0]*a[0]+b[1]*a[2], b[0]*a[1]+b[1]*a[3], b[2]*a[0]+b[3]*a[2], b[2]*a[1]+b[3]*a[3] ]
}

#[inline(always)]
pub fn mat_det_2( a: &[f64; 4] ) -> f64 {
    a[0]*a[3] - a[1]*a[2]
}

#[inline(always)]
pub fn mat_inv_2( a: &[f64; 4] ) -> [f64; 4] {
    let det = a[0]*a[3] - a[1]*a[2];
    [a[3]/det, -a[1]/det, -a[2]/det, a[0]/det]
}

#[inline(always)]
pub fn mat_det_inv_2( a: &[f64; 4] ) -> (f64, [f64; 4]) {
    let det = a[0]*a[3] - a[1]*a[2];
    (det, [a[3]/det, -a[1]/det, -a[2]/det, a[0]/det])
}









