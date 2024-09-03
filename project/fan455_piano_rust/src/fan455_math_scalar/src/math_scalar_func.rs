use super::math_scalar_trait::*;
use super::math_scalar_type::*;
use std::iter::zip;
use num_traits::PrimInt;


#[inline(always)]
pub fn div_round( a: f64, b: f64 ) -> usize {
    (a/b).round() as usize
}

#[inline(always)]
pub fn modulo<T: PrimInt>( a: T, b: T ) -> (T, T) {
    (a/b, a%b)
}

#[inline(always)]
pub fn cubic_poly_one_real_root( a: f64, b: f64, c: f64, d: f64 ) -> f64 {
    let p = (3.*a*c - b.powi(2)) / (3.*a.powi(2));
    let q = (27.*a.powi(2)*d - 9.*a*b*c + 2.*b.powi(3)) / (27.*a.powi(3));
    let tmp = ((q/2.).powi(2) + (p/3.).powi(3)).sqrt();
    -b/(3.*a) + (-q/2. + tmp).cbrt() + (-q/2. - tmp).cbrt()
}

#[inline(always)]
pub fn quadratic_poly( a: f64, b: f64, c: f64 ) -> [f64; 2] {
    let tmp = (b.powi(2) - 4.*a*c).sqrt();
    [0.5*(-b+tmp)/a, 0.5*(-b-tmp)/a]
}

#[inline(always)]
pub fn cross_product_2d( x1: f64, y1: f64, x2: f64, y2: f64 ) -> f64 {
    x1*y2 - x2*y1
}
    
#[inline(always)]
pub fn dot_product_2d( x1: f64, y1: f64, x2: f64, y2: f64 ) -> f64 {
    x1*x2 + y1*y2
}

#[inline(always)]
pub fn distance_2d( x1: f64, y1: f64, x2: f64, y2: f64 ) -> f64 {
    ((x2-x1).powi(2) + (y2-y1).powi(2)).sqrt()
}

#[inline(always)]
pub fn normal_vec2( a: f64, b: f64 ) -> [f64; 2] {
    // Compute the normal vector (x, y) to vector (a, b), with a positive direction (y >= 0)
    if a.is_zero_float() {
        [1., 0.]
    } else {
        let y = (1.+b.powi(2)/a.powi(2)).sqrt().recip();
        let x = -y*b/a;
        [x, y]
    }
}

#[inline(always)]
pub fn angle_of_vec2_npi_pi( x: f64, y: f64 ) -> f64 {
    y.atan2(x)
}

#[inline(always)]
pub fn norm_of_vec2( x: f64, y: f64 ) -> f64 {
    (x.powi(2)+y.powi(2)).sqrt()
}

#[inline(always)]
pub fn angle_of_vec2_0_2pi( x: f64, y: f64 ) -> f64 {
    let angle = y.atan2(x);
    match angle > 0. {
        true => angle,
        false => angle + TWO_PI
    }
}

#[inline(always)]
pub fn deg2rad( deg: f64 ) -> f64 {
    deg * PI / 180.
}

#[inline(always)]
pub fn rad2deg( rad: f64 ) -> f64 {
    rad * 180. / PI
}

#[inline(always)]
pub fn is_in_triangle(
    x: f64, y: f64, x1: f64, y1: f64, mut x2: f64, mut y2: f64, mut x3: f64, mut y3: f64
) -> bool {
    if cross_product_2d(x2-x1, y2-y1, x3-x1, y3-y1) < 0. {
        std::mem::swap(&mut x2, &mut x3);
        std::mem::swap(&mut y2, &mut y3);
    } // ensure counterclock
    if !point_is_left_to_line(x, y, x1, y1, x2, y2) {
        return false;
    } else if !point_is_left_to_line(x, y, x2, y2, x3, y3) {
        return false;
    } else if !point_is_left_to_line(x, y, x3, y3, x1, y1) {
        return false;
    } else {
        return true;
    }
}

#[inline(always)]
pub fn sort_four_points_counterclock(
    x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, x4: f64, y4: f64
) -> [usize; 4] {
    let x0 = (x1+x2+x3+x4)/4.;
    let y0 = (y1+y2+y3+y4)/4.;
    let angles = [
        (y1-y0).atan2(x1-x0),
        (y2-y0).atan2(x2-x0),
        (y3-y0).atan2(x3-x0),
        (y4-y0).atan2(x4-x0),
    ];
    let mut idx: [usize; 4] = [0, 1, 2, 3];
    idx.sort_by( |i, j| angles[*i].partial_cmp(&angles[*j]).unwrap() );
    idx
}

#[inline]
pub fn point_is_left_to_line( x: f64, y: f64, x1: f64, y1: f64, x2: f64, y2: f64 ) -> bool {
    // (x, y) is the point, (x1, y1)->(x2, y2) is the line with direction
    cross_product_2d(x2-x1, y2-y1, x-x1, y-y1) > 0.
}

#[inline(always)]
pub fn sort_vars_by_idx<T: Default+Copy, const N: usize>( x: [&mut T; N], idx: &[usize; N] ) {
    let mut s: [T; N] = [T::default(); N];
    for (x_, s_) in zip(x.iter(), s.iter_mut()) {
        *s_ = **x_;
    }
    for (x_, i_) in zip(x, idx.iter()) {
        *x_ = s[*i_];
    }
}

#[inline(always)]
pub fn is_in_quadrangle(
    x: f64, y: f64, mut x1: f64, mut y1: f64, mut x2: f64, mut y2: f64, 
    mut x3: f64, mut y3: f64, mut x4: f64, mut y4: f64
) -> bool {
    let idx = sort_four_points_counterclock(x1, y1, x2, y2, x3, y3, x4, y4);
    sort_vars_by_idx([&mut x1, &mut x2, &mut x3, &mut x4], &idx);
    sort_vars_by_idx([&mut y1, &mut y2, &mut y3, &mut y4], &idx);

    if !point_is_left_to_line(x, y, x1, y1, x2, y2) {
        return false;
    } else if !point_is_left_to_line(x, y, x2, y2, x3, y3) {
        return false;
    } else if !point_is_left_to_line(x, y, x3, y3, x4, y4) {
        return false;
    } else if !point_is_left_to_line(x, y, x4, y4, x1, y1) {
        return false;
    } else {
        return true;
    }
}

#[inline(always)]
pub fn is_in_rectangle( x: f64, y: f64, x_lb: f64, x_ub: f64, y_lb: f64, y_ub: f64 ) -> bool {
    if x < x_lb {
        return false;
    } else if x > x_ub {
        return false;
    } else if y < y_lb {
        return false;
    } else if y > y_ub {
        return false;
    } else {
        return true;
    }
}

#[inline(always)]
pub fn ensure_three_points_counterclock(
    x1: &f64, y1: &f64, x2: &mut f64, y2: &mut f64, x3: &mut f64, y3: &mut f64
) -> bool {
    // If swap happened, return true
    if cross_product_2d(*x2-x1, *y2-y1, *x3-x1, *y3-y1) < 0. {
        std::mem::swap(x2, x3);
        std::mem::swap(y2, y3);
        true
    } else {
        false
    }
}

#[inline(always)]
pub fn ensure_three_points_counterclock_by<T: Sized>(
    x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, _p1: &T, p2: &mut T, p3: &mut T,
) -> bool {
    // If swap happened, return true
    if cross_product_2d(x2-x1, y2-y1, x3-x1, y3-y1) < 0. {
        std::mem::swap(p2, p3);
        true
    } else {
        false
    }
}

#[inline(always)]
pub fn ensure_three_points_counterclock_with_index(
    x1: &f64, y1: &f64, x2: &mut f64, y2: &mut f64, x3: &mut f64, y3: &mut f64,
    _i1: &usize, i2: &mut usize, i3: &mut usize,
) -> bool {
    // If swap happened, return true
    if cross_product_2d(*x2-x1, *y2-y1, *x3-x1, *y3-y1) < 0. {
        std::mem::swap(x2, x3);
        std::mem::swap(y2, y3);
        std::mem::swap(i2, i3);
        true
    } else {
        false
    }
}

#[inline(always)]
pub fn area_of_triangle( x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64 ) -> f64 {
    0.5*((x1*y2+x2*y3+x3*y1)-(y1*x2+y2*x3+y3*x1))
}

#[inline(always)]
pub fn center_of_triangle( x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64 ) -> [f64; 2] {
    [(x1+x2+x3)/3., (y1+y2+y3)/3.]
}

#[inline(always)]
pub fn center_of_quadrangle( x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, x4: f64, y4: f64 ) -> [f64; 2] {
    [(x1+x2+x3+x4)/4., (y1+y2+y3+y4)/3.]
}

#[inline(always)]
pub fn line_eq_point_slope( x0: f64, y0: f64, k: f64 ) -> f64 {
    // y = k*x + b, return b
    y0 - k*x0
}

#[inline(always)]
pub fn line_eq_two_points( x0: f64, y0: f64, x1: f64, y1: f64 ) -> [f64; 2] {
    // y = k*x + b, return k, b
    let k = (y1 - y0) / (x1 - x0);
    let b = y0 - k*x0;
    [k, b]
}

#[inline(always)]
pub fn mid_point_2d( x1: f64, y1: f64, x2: f64, y2: f64, r: f64 ) -> [f64; 2] {
    // r is the ratio
    [(1.-r)*x1+r*x2, (1.-r)*y1+r*y2]
}

#[inline(always)]
pub fn mid_point_3d( x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64, r: f64 ) -> [f64; 3] {
    // r is the ratio
    [(1.-r)*x1+r*x2, (1.-r)*y1+r*y2, (1.-r)*z1+r*z2]
}

#[inline(always)]
pub fn intersect_of_two_lines( k0: f64, b0: f64, k1: f64, b1: f64 ) -> [f64; 2] {
    let x = (b1 - b0) / (k0 - k1);
    let y = k0 * x + b0;
    [x, y]
}

#[inline(always)]
pub fn j_mul( x: c128 ) -> c128 {
    c128 { re: -x.im, im: x.re }
}

#[inline(always)]
pub fn neg_j_mul( x: c128 ) -> c128 {
    c128 { re: x.im, im: -x.re }
}

#[inline(always)]
pub fn expj( x: f64 ) -> c128 {
    c128 { re: x.cos(), im: x.sin() }
}

#[inline(always)]
pub fn j_expj( x: f64 ) -> c128 {
    c128 { re: -x.sin(), im: x.cos() }
}

#[inline(always)]
pub fn neg_j_expj( x: f64 ) -> c128 {
    c128 { re: x.sin(), im: -x.cos() }
}


#[inline(always)]
pub fn j_mul_re( x: c128 ) -> f64 {
    -x.im
}

#[inline(always)]
pub fn neg_j_mul_re( x: c128 ) -> f64 {
    x.im
}

#[inline(always)]
pub fn expj_re( x: f64 ) -> f64 {
    x.cos()
}

#[inline(always)]
pub fn j_expj_re( x: f64 ) -> f64 {
    -x.sin()
}

#[inline(always)]
pub fn neg_j_expj_re( x: f64 ) -> f64 {
    x.sin()
}

#[inline(always)]
pub fn complex_mul_re( x: c128, y: c128 ) -> f64 {
    x.re * y.re - x.im * y.im
}


