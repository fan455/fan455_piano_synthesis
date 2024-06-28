use super::math_scalar_type::{fsize, csize};
use std::ops::Fn;


#[inline(always)]
pub fn cubic_poly_one_real_root( a: fsize, b: fsize, c: fsize, d: fsize ) -> fsize {
    let p = (3.*a*c - b.powi(2)) / (3.*a.powi(2));
    let q = (27.*a.powi(2)*d - 9.*a*b*c + 2.*b.powi(3)) / (27.*a.powi(3));
    let tmp = ((q/2.).powi(2) + (p/3.).powi(3)).sqrt();
    -b/(3.*a) + (-q/2. + tmp).cbrt() + (-q/2. - tmp).cbrt()
}

#[inline(always)]
pub fn quadratic_poly( a: fsize, b: fsize, c: fsize ) -> [fsize; 2] {
    let tmp = (b.powi(2) - 4.*a*c).sqrt();
    [0.5*(-b+tmp)/a, 0.5*(-b-tmp)/a]
}

#[inline(always)]
pub fn cross_product_2d( x1: fsize, y1: fsize, x2: fsize, y2: fsize ) -> fsize {
    x1*y2 - x2*y1
}
    
#[inline(always)]
pub fn dot_product_2d( x1: fsize, y1: fsize, x2: fsize, y2: fsize ) -> fsize {
    x1*x2 + y1*y2
}

#[inline(always)]
pub fn is_in_triangle(
    x: fsize, y: fsize, x1: fsize, y1: fsize, mut x2: fsize, mut y2: fsize, mut x3: fsize, mut y3: fsize
) -> bool {
    if cross_product_2d(x2-x1, y2-y1, x3-x1, y3-y1) < 0. {
        std::mem::swap(&mut x2, &mut x3);
        std::mem::swap(&mut y2, &mut y3);
    }
    if cross_product_2d(x2-x1, y2-y1, x-x1, y-y1) < 0. {
        return false;
    } else if cross_product_2d(x3-x2, y3-y2, x-x2, y-y2) < 0. {
        return false;
    } else if cross_product_2d(x1-x3, y1-y3, x-x3, y-y3) < 0. {
        return false;
    }
    true
}

#[inline(always)]
pub fn ensure_three_points_counterclock(
    x1: &fsize, y1: &fsize, x2: &mut fsize, y2: &mut fsize, x3: &mut fsize, y3: &mut fsize
) {
    if cross_product_2d(*x2-x1, *y2-y1, *x3-x1, *y3-y1) < 0. {
        std::mem::swap(x2, x3);
        std::mem::swap(y2, y3);
    }
}

#[inline(always)]
pub fn ensure_three_points_counterclock_with_index(
    x1: &fsize, y1: &fsize, x2: &mut fsize, y2: &mut fsize, x3: &mut fsize, y3: &mut fsize,
    _i1: &usize, i2: &mut usize, i3: &mut usize,
) {
    if cross_product_2d(*x2-x1, *y2-y1, *x3-x1, *y3-y1) < 0. {
        std::mem::swap(x2, x3);
        std::mem::swap(y2, y3);
        std::mem::swap(i2, i3);
    }
}

#[inline(always)]
pub fn area_of_triangle( x1: fsize, y1: fsize, x2: fsize, y2: fsize, x3: fsize, y3: fsize ) -> fsize {
    0.5*((x1*y2+x2*y3+x3*y1)-(y1*x2+y2*x3+y3*x1))
}

#[inline(always)]
pub fn center_of_triangle( x1: fsize, y1: fsize, x2: fsize, y2: fsize, x3: fsize, y3: fsize ) -> [fsize; 2] {
    [(x1+x2+x3)/3., (y1+y2+y3)/3.]
}

#[inline(always)]
pub fn center_of_quadrangle( x1: fsize, y1: fsize, x2: fsize, y2: fsize, x3: fsize, y3: fsize, x4: fsize, y4: fsize ) -> [fsize; 2] {
    [(x1+x2+x3+x4)/4., (y1+y2+y3+y4)/3.]
}

#[inline(always)]
pub fn line_eq_point_slope( x0: fsize, y0: fsize, k: fsize ) -> fsize {
    // y = k*x + b, return b
    y0 - k*x0
}

#[inline(always)]
pub fn line_eq_two_points( x0: fsize, y0: fsize, x1: fsize, y1: fsize ) -> [fsize; 2] {
    // y = k*x + b, return k, b
    let k = (y1 - y0) / (x1 - x0);
    let b = y0 - k*x0;
    [k, b]
}

#[inline(always)]
pub fn point_between_two( x1: fsize, y1: fsize, x2: fsize, y2: fsize, r: fsize ) -> [fsize; 2] {
    // r is the ratio
    [(1.-r)*x1+r*x2, (1.-r)*y1+r*y2]
}

#[inline(always)]
pub fn intersect_of_two_lines( k0: fsize, b0: fsize, k1: fsize, b1: fsize ) -> [fsize; 2] {
    let x = (b1 - b0) / (k0 - k1);
    let y = k0 * x + b0;
    [x, y]
}

#[inline(always)]
pub fn j_mul( x: csize ) -> csize {
    csize { re: -x.im, im: x.re }
}

#[inline(always)]
pub fn neg_j_mul( x: csize ) -> csize {
    csize { re: x.im, im: -x.re }
}

#[inline(always)]
pub fn expj( x: fsize ) -> csize {
    csize { re: x.cos(), im: x.sin() }
}

#[inline(always)]
pub fn j_expj( x: fsize ) -> csize {
    csize { re: -x.sin(), im: x.cos() }
}

#[inline(always)]
pub fn neg_j_expj( x: fsize ) -> csize {
    csize { re: x.sin(), im: -x.cos() }
}


#[inline(always)]
pub fn j_mul_re( x: csize ) -> fsize {
    -x.im
}

#[inline(always)]
pub fn neg_j_mul_re( x: csize ) -> fsize {
    x.im
}

#[inline(always)]
pub fn expj_re( x: fsize ) -> fsize {
    x.cos()
}

#[inline(always)]
pub fn j_expj_re( x: fsize ) -> fsize {
    -x.sin()
}

#[inline(always)]
pub fn neg_j_expj_re( x: fsize ) -> fsize {
    x.sin()
}

#[inline(always)]
pub fn complex_mul_re( x: csize, y: csize ) -> fsize {
    x.re * y.re - x.im * y.im
}


// Analytic integration.
#[inline(always)]
pub fn quad_sinpx( x0: fsize, x1: fsize, p: fsize ) -> fsize {
    ( (p*x0).cos() - (p*x1).cos() ) / p
}

#[inline(always)]
pub fn quad_cospx( x0: fsize, x1: fsize, p: fsize ) -> fsize {
    ( (p*x1).sin() - (p*x0).sin() ) / p
}

#[inline(always)]
pub fn quad_sinpx_phase( x0: fsize, x1: fsize, p: fsize, phi: fsize ) -> fsize {
    ( (p*x0+phi).cos() - (p*x1+phi).cos() ) / p
}

#[inline(always)]
pub fn quad_cospx_phase( x0: fsize, x1: fsize, p: fsize, phi: fsize ) -> fsize {
    ( (p*x1+phi).sin() - (p*x0+phi).sin() ) / p
}

#[inline(always)]
pub fn quad_sinpx_sinqx( x0: fsize, x1: fsize, p: fsize, q: fsize ) -> fsize {
    let r1 = p - q;
    let r2 = p + q;
    0.5*( (r1*x1).sin()/r1 - (r2*x1).sin()/r2 - (r1*x0).sin()/r1 + (r2*x0).sin()/r2 )
}

#[inline(always)]
pub fn quad_sinpx_sinqx_phase( x0: fsize, x1: fsize, p: fsize, phi_p: fsize, q: fsize, phi_q: fsize ) -> fsize {
    let r1 = p - q;
    let r2 = p + q;
    let s1 = phi_p - phi_q;
    let s2 = phi_p + phi_q;
    0.5*( (r1*x1+s1).sin()/r1 - (r2*x1+s2).sin()/r2 - (r1*x0+s1).sin()/r1 + (r2*x0+s2).sin()/r2 )
}

#[inline(always)]
pub fn quad_cospx_cosqx_phase( x0: fsize, x1: fsize, p: fsize, phi_p: fsize, q: fsize, phi_q: fsize ) -> fsize {
    let r1 = p - q;
    let r2 = p + q;
    let s1 = phi_p - phi_q;
    let s2 = phi_p + phi_q;
    0.5*( (r1*x1+s1).sin()/r1 + (r2*x1+s2).sin()/r2 - (r1*x0+s1).sin()/r1 - (r2*x0+s2).sin()/r2 )
}

#[inline(always)]
pub fn quad_sinpx_square_phase( x0: fsize, x1: fsize, p: fsize, phi: fsize ) -> fsize {
    0.5*( x1 - (2.*p*x1+2.*phi).cos()/(2.*p) - x0 + (2.*p*x0+2.*phi).cos()/(2.*p) )
}

#[inline(always)]
pub fn quad_sinkx_sinpx_sinqx( x0: fsize, x1: fsize, k: fsize, p: fsize, q: fsize ) -> fsize {
    0.5*( quad_sinpx_cosqx(x0, x1, k, p-q) - quad_sinpx_cosqx(x0, x1, k, p+q) )
}

#[inline(always)]
pub fn quad_sinkx_sinpx_sinqx_phase( 
    x0: fsize, x1: fsize, k: fsize, phi_k: fsize, p: fsize, phi_p: fsize, q: fsize, phi_q: fsize 
) -> fsize {
    0.5*( quad_sinpx_cosqx_phase(x0, x1, k, phi_k, p-q, phi_p-phi_q) - quad_sinpx_cosqx_phase(x0, x1, k, phi_k, p+q, phi_p+phi_q) )
}

#[inline(always)]
pub fn quad_sinkx_sinpx_square_phase( x0: fsize, x1: fsize, k: fsize, phi_k: fsize, p: fsize, phi_p: fsize ) -> fsize {
    0.5*( quad_sinpx_phase(x0, x1, k, phi_k) - quad_sinpx_cosqx_phase(x0, x1, k, phi_k, 2.*p, 2.*phi_p) )
}

#[inline(always)]
pub fn quad_sinpx_cosqx( x0: fsize, x1: fsize, p: fsize, q: fsize ) -> fsize {
    let r1 = p + q;
    let r2 = p - q;
    -0.5*( (r1*x1).cos()/r1 + (r2*x1).cos()/r2 - (r1*x0).cos()/r1 - (r2*x0).cos()/r2 )
}

#[inline(always)]
pub fn quad_sinpx_cosqx_phase( x0: fsize, x1: fsize, p: fsize, phi_p: fsize, q: fsize, phi_q: fsize ) -> fsize {
    let r1 = p + q;
    let r2 = p - q;
    let s1 = phi_p + phi_q;
    let s2 = phi_p - phi_q;
    -0.5*( (r1*x1+s1).cos()/r1 + (r2*x1+s2).cos()/r2 - (r1*x0+s1).cos()/r1 - (r2*x0+s2).cos()/r2 )
}

#[inline(always)]
pub fn quad_line_sinpx_sinqx_phase( 
    x0: fsize, x1: fsize, k: fsize, b: fsize, p: fsize, phi_p: fsize, q: fsize, phi_q: fsize 
) -> fsize {
    0.5 * ( quad_line_cospx_phase(x0, x1, k, b, p-q, phi_p-phi_q) - quad_line_cospx_phase(x0, x1, k, b, p+q, phi_p+phi_q) )
}

#[inline(always)]
pub fn quad_line_sinpx_square_phase( x0: fsize, x1: fsize, k: fsize, b: fsize, p: fsize, phi: fsize ) -> fsize {
    0.5 * ( quad_line(x0, x1, k, b) - quad_line_cospx_phase(x0, x1, k, b, 2.*p, 2.*phi) )
}

#[inline(always)]
pub fn quad_line( x0: fsize, x1: fsize, k: fsize, b: fsize ) -> fsize {
    0.5*k*x1.powi(2) + b*x1 - 0.5*k*x0.powi(2) - b*x0
}

#[inline(always)]
pub fn quad_x_cospx_phase( x0: fsize, x1: fsize, p: fsize, phi: fsize ) -> fsize {
    ( x1*(p*x1+phi).sin() - x0*(p*x0+phi).sin() - quad_sinpx_phase(x0, x1, p, phi) ) / p
}

#[inline(always)]
pub fn quad_line_cospx_phase( x0: fsize, x1: fsize, k: fsize, b: fsize, p: fsize, phi: fsize ) -> fsize {
    k*quad_x_cospx_phase(x0, x1, p, phi) + b*quad_cospx_phase(x0, x1, p, phi)
}

#[inline(always)]
pub fn quad_expkx( x0: fsize, x1: fsize, k: fsize ) -> fsize {
    ( (k*x1).exp() - (k*x0).exp() ) / k
}

#[inline(always)]
pub fn quad_expkx_sinpx_phase( x0: fsize, x1: fsize, k: fsize, p: fsize, phi: fsize ) -> fsize {
    (
        (k*x1).exp() * (k*(p*x1+phi).sin() - p*(p*x1+phi).cos()) - 
        (k*x0).exp() * (k*(p*x0+phi).sin() - p*(p*x0+phi).cos())
    ) / (k.powi(2) + p.powi(2))
}

#[inline(always)]
pub fn quad_expkx_cospx_phase( x0: fsize, x1: fsize, k: fsize, p: fsize, phi: fsize ) -> fsize {
    (
        (k*x1).exp() * (p*(p*x1+phi).sin() + k*(p*x1+phi).cos()) - 
        (k*x0).exp() * (p*(p*x0+phi).sin() + k*(p*x0+phi).cos())
    ) / (k.powi(2) + p.powi(2))
}

#[inline(always)]
pub fn quad_expkx_sinpx_sinqx_phase( 
    x0: fsize, x1: fsize, k: fsize, p: fsize, phi_p: fsize, q: fsize, phi_q: fsize 
) -> fsize {
    let r1 = p - q;
    let r2 = p + q;
    let s1 = phi_p - phi_q;
    let s2 = phi_p + phi_q;
    0.5 * ( quad_expkx_cospx_phase(x0, x1, k, r1, s1) - quad_expkx_cospx_phase(x0, x1, k, r2, s2) )
}

#[inline(always)]
pub fn quad_expkx_sinpx_square_phase( x0: fsize, x1: fsize, k: fsize, p: fsize, phi: fsize ) -> fsize {
    0.5 * ( quad_expkx(x0, x1, k) - quad_expkx_cospx_phase(x0, x1, k, 2.*p, 2.*phi) )
}

#[inline(always)]
pub fn quad_sinkx_4_phase( x0: fsize, x1: fsize, k1: fsize, b1: fsize, k2: fsize, b2: fsize, k3: fsize, b3: fsize, k4: fsize, b4: fsize ) -> fsize {
    let r1 = k1 - k2;
    let r2 = k1 + k2;
    let r3 = k3 - k4;
    let r4 = k3 + k4;
    let s1 = b1 - b2;
    let s2 = b1 + b2;
    let s3 = b3 - b4;
    let s4 = b3 + b4;
    0.25 * (
        quad_cospx_cosqx_phase(x0, x1, r1, s1, r3, s3) - 
        quad_cospx_cosqx_phase(x0, x1, r1, s1, r4, s4) - 
        quad_cospx_cosqx_phase(x0, x1, r2, s2, r3, s3) + 
        quad_cospx_cosqx_phase(x0, x1, r2, s2, r4, s4)
    )
}

#[inline(always)]
pub fn quad_sinkx_square_2_phase( x0: fsize, x1: fsize, k1: fsize, b1: fsize, k2: fsize, b2: fsize ) -> fsize {
    0.25 * (
        x1 - x0 - 
        quad_cospx_phase(x0, x1, 2.*k2, 2.*b2) - 
        quad_cospx_phase(x0, x1, 2.*k1, 2.*b1) + 
        quad_cospx_cosqx_phase(x0, x1, 2.*k1, 2.*b1, 2.*k2, 2.*b2)
    )
}

#[inline(always)]
pub fn quad_sinkx_square_2_phase2( x0: fsize, x1: fsize, k1: fsize, b1: fsize, b1_: fsize, k2: fsize, b2: fsize, b2_: fsize ) -> fsize {
    let c1 = (b1 - b1_).cos();
    let c2 = (b2 - b2_).cos();
    0.25 * (
        c1 * c2 * (x1 - x0) - 
        c1 * quad_cospx_phase(x0, x1, 2.*k2, b2+b2_) - 
        c2 * quad_cospx_phase(x0, x1, 2.*k1, b1+b1_) + 
        quad_cospx_cosqx_phase(x0, x1, 2.*k1, b1+b1_, 2.*k2, b2+b2_)
    )
}

#[inline(always)]
pub fn quad_sinkx_2_sinkx_square_phase( x0: fsize, x1: fsize, k1: fsize, b1: fsize, k2: fsize, b2: fsize, k3: fsize, b3: fsize ) -> fsize {
    let r1 = k1 - k2;
    let r2 = k1 + k2;
    let s1 = b1 - b2;
    let s2 = b1 + b2;
    0.25 * (
        quad_cospx_phase(x0, x1, r1, s1) - 
        quad_cospx_cosqx_phase(x0, x1, r1, s1, 2.*k3, 2.*b3) - 
        quad_cospx_phase(x0, x1, r2, s2) + 
        quad_cospx_cosqx_phase(x0, x1, r2, s2, 2.*k3, 2.*b3)
    )
}

#[inline(always)]
pub fn quad_sinkx_2_sinkx_square_phase2( x0: fsize, x1: fsize, k1: fsize, b1: fsize, k2: fsize, b2: fsize, k3: fsize, b3: fsize, b3_: fsize ) -> fsize {
    let r1 = k1 - k2;
    let r2 = k1 + k2;
    let s1 = b1 - b2;
    let s2 = b1 + b2;
    let c = (b3 - b3_).cos();
    0.25 * (
        c * quad_cospx_phase(x0, x1, r1, s1) - 
        quad_cospx_cosqx_phase(x0, x1, r1, s1, 2.*k3, b3+b3_) - 
        c * quad_cospx_phase(x0, x1, r2, s2) + 
        quad_cospx_cosqx_phase(x0, x1, r2, s2, 2.*k3, b3+b3_)
    )
}


pub struct EqSolver {
    pub max_iter: usize,
    pub tol: fsize,
}

impl EqSolver
{
    #[inline]
    pub fn new_default() -> Self {
        Self { max_iter: 1000, tol: 1e-4 }
    }

    #[inline]
    pub fn new( max_iter: usize, tol: fsize ) -> Self {
        Self { max_iter, tol }
    }

    #[inline(always)]
    pub fn solve<F: Fn(fsize)->[fsize; 2]>( 
        &self, f_df: F, x0: fsize 
    ) -> Result<(fsize, usize), (fsize, fsize)> {
        let mut x: fsize = x0;
        let [mut fx, mut dfx] = f_df(x);
        let mut total_iter: usize = usize::MAX;
        for i in 1..self.max_iter+1 {
            x -= fx / dfx;
            [fx, dfx] = f_df(x);
            if fx.abs() < self.tol {
                total_iter = i;
                break;
            }
        }
        match total_iter == usize::MAX {
            false => Ok((x, total_iter)),
            true => Err((x, fx)),
        }
    }
}