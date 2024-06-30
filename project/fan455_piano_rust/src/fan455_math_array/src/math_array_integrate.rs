use fan455_math_scalar::*;
use fan455_arrf64::*;
use fan455_util::*;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;
use std::ops::Fn;


pub struct FourierQuad {
    lb: fsize,
    ub: fsize,
    n: usize,
    nk: usize,
    k_end: usize,
    is_even: bool,
    fft_scale: fsize,
    fft_unscale: fsize,
    dx: fsize,
    w: fsize,
    fk_half: Vec<csize>,
    fk: Vec<csize>,
    scratch: Vec<csize>,
    planner: RealFftPlanner<fsize>,
    fft: Arc<dyn RealToComplex<fsize>>,
}


impl FourierQuad 
{
    #[inline]
    pub fn new( lb: fsize, ub: fsize, n: usize ) -> Self {
        let fft_unscale = (n as fsize).sqrt();
        let fft_scale = fft_unscale.recip();
        let dx = (ub-lb) / (n-1) as fsize;
        let w = 2.*PI / (ub-lb+dx);

        let mut planner = RealFftPlanner::new();
        let fft = planner.plan_fft_forward(n);

        let scratch_len = fft.get_scratch_len();
        let scratch: Vec<csize> = vec![C_ZERO; scratch_len];
        let nk = fft.complex_len();
        let fk_half: Vec<csize> = vec![C_ZERO; nk];
        let fk: Vec<csize> = vec![C_ZERO; n];

        let is_even = n % 2 == 0;
        let k_end = match is_even {
            true => nk - 1,
            false => nk,
        };

        Self { ub, lb, n, nk, is_even, k_end, fft_unscale, fft_scale, dx, w, fk, fk_half, scratch, planner, fft }
    }

    #[inline]
    pub fn replan( &mut self, lb: fsize, ub: fsize, n: usize ) {
        self.lb = lb;
        self.ub = ub;
        self.n = n;
        self.fft_unscale = (n as fsize).sqrt();
        self.fft_scale = self.fft_unscale.recip();
        self.dx = (ub-lb) / (n-1) as fsize;
        self.w = 2.*PI / (ub-lb+self.dx);
        
        self.fft = self.planner.plan_fft_forward(n);
        self.nk = self.fft.complex_len();
        self.fk_half.resize(self.nk, C_ZERO);
        self.fk.resize(self.n, C_ZERO);

        let scratch_len = self.fft.get_scratch_len();
        self.scratch.resize(scratch_len, C_ZERO);

        self.is_even = n % 2 == 0;
        self.k_end = match self.is_even {
            true => self.nk - 1,
            false => self.nk,
        };
    }

    #[inline]
    fn _do_fft( &mut self, fx: &mut [fsize] ) {
        self.fft.process_with_scratch(fx, &mut self.fk_half, &mut self.scratch).unwrap();
        self.fk[..self.nk].copy_from_slice(&self.fk_half);
        for elem!(fk, fk_) in mzip!(self.fk[self.nk..].iter_mut(), self.fk_half[1..self.k_end].iter().rev()) {
            *fk = fk_.conj();
        }
    }

    #[inline]
    pub fn integrate( &mut self, fx: &mut [fsize] ) -> fsize {
        self._do_fft(fx);

        let mut s: fsize = (self.ub - self.lb) / self.fft_unscale;
        for elem!(k, fk) in mzip!(1..self.n, self.fk.it()) {
            let w = (k as fsize) * self.w;
            s += complex_mul_re(*fk, neg_j_expj(w*self.ub) - neg_j_expj(w*self.lb)) / (w*self.fft_unscale);
        }
        s * self.fft_scale
    }

    #[inline]
    pub fn integrate_with_sinpx_square( &mut self, fx: &mut [fsize], p: fsize ) -> fsize {
        // assume p != q
        self._do_fft(fx);

        let mut s: fsize = (self.ub - self.lb) / self.fft_unscale;
        for elem!(k, fk) in mzip!(1..self.n, self.fk.it()) {
            let w = (k as fsize) * self.w;
            // fx = sum( sign*r*( a/r *sinwx + b/r * coswx) ) = sum( sign*r*sin(wx+phi) ), where a>0, cos(phi)=a/r
            let r = (fk.im.powi(2) + fk.re.powi(2)).sqrt();
            if r.abs() > 1e-7 {
                let [a, sign]: [fsize; 2] = match fk.im < 0. {
                    true => [ -fk.im, 1. ],
                    false => [ fk.im, -1. ],
                };
                let phi = (a/r).acos();
                s += 0.5 * sign * r * (
                    quad_sinpx_phase(self.lb, self.ub, w, phi) - 
                    quad_sinpx_cosqx_phase(self.lb, self.ub, w, phi, 2.*p, 0.)
                ) / self.fft_unscale;
            }
        }
        s * self.fft_scale
    }

    #[inline]
    pub fn integrate_with_sinpx_sinqx( &mut self, fx: &mut [fsize], p: fsize, q: fsize ) -> fsize {
        // assume p != q
        self._do_fft(fx);

        let mut s: fsize = (self.ub - self.lb) / self.fft_unscale;
        for elem!(k, fk) in mzip!(1..self.n, self.fk.it()) {
            let w = (k as fsize) * self.w;
            // fx = sum( sign*r*( a/r *sinwx + b/r * coswx) ) = sum( sign*r*sin(wx+phi) ), where a>0, cos(phi)=a/r
            let r = (fk.im.powi(2) + fk.re.powi(2)).sqrt();
            if r.abs() > 1e-7 {
                let [a, sign]: [fsize; 2] = match fk.im < 0. {
                    true => [ -fk.im, 1. ],
                    false => [ fk.im, -1. ],
                };
                let phi = (a/r).acos();
                s += 0.5 * sign * r * (
                    quad_sinpx_cosqx_phase(self.lb, self.ub, w, phi, p-q, 0.) - 
                    quad_sinpx_cosqx_phase(self.lb, self.ub, w, phi, p+q, 0.)
                ) / self.fft_unscale;
            }
        }
        s * self.fft_scale
    }
}


pub struct LobattoQuad<const N: usize, const N_FREE: usize> {
    points: [fsize; N],
    weights: [fsize; N],
    points_free: [fsize; N_FREE],
    weights_free: [fsize; N_FREE],
}


impl<const N: usize, const N_FREE: usize> LobattoQuad<N, N_FREE>
{
    #[inline]
    pub fn new() -> Self {
        let points: [fsize; N];
        let weights: [fsize; N];
        if N == 3 {
            points = vec![0., 1., -1.].try_into().unwrap();
            weights = vec![4./3., 1./3., 1./3.].try_into().unwrap();
        } else if N == 4 {
            points = vec![5_f64.sqrt()/5., -5_f64.sqrt()/5., 1., -1.].try_into().unwrap();
            weights = vec![5./6., 5./6., 1./6., 1./6.].try_into().unwrap();
        } else if N == 5 {
            points = vec![0., (3_f64/7.).sqrt(), -(3_f64/7.).sqrt(), 1., -1.].try_into().unwrap();
            weights = vec![32./45., 49./90., 49./90., 1./10., 1./10.].try_into().unwrap();
        } else if N == 6 {
            points = vec![
                ((7.-2.*7_f64.sqrt())/21.).sqrt(), -((7.-2.*7_f64.sqrt())/21.).sqrt(),
                ((7.+2.*7_f64.sqrt())/21.).sqrt(), -((7.+2.*7_f64.sqrt())/21.).sqrt(), 1., -1.
            ].try_into().unwrap();
            weights = vec![
                (14.+7_f64.sqrt())/30., (14.+7_f64.sqrt())/30., 
                (14.-7_f64.sqrt())/30., (14.-7_f64.sqrt())/30., 1./15., 1./15.
            ].try_into().unwrap();
        } else {
            panic!("N < 3 or N > 6 is not supported.");
        }
        let points_free: [fsize; N_FREE] = points[..N-2].try_into().unwrap();
        let weights_free: [fsize; N_FREE] = weights[..N-2].try_into().unwrap();
        Self { points, weights, points_free, weights_free }
    }

    #[inline]
    pub fn order( &self ) -> usize {
        N
    }

    #[inline]
    pub fn integrate_special_01( &self, fx: &[[fsize; N_FREE]], each_len: fsize ) -> fsize {
        let mut s: fsize = 0.;
        for fx_ in fx.iter() {
            for elem!(fx__, w) in mzip!(fx_, self.weights_free) {
                s += w * fx__;
            }
        }
        0.5*each_len * s
    }

    #[inline]
    pub fn integrate<F>( &self, a: fsize, b: fsize, fun: F ) -> fsize where F: Copy + Fn(fsize)->fsize {
        let mut s: fsize = 0.;
        for elem!(x_, w_) in mzip!(self.points.it(), self.weights.it()) {
            s += w_ * fun(Self::transform_point(a, b, *x_));
        }
        Self::transform_quad(a, b, s)
    }

    #[inline]
    pub fn get_mid_points( &self, a: fsize, b: fsize ) -> [fsize; N_FREE] {
        let mut x = self.points_free;
        for x_ in x.iter_mut() {
            *x_ = Self::transform_point(a, b, *x_);
        }
        x
    }

    #[inline]
    pub fn transform_point( a: fsize, b: fsize, x: fsize ) -> fsize {
        x.mul_add(0.5*(b-a), 0.5*(b+a))
    }

    #[inline]
    pub fn transform_quad( a: fsize, b: fsize, s: fsize ) -> fsize {
        0.5 * (b - a) * s
    }
}


pub struct LegendreQuad {
    a: fsize,
    b: fsize,
    points: Vec<fsize>,
    weights: Vec<fsize>,
    x: Vec<fsize>,
}

impl LegendreQuad
{
    #[inline]
    pub fn new( points_path: &String, weights_path: &String ) -> Self {
        let points: Vec<fsize> = unsafe { read_npy_vec_tm(points_path) };
        let weights: Vec<fsize> = unsafe { read_npy_vec_tm(weights_path) };
        assert_eq!(points.len(), weights.len());
        let x: Vec<fsize> = vec![0.; points.len()];
        Self { a: -1., b: 1., points, weights, x }
    }

    #[inline]
    pub fn set_bounds( &mut self, a: fsize, b: fsize ) {
        self.a = a;
        self.b = b;
        for elem!(x_, p_) in mzip!(self.x.itm(), self.points.it()) {
            *x_ = 0.5*(b-a) * p_ + 0.5*(b+a)
        }
    }

    #[inline]
    pub fn integrate<F>( &self, fun: F ) -> fsize where F: Copy + Fn(fsize)->fsize {
        let mut s: fsize = 0.;
        for elem!(x_, w_) in mzip!(self.x.it(), self.weights.it()) {
            s += w_ * fun(*x_);
        }
        0.5*(self.b - self.a) * s
    }

    #[inline]
    pub fn integrate_complex<F>( &self, fun: F ) -> csize where F: Copy + std::ops::Fn(fsize)->csize {
        let mut s: csize = C_ZERO;
        for elem!(x_, w_) in mzip!(self.x.it(), self.weights.it()) {
            s += fun(*x_).scale(*w_);
        }
        0.5*(self.b - self.a) * s
    }
}



