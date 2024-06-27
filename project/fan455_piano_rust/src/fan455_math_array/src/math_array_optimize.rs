use fan455_math_scalar::*;
use fan455_arrf64::*;
use fan455_util::*;
//use std::ops::{Index, IndexMut};


pub trait FunMut {
    fn f<VT: RVec<fsize>>( &mut self, x: &VT ) -> fsize;
}

pub trait GradMut {
    fn df<VT1: RVec<fsize>, VT2: RVecMut<fsize>>( &mut self, x0: &VT1, y0: fsize, grad: &mut VT2 );
}

pub trait GradHessMut {
    fn df_ddf<VT1: RVec<fsize>, VT2: RVecMut<fsize>, MT: RMatMut<fsize>>(
        &mut self, x0: &VT1, y0: fsize, grad: &mut VT2, hess: &mut MT,
    );
}

pub struct GradDesc {
    pub dim: usize,
    pub tol: fsize,
    pub max_search: usize,
    pub max_iter: usize,
    pub alpha: fsize, // 一般情况下，参数alpha的取值范围在0.01到0.3之间，表示我们接受目标函数减少的预测范围在1%到30%之间。
    pub beta: fsize, // 参数beta取值范围在0.1到0.8之间。0.1对应非常粗略的搜索，0.8对应没那么粗略的搜索。
    pub lr: fsize,
    pub fx: fsize,
    pub fx_new: fsize,
    pub x_new: Arr1<fsize>,
    pub grad: Arr1<fsize>,
}

impl GradDesc
{
    #[inline]
    pub fn new( dim: usize ) -> Self {
        let x_new: Arr1<fsize> = Arr1::new(dim);
        let grad: Arr1<fsize> = Arr1::new(dim);
        Self { dim, tol: 1e-4, max_search: 100, max_iter: 1000, alpha: 0.1, beta: 0.5, 
            lr: 0.001, fx: 0., fx_new: 0., x_new, grad }
    }

    #[inline]
    pub fn run_mut<F: FunMut + GradMut, VT: RVecMut<fsize>>(
        &mut self,
        fun: &mut F,
        x: &mut VT,
        prog: usize,
    ) -> Result<usize, fsize> {
        let mut res: fsize = fsize::INFINITY;
        let mut total_iter: usize = self.max_iter + 1;
        self.fx = fun.f(x);

        for i in 1..self.max_iter+1 {
            fun.df(x, self.fx, &mut self.grad);
            res = dnrm2(&self.grad);//self.grad.norm();
            if res < self.tol {
                total_iter = i;
                break;
            }

            x.subassign_scale(&self.grad, self.lr);
            self.fx = fun.f(x);

            /*let s0 = dnrm2(&self.grad);
            let mut search: usize = 0;
            let mut t: fsize = self.lr;
            loop {
                self.x_new.assign_sub_scale(x, &self.grad, t);
                self.fx_new = fun.f(&self.x_new);
                if self.fx_new > self.fx - self.alpha * t * s0 {
                    t *= self.beta;
                } else {
                    x.subassign_scale(&self.grad, t);
                    self.fx = self.fx_new;
                    break;
                }
                search += 1;
                if search > self.max_search {
                    panic!("Maximum line search exceeded. Current itertion: {i}.");
                }
            }*/
            if i % prog == 0 {
                println!("Iterations: {} / {}", i, self.max_iter);
            }
        }
        match total_iter > self.max_iter {
            false => Ok(total_iter),
            true => Err(res),
        }
    }
}

pub struct AdamW {
    pub dim: usize,
    pub tol: fsize,
    pub max_iter: usize,
    pub weight_decay: fsize,
    pub beta1: fsize,
    pub beta2: fsize,
    pub eps: fsize,
    pub lr: fsize,
    pub fx: fsize,
    pub grad: Arr1<fsize>,
    pub grad2: Arr1<fsize>,
    pub m1: Arr1<fsize>,
    pub m2: Arr1<fsize>,
    pub m1_hat: Arr1<fsize>,
    pub m2_hat: Arr1<fsize>,
}

impl AdamW
{
    #[inline]
    pub fn new( dim: usize ) -> Self {
        let grad: Arr1<fsize> = Arr1::new(dim);
        let grad2: Arr1<fsize> = Arr1::new(dim);
        let m1: Arr1<fsize> = Arr1::new(dim);
        let m2: Arr1<fsize> = Arr1::new(dim);
        let m1_hat: Arr1<fsize> = Arr1::new(dim);
        let m2_hat: Arr1<fsize> = Arr1::new(dim);
        Self { dim, tol: 1e-4, max_iter: 1000, weight_decay: 0.01, beta1: 0.9, beta2: 0.999, 
            eps: 1e-8, lr: 0.001, fx: 0., grad, grad2, m1, m2, m1_hat, m2_hat }
    }

    #[inline]
    pub fn run_mut<F: FunMut + GradMut, VT: RVecMut<fsize>>(
        &mut self,
        fun: &mut F,
        x: &mut VT,
        prog: usize,
    ) -> Result<usize, fsize> {
        let mut res: fsize = fsize::INFINITY;
        let mut total_iter: usize = self.max_iter + 1;

        for i in 1..self.max_iter+1 {
            self.fx = fun.f(x);
            fun.df(x, self.fx, &mut self.grad);
            res = dnrm2(&self.grad);//self.grad.norm();
            if res < self.tol {
                total_iter = i;
                break;
            }
            self.grad2.assign_powi(&self.grad, 2);
            x.scale(1.-self.lr*self.weight_decay);
            daxpby(1.-self.beta1, &self.grad, self.beta1, &mut self.m1);
            daxpby(1.-self.beta2, &self.grad2, self.beta2, &mut self.m2);
            self.m1_hat.assign_scale(&self.m1, 1./(1.-self.beta1.powi(i as i32)));
            self.m2_hat.assign_scale(&self.m2, 1./(1.-self.beta2.powi(i as i32)));
            for elem!(x_, m1_hat_, m2_hat_) in mzip!(x.itm(), self.m1_hat.it(), self.m2_hat.it()) {
                *x_ -= self.lr * m1_hat_ / (m2_hat_.sqrt() + self.eps);
            }
            if i % prog == 0 {
                println!("Iterations: {} / {}", i, self.max_iter);
            }
        }
        match total_iter > self.max_iter {
            false => Ok(total_iter),
            true => Err(res),
        }
    }
}

pub struct SR1 { // https://zhuanlan.zhihu.com/p/306635632
    pub dim: usize,
    pub tol: fsize,
    pub max_iter: usize,
    pub max_search: usize,
    pub min_denominator: fsize,
    pub hess_inv_scalar: fsize,
    pub alpha: fsize, // 一般情况下，参数alpha的取值范围在0.01到0.3之间，表示我们接受目标函数减少的预测范围在1%到30%之间。
    pub beta: fsize, // 参数beta取值范围在0.1到0.8之间。0.1对应非常粗略的搜索，0.8对应没那么粗略的搜索。
    pub fx: fsize,
    pub fx_new: fsize,
    pub x_new: Arr1<fsize>,
    pub dir: Arr1<fsize>,
    pub grad0: Arr1<fsize>,
    pub grad1: Arr1<fsize>,
    pub delta_grad: Arr1<fsize>,
    pub hess_inv: Arr2<fsize>,
}

impl SR1
{
    #[inline]
    pub fn new( dim: usize ) -> Self {
        let hess_inv_scalar: fsize = 0.01;
        let x_new: Arr1<fsize> = Arr1::new(dim);
        let dir: Arr1<fsize> = Arr1::new(dim);
        let grad0: Arr1<fsize> = Arr1::new(dim);
        let grad1: Arr1<fsize> = Arr1::new(dim);
        let delta_grad: Arr1<fsize> = Arr1::new(dim);
        let mut hess_inv: Arr2<fsize> = Arr2::new(dim, dim);
        hess_inv.set_diag(hess_inv_scalar);
        Self { dim, tol: 1e-4, max_iter: 1000, max_search: 100, alpha: 0.1, beta: 0.5, min_denominator: 1e-8,
            fx: 0., fx_new: 0., x_new, delta_grad, dir, grad0, grad1, hess_inv, hess_inv_scalar }
    }

    #[inline]
    pub fn run_mut<F: FunMut + GradMut, VT: RVecMut<fsize>>(
        &mut self,
        fun: &mut F,
        x: &mut VT,
        prog: usize,
    ) -> Result<usize, fsize> {
        let mut res: fsize = fsize::INFINITY;
        let mut total_iter: usize = self.max_iter + 1;
        self.fx = fun.f(x);
        fun.df(x, self.fx, &mut self.grad0);
        let mut rev_ref: bool = false;

        for i in 1..self.max_iter+1 {
            //pr1.print("hess_inv", &self.hess_inv);
            let (grad, grad_new) = match rev_ref {
                false => (&self.grad0, &mut self.grad1),
                true => (&self.grad1, &mut self.grad0),
            };
            dsymv(-1., &self.hess_inv, grad, 0., &mut self.dir, LOWER); // dir = - hess_inv * grad
            let s0 = ddot(grad, &self.dir);
            let mut search: usize = 0;
            let mut t: fsize = 1.;

            x.addassign_scale(&self.dir, t);
            self.fx = fun.f(x);

            /*loop {
                self.x_new.assign_add_scale(x, &self.dir, t);
                self.fx_new = fun.f(&self.x_new);
                if self.fx_new > self.fx + self.alpha * t * s0 {
                    t *= self.beta;
                } else {
                    //println!("t = {t:.4}");
                    x.addassign_scale(&self.dir, t);
                    self.fx = self.fx_new;
                    break;
                }
                search += 1;
                if search > self.max_search {
                    panic!("Maximum line search exceeded. Current itertion: {i}.");
                }
            }*/
            fun.df(x, self.fx, grad_new);
            res = grad_new.norm();
            if res < self.tol {
                total_iter = i;
                break;
            }
            // Update the Hessian inverse.
            self.delta_grad.assign_sub(grad_new, grad);
            dsymv(-1., &self.hess_inv, &self.delta_grad, t, &mut self.dir, LOWER); // Now dir = s - H * y
            let s1 = ddot(&self.dir, &self.delta_grad);
            if s1.abs() > self.min_denominator * self.delta_grad.norm() * self.dir.norm() {
                dsyr(1./s1, &self.dir, &mut self.hess_inv, LOWER); // Symmetric rand-1 update
            }
            rev_ref ^= true;
            if i % prog == 0 {
                println!("Iterations: {} / {}", i, self.max_iter);
            }
        }
        match total_iter > self.max_iter {
            false => Ok(total_iter),
            true => Err(res),
        }
    }
}

pub struct DampedNewton {
    pub dim: usize,
    pub tol: fsize,
    pub max_iter: usize,
    pub max_search: usize,
    pub alpha: fsize, // 一般情况下，参数alpha的取值范围在0.01到0.3之间，表示我们接受目标函数减少的预测范围在1%到30%之间。
    pub beta: fsize, // 参数beta取值范围在0.1到0.8之间。0.1对应非常粗略的搜索，0.8对应没那么粗略的搜索。
    pub x_new: Arr1<fsize>,
    pub fx: fsize,
    pub dir: Arr1<fsize>,
    pub grad: Arr1<fsize>,
    pub hess: Arr2<fsize>,
}

impl DampedNewton
{
    #[inline]
    pub fn new( dim: usize ) -> Self {
        let x_new: Arr1<fsize> = Arr1::new(dim);
        let grad: Arr1<fsize> = Arr1::new(dim);
        let dir: Arr1<fsize> = Arr1::new(dim);
        let hess: Arr2<fsize> = Arr2::new(dim, dim);
        Self { dim, tol: 1e-4, max_iter: 1000, max_search: 100, alpha: 0.1, beta: 0.5, fx: 0., x_new, dir, grad, hess }
    }

    #[inline]
    pub fn run_mut<F: FunMut + GradHessMut, VT: RVecMut<fsize>>(
        &mut self,
        fun: &mut F,
        x: &mut VT,
        prog: usize,
    ) -> Result<usize, fsize> {
        let mut res: fsize = fsize::INFINITY;
        let mut total_iter: usize = self.max_iter + 1;
        self.fx = fun.f(x);
        let mut fx_new: fsize;
        let mut t: fsize;
        let mut search: usize = 0;

        for i in 1..self.max_iter+1 {
            fun.df_ddf(x, self.fx, &mut self.grad, &mut self.hess);
            res = self.grad.norm();
            if res < self.tol {
                total_iter = i;
                break;
            }
            self.dir.copy(&self.grad);
            dposv(&mut self.hess, &mut self.dir, LOWER);
            let tmp = ddot(&self.grad, &self.dir);
            t = 1.;
            loop {
                self.x_new.assign_sub_scale(x, &self.dir, t);
                fx_new = fun.f(&self.x_new);
                if fx_new > self.fx - self.alpha * t * tmp {
                    t *= self.beta;
                } else {
                    self.fx = fx_new;
                    x.subassign_scale(&self.dir, t);
                    break;
                }
                search += 1;
                if search > self.max_search {
                    panic!("Maximum line search exceeded. Current itertion: {i}.");
                }
            }
            if i % prog == 0 {
                println!("Iterations: {} / {}", i, self.max_iter);
            }
        }
        match total_iter > self.max_iter {
            false => Ok(total_iter),
            true => Err(res),
        }
    }
}

pub struct BFGS {
    pub dim: usize,
    pub tol: fsize,
    pub max_iter: usize,
    pub max_search: usize,
    pub alpha: fsize, // 一般情况下，参数alpha的取值范围在0.01到0.3之间，表示我们接受目标函数减少的预测范围在1%到30%之间。
    pub beta: fsize, // 参数beta取值范围在0.1到0.8之间。0.1对应非常粗略的搜索，0.8对应没那么粗略的搜索。
    pub hess_inv_scalar: fsize,
    pub fx: fsize,
    pub fx_new: fsize,
    pub x_new: Arr1<fsize>,
    pub delta_x: Arr1<fsize>,
    pub dir: Arr1<fsize>,
    pub grad0: Arr1<fsize>,
    pub grad1: Arr1<fsize>,
    pub delta_grad: Arr1<fsize>,
    pub hess_inv0: Arr2<fsize>,
    pub hess_inv1: Arr2<fsize>,
    pub mbuf0: Arr2<fsize>,
    pub mbuf1: Arr2<fsize>,
    pub mbuf2: Arr2<fsize>,
}

impl BFGS
{
    #[inline]
    pub fn new( dim: usize ) -> Self {
        let hess_inv_scalar: fsize = 0.01;
        let x_new: Arr1<fsize> = Arr1::new(dim);
        let delta_x: Arr1<fsize> = Arr1::new(dim);
        let dir: Arr1<fsize> = Arr1::new(dim);
        let grad0: Arr1<fsize> = Arr1::new(dim);
        let grad1: Arr1<fsize> = Arr1::new(dim);
        let delta_grad: Arr1<fsize> = Arr1::new(dim);
        let mut hess_inv0: Arr2<fsize> = Arr2::new(dim, dim);
        hess_inv0.set_diag(hess_inv_scalar);
        let mut hess_inv1: Arr2<fsize> = Arr2::new(dim, dim);
        hess_inv1.set_diag(hess_inv_scalar);
        let mbuf0: Arr2<fsize> = Arr2::new(dim, dim);
        let mbuf1: Arr2<fsize> = Arr2::new(dim, dim);
        let mbuf2: Arr2<fsize> = Arr2::new(dim, dim);
        Self { dim, tol: 1e-4, max_iter: 1000, max_search: 100, alpha: 0.1, beta: 0.5, hess_inv_scalar, fx: 0., 
            fx_new: 0., x_new, delta_x, delta_grad, dir, grad0, grad1, hess_inv0, hess_inv1, mbuf0, mbuf1, mbuf2 }
    }

    #[inline]
    pub fn run_mut<F: FunMut + GradMut, VT: RVecMut<fsize>>(
        &mut self,
        fun: &mut F,
        x: &mut VT,
        prog: usize,
    ) -> Result<usize, fsize> {
        let mut res: fsize = fsize::INFINITY;
        let mut total_iter: usize = self.max_iter + 1;
        self.fx = fun.f(x);
        fun.df(x, self.fx, &mut self.grad0);
        let mut rev_ref: bool = false;

        for i in 1..self.max_iter+1 {
            let (grad, grad_new, hess_inv, hess_inv_new) = match rev_ref {
                false => (&self.grad0, &mut self.grad1, &self.hess_inv0, &mut self.hess_inv1),
                true => (&self.grad1, &mut self.grad0, &self.hess_inv1, &mut self.hess_inv0),
            };
            dsymv(-1., hess_inv, grad, 0., &mut self.dir, LOWER); // dir = - hess_inv * grad
            let s0 = ddot(grad, &self.dir);
            let mut t: fsize = 1.;
            let mut search: usize = 0;

            self.delta_x.assign_scale(&self.dir, t);
            x.addassign(&self.delta_x);
            self.fx = fun.f(x);

            /*loop {
                self.delta_x.assign_scale(&self.dir, t);
                self.x_new.assign_add(x, &self.delta_x);
                self.fx_new = fun.f(&self.x_new);
                if self.fx_new > self.fx + self.alpha * t * s0 {
                    t *= self.beta;
                } else {
                    x.addassign(&self.delta_x);
                    self.fx = self.fx_new;
                    break;
                }
                search += 1;
                if search > self.max_search {
                    panic!("Maximum line search exceeded. Current itertion: {i}.");
                }
            }*/
            fun.df(x, self.fx, grad_new);
            res = grad_new.norm();
            if res < self.tol {
                total_iter = i;
                break;
            }
            // Update the Hessian inverse.
            self.mbuf0.reset();
            self.mbuf1.reset();
            self.mbuf0.set_diag(1.);
            self.mbuf1.set_diag(1.);

            self.delta_grad.assign_sub(grad_new, grad);
            let s1 = 1./ ddot(&self.delta_x, &self.delta_grad);
            dger(-s1, &self.delta_x, &self.delta_grad, &mut self.mbuf0);
            dger(-s1, &self.delta_grad, &self.delta_x, &mut self.mbuf1);
            dsymm(1., hess_inv, &self.mbuf1, 0., &mut self.mbuf2, LEFT, LOWER);
            dgemm_notransa(1., &self.mbuf0, &self.mbuf2, 0., hess_inv_new, NO_TRANS, NO_TRANS);
            dsyr(s1, &self.delta_x, hess_inv_new, LOWER);

            rev_ref ^= true;
            if i % prog == 0 {
                println!("Iterations: {} / {}", i, self.max_iter);
            }
        }
        match total_iter > self.max_iter {
            false => Ok(total_iter),
            true => Err(res),
        }
    }
}

#[derive(Clone)]
pub struct Newton {
    pub dim: usize,
    pub tol: fsize,
    pub max_iter: usize,
    pub fx: fsize,
    pub grad: Arr1<fsize>,
    pub hess: Arr2<fsize>,
}

impl Newton
{
    #[inline]
    pub fn new( dim: usize ) -> Self {
        let grad: Arr1<fsize> = Arr1::new(dim);
        let hess: Arr2<fsize> = Arr2::new(dim, dim);
        Self { dim, tol: 1e-4, max_iter: 1000, fx: 0., grad, hess }
    }

    #[inline]
    pub fn run_mut<F: FunMut + GradHessMut, VT: RVecMut<fsize>>(
        &mut self,
        fun: &mut F,
        x: &mut VT,
        prog: usize,
    ) -> Result<usize, fsize> {
        let mut res: fsize = fsize::INFINITY;
        let mut total_iter: usize = self.max_iter + 1;

        for i in 1..self.max_iter+1 {
            self.fx = fun.f(x);
            fun.df_ddf(x, self.fx, &mut self.grad, &mut self.hess);
            res = self.grad.norm();
            if res < self.tol {
                total_iter = i;
                break;
            }
            dposv(&mut self.hess, &mut self.grad, LOWER);
            x.subassign(&self.grad);
            if i % prog == 0 {
                println!("Iterations: {} / {}", i, self.max_iter);
            }
        }
        match total_iter > self.max_iter {
            false => Ok(total_iter),
            true => Err(res),
        }
    }
}

#[derive(Clone)]
pub struct NumGrad { // Central differentiation
    pub dim: usize,
    pub x: Vec<fsize>,
    pub h: Vec<fsize>,
}

impl NumGrad
{
    #[inline]
    pub fn default_h() -> fsize {
        fsize::EPSILON.sqrt()
    }

    #[inline]
    pub fn new( dim: usize ) -> Self {
        let x: Vec<fsize> = vec![0.; dim];
        let h: Vec<fsize> = vec![Self::default_h(); dim];
        Self { dim, x, h }
    }

    #[inline] #[cfg(not(feature="numdiff_twoside"))]
    pub fn run_mut<F: FunMut, VT1: RVec<fsize>, VT2: RVecMut<fsize>>(
        &mut self, fun: &mut F, x0: &VT1, y0: fsize, grad: &mut VT2,
    ) {
        self.x.copy(x0);
        let mut y1: fsize;
        let mut h: fsize;
        assert_multi_eq!(self.dim, self.x.size(), self.h.size(), grad.size());
        for k in 0..self.dim {
            h = self.h[k];
            self.x[k] += h; // x+h
            y1 = fun.f(&self.x); // f(x+h)
            *grad.idxm(k) = (y1 - y0) / h;
            self.x[k] -= h; // x
        }
    }

    #[inline] #[cfg(feature="numdiff_twoside")]
    pub fn run_mut<F: FunMut, VT1: RVec<fsize>, VT2: RVecMut<fsize>>(
        &mut self, fun: &mut F, x0: &VT1, _y0: fsize, grad: &mut VT2,
    ) {
        self.x.copy(x0);
        let mut y1: fsize;
        let mut y2: fsize;
        let mut h: fsize;
        assert_multi_eq!(self.dim, self.x.size(), self.h.size(), grad.size());
        for k in 0..self.dim {
            h = self.h[k];
            self.x[k] += h; // x+h
            y1 = fun.f(&self.x); // f(x+h)
            self.x[k] -= 2.* h; // x-h
            y2 = fun.f(&self.x); // f(x-h)
            *grad.idxm(k) = (y1 - y2) / (2.* h);
            self.x[k] += h; // x
        }
    }
}

#[derive(Clone)]
pub struct NumHess {
    pub dim: usize,
    pub x: Vec<fsize>,
    pub h: Vec<fsize>,
    #[cfg(not(feature="numdiff_twoside"))] pub yh: Vec<fsize>,
}

impl NumHess
{
    #[inline]
    pub fn new( dim: usize ) -> Self {
        let x: Vec<fsize> = vec![0.; dim];
        let h: Vec<fsize> = vec![1e-6; dim];
        #[cfg(not(feature="numdiff_twoside"))] let yh: Vec<fsize> = vec![0.; dim];
        #[cfg(not(feature="numdiff_twoside"))] return Self { dim, x, h, yh };
        #[cfg(feature="numdiff_twoside")] return Self { dim, x, h };
    }

    #[inline] #[cfg(not(feature="numdiff_twoside"))]
    pub fn run_mut<F: FunMut, VT1: RVec<fsize>, VT2: RVecMut<fsize>, MT: RMatMut<fsize>>(
        &mut self, fun: &mut F, x0: &VT1, y0: fsize, grad: &mut VT2, hess: &mut MT,
    ) {
        // One-sided numeric difference, requires (k+1)(k+2)/2 function evaluations per iteration.
        assert_multi_eq!(self.dim, self.x.size(), self.h.size(), grad.size());
        //assert_multi_eq!(self.dim, ddfx.nrow(), ddfx.ncol());
        self.x.copy(x0);
        let mut y1: fsize;
        let mut y2: fsize;
        let mut h: fsize;
        for k in 0..self.dim {
            h = self.h[k];
            // gradient
            self.x[k] += h; // x+h
            y1 = fun.f(&self.x); // f(x+h)
            self.yh[k] = y1; // f(x+h)
            *grad.idxm(k) = (y1 - y0) / h;
            // Hessian, diagonal
            self.x[k] += h; // x+2h
            y2 = fun.f(&self.x); // f(x+2h)
            *hess.idxm2(k, k) = (y2 - 2.*y1 + y0) / h.powi(2);
            self.x[k] -= 2.* h; // x
        }
        for k in 0..self.dim {
            h = self.h[k];
            self.x[k] += h; // x+h
            y1 = self.yh[k];
            for i in k+1..self.dim {
                // Hessian, lower triangular
                self.x[i] += h; // x'+h
                y2 = fun.f(&self.x);  // f(x+h, x'+h)
                *hess.idxm2(i, k) = (y2 - self.yh[i] - y1 + y0) / h.powi(2);
                self.x[i] -= h; // x'
            }
            self.x[k] -= h; // x
        }
    }

    #[inline] #[cfg(feature="numdiff_twoside")]
    pub fn run_mut<F: FunMut, VT1: RVec<fsize>, VT2: RVecMut<fsize>, MT: RMatMut<fsize>>(
        &mut self, fun: &mut F, x0: &VT1, y0: fsize, grad: &mut VT2, hess: &mut MT,
    ) {
        // Two-sided numeric difference, requires 2k^2+2k+1 function evaluations per iteration.
        assert_multi_eq!(self.dim, self.x.size(), self.h.size(), grad.size());
        //assert_multi_eq!(self.dim, ddfx.nrow(), ddfx.ncol());
        self.x.copy(x0);
        let mut y1: fsize;
        let mut y2: fsize;
        let mut y3: fsize;
        let mut y4: fsize;
        let mut h: fsize;
        for k in 0..self.dim {
            h = self.h[k];
            // gradient
            self.x[k] += h; // x+h
            y1 = fun.f(&self.x); // f(x+h)
            self.x[k] -= 2.* h; // x-h
            y2 = fun.f(&self.x); // f(x-h)
            *grad.idxm(k) = (y1 - y2) / (2.* h);
            // Hessian, diagonal
            self.x[k] += 3.* h; // x+2h
            y1 = fun.f(&self.x); // f(x+2h)
            self.x[k] -= 4.* h; // x-2h
            y2 = fun.f(&self.x); // f(x-2h)
            *hess.idxm2(k, k) = (y1 - 2.*y0 + y2) / (4.* h.powi(2));
            // Hessian, lower triangular
            for i in k+1..self.dim {
                self.x[k] += 3.* h; // x+h
                self.x[i] += h; // x'+h
                y1 = fun.f(&self.x); // f(x+h, x'+h)
                self.x[k] -= 2.* h; // x-h
                y2 = fun.f(&self.x); // f(x-h, x'+h)
                self.x[k] += 2.* h; // x+h
                self.x[i] -= 2.* h; // x'-h
                y3 = fun.f(&self.x); // f(x+h, x'-h)
                self.x[k] -= 2.* h; // x-h
                y4 = fun.f(&self.x); // f(x-h, x'-h)
                *hess.idxm2(i, k) = (y1 - y2 - y3 + y4) / (4.* h.powi(2));
                self.x[k] += h; // x
                self.x[i] += h; // x'
            }
        }
    }
}

pub struct Rosenbrock {
    pub n_fcall: usize
}

impl Rosenbrock
{
    #[inline]
    pub fn new() -> Self {
        Self { n_fcall: 0 }
    }
}

impl FunMut for Rosenbrock
{
    #[inline]
    fn f<VT: RVec<fsize>>( &mut self, x: &VT ) -> fsize {
        self.n_fcall += 1;
        let x1 = x.idx(0);
        let x2 = x.idx(1);
        (1.- x1).powi(2) + 100.* (x2 - x1.powi(2)).powi(2)
    }
}

pub struct RosenbrockDiff {
    pub base: Rosenbrock,
    pub diff_grad: NumGrad,
    pub diff_hess: NumHess,
}

impl RosenbrockDiff {
    #[inline]
    pub fn new() -> Self {
        Self { base: Rosenbrock::new(), diff_grad: NumGrad::new(2), diff_hess: NumHess::new(2) }
    }
}

impl FunMut for RosenbrockDiff {
    #[inline]
    fn f<VT: RVec<fsize>>( &mut self, x: &VT ) -> fsize {
        self.base.f(x)
    }
}

impl GradMut for RosenbrockDiff {
    #[inline]
    fn df<VT1: RVec<fsize>, VT2: RVecMut<fsize>>( &mut self, x0: &VT1, y0: fsize, grad: &mut VT2 ) {
        self.diff_grad.run_mut(&mut self.base, x0, y0, grad);
    }
}

impl GradHessMut for RosenbrockDiff
{
    #[inline]
    fn df_ddf<VT1: RVec<fsize>, VT2: RVecMut<fsize>, MT: RMatMut<fsize>>(
        &mut self, x: &VT1, fx: fsize, dfx: &mut VT2, ddfx: &mut MT,
    ) {
        self.diff_hess.run_mut(&mut self.base, x, fx, dfx, ddfx);
    }
}