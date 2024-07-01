use fan455_math_scalar::*;
use fan455_arrf64::*;
//use fan455_util::{elem, mzip};
use std::iter::zip;

use crate::{mat_inv_2, mat_vec_2};


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
    pub fn solve_1d<F: Fn(fsize)->[fsize; 2]>( 
        &self, call: F, x0: fsize 
    ) -> Result<(fsize, usize), (fsize, fsize)> {
        // "call" computes zero order and first order.
        let mut x: fsize = x0;
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
    pub fn solve_2d<F: Fn([fsize; 2])->([fsize; 2], [fsize; 4])>( 
        &self, call: F, p0: [fsize; 2] 
    ) -> Result<([fsize; 2], usize), ([fsize; 2], [fsize; 2])> {
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











