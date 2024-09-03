use std::ops::Add;
use num_complex::Complex;
use num_traits::{NumRef, NumAssignRef, MulAdd, MulAddAssign};

#[allow(non_camel_case_types)] 
pub type c64 = Complex<f32>;

#[allow(non_camel_case_types)] 
pub type c128 = Complex<f64>;

#[allow(non_camel_case_types)] 
pub const PI: f64 = std::f64::consts::PI;

#[allow(non_camel_case_types)] 
pub const C_ZERO: c128 = c128{re: 0., im: 0.};

#[allow(non_camel_case_types)] 
pub const C_ONE: c128 = c128{re: 1., im: 0.};

pub trait General: Sized + Default + Copy + Clone {}
impl<T: Sized + Default + Copy + Clone> General for T {}

pub trait Numeric: General + NumRef + NumAssignRef + MulAdd<Output=Self> + MulAddAssign {}
impl<T: General + NumRef + NumAssignRef + MulAdd<Output=T> + MulAddAssign> Numeric for T {}

pub trait Float: Numeric + num_traits::Float {}
impl<T: Numeric + num_traits::Float> Float for T {}

pub const HALF_PI: f64 = 0.5*PI;
pub const TWO_PI: f64 = 2.*PI;
pub const THREE_HALF_PI: f64 = 1.5*PI;


#[derive(Copy, Clone)]
pub enum MayBeZero {
    Zero,
    NonZero(f64),
}

impl Add for MayBeZero
{
    type Output = MayBeZero;
    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Zero, Self::Zero) => Self::Zero,
            (Self::NonZero(x), Self::Zero) => Self::NonZero(x),
            (Self::Zero, Self::NonZero(x)) => Self::NonZero(x),
            (Self::NonZero(x), Self::NonZero(y)) => Self::NonZero(x+y),
        }
    }
}
