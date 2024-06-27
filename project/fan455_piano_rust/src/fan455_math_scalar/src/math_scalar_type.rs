use num_complex::Complex;
use num_traits::{NumRef, NumAssignRef, MulAdd, MulAddAssign};

#[allow(non_camel_case_types)] 
pub type c64 = Complex<f32>;

#[allow(non_camel_case_types)] 
pub type c128 = Complex<f64>;

#[cfg(not(feature="use_32bit_float"))] #[allow(non_camel_case_types)]
pub type fsize = f64;

#[cfg(feature="use_32bit_float")] #[allow(non_camel_case_types)]
pub type fsize = f32;

#[cfg(not(feature="use_32bit_float"))] #[allow(non_camel_case_types)] 
pub type csize = c128;

#[cfg(feature="use_32bit_float")] #[allow(non_camel_case_types)]
pub type csize = c64;

#[cfg(not(feature="use_32bit_float"))] #[allow(non_camel_case_types)] 
pub const PI: f64 = std::f64::consts::PI;

#[cfg(feature="use_32bit_float")] #[allow(non_camel_case_types)]
pub const PI: f32 = std::f32::consts::PI;

#[cfg(not(feature="use_32bit_float"))] #[allow(non_camel_case_types)] 
pub const C_ZERO: c128 = c128{re: 0., im: 0.};

#[cfg(feature="use_32bit_float")] #[allow(non_camel_case_types)]
pub const C_ZERO: c64 = c64{re: 0., im: 0.};

#[cfg(not(feature="use_32bit_float"))] #[allow(non_camel_case_types)] 
pub const C_ONE: c128 = c128{re: 1., im: 0.};

#[cfg(feature="use_32bit_float")] #[allow(non_camel_case_types)]
pub const C_ONE: c64 = c64{re: 1., im: 0.};

pub trait General: Sized + Default + Copy + Clone {}
impl<T: Sized + Default + Copy + Clone> General for T {}

pub trait Numeric: General + NumRef + NumAssignRef + MulAdd<Output=Self> + MulAddAssign {}
impl<T: General + NumRef + NumAssignRef + MulAdd<Output=T> + MulAddAssign> Numeric for T {}

pub trait Float: Numeric + num_traits::Float {}
impl<T: Numeric + num_traits::Float> Float for T {}

pub const HALF_PI: fsize = PI/2.;
pub const TWO_PI: fsize = 2.*PI;