//use super::util_basic::*;
use std::iter::{Iterator, Repeat};


pub trait ScalarTrait: Clone {}
pub trait ArrayTrait {}

impl ScalarTrait for usize {}
impl ScalarTrait for f64 {}
impl<T: ScalarTrait, const N: usize> ArrayTrait for [T; N] {}
impl<T: ScalarTrait> ArrayTrait for Vec<T> {}


pub trait ScalarOrArray<'a, T:'a> 
{
    type Iter: Iterator<Item = &'a T>;
    fn it_scalar_or_array( &'a self ) -> Self::Iter;
}

impl<'a, T: 'a+ScalarTrait> ScalarOrArray<'a, T> for T 
{
    type Iter = Repeat<&'a T>;

    #[inline]
    fn it_scalar_or_array( &'a self ) -> Self::Iter {
        std::iter::repeat(self)
    }
}

impl<'a, T: 'a+ScalarTrait, const N: usize> ScalarOrArray<'a, T> for [T; N] 
{
    type Iter = std::slice::Iter<'a, T>;
    
    #[inline]
    fn it_scalar_or_array( &'a self ) -> Self::Iter {
        self.iter()
    }
}

impl<'a, T: 'a+ScalarTrait> ScalarOrArray<'a, T> for Vec<T> 
{
    type Iter = std::slice::Iter<'a, T>;
    
    #[inline]
    fn it_scalar_or_array( &'a self ) -> Self::Iter {
        self.iter()
    }
}

/*pub fn print_iter<'a, I1: ScalarOrArray<'a, usize>, I2: ScalarOrArray<'a, f64>>(x: &'a I1, y: &'a I2) {
    for (x_, y_) in std::iter::zip(x.it_scalar_or_array(), y.it_scalar_or_array()) {
        let c = 1.3*y_;
        println!("x_ = {x_}");
        println!("y_ = {c}");
    }
}*/
