
pub trait FloatIsZero 
{
    fn is_zero_float( self ) -> bool;
}

impl FloatIsZero for f32
{
    #[inline]
    fn is_zero_float( self ) -> bool {
        self.abs() < f32::EPSILON
    }
}

impl FloatIsZero for f64
{
    #[inline]
    fn is_zero_float( self ) -> bool {
        self.abs() < f64::EPSILON
    }
}


pub trait IntToBool 
{
    fn to_bool( self ) -> bool;
}

impl IntToBool for u8 { fn to_bool( self ) -> bool { self != 0 } }
impl IntToBool for usize { fn to_bool( self ) -> bool { self != 0 } }
