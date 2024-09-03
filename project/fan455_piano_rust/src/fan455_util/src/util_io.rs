use std::fs::File;
use std::io::{Read, Write};
use fan455_util_macro::*;
use crate::{elem, mzip};
use super::util_basic::*;
use std::mem::size_of;


pub struct StaBinFile {
    pub sta_bytes: Vec<u8>,
}

impl StaBinFile
{
    #[inline]
    pub fn write( &self, path: &str ) {
        let mut file = File::create(path).unwrap();
        // The bytes 0..8 store the length of sta_bytes.
        file.write( &usize::to_le_bytes(self.sta_bytes.len()) ).unwrap();
        file.write( &self.sta_bytes ).unwrap();
    }

    #[inline]
    pub fn read( path: &str ) -> Self {
        let mut file = File::open(path).unwrap();

        let mut b0: [u8; 8] = [0; 8];
        file.read_exact(&mut b0).unwrap();
        let sta_len = usize::from_le_bytes(b0);

        let mut sta_bytes: Vec<u8> = vec![0; sta_len];
        file.read_exact(&mut sta_bytes).unwrap();
        
        Self {sta_bytes}
    }
}


pub trait StaBinStruct<const FIELDS_N: usize> 
where Self: Sized,
{
    const SIZES: [usize; FIELDS_N];
    const POS: [usize; FIELDS_N] = const_lens_to_begs(&Self::SIZES);
    const STA_BIN_SIZE: usize = const_sum_array_usize(&Self::SIZES);

    fn struct_to_bytes( &self, 
        sta_bytes: &mut [u8], sta_pos: &[usize; FIELDS_N],
    );
    fn bytes_to_struct( 
        sta_bytes: &[u8], sta_pos: &[usize; FIELDS_N],
    ) -> Self;

    #[inline]
    fn sta_len() -> usize {
        let mut s = 0;
        for s_ in Self::SIZES.iter() {
            s += s_;
        }
        s
    }

    #[inline]
    fn to_bin_vec( vec: &Vec<Self> ) -> StaBinFile {
        let sta_len = Self::sta_len();
        let mut sta_bytes: Vec::<u8> = vec![0; vec.len()*sta_len];
        let mut sta_pos: [usize; FIELDS_N] = [0; FIELDS_N];
        let mut beg_0: usize = 0;

        for s in vec.iter() {
            for i in 0..FIELDS_N {
                sta_pos[i] = beg_0;
                beg_0 += Self::SIZES[i];
            }
            s.struct_to_bytes(&mut sta_bytes, &sta_pos);
        }
        StaBinFile {sta_bytes}
    }

    #[inline]
    fn from_bin_vec( bin: &StaBinFile ) -> Vec<Self> {
        let n0 = bin.sta_bytes.len();
        let n1 = Self::sta_len();
        assert_eq!(n0 % n1, 0);

        let n = n0 / n1;
        let mut vec = Vec::<Self>::with_capacity(n);
        let mut sta_pos: [usize; FIELDS_N] = [0; FIELDS_N];
        let mut beg_0: usize = 0;

        for _ in 0..n {
            for i in 0..FIELDS_N {
                sta_pos[i] = beg_0;
                beg_0 += Self::SIZES[i]
            }
            vec.push( Self::bytes_to_struct(&bin.sta_bytes, &sta_pos) );
        }
        vec
    }

    #[inline]
    fn write_bin_vec( path: &str, vec: &Vec<Self> ) {
        let bin = Self::to_bin_vec(vec);
        bin.write(path);
    }

    #[inline]
    fn read_bin_vec( path: &str ) -> Vec<Self> {
        let bin = StaBinFile::read(path);
        Self::from_bin_vec(&bin)
    }

    #[inline]
    fn to_bytes_sl( &self, bytes: &mut [u8] ) {
        self.struct_to_bytes(bytes, &Self::POS);
    }

    #[inline]
    fn from_bytes_sl( bytes: &[u8] ) -> Self {
        Self::bytes_to_struct(bytes, &Self::POS)
    }
}


pub struct DynBinFile {
    pub sta_bytes: Vec<u8>,
    pub dyn_bytes: Vec<u8>,
}

impl DynBinFile
{
    #[inline]
    pub fn write( &self, path: &str ) {
        let mut file = File::create(path).unwrap();
        // The bytes 0..8 store the length of sta_bytes, bytes 8..16 store the length of dyn_bytes, 
        file.write( &usize::to_le_bytes(self.sta_bytes.len()) ).unwrap();
        file.write( &usize::to_le_bytes(self.dyn_bytes.len()) ).unwrap();
        file.write( &self.sta_bytes ).unwrap();
        file.write( &self.dyn_bytes ).unwrap();
    }

    #[inline]
    pub fn read( path: &str ) -> Self {
        let mut file = File::open(path).unwrap();

        let mut b0: [u8; 8] = [0; 8];
        file.read_exact(&mut b0).unwrap();
        let sta_len = usize::from_le_bytes(b0);
        file.read_exact(&mut b0).unwrap();
        let dyn_len = usize::from_le_bytes(b0);

        let mut sta_bytes: Vec<u8> = vec![0; sta_len];
        let mut dyn_bytes: Vec<u8> = vec![0; dyn_len];
        file.read_exact(&mut sta_bytes).unwrap();
        file.read_exact(&mut dyn_bytes).unwrap();
        
        Self {sta_bytes, dyn_bytes}
    }
}


pub trait DynBinStruct<const FIELDS_N: usize, const DYN_N: usize> 
where Self: Sized,
{
    const SIZES: [usize; FIELDS_N]; // Not including (beg, end)

    fn dyn_sizes( &self ) -> [usize; DYN_N];

    fn struct_to_bytes( &self, 
        sta_bytes: &mut [u8], sta_pos: &[usize; FIELDS_N], 
        dyn_bytes: &mut [u8], dyn_pos: &[[usize; 2]; DYN_N],
    );
    fn bytes_to_struct( 
        sta_bytes: &[u8], sta_pos: &[usize; FIELDS_N], 
        dyn_bytes: &[u8], dyn_pos: &[[usize; 2]; DYN_N],
    ) -> Self;

    #[inline]
    fn dyn_len( &self ) -> usize {
        let sizes = self.dyn_sizes();
        let mut s: usize = 0;
        for s_ in sizes {
            s += s_;
        }
        s
    }

    #[inline]
    fn sta_len() -> usize {
        let mut s = 16*DYN_N;
        for s_ in Self::SIZES.iter() {
            s += s_;
        }
        s
    }

    #[inline]
    fn to_bin_vec( vec: &Vec<Self> ) -> DynBinFile {
        let sta_len = Self::sta_len();
        let mut dyn_len_sum: usize = 0;
        for s in vec.iter() {
            let s_ = s.dyn_len();
            dyn_len_sum += s_;
        }
        let mut sta_bytes: Vec::<u8> = vec![0; vec.len()*sta_len];
        let mut dyn_bytes: Vec::<u8> = vec![0; dyn_len_sum];

        let mut sta_pos: [usize; FIELDS_N] = [0; FIELDS_N];
        let mut dyn_pos: [[usize; 2]; DYN_N] = [[0, 0]; DYN_N];

        let mut beg_0: usize = 0;
        let mut beg_1: usize = 0;
        let mut end_1: usize;

        for s in vec.iter() {
            for i in 0..FIELDS_N {
                sta_pos[i] = beg_0;
                beg_0 += Self::SIZES[i];
            }
            let dyn_sizes = s.dyn_sizes();
            for i in 0..DYN_N {
                end_1 = beg_1 + dyn_sizes[i];
                dyn_pos[i] = [beg_1, end_1];

                sta_bytes[beg_0..beg_0+8].copy_from_slice( &usize::to_le_bytes(beg_1) );
                beg_0 += 8;

                sta_bytes[beg_0..beg_0+8].copy_from_slice( &usize::to_le_bytes(end_1) );
                beg_0 += 8;

                beg_1 = end_1;
            }
            s.struct_to_bytes(&mut sta_bytes, &sta_pos, &mut dyn_bytes, &dyn_pos);
        }
        DynBinFile{ sta_bytes, dyn_bytes }
    }

    #[inline]
    fn from_bin_vec( bin: &DynBinFile ) -> Vec<Self> {
        let n0 = bin.sta_bytes.len();
        let n1 = Self::sta_len();
        assert_eq!(n0 % n1, 0);

        let n = n0 / n1;
        let n2 = n1 - 16*DYN_N;
        let mut vec = Vec::<Self>::with_capacity(n);

        let mut sta_pos: [usize; FIELDS_N] = [0; FIELDS_N];
        let mut dyn_pos: [[usize; 2]; DYN_N] = [[0, 0]; DYN_N];

        let mut beg_0: usize = 0;
        let mut beg_1: usize;
        let mut end_1: usize;
        let mut beg_: usize;
        let mut end_: usize;
        let mut u8_8: [u8; 8] = [0; 8];

        for s in bin.sta_bytes.chunks_exact(n1) {
            for i in 0..FIELDS_N {
                sta_pos[i] = beg_0;
                beg_0 += Self::SIZES[i]
            }
            beg_ = n2;
            for i in 0..DYN_N {
                beg_0 += 16;

                end_ = beg_ + 8;
                u8_8.copy_from_slice( &s[beg_..end_] );
                beg_1 = usize::from_le_bytes(u8_8);
                beg_ = end_;

                end_ = beg_ + 8;
                u8_8.copy_from_slice( &s[beg_..end_] );
                end_1 = usize::from_le_bytes(u8_8);
                beg_ = end_;

                dyn_pos[i] = [beg_1, end_1];
            }
            vec.push( Self::bytes_to_struct(&bin.sta_bytes, &sta_pos, &bin.dyn_bytes, &dyn_pos) );
        }
        vec
    }

    #[inline]
    fn write_bin_vec( path: &str, vec: &Vec<Self> ) {
        let bin = Self::to_bin_vec(vec);
        bin.write(path);
    }

    #[inline]
    fn read_bin_vec( path: &str ) -> Vec<Self> {
        let bin = DynBinFile::read(path);
        Self::from_bin_vec(&bin)
    }
}


pub trait StaBinSize {
    const STA_BIN_SIZE: usize;
}

pub trait DynBinSize {
    fn dyn_bin_size( &self ) -> usize;
}

impl StaBinSize for bool { const STA_BIN_SIZE: usize = 1; }
impl StaBinSize for u8 { const STA_BIN_SIZE: usize = 1; }
impl StaBinSize for usize { const STA_BIN_SIZE: usize = 8; }
impl StaBinSize for isize { const STA_BIN_SIZE: usize = 8; }
impl StaBinSize for f64 { const STA_BIN_SIZE: usize = 8; }
impl<T: StaBinSize, const N: usize> StaBinSize for [T; N] { const STA_BIN_SIZE: usize = N*T::STA_BIN_SIZE; }

impl<T> StaBinSize for Vec<T> { const STA_BIN_SIZE: usize = 0; }
impl<T: Sized> DynBinSize for Vec<T> 
{ 
    #[inline] 
    fn dyn_bin_size( &self ) -> usize { self.len()*size_of::<T>() } 
}


pub trait ToBytesSl where Self: Sized {
    fn to_bytes_sl( &self, bytes: &mut [u8] );
}

pub trait FromBytesSl where Self: Sized {
    fn from_bytes_sl( bytes: &[u8] ) -> Self;
}

impl ToBytesSl for u8 
{
    #[inline]
    fn to_bytes_sl( &self, bytes: &mut [u8] ) { bytes[0] = *self; }
}
impl FromBytesSl for u8 
{
    #[inline]
    fn from_bytes_sl( bytes: &[u8] ) -> Self { bytes[0] }
}

impl ToBytesSl for bool 
{
    #[inline]
    fn to_bytes_sl( &self, bytes: &mut [u8] ) { bytes[0] = u8::from(*self); }
}
impl FromBytesSl for bool 
{
    #[inline]
    fn from_bytes_sl( bytes: &[u8] ) -> Self { bytes[0] != 0 }
}

impl_ToBytesSl_for!( usize, isize, f64 );
impl_FromBytesSl_for!( usize, isize, f64 );

impl_ToBytesSl_for_array!( usize, isize, f64 );
impl_FromBytesSl_for_array!( usize, isize, f64 );


#[inline]
pub fn vec_to_bytes<T>( vec: &[T], bytes: &mut [u8] ) 
where T: Sized + Copy + ToBytesSl,
{
    for elem!(vec_, bytes_) in mzip!(vec.iter(), bytes.chunks_exact_mut(size_of::<T>())) {
        vec_.to_bytes_sl(bytes_);
    }
}

#[inline]
pub fn bytes_to_vec<T>( bytes: &[u8], vec: &mut [T] ) 
where T: Sized + Copy + FromBytesSl,
{
    for elem!(vec_, bytes_) in mzip!(vec.iter_mut(), bytes.chunks_exact(size_of::<T>())) {
        *vec_ = T::from_bytes_sl(bytes_);
    }
}