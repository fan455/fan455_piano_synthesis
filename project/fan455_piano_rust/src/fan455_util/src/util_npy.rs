use super::util_string::vec_from_string_no_bracket;
use std::default::Default;
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::marker::PhantomData;
use regex::Regex;
use std::iter::zip;
use std::clone::Clone;
use fan455_math_scalar::{c64, c128};


macro_rules! elem {
    ($x:tt) => {
        ($x)
    };
    ($x:tt, $y:tt) => {
        ($x, $y)
    };
    ($x:tt, $y:tt, $($rest:tt),+) => {
        ($x, elem!($y, $($rest),+))
    };
}


macro_rules! mzip {
    ($x:expr) => {
        ($x)
    };
    ($x:expr, $y:expr) => {
        std::iter::zip($x, $y)
    };
    ($x:expr, $y:expr, $($rest:expr),+) => {
        std::iter::zip($x, mzip!($y, $($rest),+))
    };
}


pub fn shape_to_size( shape: &Vec<usize> ) -> usize {
    let mut size: usize = 1;
    for x in shape {
        size *= x;
    }
    size
}


pub trait NpyVecLenScaler {
    fn npy_vec_len( len: usize ) -> usize;
}

impl NpyVecLenScaler for f32 {
    fn npy_vec_len( len: usize ) -> usize { len }
}

impl NpyVecLenScaler for f64 {
    fn npy_vec_len( len: usize ) -> usize { len }
}

impl NpyVecLenScaler for u8 {
    fn npy_vec_len( len: usize ) -> usize { len }
}

impl NpyVecLenScaler for usize {
    fn npy_vec_len( len: usize ) -> usize { len }
}

impl NpyVecLenScaler for isize {
    fn npy_vec_len( len: usize ) -> usize { len }
}

impl<const N: usize> NpyVecLenScaler for [f32; N] {
    fn npy_vec_len( len: usize ) -> usize { len/N + (len%N!=0) as usize }
}

impl<const N: usize> NpyVecLenScaler for [f64; N] {
    fn npy_vec_len( len: usize ) -> usize { len/N + (len%N!=0) as usize }
}

impl<const N: usize> NpyVecLenScaler for [u8; N] {
    fn npy_vec_len( len: usize ) -> usize { len/N + (len%N!=0) as usize }
}

impl<const N: usize> NpyVecLenScaler for [usize; N] {
    fn npy_vec_len( len: usize ) -> usize { len/N + (len%N!=0) as usize }
}

impl<const N: usize> NpyVecLenScaler for [isize; N] {
    fn npy_vec_len( len: usize ) -> usize { len/N + (len%N!=0) as usize }
}


pub trait NpyDescrGetter {
    const NPY_DESCR: &'static str;
}

impl NpyDescrGetter for u8 {
    const NPY_DESCR: &'static str = "'|u1'";
}

impl NpyDescrGetter for usize {
    const NPY_DESCR: &'static str = "'<u8'";
}

impl NpyDescrGetter for f32 {
    const NPY_DESCR: &'static str = "'<f4'";
}

impl NpyDescrGetter for f64 {
    const NPY_DESCR: &'static str = "'<f8'";
}

impl NpyDescrGetter for c64 {
    const NPY_DESCR: &'static str = "'<c8'";
}

impl NpyDescrGetter for c128 {
    const NPY_DESCR: &'static str = "'<c16'";
}

impl<const N: usize> NpyDescrGetter for [u8; N] {
    const NPY_DESCR: &'static str = "'|u1'";
}

impl<const N: usize> NpyDescrGetter for [usize; N] {
    const NPY_DESCR: &'static str = "'<u8'";
}

impl<const N: usize> NpyDescrGetter for [f32; N] {
    const NPY_DESCR: &'static str = "'<f4'";
}

impl<const N: usize> NpyDescrGetter for [f64; N] {
    const NPY_DESCR: &'static str = "'<f8'";
}


pub struct NpyObject<T> {
    pub npy_file: File,
    pub npy_version: [u8; 2],
    pub header_len: usize,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
    pub size: usize,
    phantom: PhantomData<T>,
}


impl<T> NpyObject<T>
{
    #[inline]
    pub fn new_reader( npy_path: &String ) -> Self {
        Self {
            npy_file: File::open(npy_path).unwrap(),
            npy_version: [1, 0],
            header_len: 0,
            fortran_order: false,
            shape: Vec::new(),
            size: 1,
            phantom: PhantomData,
        }
    }


    #[inline]
    pub fn change_file( &mut self, npy_path: &String ) {
        self.npy_file = File::open(npy_path).unwrap();
    }


    #[inline]
    pub fn seek_to_data( &mut self ) {
        self.npy_file.seek( std::io::SeekFrom::Start((8+self.header_len) as u64) ).unwrap();
    }


    #[inline]
    pub fn new_writer(
        npy_path: &String,
        npy_version: [u8; 2],
        fortran_order: bool,
        shape: Vec<usize>,
    ) -> Self {
        let size: usize = shape_to_size(&shape);
        Self {
            npy_file: File::create(npy_path).unwrap(),
            npy_version,
            header_len: 0,
            fortran_order,
            shape,
            size,
            phantom: PhantomData,
        }
    }
    

    #[inline]
    pub fn read_header( &mut self ) -> Result<(), String> {
        // Read magic.
        let mut magic_bytes: [u8; 8] = [0; 8];
        self.npy_file.read(magic_bytes.as_mut_slice()).unwrap();
        self.npy_version[0] = magic_bytes[6];
        self.npy_version[1] = magic_bytes[7];

        // Read header length.
        if self.npy_version == [1, 0] {
            let mut header_len_bytes: [u8; 2] = [0; 2];
            self.npy_file.read(header_len_bytes.as_mut_slice()).unwrap();
            self.header_len = u16::from_le_bytes(header_len_bytes) as usize;

        } else if self.npy_version == [2, 0] || self.npy_version == [3, 0] {
            let mut header_len_bytes: [u8; 4] = [0; 4];
            self.npy_file.read(header_len_bytes.as_mut_slice()).unwrap();
            self.header_len = u32::from_le_bytes(header_len_bytes) as usize;

        } else {
            return Err(format!("Unsupported npy version: {}.{}. Only supports version 1.0, 2.0 and 3.0.", 
                self.npy_version[0], self.npy_version[1]));
        }

        // Allocate header buffer and read.
        let mut header_bytes: Vec<u8> = vec![0; self.header_len];
        self.npy_file.read(header_bytes.as_mut_slice()).unwrap();

        let mut sep: usize = header_bytes.len()-1;
        let mut header_bytes_iter = (&header_bytes[..header_bytes.len()-1]).iter();
        while header_bytes_iter.next_back() == Some(&32) { // 32 -> '\x20'
            sep -= 1;
        }
        header_bytes.truncate(sep);
        let header_str = String::from_utf8(header_bytes).unwrap();

        let re = Regex::new(r"'fortran_order': ?(.+), ?'shape': ?\((.+)\)").unwrap();
        //let re = Regex::new(r"'descr': ?(.+), ?'fortran_order': ?(.+), ?'shape': ?\((.+)\)").unwrap();
        //self.descr = re.captures(&header_str).unwrap().get(1).unwrap().as_str().to_string();
        self.fortran_order = match re.captures(&header_str).unwrap().get(1).unwrap().as_str() {
            "False" => false,
            "True" => true,
            _ => panic!("Error: when parsing 'fortran_order', 'True' or 'False' is not found."),
        };
        let mut shape_str: String = re.captures(&header_str).unwrap().get(2).unwrap().as_str().to_string();
        shape_str.retain(|c| !c.is_whitespace());
        if shape_str.chars().next_back() == Some(',') {
            shape_str.truncate(shape_str.len()-1);
        }
        vec_from_string_no_bracket(&mut self.shape, &shape_str);
        self.size = shape_to_size(&self.shape);

        Ok(())
    }

    
    #[inline]
    pub unsafe fn write_tm( &mut self, arr: &[T] ) {
        let slice_u8 = std::slice::from_raw_parts(
            arr.as_ptr() as *const u8, 
            self.size * std::mem::size_of::<T>()
        );
        self.npy_file.write(slice_u8).unwrap();
    }
}


impl<T: NpyDescrGetter> NpyObject<T>
{
    #[inline]
    pub fn write_header( &mut self ) -> Result<(), String> {
        let magic_bytes: [u8; 8] = [147, 78, 85, 77, 80, 89, self.npy_version[0], self.npy_version[1]]; // "\93NUMPY" + version
        self.npy_file.write(&magic_bytes).unwrap();

        let descr_str = T::NPY_DESCR;
        let fortran_order_str = match self.fortran_order {
            true => "True",
            false => "False",
        };
        let mut shape_str = String::new();

        if self.shape.len() == 1 {
            shape_str.push_str( format!("({},)", self.shape[0]).as_str() );
        } else {
            shape_str.push('(');
            for x in &self.shape[..self.shape.len()-1] {
                shape_str.push_str( format!("{}, ", x).as_str() )
            }
            shape_str.push_str( format!("{})", self.shape[self.shape.len()-1]).as_str() )
        }

        let header_str = format!(
            "{{'descr': {descr_str}, 'fortran_order': {fortran_order_str}, 'shape': {shape_str}, }}"
        );
        let n1: usize = header_str.len(); // header size without padding and '\n'
        self.header_len = ((10 + n1 + 1 + 63) / 64) * 64 - 10; // header_len, +1 is for '\n'
        let n2: usize = self.header_len - n1 - 1; // pad size, -1 is for '\n'

        if self.npy_version == [1, 0] {
            let tmp: u16 = self.header_len as u16;
            let header_len_bytes: [u8; 2] = tmp.to_le_bytes();
            self.npy_file.write(&header_len_bytes).unwrap();

        } else if (self.npy_version == [2, 0]) || (self.npy_version == [3, 0]) {
            let tmp: u32 = self.header_len as u32;
            let header_len_bytes: [u8; 4] = tmp.to_le_bytes();
            self.npy_file.write(&header_len_bytes).unwrap();

        } else {
            return Err(format!("Unsupported npy version: {}.{}. Only supports version 1.0, 2.0 and 3.0.", 
                self.npy_version[0], self.npy_version[1]));
        }
        let mut header_bytes = header_str.into_bytes();
        if n2 != 0 {
            header_bytes.resize(n1+n2+1, 32); // Append pad bytes, 32 -> '\x20'.
            header_bytes[n1+n2] = 10; // 10 -> '\n'
        } else {
            header_bytes.push(10); // 10 -> '\n'
        }
        self.npy_file.write(header_bytes.as_slice()).unwrap();
        Ok(())
    }
}


impl NpyObject<f64>
{
    #[inline]
    pub fn read( &mut self ) -> Vec<f64> {
        let mut buf: Vec<u8> = vec![0; 8*self.size];
        self.npy_file.read(buf.as_mut_slice()).unwrap();
        let buf_iter = buf.as_slice().chunks_exact(8);

        let mut arr: Vec<f64> = vec![0.; self.size];
        let arr_iter = arr.as_mut_slice().iter_mut();
        let mut chunk: [u8; 8] = [0; 8];

        for (x, y) in zip(buf_iter, arr_iter) {
            chunk.copy_from_slice(x);
            *y = f64::from_le_bytes(chunk);
        }
        arr
    }

    #[inline]
    pub fn write( &mut self, arr: &[f64] ) {
        let mut buf: Vec<u8> = Vec::with_capacity(8*self.size);        
        for x in arr.iter() {
            buf.extend_from_slice( &x.to_le_bytes() );
        }
        self.npy_file.write(buf.as_slice()).unwrap();
    }
}


impl NpyObject<f32>
{
    #[inline]
    pub fn read( &mut self ) -> Vec<f32> {
        let mut buf: Vec<u8> = vec![0; 4*self.size];
        self.npy_file.read(buf.as_mut_slice()).unwrap();
        let buf_iter = buf.as_slice().chunks_exact(4);

        let mut arr: Vec<f32> = vec![0.; self.size];
        let arr_iter = arr.as_mut_slice().iter_mut();
        let mut chunk: [u8; 4] = [0; 4];

        for (x, y) in zip(buf_iter, arr_iter) {
            chunk.copy_from_slice(x);
            *y = f32::from_le_bytes(chunk);
        }
        arr
    }

    #[inline]
    pub fn write( &mut self, arr: &[f32] ) {
        let mut buf: Vec<u8> = Vec::with_capacity(4*self.size);        
        for x in arr.iter() {
            buf.extend_from_slice( &x.to_le_bytes() );
        }
        self.npy_file.write(buf.as_slice()).unwrap();
    }
}


impl NpyObject<usize>
{
    #[inline]
    pub fn read( &mut self ) -> Vec<usize> {
        let mut buf: Vec<u8> = vec![0; 8*self.size];
        self.npy_file.read(buf.as_mut_slice()).unwrap();
        let buf_iter = buf.as_slice().chunks_exact(8);

        let mut arr: Vec<usize> = vec![0; self.size];
        let arr_iter = arr.as_mut_slice().iter_mut();
        let mut chunk: [u8; 8] = [0; 8];

        for (x, y) in zip(buf_iter, arr_iter) {
            chunk.copy_from_slice(x);
            *y = usize::from_le_bytes(chunk);
        }
        arr
    }

    #[inline]
    pub fn write( &mut self, arr: &[usize] ) {
        let mut buf: Vec<u8> = Vec::with_capacity(8*self.size);        
        for x in arr.iter() {
            buf.extend_from_slice( &x.to_le_bytes() );
        }
        self.npy_file.write(buf.as_slice()).unwrap();
    }
}


impl NpyObject<u8>
{
    #[inline]
    pub fn read( &mut self ) -> Vec<u8> {
        let mut buf: Vec<u8> = vec![0; self.size];
        self.npy_file.read(buf.as_mut_slice()).unwrap();
        buf
    }

    #[inline]
    pub fn write( &mut self, arr: &[u8] ) {
        self.npy_file.write(arr).unwrap();
    }
}


impl<const N: usize> NpyObject<[f64; N]>
{
    #[inline]
    pub fn read( &mut self ) -> Vec<[f64; N]> {
        let mut buf: Vec<u8> = vec![0; 8*self.size];
        self.npy_file.read(buf.as_mut_slice()).unwrap();
        let buf_iter = buf.as_slice().chunks_exact(8*N);

        let mut arr: Vec<[f64; N]> = vec![[0.; N]; <[f64; N]>::npy_vec_len(self.size)];
        let arr_iter = arr.as_mut_slice().iter_mut();
        let mut chunk: [u8; 8] = [0; 8];

        for (x, y) in zip(buf_iter, arr_iter) {
            for elem!(x_, y_) in mzip!(x.chunks_exact(8), y.iter_mut()) {
                chunk.copy_from_slice(x_);
                *y_ = f64::from_le_bytes(chunk);
            }
        }
        arr
    }

    #[inline]
    pub fn write( &mut self, arr: &[[f64; N]] ) {
        let mut buf: Vec<u8> = Vec::with_capacity(8*self.size);        
        for x in arr.iter() {
            for x_ in x.iter() {
                buf.extend_from_slice( &x_.to_le_bytes() );
            }
        }
        self.npy_file.write(buf.as_slice()).unwrap();
    }
}


impl<const N: usize> NpyObject<[f32; N]>
{
    #[inline]
    pub fn read( &mut self ) -> Vec<[f32; N]> {
        let mut buf: Vec<u8> = vec![0; 4*self.size];
        self.npy_file.read(buf.as_mut_slice()).unwrap();
        let buf_iter = buf.as_slice().chunks_exact(4*N);

        let mut arr: Vec<[f32; N]> = vec![[0.; N]; <[f32; N]>::npy_vec_len(self.size)];
        let arr_iter = arr.as_mut_slice().iter_mut();
        let mut chunk: [u8; 4] = [0; 4];

        for (x, y) in zip(buf_iter, arr_iter) {
            for elem!(x_, y_) in mzip!(x.chunks_exact(4), y.iter_mut()) {
                chunk.copy_from_slice(x_);
                *y_ = f32::from_le_bytes(chunk);
            }
        }
        arr
    }

    #[inline]
    pub fn write( &mut self, arr: &[[f32; N]] ) {
        let mut buf: Vec<u8> = Vec::with_capacity(4*self.size);        
        for x in arr.iter() {
            for x_ in x.iter() {
                buf.extend_from_slice( &x_.to_le_bytes() );
            }
        }
        self.npy_file.write(buf.as_slice()).unwrap();
    }
}


impl<const N: usize> NpyObject<[usize; N]>
{
    #[inline]
    pub fn read( &mut self ) -> Vec<[usize; N]> {
        let mut buf: Vec<u8> = vec![0; 8*self.size];
        self.npy_file.read(buf.as_mut_slice()).unwrap();
        let buf_iter = buf.as_slice().chunks_exact(8*N);

        let mut arr: Vec<[usize; N]> = vec![[0; N]; <[usize; N]>::npy_vec_len(self.size)];
        let arr_iter = arr.as_mut_slice().iter_mut();
        let mut chunk: [u8; 8] = [0; 8];

        for (x, y) in zip(buf_iter, arr_iter) {
            for elem!(x_, y_) in mzip!(x.chunks_exact(8), y.iter_mut()) {
                chunk.copy_from_slice(x_);
                *y_ = usize::from_le_bytes(chunk);
            }
        }
        arr
    }

    #[inline]
    pub fn write( &mut self, arr: &[[usize; N]] ) {
        let mut buf: Vec<u8> = Vec::with_capacity(8*self.size);        
        for x in arr.iter() {
            for x_ in x.iter() {
                buf.extend_from_slice( &x_.to_le_bytes() );
            }
        }
        self.npy_file.write(buf.as_slice()).unwrap();
    }
}


impl NpyObject<c128>
{
    #[inline]
    pub fn read( &mut self ) -> Vec<c128> {
        let mut buf: Vec<u8> = vec![0; 16*self.size];
        self.npy_file.read(buf.as_mut_slice()).unwrap();
        let buf_iter = buf.as_slice().chunks_exact(16);

        let mut arr: Vec<c128> = vec![c128::default(); self.size];
        let arr_iter = arr.as_mut_slice().iter_mut();
        let mut chunk0: [u8; 8] = [0; 8];
        let mut chunk1: [u8; 8] = [0; 8];

        for (x, y) in zip(buf_iter, arr_iter) {
            chunk0.copy_from_slice(&x[0..8]);
            chunk1.copy_from_slice(&x[8..16]);
            *y = c128::new(f64::from_le_bytes(chunk0), f64::from_le_bytes(chunk1));
        }
        arr
    }

    #[inline]
    pub fn write( &mut self, arr: &[c128] ) {
        let mut buf: Vec<u8> = Vec::with_capacity(16*self.size);        
        for x in arr.iter() {
            buf.extend_from_slice( &x.re.to_le_bytes() );
            buf.extend_from_slice( &x.im.to_le_bytes() );
        }
        self.npy_file.write(&buf).unwrap();
    }
}


impl<T: Default + Clone + NpyVecLenScaler> NpyObject<T>
{
    #[inline]
    pub unsafe fn read_tm( &mut self ) -> Vec<T> {
        let mut arr: Vec<T> = vec![Default::default(); T::npy_vec_len(self.size)];
        let slice_u8 = std::slice::from_raw_parts_mut(
            arr.as_mut_ptr() as *mut u8, 
            self.size * std::mem::size_of::<T>()
        );
        self.npy_file.read(slice_u8).unwrap();
        arr
    }
}


impl<T: Default + Clone> NpyObject<T>
{
    #[inline]
    pub unsafe fn read2_tm( &mut self, arr: &mut [T] ) {
        let slice_u8 = std::slice::from_raw_parts_mut(
            arr.as_mut_ptr() as *mut u8, 
            self.size * std::mem::size_of::<T>()
        );
        self.npy_file.read(slice_u8).unwrap();
    }
}



#[inline]
pub unsafe fn read_npy_tm<T: Default + Clone + NpyVecLenScaler>( path: &String ) -> Vec<T> {
    let mut npy = NpyObject::<T>::new_reader(path);
    npy.read_header().unwrap();
    npy.read_tm()
}


#[inline]
pub unsafe fn write_npy_tm<T: NpyDescrGetter>( path: &String, arr: &[T] ) {
    let mut npy = NpyObject::<T>::new_writer(path, [1,0], false, vec![arr.len()]);
    npy.write_header().unwrap();
    npy.write_tm(arr);
}