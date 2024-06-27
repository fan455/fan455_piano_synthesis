//#![allow(warnings)]
//#[allow(dead_code, unused_imports, non_snake_case, non_upper_case_globals, nonstandard_style)]

use fan455_math_scalar::*;
use fan455_arrf64::*;
use fan455_math_array::*;
use fan455_util::*;


#[allow(non_snake_case)]
fn main()  {

    let mut x = CsrMat::<isize> {
        nrow: 5,
        ncol: 5,
        nnz: 13,
        data: vec![1, -1, -3, -2, 5, 4, 6, 4, -4, 2, 7, 8, -5],
        row_pos: vec![0, 3, 5, 8, 11, 13],
        row_idx: vec![0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        col_idx: vec![0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4],
    };

    println!("x.idx2(3, 2) = {}", *x.idx2(3, 2)); // should be 2
    *x.idxm2(3, 2) = 100;
    println!("x.idx2(3, 2) = {}", *x.idx2(3, 2)); // should be 100

    println!("x.idx2(4, 4) = {}", *x.idx2(4, 4)); // should be -5
    *x.idxm2(4, 4) = 100;
    println!("x.idx2(4, 4) = {}", *x.idx2(4, 4)); // should be 100
}
