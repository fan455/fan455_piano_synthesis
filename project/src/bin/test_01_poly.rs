//#![allow(warnings)]
//#[allow(dead_code, unused_imports, non_snake_case, non_upper_case_globals, nonstandard_style)]

use std::vec;

use fan455_math_scalar::*;
use fan455_arrf64::*;
use fan455_math_array::*;
use fan455_util::*;


#[allow(non_snake_case)]
fn main()  {

    /*
    evaluate (10) + (y) + (4 * y^2) + (5 * x) + (3 * x^2 * y) + (2 * x^3)
    */

    let pr0 = RVecPrinter::new(12, 4);
    //let pr1 = RMatPrinter::new(12, 3);
    

    let order: usize = 3;
    let n = eval_poly2_tr_size!(order);
    let mut f_px: Vec<i32> = Vec::with_capacity(n);
    let mut f_py: Vec<i32> = Vec::with_capacity(n);
    {
        for ix in 0..order+1 {
            for iy in 0..order+1 {
                if ix + iy <= order {
                    let px = ix as i32;
                    let py = iy as i32;
                    f_px.push(px);
                    f_py.push(py);
                }
            }
        }
    }
    let mut f_co: Vec<fsize> = vec![0.; n];
    for elem!(px, py, f_co_) in mzip!(f_px.iter(), f_py.iter(), f_co.iter_mut()) {
        if *px == 0 && *py == 0 {
            *f_co_ = 10.;
        } else if *px == 0 && *py == 1 {
            *f_co_ = 1.;
        } else if *px == 0 && *py == 2 {
            *f_co_ = 4.;
        } else if *px == 1 && *py == 0 {
            *f_co_ = 5.;
        } else if *px == 2 && *py == 1 {
            *f_co_ = 3.;
        } else if *px == 3 {
            *f_co_ = 2.;
        }
    }
    println!("f_px = \n{f_px:?}");
    println!("f_py = \n{f_py:?}");
    pr0.print("f_co", &f_co);

    let fx_n = eval_poly2_tr_fx_size!(order, n);
    let fy_n = eval_poly2_tr_fx_size!(order, n);
    let fxx_n = eval_poly2_tr_fxx_size!(order, n);
    let fyy_n = eval_poly2_tr_fxx_size!(order, n);
    let fxy_n = eval_poly2_tr_fxy_size!(order, n);

    let mut fx_idx: Vec<usize> = vec![0; fx_n];
    let mut fy_idx: Vec<usize> = vec![0; fy_n];
    let mut fxx_idx: Vec<usize> = vec![0; fxx_n];
    let mut fyy_idx: Vec<usize> = vec![0; fyy_n];
    let mut fxy_idx: Vec<usize> = vec![0; fxy_n];

    let mut fx_px: Vec<i32> = vec![0; fx_n]; let mut fx_py: Vec<i32> = vec![0; fx_n];
    let mut fy_px: Vec<i32> = vec![0; fy_n]; let mut fy_py: Vec<i32> = vec![0; fy_n];
    let mut fxx_px: Vec<i32> = vec![0; fxx_n]; let mut fxx_py: Vec<i32> = vec![0; fxx_n];
    let mut fyy_px: Vec<i32> = vec![0; fyy_n]; let mut fyy_py: Vec<i32> = vec![0; fyy_n];
    let mut fxy_px: Vec<i32> = vec![0; fxy_n]; let mut fxy_py: Vec<i32> = vec![0; fxy_n];

    poly2_fx_idx(f_px.sl(), fx_idx.slm());
    poly2_fx_pow(f_px.sl(), f_py.sl(), fx_px.slm(), fx_py.slm(), fx_idx.sl());
    // Compxte fy
    poly2_fy_idx(f_py.sl(), fy_idx.slm());
    poly2_fy_pow(f_px.sl(), f_py.sl(), fy_px.slm(), fy_py.slm(), fy_idx.sl());
    // Compxte fxx
    poly2_fx_idx(fx_px.sl(), fxx_idx.slm());
    poly2_fx_pow(fx_px.sl(), fx_py.sl(), fxx_px.slm(), fxx_py.slm(), fxx_idx.sl());
    // Compxte fyy
    poly2_fy_idx(fy_py.sl(), fyy_idx.slm());
    poly2_fy_pow(fy_px.sl(), fy_py.sl(), fyy_px.slm(), fyy_py.slm(), fyy_idx.sl());
    // Compxte fxy
    poly2_fy_idx(fx_py.sl(), fxy_idx.slm());
    poly2_fy_pow(fx_px.sl(), fx_py.sl(), fxy_px.slm(), fxy_py.slm(), fxy_idx.sl());


    let mut fx_co: Arr2<fsize> = Arr2::new(fx_n, n);
    let mut fy_co: Arr2<fsize> = Arr2::new(fy_n, n);
    let mut fxx_co: Arr2<fsize> = Arr2::new(fxx_n, n);
    let mut fyy_co: Arr2<fsize> = Arr2::new(fyy_n, n);
    let mut fxy_co: Arr2<fsize> = Arr2::new(fxy_n, n);

    poly2_fx_coef(f_co.sl(), fx_co.slm(), fx_px.sl(), fx_idx.sl());
    // compxte fy
    poly2_fy_coef(f_co.sl(), fy_co.slm(), fy_py.sl(), fy_idx.sl());
    // compxte fxx
    poly2_fx_coef(fx_co.sl(), fxx_co.slm(), fxx_px.sl(), fxx_idx.sl());
    // compxte fyy
    poly2_fy_coef(fy_co.sl(), fyy_co.slm(), fyy_py.sl(), fyy_idx.sl());
    // compxte fxy
    poly2_fy_coef(fx_co.sl(), fxy_co.slm(), fxy_py.sl(), fxy_idx.sl());


    let x_vec: Vec<fsize> = vec![2., 7.5, 3.6];
    let y_vec: Vec<fsize> = vec![6., 1.5, 4.15];
    let n_points = x_vec.len();
    assert_eq!(n_points, y_vec.len());
    
    let mut f_vec: Vec<fsize> = vec![0.; n_points];
    let mut fx_vec: Vec<fsize> = vec![0.; n_points];
    let mut fy_vec: Vec<fsize> = vec![0.; n_points];
    let mut fxx_vec: Vec<fsize> = vec![0.; n_points];
    let mut fyy_vec: Vec<fsize> = vec![0.; n_points];
    let mut fxy_vec: Vec<fsize> = vec![0.; n_points];

    poly2_batch(f_co.sl(), f_px.sl(), f_py.sl(), x_vec.sl(), y_vec.sl(), f_vec.slm());
    poly2_batch(fx_co.sl(), fx_px.sl(), fx_py.sl(), x_vec.sl(), y_vec.sl(), fx_vec.slm());
    poly2_batch(fy_co.sl(), fy_px.sl(), fy_py.sl(), x_vec.sl(), y_vec.sl(), fy_vec.slm());
    poly2_batch(fxx_co.sl(), fxx_px.sl(), fxx_py.sl(), x_vec.sl(), y_vec.sl(), fxx_vec.slm());
    poly2_batch(fyy_co.sl(), fyy_px.sl(), fyy_py.sl(), x_vec.sl(), y_vec.sl(), fyy_vec.slm());
    poly2_batch(fxy_co.sl(), fxy_px.sl(), fxy_py.sl(), x_vec.sl(), y_vec.sl(), fxy_vec.slm());

    println!("f(2,3) = {:.4}", poly2(&f_co, &f_px, &f_py, 2., 3.));

    pr0.print("f_vec", &f_vec);
    pr0.print("fx_vec", &fx_vec);
    pr0.print("fy_vec", &fy_vec);
    pr0.print("fxx_vec", &fxx_vec);
    pr0.print("fyy_vec", &fyy_vec);
    pr0.print("fxy_vec", &fxy_vec);

}
