//#![allow(warnings)]
//#[allow(dead_code, unused_imports, non_snake_case, non_upper_case_globals, nonstandard_style)]

use fan455_arrf64::*;
use fan455_util::*;
use fan455_util_macro::*;

#[allow(non_snake_case)]
fn main()  {

    parse_cmd_args!();

    cmd_arg!(a_dense_path, String);
    cmd_arg!(a_row_pos_path, String);
    cmd_arg!(a_row_idx_path, String);
    cmd_arg!(a_col_idx_path, String);
    cmd_arg!(a_data_path, String);

    cmd_arg!(b_dense_path, String);
    cmd_arg!(b_row_pos_path, String);
    cmd_arg!(b_row_idx_path, String);
    cmd_arg!(b_col_idx_path, String);
    cmd_arg!(b_data_path, String);

    cmd_arg!(eigval_lb, f64);
    cmd_arg!(eigval_ub, f64);
    cmd_arg!(eigval_n_guess, usize);

    cmd_arg!(runtime_print, bool, true);
    cmd_arg!(tol, BlasInt, 3);
    cmd_arg!(max_loops, BlasInt, 5);
    cmd_arg!(stop_type, BlasInt, 0);
    cmd_arg!(num_contour_points, BlasInt, 8);
    cmd_arg!(sparse_mat_check, bool, true);
    cmd_arg!(positive_mat_check, bool, true);

    cmd_arg!(print_width, usize, 12);
    cmd_arg!(print_prec, usize, 4);

    unknown_cmd_args!();

    println!("\nFinished parsing cmd args.\n");
    let pr0 = RVecPrinter::new(print_width, print_prec);
    let pr1 = RMatPrinter::new(print_width, print_prec);

    let a_dense = Arr2::<f64>::read_npy(&a_dense_path);
    let mut a = CsrMat::<f64>::read_npy_square(
        &a_row_pos_path, &a_row_idx_path, &a_col_idx_path, &a_data_path
    );
    println!("Finished reading sparse matrix a data.");

    let b_dense = Arr2::<f64>::read_npy(&b_dense_path);
    let mut b = CsrMat::<f64>::read_npy_square(
        &b_row_pos_path, &b_row_idx_path, &b_col_idx_path, &b_data_path
    );
    println!("Finished reading sparse matrix b data.\n");
    assert_multi_eq!(a_dense.nrow(), a_dense.ncol(), a.nrow, b.nrow, b_dense.nrow(), b_dense.ncol());

    println!("a.nrow = {}, a.ncol = {}, a.nnz = {}\n", a.nrow, a.ncol, a.nnz);
    println!("a.row_pos = {:?}\n", a.row_pos);
    println!("a.row_idx = {:?}\n", a.row_idx);
    println!("a.col_idx = {:?}\n", a.col_idx);
    println!("a.data = {:.4?}\n\n", a.data);

    println!("b.nrow = {}, b.ncol = {}, b.nnz = {}\n", b.nrow, b.ncol, b.nnz);
    println!("b.row_pos = {:?}\n", b.row_pos);
    println!("b.row_idx = {:?}\n", b.row_idx);
    println!("b.col_idx = {:?}\n", b.col_idx);
    println!("b.data = {:.4?}\n\n", b.data);

    let n = a.nrow;
    let mut eigval = Arr1::<f64>::new(n);
    let mut eigvec = Arr2::<f64>::new(n, n);

    let mut eigsol = FeastSparse::<f64>::new();
    eigsol.uplo = FULL;
    eigsol.set_runtime_print(runtime_print);
    eigsol.set_tol(tol);
    eigsol.set_max_loops(max_loops);
    eigsol.set_stop_type(stop_type);
    eigsol.set_num_contour_points(num_contour_points);
    eigsol.set_sparse_mat_check(sparse_mat_check);
    eigsol.set_positive_mat_check(positive_mat_check);
    println!("Finished creating MKL FEAST solver and setting parameters.");

    a.change_to_one_based_index();
    b.change_to_one_based_index();
    println!("Finished change index to one-based.");

    eigsol.init_guess(eigval_lb, eigval_ub, eigval_n_guess);
    //eigsol.solve(&a, &mut eigval, &mut eigvec);
    eigsol.solve_generalized(&a, &b, &mut eigval, &mut eigvec);
    eigsol.report();

    pr0.print("\neigval", &eigval);
    pr1.print("\neigvec", &eigvec);

    let eig_n = eigsol.eig_n;
    eigvec.truncate(n, eig_n);
    let mut lhs = Arr2::<f64>::new(eig_n, eig_n);
    let mut tmp = Arr2::<f64>::new(eig_n, n);
    dgemm(1., &eigvec, &b_dense, 0., &mut tmp, TRANS, NO_TRANS);
    dgemm(1., &tmp, &eigvec, 0., &mut lhs, NO_TRANS, NO_TRANS);

    pr1.print("\nlhs", &lhs);
    println!("\nlhs.diag = {:.3?}", lhs.diag());

    println!("\nFinished.");
}
