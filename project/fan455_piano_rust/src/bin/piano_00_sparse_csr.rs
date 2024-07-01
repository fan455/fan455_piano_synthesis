use fan455_util::*;
use fan455_util_macro::*;
use fan455_piano_lib::*;


fn main() {

    // Read parameters from toml file.
    parse_cmd_args!();
    cmd_arg!(elems_nodes_path, String);
    cmd_arg!(nodes_n, usize);
    cmd_arg!(free_nodes_n, usize);

    cmd_arg!(row_pos_path, String);
    cmd_arg!(row_idx_path, String);
    cmd_arg!(col_idx_path, String);
    unknown_cmd_args!();

    println!();

    compute_mass_stiff_mat_sparse_idx(
        &elems_nodes_path, 
        nodes_n, 
        free_nodes_n, 
        &row_pos_path, 
        &row_idx_path, 
        &col_idx_path
    );

    println!("Program ended successfully.");
}