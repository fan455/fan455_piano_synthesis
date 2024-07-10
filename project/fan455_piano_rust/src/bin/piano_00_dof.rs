use fan455_util::*;
use fan455_util_macro::*;
use fan455_piano_lib::*;
use std::fs::read_to_string;


fn main() {

    // Read parameters from toml file.
    println!("Reading toml file of piano parameters...");
    parse_cmd_args!();
    cmd_arg!(piano_toml, String);
    unknown_cmd_args!();

    let piano_toml_string = read_to_string(&piano_toml).unwrap();
    let args: PianoParamsIn = toml::from_str(&piano_toml_string).unwrap();
    println!("Finished.\n");

    let mesh = PianoSoundboardMesh::new(&args.mesh, true);
    compute_mass_stiff_mat_sparse_idx(&mesh, &args);

    write_npy_vec(&args.mesh.dof_kinds_path, &mesh.dof_kinds);
    write_npy_vec(&args.mesh.nodes_dof_path, &mesh.nodes_dof);

    println!("Program ended successfully.");
}