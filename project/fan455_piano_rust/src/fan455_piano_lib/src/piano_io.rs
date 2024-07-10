use serde::{Serialize, Deserialize};
use std::collections::HashMap;


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoParamsOut {
    //pub sb: PianoSoundboardParamsOut,
    pub mesh: PianoMeshParamsOut,
    pub vib: PianoVibrationParamsOut,
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoParamsIn {
    pub stage: i8,
    pub data_dir: String,
    pub n_prog: usize,
    pub normalize: f64,

    pub mesh: PianoMeshParamsIn,
    pub sb: PianoSoundboardParamsIn,
    pub ribs: PianoRibsParamsIn,
    pub bridges: PianoBridgesParamsIn,
    pub vib: PianoVibrationParamsIn,
    pub rad: PianoRadiationParamsIn,
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoRadiationParamsIn {
    pub sound_speed: f64,
    pub bridge_pos: [f64; 2],
    pub listen_pos: [f64; 3],
    pub response_dir: String,
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoMeshParamsIn {
    pub dof_kinds_path: String,

    pub nodes_dof_path: String,
    pub nodes_kinds_path: String,
    pub nodes_xy_path: String,

    pub elems_nodes_path: String,
    pub elems_groups_path: String,

    pub quad_points_path: String,
    pub quad_weights_path: String,

    pub groups_ribs: HashMap<String, [usize; 2]>,
    pub groups_bridges: HashMap<String, usize>,
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoMeshParamsOut {
    pub order: [usize; 5],
    pub dof: usize,
    pub edof_max: usize,
    pub edofs_max: [usize; 5],
    pub edof_max_unique: usize,
    pub edofs_max_unique: [usize; 3],
    pub enn: usize,
    pub nodes_n: usize,
    pub elems_n: usize,
    pub quad_n: usize,
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoVibrationParamsIn {
    pub modal_freq_ub: f64,
    pub sample_rate: usize,
    pub duration: f64,
    pub print_first_freq: usize,
    pub damp_path: String,
    pub init_vel_factor: f64,

    pub eigfreq_lb: f64, 
    pub eigfreq_ub: f64, 
    pub eig_n_guess: usize,
    pub eigsol: EigenSolverParamsIn,
    
    pub truncate_eigvec: bool,
    pub truncate_eigval: bool,
    pub truncate_modal_freq: bool,
    pub truncate_modal_damp: bool,

    pub mass_mat_row_pos_path: String,
    pub mass_mat_row_idx_path: String,
    pub mass_mat_col_idx_path: String,

    pub stiff_mat_row_pos_path: String,
    pub stiff_mat_row_idx_path: String,
    pub stiff_mat_col_idx_path: String,
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoVibrationParamsOut {
    pub eig_n: usize,
    pub modes_n: usize,
    pub modal_freq_range: [f64; 2],
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct EigenSolverParamsIn {
    pub runtime_print: bool,
    pub num_contour_points: usize,
    pub tol: usize,
    pub max_loops: usize,
    pub stop_type: usize,
    pub sparse_mat_check: bool,
    pub positive_mat_check: bool,
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoSoundboardParamsIn {
    pub angle: f64,
    pub density: f64,
    pub thickness: f64,
    pub young_modulus: [f64; 2], // E_x, E_y
    pub shear_modulus: [f64; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: f64, // v_xy, v_yx
    pub shear_correct: [f64; 2], // (k_x)^2, (k_y)^2
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoSoundboardParamsOut {
    //pub mass_co: [f64; 2],
    pub stiff_co: [f64; 9],
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoRibsParamsIn {
    pub num: usize,
    pub angle: f64, 
    pub density: f64,
    pub height: [f64; 2], // height at the middle par; height at the two ends

    pub young_modulus: [f64; 2], // E_x, E_y
    pub shear_modulus: [f64; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: f64, // v_xy, v_yx
    pub shear_correct: [f64; 2], // k_x, k_y

    pub beg_xy_path: String,
    pub end_xy_path: String,
    pub mid1_xy_path: String,
    pub mid2_xy_path: String,
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoBridgesParamsIn {
    pub num: usize,
    pub angle: Vec<f64>,
    pub density: f64,
    pub height: f64, // homogenous height

    pub young_modulus: [f64; 2], // E_x, E_y
    pub shear_modulus: [f64; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: f64, // v_xy, v_yx
    pub shear_correct: [f64; 2], // k_x, k_y

}