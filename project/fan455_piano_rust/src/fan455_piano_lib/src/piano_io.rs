use serde::{Serialize, Deserialize};
use std::collections::HashMap;


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoParamsOut {
    //pub sb: PianoSoundboardParamsOut,
    pub mesh: PianoSoundboardMeshParamsOut,
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
    pub air: PianoAirParamsIn,
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoMeshParamsIn {
    pub str: PianoStringMeshParamsIn,
    pub sb: PianoSoundboardMeshParamsIn,
    pub air: PianoAirMeshParamsIn,
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoAirParamsIn {
    pub sound_speed: f64,
    pub density: f64,
    pub bridge_pos: [f64; 2],
    pub listen_pos: [f64; 3],
    pub response_dir: String,
}

#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoAirMeshParamsIn {
    /*pub points_tag_path: String,
    pub points_kind_path: String,
    pub points_coord_path: String,
    pub elems_points_tag_path: String,
    pub elems_groups_path: String,*/

    pub dofs_path: String,
    pub nodes_path: String,
    pub elems_path: String,

    pub quad_points_path: String,
    pub quad_weights_path: String,
}

#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoStringMeshParamsIn {
    pub length: f64,
    pub delta_length: f64,
    pub hammer_pos_rel: f64,

    pub quad_points_path: String,
    pub quad_weights_path: String,
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoSoundboardMeshParamsIn {
    pub points_tag_path: String,
    pub points_kind_path: String,
    pub points_coord_path: String,
    pub elems_points_tag_path: String,
    pub elems_groups_path: String,

    pub dofs_path: String,
    pub nodes_path: String,
    pub elems_path: String,

    pub quad_points_path: String,
    pub quad_weights_path: String,

    pub ribs_beg_map: HashMap<String, usize>, // kind -> idx
    pub ribs_end_map: HashMap<String, usize>, // kind -> idx
    pub ribs_mid1_map: HashMap<String, usize>, // kind -> idx
    pub ribs_mid2_map: HashMap<String, usize>, // kind -> idx
    pub groups_ribs: HashMap<String, [usize; 2]>, // group -> idx
    pub groups_bridges: HashMap<String, usize>, // group -> idx
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoSoundboardMeshParamsOut {
    pub order: [[usize; 2]; 3], // each is [order_xy, order_z]
    pub dofs_n: usize, // Total dof
    pub edofs_max: usize,
    pub enodes_n: usize,
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

    pub tension: [f64; 2], // T_x, T_y
    pub young_modulus: [f64; 3], // E_x, E_y, E_z
    pub shear_modulus: [f64; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: [f64; 3], // nu_xy, nu_xz, nu_yz
    pub shear_correct: [f64; 2], // k_xz, k_yz
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoStringParamsIn {
    pub length: f64, // L
    pub density: f64, // rho
    pub diameter: f64, // d
    pub tension: f64, // T_x
    pub young_modulus: f64, // E_x
    pub shear_modulus: [f64; 2], // G_xy, G_xz
    pub shear_correct: f64, // k_xz
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoSoundboardParamsOut {
    pub tension_co: [f64; 2],
    pub stiff_co: [f64; 13],
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoRibsParamsIn {
    pub num: usize,
    pub angle: f64, 
    pub density: f64,
    pub height: [f64; 2], // height at the middle par; height at the two ends

    pub young_modulus: [f64; 3], // E_x, E_y, E_z
    pub shear_modulus: [f64; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: [f64; 3], // nu_xy, nu_xz, nu_yz
    pub shear_correct: [f64; 2], // k_xz, k_yz
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoBridgesParamsIn {
    pub num: usize,
    pub angle: Vec<f64>,
    pub density: f64,
    pub height: f64, // homogenous height

    pub young_modulus: [f64; 3], // E_x, E_y, E_z
    pub shear_modulus: [f64; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: [f64; 3], // nu_xy, nu_xz, nu_yz
    pub shear_correct: [f64; 2], // k_xz, k_yz
}