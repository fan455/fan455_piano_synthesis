use fan455_math_scalar::*;
use serde::{Serialize, Deserialize};


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoParamsOut {
    pub sb: PianoSoundboardParamsOut,
    pub vib: PianoVibrationParamsOut,
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoParamsIn {
    pub stage: i8,
    pub data_dir: String,
    pub n_prog: usize,
    pub normalize: fsize,

    pub elems: PianoFiniteElementParamsIn,
    pub sb: PianoSoundboardParamsIn,
    pub ribs: PianoRibsParamsIn,
    pub bridges: PianoBridgesParamsIn,
    pub vib: PianoVibrationParamsIn,
    pub rad: PianoRadiationParamsIn,
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoRadiationParamsIn {
    pub sound_speed: fsize,
    pub bridge_pos: [fsize; 2],
    pub listen_pos: [fsize; 3],
    pub response_dir: String,
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoFiniteElementParamsIn {
    pub element_type: u8,
    pub order: usize,
    pub nodes_ordering: u8,
    pub free_nodes_n: usize,

    pub nodes_xy_path: String,
    pub elems_nodes_path: String,
    //pub elems_groups_path: String,
    pub groups_elems_idx_path: String,
    //pub quad_points_path: String,
    //pub quad_weights_path: String,
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoVibrationParamsIn {
    pub modal_freq_ub: fsize,
    pub sample_rate: usize,
    pub duration: fsize,
    pub print_first_freq: usize,
    pub damp_path: String,
    

    pub stiff_mat_row_pos_path: String,
    pub stiff_mat_row_idx_path: String,
    pub stiff_mat_col_idx_path: String,

    pub eigfreq_lb: fsize, 
    pub eigfreq_ub: fsize, 
    pub eig_n_guess: usize,
    pub eigsol: EigenSolverParamsIn,
    
    pub truncate_eigvec: bool,
    pub truncate_eigval: bool,
    pub truncate_modal_freq: bool,
    pub truncate_modal_damp: bool,
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
pub struct PianoVibrationParamsOut {
    pub eig_n: usize,
    pub modes_n: usize,
    pub modal_freq_range: [fsize; 2],
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoSoundboardParamsIn {
    pub angle: fsize,
    pub density: fsize,
    pub thickness: fsize,
    pub young_modulus: [fsize; 2], // E_x, E_y
    pub shear_modulus: [fsize; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: fsize, // v_xy, v_yx
    pub shear_correct: [fsize; 2], // (k_x)^2, (k_y)^2
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoSoundboardParamsOut {
    pub mass_co: [fsize; 2],
    pub stiff_co: [fsize; 6],
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoRibsParamsIn {
    pub num: usize,
    pub angle: fsize, 
    pub density: fsize,
    pub height: [fsize; 2], // height at the middle par; height at the two ends

    pub young_modulus: [fsize; 2], // E_x, E_y
    pub shear_modulus: [fsize; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: fsize, // v_xy, v_yx
    pub shear_correct: [fsize; 2], // (k_x)^2, (k_y)^2

    pub beg_xy_path: String,
    pub end_xy_path: String,
    pub mid1_xy_path: String,
    pub mid2_xy_path: String,

    pub group_range: Vec<Vec<[usize; 2]>>, // group index range, not including the right one.
}


#[derive(Serialize, Deserialize, Debug, Default)] #[allow(non_snake_case)]
pub struct PianoBridgesParamsIn {
    pub num: usize,
    pub angle: Vec<fsize>,
    pub density: fsize,
    pub height: fsize, // homogenous height

    pub young_modulus: [fsize; 2], // E_x, E_y
    pub shear_modulus: [fsize; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: fsize, // v_xy, v_yx
    pub shear_correct: [fsize; 2], // (k_x)^2, (k_y)^2
    
    pub group_range: Vec<Vec<[usize; 2]>>,
}