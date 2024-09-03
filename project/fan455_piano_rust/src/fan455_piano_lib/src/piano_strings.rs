use fan455_arrf64::*;
//use fan455_math_array::*;
//use fan455_math_scalar::*;
use fan455_util::*;
use indicatif::ProgressBar;
use super::piano_io::*;
use super::piano_model::*;
use super::piano_fem_1d::*;


pub struct PianoString {
    pub length: f64, // L
    pub density: f64, // rho
    pub diameter: f64, // d
    pub tension: f64, // T_x
    pub young_modulus: f64, // E_x
    pub shear_modulus: [f64; 2], // G_xy, G_xz
    pub shear_correct: f64, // k_xz
    pub stiff_co: [f64; 3], // E_x, G_xy, G_xz
}


impl PianoString
{
    #[inline] #[allow(non_snake_case)]
    pub fn new( data: &PianoStringParamsIn ) -> Self {
        println!("Initializing piano string parameters...");
        let E_x = data.young_modulus;
        let [G_xy, G_xz] = data.shear_modulus;
        let k_x = data.shear_correct;
        let stiff_co = [E_x, G_xy, k_x*G_xz];

        println!("Finished.\n");
        Self { length: data.length, density: data.density, diameter: data.diameter, tension: data.tension, young_modulus: data.young_modulus, shear_modulus: data.shear_modulus, shear_correct: data.shear_correct, stiff_co }
    }


    #[inline] #[allow(non_snake_case)]
    pub fn compute_mass_stiff( &self, 
        mass_mat: &mut SparseRowMat<f64>, 
        stiff_mat: &mut SparseRowMat<f64>,
        mesh: &PianoStringMesh,
        mbuf: &mut PianoStringMeshBuf,
        _normalize: f64, n_prog: usize,
    ) {
        println!("Computing the string's contribution to the mass matrix and stiffness matrix...");
        assert_multi_eq!(mesh.dofs_n, mass_mat.nrow, mass_mat.ncol, stiff_mat.nrow, stiff_mat.ncol);

        let model = StringModel::new(
            self.diameter, self.density, self.tension, self.stiff_co,
        );

        let total_work = mesh.elems_n;
        let prog_size = std::cmp::max(1, (total_work as f64 / n_prog as f64).ceil() as usize);
        let prog_bar = ProgressBar::new(total_work as u64);
        let mut curr_work: usize;

        for i_elem in 0..mesh.elems_n {
            // Compute the isoparametric mapping.
            mbuf.process_elem(i_elem, mesh);
            mbuf.compute_mapping();

            // Compute the gradient and hessian of basis functions, if needed.
            mbuf.compute_gradient(mesh);
            
            // Iterate dof to assemble the mass and stiffness matrices.
            for dof_i in mbuf.iter_edofs() {
                // Row index i is for the trial function.
                let [i_kind, i_global, i_local] = *dof_i;
                for dof_j in mbuf.iter_edofs() {
                    // Column index j is for the basis function.
                    let [j_kind, j_global, j_local] = *dof_j;
                    if i_global >= j_global {

                        let mass_val = model.compute_mass_mat_entry(
                            i_kind, j_kind, i_local, j_local, mesh, mbuf
                        );
                        let stiff_val = model.compute_stiff_mat_entry(
                            i_kind, j_kind, i_local, j_local, mesh, mbuf
                        );
                        mass_mat.addassign_at_option(i_global, j_global, mass_val);
                        stiff_mat.addassign_at_option(i_global, j_global, stiff_val);
                    }
                }
            }
            curr_work = i_elem + 1;
            if curr_work % prog_size == 0 {
                prog_bar.inc(prog_size as u64);
            }
        }
        prog_bar.finish();
        println!("Finished.\n");
    }


    #[inline]
    pub fn compute_bridge_force( &self ) {
        
    }
}


