use fan455_arrf64::*;
//use fan455_math_array::*;
use fan455_math_scalar::*;
use fan455_util::*;
use indicatif::ProgressBar;
use super::piano_io::*;
use super::piano_fem_3d::*;
use super::piano_model::*;


#[allow(non_snake_case)]
pub struct PianoSoundboard {   
    pub angle: f64, 
    // The angle of soundboard fibers to the old x axis, in range [0, pi], will be the new x axis.
    pub density: f64,
    pub thickness: f64,

    pub tension: [f64; 2], // T_x, T_y
    pub young_modulus: [f64; 3], // E_x, E_y, E_z
    pub shear_modulus: [f64; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: [f64; 3], // nu_xy, nu_xz, nu_yz
    pub shear_correct: [f64; 2], // k_xz, k_yz

    pub tension_co: [f64; 2], // T_x, T_y (rotated), auto computed
    pub stiff_co: [f64; 13], // D_11 to D_66 (rotated, symmetric, non-zero), auto computed
}

impl PianoSoundboard
{
    #[inline] #[allow(non_snake_case)]
    pub fn new( data: &PianoSoundboardParamsIn ) -> Self {
        println!("Initializing soundboard parameters...");
        let angle = data.angle;

        let stiff_co = PlateModel::compute_stiff_co_rotated(
            data.young_modulus, data.shear_modulus, data.poisson_ratio, 
            data.shear_correct, angle-PI
        );
        let tension_co = PlateModel::compute_tension_co_rotated(
            data.tension, angle-PI
        );
        println!("Finished.\n");
        Self {
            angle, density: data.density, thickness: data.thickness, tension: data.tension,
            young_modulus: data.young_modulus, shear_modulus: data.shear_modulus, 
            poisson_ratio: data.poisson_ratio, shear_correct: data.shear_correct, 
            tension_co, stiff_co,
        }
    }

    #[inline]
    pub fn output_params( &self ) -> PianoSoundboardParamsOut {
        PianoSoundboardParamsOut {
            tension_co: self.tension_co,
            stiff_co: self.stiff_co,
        }
    }

    #[inline] #[allow(non_snake_case)]
    pub fn compute_mass_stiff( &self, 
        mass_mat: &mut SparseRowMat<f64>, 
        stiff_mat: &mut SparseRowMat<f64>,
        mesh: &PianoSoundboardMesh,
        mbuf: &mut PianoSoundboardMeshBuf,
        _normalize: f64, n_prog: usize,
    ) {
        println!("Computing the plate's contribution to the mass matrix and stiffness matrix...");
        //let pr0 = RVecPrinter::new(16, 4);
        assert_multi_eq!(mesh.dofs_n, mass_mat.nrow, mass_mat.ncol, stiff_mat.nrow, stiff_mat.ncol);

        let model = PlateModel::new(self.density, self.tension_co, self.stiff_co);

        let total_work = mesh.elems_n;
        let prog_size = std::cmp::max(1, (total_work as f64 / n_prog as f64).ceil() as usize);
        let prog_bar = ProgressBar::new(total_work as u64);
        let mut curr_work: usize;

        for i_elem in 0..mesh.elems_n {
            // Compute the isoparametric mapping.
            mbuf.process_elem(i_elem, mesh);
            mbuf.compute_mapping(mesh);
            mbuf.compute_jacobian(mesh);
            for det in mbuf.map.jac_det.iter() {
                assert!(*det > 0., "Element {i_elem} has negative jacobian determinant value: {det:.6}");
            }
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
}




