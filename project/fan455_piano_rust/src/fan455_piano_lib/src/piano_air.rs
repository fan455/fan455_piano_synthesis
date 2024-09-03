use fan455_arrf64::*;
//use fan455_math_array::*;
//use fan455_math_scalar::*;
use fan455_util::*;
use indicatif::ProgressBar;
use super::piano_io::*;
use super::piano_fem_3d::*;
use super::piano_model::*;


#[allow(non_snake_case)]
pub struct PianoAir {
    // The air surrounding the piano in an acoustic space.
    pub sound_speed: f64,
    pub density: f64,
}

impl PianoAir
{
    #[inline] #[allow(non_snake_case)]
    pub fn new( data: &PianoAirParamsIn ) -> Self {
        println!("Initializing air parameters...");
        println!("Finished.\n");
        Self { sound_speed: data.sound_speed, density: data.density }
    }

    #[inline] #[allow(non_snake_case)]
    pub fn compute_mass_stiff( &self, 
        mass_mat: &mut SparseRowMat<f64>, 
        stiff_mat: &mut SparseRowMat<f64>,
        mesh: &PianoAirMesh,
        mbuf: &mut PianoAirMeshBuf,
        _normalize: f64, n_prog: usize,
    ) {
        println!("Computing the plate's contribution to the mass matrix and stiffness matrix...");
        //let pr0 = RVecPrinter::new(16, 4);
        assert_multi_eq!(mesh.dofs_n, mass_mat.nrow, mass_mat.ncol, stiff_mat.nrow, stiff_mat.ncol);

        let model = WaveModel::new(self.sound_speed);

        let total_work = mesh.elems_n;
        let prog_size = std::cmp::max(1, (total_work as f64 / n_prog as f64).ceil() as usize);
        let prog_bar = ProgressBar::new(total_work as u64);
        let mut curr_work: usize;

        for i_elem in 0..mesh.elems_n {
            // Compute the isoparametric mapping.
            mbuf.process_elem(i_elem, mesh);
            mbuf.compute_mapping(mesh);
            mbuf.compute_jacobian(mesh);
            assert!(mbuf.map.jac_det > 0., "Element {i_elem} has negative jacobian determinant value: {:.6}", mbuf.map.jac_det);

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




