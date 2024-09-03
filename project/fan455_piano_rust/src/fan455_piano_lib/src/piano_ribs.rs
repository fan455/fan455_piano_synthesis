use fan455_arrf64::*;
//use fan455_math_array::*;
//use fan455_math_scalar::*;
use fan455_util::*;
use indicatif::ProgressBar;
use super::piano_io::*;
use super::piano_fem_3d::*;
use super::piano_model::*;


#[derive(Default)] #[allow(non_snake_case)]
pub struct PianoRibs {
    // Ribs are set perpendicular to the soundboard wood fibers.
    pub angle: f64, // The angle of ribs to the old x axis, in range [0, pi], will be the new x axis.
    pub num: usize,
    pub density: f64,
    pub height: [f64; 2], // height at the middle par; height at the two ends

    pub young_modulus: [f64; 3], // E_x, E_y, E_z
    pub shear_modulus: [f64; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: [f64; 3], // nu_xy, nu_xz, nu_yz
    pub shear_correct: [f64; 2], // k_xz, k_yz

    pub stiff_co: [f64; 13], // D_11 to D_66 (rotated, symmetric, non-zero), auto computed
}


impl PianoRibs
{
    #[inline] #[allow(non_snake_case)]
    pub fn new( data: &PianoRibsParamsIn ) -> Self {
        println!("Initializing ribs parameters...");
        let num = data.num;
        let angle = data.angle;
        let density = data.density;
        let height = data.height;
        assert!(is_descend!(height[0], height[1], 0.), "Heights of ribs may be incorrect.");

        let stiff_co = PlateModel::compute_stiff_co_rotated(
            data.young_modulus, data.shear_modulus, data.poisson_ratio, data.shear_correct, angle
        );
        println!("Finished.\n");
        Self { angle, num, density, height,  
            young_modulus: data.young_modulus, shear_modulus: data.shear_modulus, 
            poisson_ratio: data.poisson_ratio, shear_correct: data.shear_correct, stiff_co }
    }

    #[inline] #[allow(non_snake_case)]
    pub fn compute_mass_stiff( &self, 
        mass_mat: &mut SparseRowMat<f64>, 
        stiff_mat: &mut SparseRowMat<f64>,
        mesh: &PianoSoundboardMesh,
        mbuf: &mut PianoSoundboardMeshBuf,
        _normalize: f64, n_prog: usize,
    ) {
        println!("Computing the ribs' contribution to the mass matrix and stiffness matrix...");
        assert_multi_eq!(mesh.dofs_n, mass_mat.nrow, mass_mat.ncol, stiff_mat.nrow, stiff_mat.ncol);

        let model = PlateModel::new(self.density, [0.; 2], self.stiff_co);

        let total_work = mesh.elems_n;
        let prog_size = std::cmp::max(1, (total_work as f64 / n_prog as f64).ceil() as usize);
        let prog_bar = ProgressBar::new(total_work as u64);
        let mut curr_work: usize = 0;

        for i_elem in 0..mesh.elems_n {
            let i_group = mesh.elems[i_elem].group;

            if let Some(_) = mesh.groups_bridges.get(&i_group) {
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
            }
            curr_work += 1;
            if curr_work % prog_size == 0 {
                prog_bar.inc(prog_size as u64);
            }
        }
        prog_bar.finish();
        println!("Finished.\n");
    }
}

