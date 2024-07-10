use fan455_arrf64::*;
use fan455_math_scalar::*;
//use fan455_math_array::*;
use fan455_util::{assert_multi_eq, elem, mzip};
use indicatif::ProgressBar;
use super::piano_io::*;
use super::piano_finite_element::*;
use super::piano_model::*;


#[derive(Default)] #[allow(non_snake_case)]
pub struct PianoBridges {
    pub sb_thick: f64, // soundboard thicknes
    pub angle: Vec<f64>, // The angle of bridges to the old x axis, in range [0, pi], will be the new x axis.

    pub num: usize,
    pub height: f64, 
    pub density: f64,
    pub young_modulus: [f64; 2], // E_x, E_y
    pub shear_modulus: [f64; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: f64, // v_xy; v_yx will be autocomputed.
    pub shear_correct: [f64; 2], // (k_x)^2, (k_y)^2

    pub stiff_co: Vec<[f64; 9]>, // D1 to D9, auto computed
}


impl PianoBridges
{
    #[inline] #[allow(non_snake_case)]
    pub fn new( 
        data: &PianoBridgesParamsIn,
        sb_thick: f64,
    ) -> Self {
        println!("Initializing ribs parameters...");
        let num = data.num;
        let angle: Vec<f64> = data.angle.clone();
        assert_multi_eq!(num, angle.len());

        let mut stiff_co: Vec<[f64; 9]> = Vec::with_capacity(num);
        for angle_ in angle.iter() {
            stiff_co.push(
                PlateModel::compute_stiff_coef_rotated(
                    data.young_modulus, data.shear_modulus, data.poisson_ratio, 
                    data.shear_correct, angle_-PI
                )
            );
        }

        println!("Finished.\n");
        Self { sb_thick, num, angle, height: data.height, density: data.density, young_modulus: data.young_modulus, shear_modulus: data.shear_modulus, poisson_ratio: data.poisson_ratio, shear_correct: data.shear_correct, stiff_co }
    }


    #[inline] #[allow(non_snake_case)]
    pub fn compute_mass_stiff(
        &self, 
        mass_mat: &mut CsrMat<f64>, 
        stiff_mat: &mut CsrMat<f64>,
        mesh: &PianoSoundboardMesh,
        mbuf: &mut PianoSoundboardMeshBuf,
        _normalize: f64,
        n_prog: usize,
    ) {
        println!("Computing the bridges contribution to the mass matrix and stiffness matrix...");
        //let pr0 = RVecPrinter::new(16, 4);
        assert_multi_eq!(mesh.dof, mass_mat.nrow, mass_mat.ncol, stiff_mat.nrow, stiff_mat.ncol);
        
        let mut model = PlateModel::new(self.density, [0.; 9], mesh.quad_n);
        model.compute_quad_z_with_const_thick(self.sb_thick/2., self.height, false);
        
        let total_work = mesh.elems_n;
        let prog_size = std::cmp::max(1, (total_work as f64 / n_prog as f64).ceil() as usize);
        let prog_bar = ProgressBar::new(total_work as u64);
        let mut curr_work: usize = 0;

        let mut i_bridge: usize = 0;
        for elem!(i_elem, i_group) in mzip!(0..mesh.elems_n, mesh.elems_groups.it()) {

            if let Some(i_bridge_new) = mesh.groups_bridges.get(i_group) {
                if i_bridge != *i_bridge_new {
                    i_bridge = *i_bridge_new;
                    model.stiff_co.copy_from_slice(&self.stiff_co[i_bridge]);
                }
                // Compute the isoparametric mapping.
                mbuf.process_elem(i_elem, mesh);
                mbuf.compute_mapping();
                mbuf.compute_jacobian();
                assert!(mbuf.map.jac_det > 0., 
                    "Element {i_elem} has negative jacobian determinant value: {:.6}", mbuf.map.jac_det
                );

                // Compute the gradient and hessian of basis functions, if needed.
                mbuf.compute_gradient(mesh);
                
                // Iterate dof to assemble the mass and stiffness matrices.
                for dof_i in mbuf.iter_edof_idx() {
                    // Row index i is for the trial function.
                    let [i_kind, i_global, i_local] = *dof_i;

                    for dof_j in mbuf.iter_edof_idx() {
                        // Column index j is for the basis function.
                        let [j_kind, j_global, j_local] = *dof_j;

                        if i_global >= j_global {

                            if let MayBeZero::NonZero(mass_val) = model.compute_mass_mat_entry(
                                i_kind, j_kind, i_local, j_local, mesh, mbuf
                            ) {
                                assert!(!mass_val.is_nan(), 
                                    "Mass matrix value in bridges part got nan at index ({i_global}, {j_global})"
                                );
                                if let Some(s) = mass_mat.idxm2_option(i_global, j_global) {
                                    *s += mass_val;
                                } else {
                                    panic!("Mass matrix non-zero entry at index ({i_global}, {j_global}) does not exist, current i_elem = {i_elem}, i_kind = {i_kind}, j_kind = {j_kind}.");
                                }
                            }

                            if let MayBeZero::NonZero(stiff_val) = model.compute_stiff_mat_entry(
                                i_kind, j_kind, i_local, j_local, mesh, mbuf
                            ) {
                                assert!(!stiff_val.is_nan(), 
                                    "Stiff matrix value in bridges part got nan at index ({i_global}, {j_global})"
                                );
                                if let Some(s) = stiff_mat.idxm2_option(i_global, j_global) {
                                    *s += stiff_val;
                                } else {
                                    panic!("Stiff matrix non-zero entry at index ({i_global}, {j_global}) does not exist, current i_elem = {i_elem}, i_kind = {i_kind}, j_kind = {j_kind}.");
                                }
                            }
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
