use fan455_arrf64::*;
use fan455_math_scalar::*;
use fan455_math_array::*;
use fan455_util::read_npy_vec;
use fan455_util::{assert_multi_eq, elem, is_ascend, is_descend, mzip};
use indicatif::ProgressBar;
use super::piano_io::*;
use super::piano_finite_element::*;
use super::piano_model::*;


#[derive(Default)] #[allow(non_snake_case)]
pub struct PianoRibs {
    // Ribs are set perpendicular to the soundboard wood fibers.
    pub angle: f64, // The angle of ribs to the old x axis, in range [0, pi], will be the new x axis.
    pub rotate: CoordSysRotation,
    pub sb_thick: f64, // soundboard thicknes

    pub num: usize,
    pub density: f64,
    pub height: [f64; 2], // height at the middle part; height at the begin and end positions (same).
    pub height_decay_beg: Vec<f64>, // height decay rates at the begin positions for every rib, auto computed.
    pub height_decay_end: Vec<f64>, // height decay rates at the end positions for every rib, auto computed.

    pub young_modulus: [f64; 2], // E_x, E_y
    pub shear_modulus: [f64; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: f64, // v_xy; v_yx will be autocomputed.
    pub shear_correct: [f64; 2], // k_x, k_y

    pub stiff_co: [f64; 9], // D1 to D9, auto computed

    pub beg_xy: Vec<[f64; 2]>, 
    pub end_xy: Vec<[f64; 2]>, 
    pub mid1_xy: Vec<[f64; 2]>, 
    pub mid2_xy: Vec<[f64; 2]>, 
}


impl PianoRibs
{
    #[inline] #[allow(non_snake_case)]
    pub fn new( 
        data: &PianoRibsParamsIn,
        sb_thick: f64,
    ) -> Self {
        println!("Initializing ribs parameters...");
        let num = data.num;
        let angle = data.angle;
        let density = data.density;
        let height = data.height;
        assert!(is_descend!(height[0], height[1], 0.), "Heights of ribs may be incorrect.");

        let stiff_co = PlateModel::compute_stiff_coef_rotated(
            data.young_modulus, data.shear_modulus, data.poisson_ratio, data.shear_correct, angle
        );
        
        let mut height_decay_beg: Vec<f64> = Vec::with_capacity(num);
        let mut height_decay_end: Vec<f64> = Vec::with_capacity(num);

        let mut beg_xy: Vec<[f64; 2]> = read_npy_vec(&data.beg_xy_path);
        let mut end_xy: Vec<[f64; 2]> = read_npy_vec(&data.end_xy_path);
        let mut mid1_xy: Vec<[f64; 2]> = read_npy_vec(&data.mid1_xy_path);
        let mut mid2_xy: Vec<[f64; 2]> = read_npy_vec(&data.mid2_xy_path);
        assert_multi_eq!(num, beg_xy.size(), end_xy.size(), mid1_xy.size(), mid2_xy.size());

        // Rotate coordinates of bridge points, and compute the height curve coefficients.
        let rotate = CoordSysRotation::new(angle, CoordSysRotation::COUNTERCLOCK);
        let [h_mid, h_side] = height;
        rotate.batch_rotate(&mut beg_xy);
        rotate.batch_rotate(&mut end_xy);
        rotate.batch_rotate(&mut mid1_xy);
        rotate.batch_rotate(&mut mid2_xy);

        for elem!([beg_x_, beg_y_], [end_x_, end_y_], [mid1_x_, mid1_y_], [mid2_x_, mid2_y_]) in mzip!(
            beg_xy.it(), end_xy.it(), mid1_xy.it(), mid2_xy.it()
        ) {
            assert!((beg_y_ - end_y_).abs() < 1e-2, "The angles or coordinates of ribs may be incorrect.");
            assert!((beg_y_ - mid1_y_).abs() < 1e-2, "The angles or coordinates of ribs may be incorrect.");
            assert!((beg_y_ - mid2_y_).abs() < 1e-2, "The angles or coordinates of ribs may be incorrect.");
            assert!(is_ascend!(beg_x_, mid1_x_, mid2_x_, end_x_), "The begin or end positions of ribs may be incorrect.");

            let decay_beg = (h_side / h_mid).ln() / (beg_x_ - mid1_x_);
            let decay_end = (h_side / h_mid).ln() / (mid2_x_ - end_x_);
            height_decay_beg.push(decay_beg);
            height_decay_end.push(decay_end);
        }
        //println!("height_decay_beg = {height_decay_beg:?}");
        //println!("height_decay_end = {height_decay_end:?}");
        println!("Finished.\n");
        Self { angle, rotate, sb_thick, num, density, height, height_decay_beg, height_decay_end, 
            young_modulus: data.young_modulus, shear_modulus: data.shear_modulus, 
            poisson_ratio: data.poisson_ratio, shear_correct: data.shear_correct, stiff_co,
            beg_xy, end_xy, mid1_xy, mid2_xy }
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
        println!("Computing the ribs' contribution to the mass matrix and stiffness matrix...");
        assert_multi_eq!(mesh.dof, mass_mat.nrow, mass_mat.ncol, stiff_mat.nrow, stiff_mat.ncol);

        let mut model = PlateModel::new(self.density, self.stiff_co, mesh.quad_n);

        let mut fn_height = FnRibsHeight {
            ribs_part: FnRibsHeight::RIBS_BEG, h_sb: self.sb_thick/2., h_mid: self.height[0],
            mid1_x: {
                let mut tmp: Vec<f64> = Vec::with_capacity(self.num);
                for [x, _] in self.mid1_xy.iter() { tmp.push(*x); }
                tmp
            }, 
            mid2_x: {
                let mut tmp: Vec<f64> = Vec::with_capacity(self.num);
                for [x, _] in self.mid2_xy.iter() { tmp.push(*x); }
                tmp
            }, 
            decay_beg: self.height_decay_beg.clone(),
            decay_end: self.height_decay_end.clone(),
            rotate: self.rotate,
        };

        let total_work = mesh.elems_n;
        let prog_size = std::cmp::max(1, (total_work as f64 / n_prog as f64).ceil() as usize);
        let prog_bar = ProgressBar::new(total_work as u64);
        let mut curr_work: usize = 0;

        let mut i_rib: usize = 0;
        for elem!(i_elem, i_group) in mzip!(0..mesh.elems_n, mesh.elems_groups.it()) {

            if let Some([i_rib_new, ribs_part_new]) = mesh.groups_ribs.get(i_group) {
                if i_rib != *i_rib_new {
                    i_rib = *i_rib_new;
                }
                if fn_height.ribs_part != *ribs_part_new {
                    fn_height.ribs_part = *ribs_part_new;
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

                // Compute the position-dependent height.
                mbuf.b0.compute_quad_xy(&mesh.iso[0], &mbuf.map);
                fn_height.call(i_rib, &mbuf.b0.quad_xy, &mut model.quad_z0, &mut model.quad_z1, &mut model.quad_z2);
                
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
                                    "Mass matrix value in ribs part got nan at index ({i_global}, {j_global})"
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
                                    "Stiff matrix value in ribs part got nan at index ({i_global}, {j_global})"
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


struct FnRibsHeight {
    ribs_part: usize,
    h_sb: f64,
    h_mid: f64,
    mid1_x: Vec<f64>,
    mid2_x: Vec<f64>,
    decay_beg: Vec<f64>,
    decay_end: Vec<f64>,
    rotate: CoordSysRotation,
}

impl FnRibsHeight
{
    const RIBS_BEG: usize = 0;
    const RIBS_MID: usize = 1;
    const RIBS_END: usize = 2;

    #[inline]
    fn call( &self, i_rib: usize, xy: &[[f64; 2]], h: &mut [f64], hs: &mut [f64], hss: &mut [f64] ) {
        // Compute the heights of points on the ribs, which varys by different parts of ribs.
        if self.ribs_part == Self::RIBS_MID {
            h.fill(self.h_mid);
            hs.fill( (self.h_sb.powi(2) - (self.h_sb + self.h_mid).powi(2)) / 2. );
            hss.fill( ((self.h_sb + self.h_mid).powi(3) - self.h_sb.powi(3)) / 3. );

        } else if self.ribs_part == Self::RIBS_BEG {
            let decay = self.decay_beg[i_rib];
            let x_mid1 = self.mid1_x[i_rib];
            for elem!([x, y], h_, hs_, hss_) in mzip!(xy.iter(), h.iter_mut(), hs.iter_mut(), hss.iter_mut()) {
                let [x_, _] = self.rotate.rotate(*x, *y);
                let h_tmp = self.h_mid * (decay * (x_ - x_mid1)).exp();
                *h_ = h_tmp;
                *hs_ = (self.h_sb.powi(2) - (self.h_sb + h_tmp).powi(2)) / 2.;
                *hss_ = ((self.h_sb + h_tmp).powi(3) - self.h_sb.powi(3)) / 3.;
            }

        } else if self.ribs_part == Self::RIBS_END {
            let decay = self.decay_end[i_rib];
            let x_mid2 = self.mid2_x[i_rib];
            for elem!([x, y], h_, hs_, hss_) in mzip!(xy.iter(), h.iter_mut(), hs.iter_mut(), hss.iter_mut()) {
                let [x_, _] = self.rotate.rotate(*x, *y);
                let h_tmp = self.h_mid * (-decay * (x_ - x_mid2)).exp();
                *h_ = h_tmp;
                *hs_ = (self.h_sb.powi(2) - (self.h_sb + h_tmp).powi(2)) / 2.;
                *hss_ = ((self.h_sb + h_tmp).powi(3) - self.h_sb.powi(3)) / 3.;
            }
        } else {
            panic!("Unknown ribs part in FnRibsHeight: {}", self.ribs_part);
        }
    }
}

