use core::panic;

use fan455_arrf64::*;
use fan455_math_scalar::*;
use fan455_math_array::*;
use fan455_util::NpyObject;
use fan455_util::{assert_multi_eq, elem, is_ascend, is_descend, mzip};
use indicatif::ProgressBar;
use super::piano_io::*;
use super::piano_finite_element::*;


#[derive(Default)] #[allow(non_snake_case)]
pub struct PianoRibs {
    // Ribs are set perpendicular to the soundboard wood fibers.
    pub angle: fsize, // The angle of ribs to the old x axis, in range [0, pi], will be the new x axis.
    pub rotate: CoordSysRotation,
    pub sb_thickness: fsize, // soundboard thicknes

    pub num: usize,
    pub density: fsize,
    pub height: [fsize; 2], // height at the middle part; height at the begin and end positions (same).
    pub height_decay_beg: Vec<fsize>, // height decay rates at the begin positions for every rib, auto computed.
    pub height_decay_end: Vec<fsize>, // height decay rates at the end positions for every rib, auto computed.

    pub young_modulus: [fsize; 2], // E_x, E_y
    pub shear_modulus: [fsize; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: fsize, // v_xy; v_yx will be autocomputed.
    pub shear_correct: [fsize; 2], // (k_x)^2, (k_y)^2

    pub mass_co: [fsize; 2], // M1, M2, auto computed
    pub stiff_co: [fsize; 6], // D1, D2, D3, D4, D5, D6, auto computed

    pub group_range: Vec<Vec<[usize; 2]>>,
    pub beg_xy: Vec<[fsize; 2]>, 
    pub end_xy: Vec<[fsize; 2]>, 
    pub mid1_xy: Vec<[fsize; 2]>, 
    pub mid2_xy: Vec<[fsize; 2]>, 
}


impl PianoRibs
{
    #[inline] #[allow(non_snake_case)]
    pub fn new( 
        data: &PianoRibsParamsIn,
        sb_thickness: fsize,
    ) -> Self {
        println!("Initializing ribs parameters...");
        let num = data.num;
        let angle = data.angle;
        let density = data.density;
        let height = data.height;
        assert!(is_descend!(height[0], height[1], 0.), "Heights of ribs may be incorrect.");

        let rho = data.density;
        let [E_x, E_y] = data.young_modulus;
        let [G_xy, G_xz, G_yz] = data.shear_modulus;
        let nu_xy = data.poisson_ratio;
        let [k2_x, k2_y] = data.shear_correct;

        let nu_yx = nu_xy * E_y / E_x;

        let mass_co: [fsize; 2] = [rho, rho];
        let stiff_co: [fsize; 6] = [
            E_x / (1.- nu_xy * nu_yx),
            E_x * nu_yx / (1.- nu_xy * nu_yx),
            E_y / (1.- nu_xy * nu_yx),
            2.* G_xy,
            k2_x * G_xz,
            k2_y * G_yz,
        ];
        
        let mut height_decay_beg: Vec<fsize> = Vec::with_capacity(num);
        let mut height_decay_end: Vec<fsize> = Vec::with_capacity(num);

        let mut beg_xy: Vec<[fsize; 2]> = {
            let mut npy = NpyObject::<[fsize; 2]>::new_reader(&data.beg_xy_path);
            npy.read_header().unwrap();
            npy.read()
        };
        let mut end_xy: Vec<[fsize; 2]> = {
            let mut npy = NpyObject::<[fsize; 2]>::new_reader(&data.end_xy_path);
            npy.read_header().unwrap();
            npy.read()
        };
        let mut mid1_xy: Vec<[fsize; 2]> = {
            let mut npy = NpyObject::<[fsize; 2]>::new_reader(&data.mid1_xy_path);
            npy.read_header().unwrap();
            npy.read()
        };
        let mut mid2_xy: Vec<[fsize; 2]> = {
            let mut npy = NpyObject::<[fsize; 2]>::new_reader(&data.mid2_xy_path);
            npy.read_header().unwrap();
            npy.read()
        };
        assert_multi_eq!(num, beg_xy.size(), end_xy.size(), mid1_xy.size(), mid2_xy.size());

        let group_range = data.group_range.clone();

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
        Self { angle, rotate, sb_thickness, num, density, height, height_decay_beg, height_decay_end, 
            young_modulus: data.young_modulus, shear_modulus: data.shear_modulus, 
            poisson_ratio: data.poisson_ratio, shear_correct: data.shear_correct, mass_co, stiff_co,
            beg_xy, end_xy, mid1_xy, mid2_xy, group_range }
    }


    #[inline] #[allow(non_snake_case)]
    pub fn compute_mass_stiff(
        &self, 
        mass_diag: &mut Arr1<fsize>, 
        stiff_mat: &mut CsrMat<fsize>,
        iso: &IsoElement,
        mesh: &Mesh2d,
        mbuf: &mut Mesh2dBuf,
        _normalize: fsize,
        n_prog: usize,
    ) {
        println!("Computing the ribs' contribution to the mass matrix and stiffness matrix...");
        let free_nodes_n = mesh.free_nodes_n;
        let nodes_n = mesh.nodes_n;
        //let elems_n = mesh.elems_n;
        let quad_n = mesh.quad_n;
        let en_n = mesh.n;
        let dof = free_nodes_n + 2*nodes_n;
        assert_multi_eq!(dof, mass_diag.size(), stiff_mat.nrow, stiff_mat.ncol);

        let [C1, C2] = self.mass_co;
        let [D1, D2, D3, D4, D5, D6] = self.stiff_co;

        let h_sb = self.sb_thickness;
        let h_mid = self.height[0];
        let mut h: Vec<fsize> = vec![0.; quad_n];
        let mut hs: Vec<fsize> = vec![0.; quad_n];

        // f is trial function, g is basis function.
        let mut f_g: Vec<fsize> = vec![0.; quad_n];
        let mut f_gx: Vec<fsize> = vec![0.; quad_n];
        let mut f_gy: Vec<fsize> = vec![0.; quad_n];
        //let mut fx_g: Vec<fsize> = vec![0.; quad_n];
        let mut fx_gx: Vec<fsize> = vec![0.; quad_n];
        let mut fx_gy: Vec<fsize> = vec![0.; quad_n];
        //let mut fy_g: Vec<fsize> = vec![0.; quad_n];
        let mut fy_gx: Vec<fsize> = vec![0.; quad_n];
        let mut fy_gy: Vec<fsize> = vec![0.; quad_n];

        let en_idx_local: Vec<usize> = {
            let mut s = Vec::<usize>::with_capacity(en_n);
            for i in 0..en_n {
                s.push(i);
            }
            s
        };

        let mut fn_height = FnRibsHeight {
            ribs_part: FnRibsHeight::RIBS_BEG, h_sb, h_mid,
            mid1_x: {
                let mut tmp: Vec<fsize> = Vec::with_capacity(self.num);
                for [x, _] in self.mid1_xy.iter() { tmp.push(*x); }
                tmp
            }, 
            mid2_x: {
                let mut tmp: Vec<fsize> = Vec::with_capacity(self.num);
                for [x, _] in self.mid2_xy.iter() { tmp.push(*x); }
                tmp
            }, 
            decay_beg: self.height_decay_beg.clone(),
            decay_end: self.height_decay_end.clone(),
        };

        let total_work = {
            let mut s: usize = 0;
            for g in self.group_range.iter() {
                for g_ in g.iter() {
                    for g__ in g_[0]..g_[1] {
                        let [i_elem_lb, i_elem_ub] = mesh.groups_elems_idx[g__];
                        s += i_elem_ub - i_elem_lb;
                    }
                }
            }
            s
        };
        let prog_size = std::cmp::max(1, (total_work as fsize / n_prog as fsize).ceil() as usize);
        let prog_bar = ProgressBar::new(total_work as u64);
        let mut curr_work: usize = 0;

        let mut fn_tmp = |group_range: [usize; 2], fn_h: &FnRibsHeight| {

            for elem!(i_rib, [i_elem_lb, i_elem_ub]) in mzip!(
                0..self.num, mesh.groups_elems_idx[group_range[0]..group_range[1]].iter()
            ) {
                for i_elem in *i_elem_lb..*i_elem_ub {
                    // Compute the isoparametric mapping.
                    let en_idx = mesh.elems_nodes.col(i_elem);
                    index_vec2_unbind(&mesh.nodes_xy, en_idx.sl(), &mut mbuf.en_x, &mut mbuf.en_y);
                    self.rotate.rotate_vec(&mut mbuf.en_x, &mut mbuf.en_y);
                    mbuf.compute_mapping(&iso);
                    mbuf.compute_jacobian(&iso);

                    if mbuf.jac_det < 0. {
                        panic!("Negative jacobian determinant encountered.");
                    }

                    // Compute the gradient and hessian of basis functions, if needed.
                    mbuf.compute_gradient(&iso, &en_idx_local);
                    mbuf.compute_quad_x(&iso);
                    fn_h.call(i_rib, &mbuf.quad_x, &mut h, &mut hs);
                    
                    for elem!(i, i_global) in mzip!(0..en_n, en_idx.it()) {
                        // Row index i is for the trial function.
                        for elem!(j, j_global) in mzip!(0..en_n, en_idx.it()) {
                            // Column index j is for the basis function.
                            let i1 = *i_global;
                            let j1 = *j_global;

                            if i1 >= j1 {
                                f_g.  assign_mul( &iso.f.col(i)  , &iso.f.col(j)   );
                                f_gx. assign_mul( &iso.f.col(i)  , &mbuf.fx.col(j) );
                                f_gy. assign_mul( &iso.f.col(i)  , &mbuf.fy.col(j) );
                                //fx_g. assign_mul( &mbuf.fx.col(i), &iso.f.col(j)   );
                                fx_gx.assign_mul( &mbuf.fx.col(i), &mbuf.fx.col(j) );
                                fx_gy.assign_mul( &mbuf.fx.col(i), &mbuf.fy.col(j) );
                                //fy_g. assign_mul( &mbuf.fy.col(i), &iso.f.col(j)   );
                                fy_gx.assign_mul( &mbuf.fy.col(i), &mbuf.fx.col(j) );
                                fy_gy.assign_mul( &mbuf.fy.col(i), &mbuf.fy.col(j) );

                                let quad_f_g_h    = iso.quad2(&f_g  , &h , mbuf.jac_det);
                                let quad_f_gx_h   = iso.quad2(&f_gx , &h , mbuf.jac_det);
                                let quad_f_gy_h   = iso.quad2(&f_gy , &h , mbuf.jac_det);
                                //let quad_fx_g_h   = iso.quad2(&fx_g , &h,  mbuf.jac_det);
                                let quad_fx_gx_h  = iso.quad2(&fx_gx, &h , mbuf.jac_det);
                                //let quad_fx_gy_h  = iso.quad2(&fx_gy, &h , mbuf.jac_det);
                                //let quad_fy_g_h   = iso.quad2(&fy_g , &h , mbuf.jac_det);
                                //let quad_fy_gx_h  = iso.quad2(&fy_gx, &h , mbuf.jac_det);
                                let quad_fy_gy_h  = iso.quad2(&fy_gy, &h , mbuf.jac_det);

                                let quad_f_g_hs   = iso.quad2(&f_g  , &hs, mbuf.jac_det);
                                //let quad_f_gx_hs  = iso.quad2(&f_gx , &hs, mbuf.jac_det);
                                //let quad_f_gy_hs  = iso.quad2(&f_gy , &hs, mbuf.jac_det);
                                //let quad_fx_g_hs  = iso.quad2(&fx_g , &hs, mbuf.jac_det);
                                let quad_fx_gx_hs = iso.quad2(&fx_gx, &hs, mbuf.jac_det);
                                let quad_fx_gy_hs = iso.quad2(&fx_gy, &hs, mbuf.jac_det);
                                //let quad_fy_g_hs  = iso.quad2(&fy_g , &hs, mbuf.jac_det);
                                let quad_fy_gx_hs = iso.quad2(&fy_gx, &hs, mbuf.jac_det);
                                let quad_fy_gy_hs = iso.quad2(&fy_gy, &hs, mbuf.jac_det);
                                

                                let i2 = i1 + free_nodes_n;
                                let i3 = i2 + nodes_n;
                                let j2 = j1 + free_nodes_n;
                                let j3 = j2 + nodes_n;

                                if i1 == j1 {
                                    let M1 = C1 * quad_f_g_h;
                                    let M2 = C2 * quad_f_g_hs;
                                    let M3 = C2 * quad_f_g_hs;

                                    if M1.is_nan() {
                                        panic!("Mass matrix value at element {} and global index ({}, {}) got nan, 
                                        in soundboard's plate part.", i_elem, i1, i1);
                                    }
                                    if M2.is_nan() {
                                        panic!("Mass matrix value at element {} and global index ({}, {}) got nan, 
                                        in soundboard's plate part.", i_elem, i2, i2);
                                    }
                                    if M3.is_nan() {
                                        panic!("Mass matrix value at element {} and global index ({}, {}) got nan, 
                                        in soundboard's plate part.", i_elem, i3, i3);
                                    }
                                    *mass_diag.idxm(i1) += M1;
                                    *mass_diag.idxm(i2) += M2;
                                    *mass_diag.idxm(i3) += M3;
                                }

                                let i_is_free = i1 < free_nodes_n;
                                let j_is_free = j1 < free_nodes_n;

                                if i_is_free && j_is_free {
                                    let K1 = D5 * quad_fx_gx_h + D6 * quad_fy_gy_h;
                                    if K1.is_nan() {
                                        panic!("Stiffness matrix value at element {} and global index ({}, {}) got nan, 
                                        in soundboard's plate part.", i_elem, i1, j1);
                                    }
                                    *stiff_mat.idxm2(i1, j1) += K1;
                                }
                                if j_is_free {
                                    let K2 = D5 * quad_f_gx_h;
                                    if K2.is_nan() {
                                        panic!("Stiffness matrix value at element {} and global index ({}, {}) got nan, 
                                        in soundboard's plate part.", i_elem, i3, j1);
                                    }

                                    let K3 = D6 * quad_f_gy_h;
                                    if K3.is_nan() {
                                        panic!("Stiffness matrix value at element {} and global index ({}, {}) got nan, 
                                        in soundboard's plate part.", i_elem, i2, j1);
                                    }
                                    *stiff_mat.idxm2(i2, j1) += K3;
                                    *stiff_mat.idxm2(i3, j1) += K2;
                                }
                                let K4 = D5 * quad_f_g_h + D1 * quad_fx_gx_hs + D4 * quad_fy_gy_hs;
                                if K4.is_nan() {
                                    panic!("Stiffness matrix value at element {} and global index ({}, {}) got nan, 
                                    in soundboard's plate part.", i_elem, i3, j3);
                                }

                                let K5 = D4 * quad_fx_gy_hs + D2 * quad_fy_gx_hs;
                                if K5.is_nan() {
                                    panic!("Stiffness matrix value at element {} and global index ({}, {}) got nan, 
                                    in soundboard's plate part.", i_elem, i3, j2);
                                }

                                let K6 = D6 * quad_f_g_h + D3 * quad_fy_gy_hs + D4 * quad_fx_gx_hs;
                                if K6.is_nan() {
                                    panic!("Stiffness matrix value at element {} and global index ({}, {}) got nan, 
                                    in soundboard's plate part.", i_elem, i2, j2);
                                }
                                *stiff_mat.idxm2(i2, j2) += K6;
                                *stiff_mat.idxm2(i3, j2) += K5;
                                *stiff_mat.idxm2(i3, j3) += K4;
                            }
                        }
                    }
                    curr_work += 1;
                    if curr_work % prog_size == 0 {
                        prog_bar.inc(prog_size as u64);
                    }
                }
            }
        };
        fn_tmp(self.group_range[0][0], &fn_height);

        fn_height.ribs_part = FnRibsHeight::RIBS_MID;
        fn_tmp(self.group_range[1][0], &fn_height);

        fn_height.ribs_part = FnRibsHeight::RIBS_END;
        fn_tmp(self.group_range[2][0], &fn_height);

        prog_bar.finish_with_message("Finished.\n");
    }
}


struct FnRibsHeight {
    ribs_part: u8,
    h_sb: fsize,
    h_mid: fsize,
    mid1_x: Vec<fsize>,
    mid2_x: Vec<fsize>,
    decay_beg: Vec<fsize>,
    decay_end: Vec<fsize>,
}

impl FnRibsHeight
{
    const RIBS_BEG: u8 = 1;
    const RIBS_MID: u8 = 2;
    const RIBS_END: u8 = 3;

    #[inline]
    fn call( &self, i_rib: usize, x: &[fsize], h: &mut [fsize], hs: &mut [fsize] ) {
        // Compute the heights of points on the ribs, which varys by different parts of ribs.
        if self.ribs_part == Self::RIBS_MID {
            h.fill(self.h_mid);
            hs.fill( ((self.h_mid + 0.5*self.h_sb).powi(3) - (0.5*self.h_sb).powi(3)) / 3. );

        } else if self.ribs_part == Self::RIBS_BEG {
            let decay = self.decay_beg[i_rib];
            let x_mid1 = self.mid1_x[i_rib];
            for elem!(x_, h_, hs_) in mzip!(x.iter(), h.iter_mut(), hs.iter_mut()) {
                *h_ = self.h_mid * (decay * (x_ - x_mid1)).exp();
                *hs_ = ((*h_ + 0.5*self.h_sb).powi(3) - (0.5*self.h_sb).powi(3)) / 3.;
            }

        } else if self.ribs_part == Self::RIBS_END {
            let decay = self.decay_end[i_rib];
            let x_mid2 = self.mid2_x[i_rib];
            for elem!(x_, h_, hs_) in mzip!(x.iter(), h.iter_mut(), hs.iter_mut()) {
                *h_ = self.h_mid * (-decay * (x_ - x_mid2)).exp();
                *hs_ = ((*h_ + 0.5*self.h_sb).powi(3) - (0.5*self.h_sb).powi(3)) / 3.;
            }
            /*if i_rib == 3 {
                println!("decay = {decay:.6}");
                println!("h_mid = {:.6}", self.h_mid);
                println!("x_mid2 = {x_mid2:.6}");
                println!("x = {x:.4?}");
                println!("h = {h:.4?}");
                println!("hs = {hs:.4?}");
                panic!("Stop here");
            }*/
        } else {
            panic!("Unknown ribs part in FnRibsHeight: {}", self.ribs_part);
        }
    }
}

