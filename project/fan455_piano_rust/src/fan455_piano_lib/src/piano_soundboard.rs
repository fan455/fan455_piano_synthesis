use fan455_arrf64::*;
use fan455_math_array::*;
use fan455_math_scalar::*;
use fan455_math_array::CoordSysRotation;
use fan455_util::{assert_multi_eq, elem, mzip};
use indicatif::ProgressBar;
use super::piano_finite_element::*;
use super::piano_io::*;


#[derive(Default)] #[allow(non_snake_case)]
pub struct PianoSoundboard {   
    pub angle: fsize, 
    // The angle of soundboard fibers to the old x axis, in range [0, pi], will be the new x axis.
    pub rotate: CoordSysRotation,

    pub density: fsize,
    pub thickness: fsize,
    pub young_modulus: [fsize; 2], // E_x, E_y
    pub shear_modulus: [fsize; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: fsize, // v_xy; v_yx will be autocomputed.
    pub shear_correct: [fsize; 2], // (k_x)^2, (k_y)^2

    pub mass_co: [fsize; 2], // M1, M2, auto computed
    pub stiff_co: [fsize; 6], // D1, D2, D3, D4, D5, D6, auto computed
}


impl PianoSoundboard
{
    #[inline] #[allow(non_snake_case)]
    pub fn new( data: &PianoSoundboardParamsIn ) -> Self {
        println!("Initializing soundboard parameters...");
        let angle = data.angle;
        let rotate = CoordSysRotation::new(PI-angle, CoordSysRotation::CLOCKWISE);

        let rho = data.density;
        let h = data.thickness;
        let h3 = h.powi(3);
        let [E_x, E_y] = data.young_modulus;
        let [G_xy, G_xz, G_yz] = data.shear_modulus;
        let nu_xy = data.poisson_ratio;
        let [k2_x, k2_y] = data.shear_correct;

        let nu_yx = nu_xy * E_y / E_x;

        let mass_co: [fsize; 2] = [rho*h, rho*h3/12.];
        let stiff_co: [fsize; 6] = [
            (h3 * E_x) / (12.* (1.- nu_xy * nu_yx)),
            (h3 * E_x * nu_yx) / (12.* (1.- nu_xy * nu_yx)),
            (h3 * E_y) / (12.* (1.- nu_xy * nu_yx)),
            (h3 * G_xy) / 6.,
            h * k2_x * G_xz,
            h * k2_y * G_yz,
        ];
        println!("Finished.\n");

        Self {
            angle, rotate, 
            density: data.density, thickness: data.thickness,
            young_modulus: data.young_modulus, shear_modulus: data.shear_modulus, 
            poisson_ratio: data.poisson_ratio, shear_correct: data.shear_correct, 
            mass_co, stiff_co,
        }
    }


    #[inline]
    pub fn output_params( &self ) -> PianoSoundboardParamsOut {
        PianoSoundboardParamsOut {
            mass_co: self.mass_co,
            stiff_co: self.stiff_co,
        }
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
        println!("Computing the plate's contribution to the mass matrix and stiffness matrix...");
        //let pr0 = RVecPrinter::new(16, 4);
        assert_multi_eq!(mesh.dof, mass_diag.size(), stiff_mat.nrow, stiff_mat.ncol);

        let [C1, C2] = self.mass_co;
        let [D1, D2, D3, D4, D5, D6] = self.stiff_co;

        // f is trial function, g is basis function.
        let mut f_g: Vec<fsize> = vec![0.; mesh.quad_n];
        let mut f_gx: Vec<fsize> = vec![0.; mesh.quad_n];
        let mut f_gy: Vec<fsize> = vec![0.; mesh.quad_n];
        //let mut fx_g: Vec<fsize> = vec![0.; mesh.quad_n];
        let mut fx_gx: Vec<fsize> = vec![0.; mesh.quad_n];
        let mut fx_gy: Vec<fsize> = vec![0.; mesh.quad_n];
        //let mut fy_g: Vec<fsize> = vec![0.; mesh.quad_n];
        let mut fy_gx: Vec<fsize> = vec![0.; mesh.quad_n];
        let mut fy_gy: Vec<fsize> = vec![0.; mesh.quad_n];

        let en_idx_local: Vec<usize> = {
            let mut s = Vec::<usize>::with_capacity(mesh.n);
            for i in 0..mesh.n {
                s.push(i);
            }
            s
        };

        let total_work = mesh.elems_n;
        let prog_size = std::cmp::max(1, (total_work as fsize / n_prog as fsize).ceil() as usize);
        let prog_bar = ProgressBar::new(total_work as u64);
        let mut curr_work: usize;

        for i_elem in 0..mesh.elems_n {
            // Compute the isoparametric mapping.
            let en_idx = mesh.elems_nodes.col(i_elem);
            index_vec2_unbind(&mesh.nodes_xy, en_idx.sl(), &mut mbuf.en_x, &mut mbuf.en_y);
            self.rotate.rotate_vec(&mut mbuf.en_x, &mut mbuf.en_y);
            mbuf.compute_mapping(&iso);
            mbuf.compute_jacobian(&iso);

            if mbuf.jac_det < 0. {
                panic!("Element {i_elem} has negative jacobian determinant value: {:.6}", mbuf.jac_det);
            }

            // Compute the gradient and hessian of basis functions, if needed.
            mbuf.compute_gradient(&iso, &en_idx_local);
            
            for elem!(i, i_global) in mzip!(0..mesh.n, en_idx.it()) {
                // Row index i is for the trial function.
                for elem!(j, j_global) in mzip!(0..mesh.n, en_idx.it()) {
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

                        let quad_f_g   = iso.quad(&f_g  , mbuf.jac_det);
                        let quad_f_gx  = iso.quad(&f_gx , mbuf.jac_det);
                        let quad_f_gy  = iso.quad(&f_gy , mbuf.jac_det);
                        //let quad_fx_g  = iso.quad(&fx_g , mbuf.jac_det);
                        let quad_fx_gx = iso.quad(&fx_gx, mbuf.jac_det);
                        let quad_fx_gy = iso.quad(&fx_gy, mbuf.jac_det);
                        //let quad_fy_g  = iso.quad(&fy_g , mbuf.jac_det);
                        let quad_fy_gx = iso.quad(&fy_gx, mbuf.jac_det);
                        let quad_fy_gy = iso.quad(&fy_gy, mbuf.jac_det);

                        let M1 = C1 * quad_f_g;
                        let M2 = C2 * quad_f_g;
                        let M3 = C2 * quad_f_g;

                        let K1 = D5 * quad_fx_gx + D6 * quad_fy_gy;
                        let K2 = D5 * quad_f_gx;
                        let K3 = D6 * quad_f_gy;
                        let K4 = D5 * quad_f_g + D1 * quad_fx_gx + D4 * quad_fy_gy;
                        let K5 = D4 * quad_fx_gy + D2 * quad_fy_gx;
                        let K6 = D6 * quad_f_g + D3 * quad_fy_gy + D4 * quad_fx_gx;

                        let i_is_free = i1 < mesh.free_nodes_n;
                        let j_is_free = j1 < mesh.free_nodes_n;

                        #[cfg(not(feature="clamped_plate"))] {
                            let i2 = i1 + mesh.free_nodes_n;
                            let i3 = i2 + mesh.nodes_n;
                            let j2 = j1 + mesh.free_nodes_n;
                            let j3 = j2 + mesh.nodes_n;

                            if i_is_free && j_is_free {
                                if i1 == j1 {
                                    *mass_diag.idxm(i1) += M1;
                                }
                                *stiff_mat.idxm2(i1, j1) += K1;
                            }
                            if i1 == j1 {
                                *mass_diag.idxm(i2) += M2;
                                *mass_diag.idxm(i3) += M3;
                            }
                            if j_is_free {
                                *stiff_mat.idxm2(i2, j1) += K2;
                                *stiff_mat.idxm2(i3, j1) += K3;
                            }
                            *stiff_mat.idxm2(i2, j2) += K4;
                            *stiff_mat.idxm2(i3, j2) += K5;
                            *stiff_mat.idxm2(i3, j3) += K6;
                        }

                        #[cfg(feature="clamped_plate")] {
                            let i2 = i1 + mesh.free_nodes_n;
                            let i3 = i2 + mesh.free_nodes_n;
                            let j2 = j1 + mesh.free_nodes_n;
                            let j3 = j2 + mesh.free_nodes_n;

                            if i_is_free && j_is_free {
                                if i1 == j1 {
                                    *mass_diag.idxm(i1) += M1;
                                    *mass_diag.idxm(i2) += M2;
                                    *mass_diag.idxm(i3) += M3;
                                }
                                *stiff_mat.idxm2(i1, j1) += K1;
                                *stiff_mat.idxm2(i2, j1) += K2;
                                *stiff_mat.idxm2(i3, j1) += K3;
                                *stiff_mat.idxm2(i2, j2) += K4;
                                *stiff_mat.idxm2(i3, j2) += K5;
                                *stiff_mat.idxm2(i3, j3) += K6;
                            }
                        }
                    }
                }
            }         
            curr_work = i_elem + 1;
            if curr_work % prog_size == 0 {
                prog_bar.inc(prog_size as u64);
            }
        }
        prog_bar.finish_with_message("Finished.\n");
    }


    #[inline]
    pub fn compute_modal_quad
    <MT1: RMatMut<fsize>, MT2: RMat<fsize>>
    ( 
        &self, 
        modal_quad: &mut MT1, // (modes_n, elems_n)
        eigvec_trans: &MT2, // (modes_n, dof)
        iso: &IsoElement,
        mesh: &Mesh2d,
        mbuf: &mut Mesh2dBuf,
        _normalize: fsize,
        n_prog: usize,
    ) {
        println!("Computing element modal quadratures...");
        let dof = mesh.dof;
        let elems_n = mesh.elems_n;
        let modes_n = modal_quad.nrow();
        assert_eq!(modes_n, eigvec_trans.nrow());
        assert_eq!(dof, eigvec_trans.ncol());

        let mut quad_buf: Vec<fsize> = vec![0.; mesh.n]; // (n,)

        let total_work = elems_n;
        let prog_size = std::cmp::max(1, (total_work as fsize / n_prog as fsize).ceil() as usize);
        let prog_bar = ProgressBar::new(total_work as u64);
        let mut curr_work: usize;

        for i_elem in 0..elems_n {

            let mut modal_quad_ = modal_quad.col_mut(i_elem);
            let en_idx = mesh.elems_nodes.col(i_elem);
            index_vec2_unbind(&mesh.nodes_xy, en_idx.sl(), &mut mbuf.en_x, &mut mbuf.en_y);
            self.rotate.rotate_vec(&mut mbuf.en_x, &mut mbuf.en_y);
            mbuf.compute_mapping(&iso);
            mbuf.compute_jacobian(&iso);

            for elem!(i, s_) in mzip!(0..mesh.n, quad_buf.itm()) {
                *s_ = iso.quad(iso.f.col(i).sl(), mbuf.jac_det);
            }

            for elem!(i, i_global, s_) in mzip!(0..mesh.n, en_idx.it(), quad_buf.itm()) {
                let i1 = *i_global;

                let s__ = iso.quad(iso.f.col(i).sl(), mbuf.jac_det);
                *s_ = s__;
                
                if i1 < mesh.free_nodes_n {
                    modal_quad_.addassign_scale(&eigvec_trans.col(i1), s__);
                }
            }

            curr_work = i_elem + 1;
            if curr_work % prog_size == 0 {
                prog_bar.inc(prog_size as u64);
            }
        }
        prog_bar.finish_with_message("Finished.\n");
    }


    #[inline]
    pub fn compute_response
    <MT1: RMat<fsize>, MT2: RMat<fsize>, MT3: RMat<fsize>>
    (
        &self,
        dt: fsize,
        sound_speed: fsize,
        bridge_pos: [fsize; 2], // This point is supposed to be within one of the bridge elements.
        bridge_group_range: &Vec<Vec<[usize; 2]>>,
        listen_pos: [fsize; 3],
        elems_center_xy: &[[fsize; 2]], // (elems_n,)

        response_0: &mut [fsize], // (n_time,), for transverse movement.
        response_1: &mut [fsize], // (n_time,), for shear x movement.
        response_2: &mut [fsize], // (n_time,), for shear y movement.

        modal_force_0: &mut [fsize], // (modes_n,), should have been set to zeros.
        modal_force_1: &mut [fsize], // (modes_n,), should have been set to zeros.
        modal_force_2: &mut [fsize], // (modes_n,), should have been set to zeros.
        
        modal_buf_0: &mut [fsize], // (modes_n,), temporary buffer.
        modal_buf_1: &mut [fsize], // (modes_n,), temporary buffer.
        modal_buf_2: &mut [fsize], // (modes_n,), temporary buffer.

        modal_quad: &MT1, // (modes_n, elems_n)
        eigvec_trans: &MT2, // (modes_n, dof)
        modal_move: &MT3, // (modes_n, n_time)

        iso: &IsoElement,
        mesh: &Mesh2d,
        mbuf: &mut Mesh2dBuf,

        _normalize: fsize,
        n_prog: usize,
    ) {
        let elems_n = mesh.elems_n;
        let modes_n = modal_quad.nrow();
        let nt = modal_move.ncol();
        assert_multi_eq!(elems_n, elems_center_xy.len(), modal_quad.ncol());
        assert_multi_eq!(
            modes_n, modal_move.nrow(), eigvec_trans.nrow(), 
            modal_force_0.len(), modal_force_1.len(), modal_force_2.len(), 
            modal_buf_0.len(), modal_buf_1.len(), modal_buf_2.len()
        );
        assert_multi_eq!(nt, response_0.len(), response_1.len(), response_2.len());
        assert_eq!(mesh.dof, eigvec_trans.ncol());

        // Find which element the input bridge position belongs to.
        println!("Finding the element index of bridge position...");
        let mut bridge_elem: usize = usize::MAX;
        let [u0, v0]: [fsize; 2] = [0., 0.];
        {
            let [x0, y0] = bridge_pos;

            // Search which element the bridge point is in.
            'outer: for ranges in bridge_group_range.iter() {
                for group_range in ranges.iter() {
    
                    for [i_elem_lb, i_elem_ub] in mesh.groups_elems_idx[group_range[0]..group_range[1]].iter() {
                        
                        for i_elem in *i_elem_lb..*i_elem_ub {
                            let corners = mesh.elems_nodes.subvec2(0, i_elem, 3, i_elem);
                            let i1 = *corners.idx(0);
                            let i2 = *corners.idx(1);
                            let i3 = *corners.idx(2);
    
                            let [x1, y1] = mesh.nodes_xy[i1];
                            let [x2, y2] = mesh.nodes_xy[i2];
                            let [x3, y3] = mesh.nodes_xy[i3];
    
                            if is_in_triangle(x0, y0, x1, y1, x2, y2, x3, y3) {
                                bridge_elem = i_elem;
                                break 'outer;
                            }
                        }
                    }
                }
            }
            if bridge_elem == usize::MAX {
                panic!("The input bridge position is not within any bridge element.");
            }
            let en_idx = mesh.elems_nodes.col(bridge_elem);
            index_vec2_unbind(&mesh.nodes_xy, en_idx.sl(), &mut mbuf.en_x, &mut mbuf.en_y);
            mbuf.compute_mapping(&iso);
            //mbuf.compute_jacobian(&iso);
            mbuf.compute_inv_mapping(&iso);
            iso.xy_to_uv(&[bridge_pos], &mut [[u0, v0]], &mbuf.u_co, &mbuf.v_co);

            println!("Finished.\n");
        }

        // Compute the distribution of external force onto the modal basis.
        {
            println!("Computing modal forces...");
            let en_idx = mesh.elems_nodes.col(bridge_elem);

            for elem!(i, i_global) in mzip!(0..mesh.n, en_idx.it()) {
                let i1 = *i_global;

                let val = iso.f_at_point(i, u0, v0);

                if i1 < mesh.free_nodes_n {
                    for elem!(e, s) in mzip!(eigvec_trans.col(i1).it(), modal_force_0.iter_mut()) {
                        *s += e * val;
                    }
                    #[cfg(feature="clamped_plate")] {
                        let i2 = i1 + mesh.free_nodes_n;
                        let i3 = i2 + mesh.free_nodes_n;

                        for elem!(e, s) in mzip!(eigvec_trans.col(i2).it(), modal_force_1.iter_mut()) {
                            *s += e * val;
                        }
                        for elem!(e, s) in mzip!(eigvec_trans.col(i3).it(), modal_force_2.iter_mut()) {
                            *s += e * val;
                        }
                    }
                }
                #[cfg(not(feature="clamped_plate"))] {
                    let i2 = i1 + mesh.free_nodes_n;
                    let i3 = i2 + mesh.nodes_n;

                    for elem!(e, s) in mzip!(eigvec_trans.col(i2).it(), modal_force_1.iter_mut()) {
                        *s += e * val;
                    }
                    for elem!(e, s) in mzip!(eigvec_trans.col(i3).it(), modal_force_2.iter_mut()) {
                        *s += e * val;
                    }
                }
            }
            println!("Finished.\n");
        }

        // Compute sound radiation in the air using Rayleigh integral.
        {
            println!("Computing sound radiation in the air...");
            let [x0, y0, z0] = listen_pos;

            let total_work = elems_n;
            let prog_size = std::cmp::max(1, (total_work as fsize / n_prog as fsize).ceil() as usize);
            let prog_bar = ProgressBar::new(total_work as u64);
            let mut curr_work: usize;

            for elem!(i_elem, [x1, y1]) in mzip!(0..elems_n, elems_center_xy.iter()) {
                let distance = ((x1-x0).powi(2) + (y1-y0).powi(2) + z0.powi(2)).sqrt();
                let delay = (distance / (sound_speed * dt)).round() as usize;

                for elem!(modal_quad_, f0, f1, f2, b0, b1, b2) in mzip!(
                    modal_quad.col(i_elem).it(), 
                    modal_force_0.iter(), modal_force_1.iter(), modal_force_2.iter(),
                    modal_buf_0.iter_mut(), modal_buf_1.iter_mut(), modal_buf_2.iter_mut()
                ) {
                    *b0 = f0 * modal_quad_ / distance;
                    *b1 = f1 * modal_quad_ / distance;
                    *b2 = f2 * modal_quad_ / distance;
                }

                for elem!(t, r0, r1, r2) in mzip!(
                    0..nt-delay, response_0[delay..].iter_mut(), response_1[delay..].iter_mut(), 
                    response_2[delay..].iter_mut()
                ) {
                    for elem!(modal_move_, b0, b1, b2) in mzip!(
                        modal_move.col(t).it(), modal_buf_0.iter(), modal_buf_1.iter(), modal_buf_2.iter()
                    ) {
                        *r0 += modal_move_ * b0;
                        *r1 += modal_move_ * b1;
                        *r2 += modal_move_ * b2;
                    }
                }
                curr_work = i_elem + 1;
                if curr_work % prog_size == 0 {
                    prog_bar.inc(prog_size as u64);
                }
            }
            prog_bar.finish_with_message("Finished.\n");
        }
    }
}
