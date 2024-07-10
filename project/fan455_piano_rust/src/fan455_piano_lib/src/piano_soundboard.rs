use fan455_arrf64::*;
//use fan455_math_array::*;
use fan455_math_scalar::*;
use fan455_util::{assert_multi_eq, elem, mzip};
use indicatif::ProgressBar;
use super::piano_io::*;
use super::piano_finite_element::*;
use super::piano_model::*;


#[derive(Default)] #[allow(non_snake_case)]
pub struct PianoSoundboard {   
    pub angle: f64, 
    // The angle of soundboard fibers to the old x axis, in range [0, pi], will be the new x axis.

    pub density: f64,
    pub thickness: f64,
    pub young_modulus: [f64; 2], // E_x, E_y
    pub shear_modulus: [f64; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: f64, // v_xy; v_yx will be autocomputed.
    pub shear_correct: [f64; 2], // k_x, k_y

    pub stiff_co: [f64; 9], // D1 to D9, auto computed
}


impl PianoSoundboard
{
    #[inline] #[allow(non_snake_case)]
    pub fn new( data: &PianoSoundboardParamsIn ) -> Self {
        println!("Initializing soundboard parameters...");
        let angle = data.angle;

        let stiff_co = PlateModel::compute_stiff_coef_rotated(
            data.young_modulus, data.shear_modulus, data.poisson_ratio, 
            data.shear_correct, angle-PI
        );
        
        println!("Finished.\n");

        Self {
            angle,  
            density: data.density, thickness: data.thickness,
            young_modulus: data.young_modulus, shear_modulus: data.shear_modulus, 
            poisson_ratio: data.poisson_ratio, shear_correct: data.shear_correct, 
            stiff_co,
        }
    }


    /*#[inline]
    pub fn output_params( &self ) -> PianoSoundboardParamsOut {
        PianoSoundboardParamsOut {
            mass_co: self.mass_co,
            stiff_co: self.stiff_co,
        }
    }*/


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
        println!("Computing the plate's contribution to the mass matrix and stiffness matrix...");
        //let pr0 = RVecPrinter::new(16, 4);
        assert_multi_eq!(mesh.dof, mass_mat.nrow, mass_mat.ncol, stiff_mat.nrow, stiff_mat.ncol);

        let mut model = PlateModel::new(self.density, self.stiff_co, mesh.quad_n);
        model.compute_quad_z_with_const_thick(-self.thickness/2., self.thickness/2., true);
        println!("called model.compute_quad_z_with_const_thick");

        let total_work = mesh.elems_n;
        let prog_size = std::cmp::max(1, (total_work as f64 / n_prog as f64).ceil() as usize);
        let prog_bar = ProgressBar::new(total_work as u64);
        let mut curr_work: usize;

        for i_elem in 0..mesh.elems_n {
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
                                "Mass matrix value in soundboard part got nan at index ({i_global}, {j_global}), current i_elem = {i_elem}."
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
                                "Stiff matrix value in soundboard part got nan at index ({i_global}, {j_global}), current i_elem = {i_elem}."
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
            curr_work = i_elem + 1;
            if curr_work % prog_size == 0 {
                prog_bar.inc(prog_size as u64);
            }
        }
        prog_bar.finish();
        println!("Finished.\n");
    }


    #[inline]
    pub fn compute_modal_quad
    <MT1: RMatMut<f64>, MT2: RMat<f64>>
    ( 
        &self, 
        modal_quad: &mut MT1, // (modes_n, elems_n)
        eigvec_trans: &MT2, // (modes_n, dof)
        mesh: &PianoSoundboardMesh,
        mbuf: &mut PianoSoundboardMeshBuf,
        _normalize: f64,
        n_prog: usize,
    ) {
        macro_rules! quad {
            ($f1:expr) => {
                mesh.iso[0].quad($f1, mbuf.map.jac_det)
            };
        }
        macro_rules! f  { ($i:ident) => { mbuf.b0.f. col($i).sl() }; }
        macro_rules! g  { ($i:ident) => { mbuf.b1.f. col($i).sl() }; }
        macro_rules! h  { ($i:ident) => { mbuf.b2.f. col($i).sl() }; }

        println!("Computing element modal quadratures...");
        let modes_n = modal_quad.nrow();
        assert_eq!(modes_n, eigvec_trans.nrow());
        assert_eq!(mesh.dof, eigvec_trans.ncol());

        let total_work = mesh.elems_n;
        let prog_size = std::cmp::max(1, (total_work as f64 / n_prog as f64).ceil() as usize);
        let prog_bar = ProgressBar::new(total_work as u64);
        let mut curr_work: usize;

        for i_elem in 0..mesh.elems_n {
            let mut modal_quad_ = modal_quad.col_mut(i_elem);

            mbuf.process_elem(i_elem, mesh);
            mbuf.compute_mapping();
            mbuf.compute_jacobian();
            assert!(mbuf.map.jac_det > 0., 
                "Element {i_elem} has negative jacobian determinant value: {:.6}", mbuf.map.jac_det
            );

            for dof_i in mbuf.iter_edof_idx() {
                let [i_kind, i_global, i_local] = *dof_i;
                let s = match i_kind {
                    0     => quad!( f![i_local] ),
                    1 | 2 => quad!( g![i_local] ),
                    3 | 4 => quad!( h![i_local] ),
                    _ => panic!("Invaild kind {i_kind} in computing modal_quad."),
                };
                modal_quad_.addassign_scale(&eigvec_trans.col(i_global), s);
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
    pub fn compute_response
    <MT1: RMat<f64>, MT2: RMat<f64>, MT3: RMat<f64>,
    MT4: RMatMut<f64>, MT5: RMatMut<f64>, MT6: RMatMut<f64>>
    (
        &self,
        dt: f64,
        sound_speed: f64,
        bridge_pos: [f64; 2], // This point is supposed to be within one of the bridge elements.
        listen_pos: [f64; 3],
        elems_center_xy: &[[f64; 2]], // (elems_n,)

        response: &mut MT4, // (n_time, 5)
        modal_force: &mut MT5, // (modes_n, 5)
        modal_buf: &mut MT6, // (modes_n, 5), temporary buffer.

        modal_quad: &MT1, // (modes_n, elems_n)
        eigvec_trans: &MT2, // (modes_n, dof)
        modal_move: &MT3, // (modes_n, n_time)

        mesh: &PianoSoundboardMesh,
        mbuf: &mut PianoSoundboardMeshBuf,

        _normalize: f64,
        n_prog: usize,
    ) {
        let elems_n = mesh.elems_n;
        let modes_n = modal_quad.nrow();
        let nt = modal_move.ncol();
        assert_multi_eq!(elems_n, elems_center_xy.len(), modal_quad.ncol());
        assert_multi_eq!(5, response.ncol(), modal_force.ncol(), modal_buf.ncol());
        assert_multi_eq!(modes_n, modal_move.nrow(), eigvec_trans.nrow(), modal_force.nrow(), modal_buf.nrow());
        assert_eq!(nt, response.nrow());
        assert_eq!(mesh.dof, eigvec_trans.ncol());

        // Find which element the input bridge position belongs to.
        println!("Finding the element index of bridge position...");
        let mut bridge_elem: usize = usize::MAX;
        {
            let [x0, y0] = bridge_pos;

            // Search which element the bridge point is in.
            'outer: for elem!(i_elem, i_group) in mzip!(0..mesh.elems_n, mesh.elems_groups.it()) {

                if let Some(_) = mesh.groups_bridges.get(i_group) {
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
            if bridge_elem == usize::MAX {
                panic!("The input bridge position is not within any bridge element.");
            }
            println!("Finished.\n");
        }

        // Compute the distribution of external force onto the modal basis.
        {
            println!("Computing modal force...");
            mbuf.process_elem(bridge_elem, mesh);
            mbuf.compute_mapping();
            mbuf.compute_jacobian();
            mbuf.compute_inv_mapping();

            let bridge_uv = TrElemC0::xy_to_uv(&bridge_pos, &mbuf.map.u_co, &mbuf.map.v_co);

            for dof_i in mbuf.iter_edof_idx() {
                let [i_kind, i_global, i_local] = *dof_i;
                let s = match i_kind {
                    0     => mesh.iso[0].f_at_point(i_local, &bridge_uv),
                    1 | 2 => mesh.iso[1].f_at_point(i_local, &bridge_uv),
                    3 | 4 => mesh.iso[2].f_at_point(i_local, &bridge_uv),
                    _ => panic!("Invaild kind {i_kind} in computing modal_force."),
                };
                modal_force.col_mut(i_kind).addassign_scale(&eigvec_trans.col(i_global), s);
            }
            println!("Finished.\n");
        }

        // Compute sound radiation in the air using Rayleigh integral.
        {
            println!("Computing sound radiation in the air...");
            let [x0, y0, z0] = listen_pos;

            let total_work = elems_n;
            let prog_size = std::cmp::max(1, (total_work as f64 / n_prog as f64).ceil() as usize);
            let prog_bar = ProgressBar::new(total_work as u64);
            let mut curr_work: usize;

            for elem!(i_elem, [x1, y1]) in mzip!(0..elems_n, elems_center_xy.iter()) {
                let distance = ((x1-x0).powi(2) + (y1-y0).powi(2) + z0.powi(2)).sqrt();
                let delay = (distance / (sound_speed * dt)).round() as usize;

                let modal_quad_ = modal_quad.col(i_elem);
                for i_kind in 0..5 {
                    for elem!(quad_, force_, buf_) in mzip!(
                        modal_quad_.it(), modal_force.col(i_kind).it(), modal_buf.col_mut(i_kind).itm()
                    ) {
                        *buf_ = force_ * quad_ / distance;
                    }
                }
                for i_kind in 0..5 {
                    let modal_buf_ = modal_buf.col(i_kind);
                    for elem!(t, r) in mzip!(0..nt-delay, response.subvec2_mut(delay, i_kind, nt, i_kind).itm()) {
                        for elem!(move_, buf_) in mzip!(modal_move.col(t).it(), modal_buf_.it()) {
                            *r += move_ * buf_;
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
}


