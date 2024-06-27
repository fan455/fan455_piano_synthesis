use fan455_arrf64::*;
use fan455_math_scalar::*;
use fan455_math_array::*;
use fan455_util::{assert_multi_eq, elem, mzip};
use indicatif::ProgressBar;
use super::piano_io::*;
use super::piano_finite_element::*;


#[derive(Default)] #[allow(non_snake_case)]
pub struct PianoBridges {
    pub sb_thickness: fsize, // soundboard thicknes
    pub angle: Vec<fsize>, // The angle of bridges to the old x axis, in range [0, pi], will be the new x axis.
    pub rotate: Vec<CoordSysRotation>, // (num,)

    pub num: usize,
    pub height: fsize, 
    pub density: fsize,
    pub young_modulus: [fsize; 2], // E_x, E_y
    pub shear_modulus: [fsize; 3], // G_xy, G_xz, G_yz
    pub poisson_ratio: fsize, // v_xy; v_yx will be autocomputed.
    pub shear_correct: [fsize; 2], // (k_x)^2, (k_y)^2

    pub mass_co: [fsize; 2], // M1, M2, auto computed
    pub stiff_co: [fsize; 6], // D1, D2, D3, D4, D5, D6, auto computed

    pub group_range: Vec<Vec<[usize; 2]>>, // (n_bridges, n_ranges_for_each_bridge)
}


impl PianoBridges
{
    #[inline] #[allow(non_snake_case)]
    pub fn new( 
        data: &PianoBridgesParamsIn,
        sb_thickness: fsize,
    ) -> Self {
        println!("Initializing ribs parameters...");
        let num = data.num;

        let rho = data.density;
        let h = data.height;
        let h_sb = sb_thickness;
        let h_tmp = ( (h+h_sb/2.).powi(3) - (h_sb/2.).powi(3) ) / 3.;
        let [E_x, E_y] = data.young_modulus;
        let [G_xy, G_xz, G_yz] = data.shear_modulus;
        let nu_xy = data.poisson_ratio;
        let [k2_x, k2_y] = data.shear_correct;

        let nu_yx = nu_xy * E_y / E_x;

        let mass_co: [fsize; 2] = [rho*h, rho*h_tmp];
        let stiff_co: [fsize; 6] = [
            (h_tmp * E_x) / (1.- nu_xy * nu_yx),
            (h_tmp * E_x * nu_yx) / (1.- nu_xy * nu_yx),
            (h_tmp * E_y) / (1.- nu_xy * nu_yx),
            2.* h_tmp * G_xy,
            h * k2_x * G_xz,
            h_tmp * k2_y * G_yz,
        ];

        let angle: Vec<fsize> = data.angle.clone();
        assert_multi_eq!(num, angle.size());

        let mut rotate: Vec<CoordSysRotation> = Vec::with_capacity(num);
        for angle_ in angle.iter() {
            rotate.push(CoordSysRotation::new(PI-angle_, CoordSysRotation::CLOCKWISE));
        }
        let group_range = data.group_range.clone();
        //println!("bridges group range = {:?}", group_range);

        println!("Finished.\n");
        Self { sb_thickness, num, angle, height: data.height, density: data.density, young_modulus: data.young_modulus, shear_modulus: data.shear_modulus, poisson_ratio: data.poisson_ratio, shear_correct: data.shear_correct, mass_co, stiff_co, rotate, group_range }
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
        println!("Computing the bridges contribution to the mass matrix and stiffness matrix...");
        //let pr0 = RVecPrinter::new(16, 4);

        let free_nodes_n = mesh.free_nodes_n;
        let nodes_n = mesh.nodes_n;
        //let elems_n = mesh.elems_n;
        let quad_n = mesh.quad_n;
        let en_n = mesh.n;
        let dof = free_nodes_n + 2*nodes_n;
        assert_multi_eq!(dof, mass_diag.size(), stiff_mat.nrow, stiff_mat.ncol);

        let [C1, C2] = self.mass_co;
        let [D1, D2, D3, D4, D5, D6] = self.stiff_co;

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

        for i_bridge in 0..self.num {
            let rotate = &self.rotate[i_bridge];

            for group_range in self.group_range[i_bridge].iter() {

                for [i_elem_lb, i_elem_ub] in mesh.groups_elems_idx[group_range[0]..group_range[1]].iter() {
                    
                    for i_elem in *i_elem_lb..*i_elem_ub {
                        // Compute the isoparametric mapping.
                        let en_idx = mesh.elems_nodes.col(i_elem);
                        index_vec2_unbind(&mesh.nodes_xy, en_idx.sl(), &mut mbuf.en_x, &mut mbuf.en_y);
                        rotate.rotate_vec(&mut mbuf.en_x, &mut mbuf.en_y);
                        mbuf.compute_mapping(&iso);
                        mbuf.compute_jacobian(&iso);

                        if mbuf.jac_det < 0. {
                            panic!("Negative jacobian determinant encountered.");
                        }

                        // Compute the gradient and hessian of basis functions, if needed.
                        mbuf.compute_gradient(&iso, &en_idx_local);
                        
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

                                    let quad_f_g   = iso.quad(&f_g  , mbuf.jac_det);
                                    let quad_f_gx  = iso.quad(&f_gx , mbuf.jac_det);
                                    let quad_f_gy  = iso.quad(&f_gy , mbuf.jac_det);
                                    //let quad_fx_g  = iso.quad(&fx_g , mbuf.jac_det);
                                    let quad_fx_gx = iso.quad(&fx_gx, mbuf.jac_det);
                                    let quad_fx_gy = iso.quad(&fx_gy, mbuf.jac_det);
                                    //let quad_fy_g  = iso.quad(&fy_g , mbuf.jac_det);
                                    let quad_fy_gx = iso.quad(&fy_gx, mbuf.jac_det);
                                    let quad_fy_gy = iso.quad(&fy_gy, mbuf.jac_det);

                                    let i2 = i1 + free_nodes_n;
                                    let i3 = i2 + nodes_n;
                                    let j2 = j1 + free_nodes_n;
                                    let j3 = j2 + nodes_n;

                                    if i1 == j1 {
                                        let M1 = C1 * quad_f_g;
                                        let M2 = C2 * quad_f_g;
                                        let M3 = C2 * quad_f_g;

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
                                        let K1 = D5 * quad_fx_gx + D6 * quad_fy_gy;
                                        if K1.is_nan() {
                                            panic!("Stiffness matrix value at element {} and global index ({}, {}) got nan, 
                                            in soundboard's plate part.", i_elem, i1, j1);
                                        }
                                        *stiff_mat.idxm2(i1, j1) += K1;
                                    }
                                    if j_is_free {
                                        let K2 = D5 * quad_f_gx;
                                        if K2.is_nan() {
                                            panic!("Stiffness matrix value at element {} and global index ({}, {}) got nan, 
                                            in soundboard's plate part.", i_elem, i2, j1);
                                        }
                                        *stiff_mat.idxm2(i2, j1) += K2;

                                        let K3 = D6 * quad_f_gy;
                                        if K3.is_nan() {
                                            panic!("Stiffness matrix value at element {} and global index ({}, {}) got nan, 
                                            in soundboard's plate part.", i_elem, i3, j1);
                                        }
                                        *stiff_mat.idxm2(i3, j1) += K3;
                                    }
                                    let K4 = D5 * quad_f_g + D1 * quad_fx_gx + D4 * quad_fy_gy;
                                    if K4.is_nan() {
                                        panic!("Stiffness matrix value at element {} and global index ({}, {}) got nan, 
                                        in soundboard's plate part.", i_elem, i2, j2);
                                    }
                                    *stiff_mat.idxm2(i2, j2) += K4;

                                    let K5 = D4 * quad_fx_gy + D2 * quad_fy_gx;
                                    if K5.is_nan() {
                                        panic!("Stiffness matrix value at element {} and global index ({}, {}) got nan, 
                                        in soundboard's plate part.", i_elem, i3, j2);
                                    }
                                    *stiff_mat.idxm2(i3, j2) += K5;

                                    let K6 = D6 * quad_f_g + D3 * quad_fy_gy + D4 * quad_fx_gx;
                                    if K6.is_nan() {
                                        panic!("Stiffness matrix value at element {} and global index ({}, {}) got nan, 
                                        in soundboard's plate part.", i_elem, i3, j3);
                                    }
                                    *stiff_mat.idxm2(i3, j3) += K6;
                                }
                            }
                        }         
                        curr_work += 1;
                        if curr_work % prog_size == 0 {
                            prog_bar.inc(prog_size as u64);
                        }
                    }
                }
            }
        }
        prog_bar.finish_with_message("Finished.\n");
    }
}
