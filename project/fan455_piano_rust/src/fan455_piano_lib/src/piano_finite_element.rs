use fan455_arrf64::*;
use fan455_math_scalar::*;
use fan455_math_array::*;
use fan455_util::{elem, mzip, NpyObject};


// Element types
pub const ELEM_TRIANGLE: u8 = 1;
pub const ELEM_QUADRANGLE: u8 = 2;
// Node types
pub const CORNER_NODE: u8 = 0;
pub const EDGE_NODE: u8 = 1;
pub const INNER_NODE: u8 = 2;
// gmsh element types
pub const GMSH_TRIANGLE_10: u8 = 10;


#[inline]
pub fn dof_reissner_mindlin_plate( free_nodes_n: usize, _nodes_n: usize ) -> usize {
    #[cfg(not(feature="clamped_plate"))] return free_nodes_n + 2*_nodes_n;
    #[cfg(feature="clamped_plate")] return free_nodes_n*3;
}


pub struct Mesh2dBuf {
    // The below data is buffer, updated for each element, so storage will be efficient.
    pub en_x: Vec<fsize>, // (n_nodes_per_elem,), element nodes x coordinates, indexed from nodes_x using elems_nodes.
    pub en_y: Vec<fsize>, // (n_nodes_per_elem,), element nodes y coordinates, indexed from nodes_y using elems_nodes.

    pub x_co: [fsize; 3], // coefficients of isoparametric transform from (u, v) to x.
    pub y_co: [fsize; 3], // coefficients of isoparametric transform from (u, v) to y.

    pub u_co: [fsize; 3], // coefficients of isoparametric transform from (x, y) to u.
    pub v_co: [fsize; 3], // coefficients of isoparametric transform from (x, y) to v.

    pub jac_det: fsize, // jacobian determinant
    pub jac_inv: [fsize; 4], // inverse jacobian

    pub f: Arr2<fsize>, // (quad_n, n_nodes_per_elem)
    pub fx: Arr2<fsize>, // (quad_n, n_nodes_per_elem)
    pub fy: Arr2<fsize>, // (quad_n, n_nodes_per_elem)
    pub fxx: Arr2<fsize>, // (quad_n, n_nodes_per_elem)
    pub fyy: Arr2<fsize>, // (quad_n, n_nodes_per_elem)
    pub fxy: Arr2<fsize>, // (quad_n, n_nodes_per_elem)

    pub quad_x: Vec<fsize>, // (quad_n,), x coordinate
    pub quad_y: Vec<fsize>, // (quad_n,), y coordinate
}


impl Mesh2dBuf
{
    #[inline]
    pub fn new() -> Self {
        println!("Initializing mesh buffer...");
        let n = IsoElement::N;
        let quad_n = IsoElement::QUAD_N;

        let en_x: Vec<fsize> = vec![0.; n];
        let en_y: Vec<fsize> = vec![0.; n];

        let x_co: [fsize; 3] = [0.; 3];
        let y_co: [fsize; 3] = [0.; 3];

        let u_co: [fsize; 3] = [0.; 3];
        let v_co: [fsize; 3] = [0.; 3];

        let jac_det: fsize = 0.;
        let jac_inv: [fsize; 4] = [0.; 4];

        let f: Arr2<fsize> = Arr2::new(quad_n, n);
        let fx: Arr2<fsize> = Arr2::new(quad_n, n);
        let fy: Arr2<fsize> = Arr2::new(quad_n, n);
        let fxx: Arr2<fsize> = Arr2::new(quad_n, n);
        let fyy: Arr2<fsize> = Arr2::new(quad_n, n);
        let fxy: Arr2<fsize> = Arr2::new(quad_n, n);

        let quad_x: Vec<fsize> = vec![0.; quad_n];
        let quad_y: Vec<fsize> = vec![0.; quad_n];
        println!("Finished.\n");

        Self { en_x, en_y, x_co, y_co, u_co, v_co, jac_det, jac_inv, f, fx, fy, fxx, fyy, fxy, quad_x, quad_y }
    }


    #[inline]
    pub fn compute_mapping( &mut self, iso: &IsoElement ) {
        let vertices: [[fsize; 2]; 3] = [
            [self.en_x[0], self.en_y[0]],
            [self.en_x[1], self.en_y[1]],
            [self.en_x[2], self.en_y[2]],
        ];
        iso.compute_mapping(&vertices, &mut self.x_co, &mut self.y_co);
    }


    #[inline]
    pub fn compute_jacobian( &mut self, iso: &IsoElement ) {
        iso.compute_jacobian(&self.x_co, &self.y_co, &mut self.jac_inv, &mut self.jac_det);
    }


    #[inline]
    pub fn compute_inv_mapping( &mut self, iso: &IsoElement ) {
        iso.compute_inv_mapping(&self.x_co, &self.y_co, &mut self.u_co, &mut self.v_co);
    }


    #[inline]
    pub fn compute_gradient(  &mut self, iso: &IsoElement, free_local: &[usize] ) {
        let en_free_n = free_local.len();
        for elem!(i, i_local) in mzip!(0..en_free_n, free_local.iter()) {
            iso.compute_gradient(*i_local, self.fx.col_mut(i).slm(), self.fy.col_mut(i).slm(), &self.jac_inv);
        }
    }


    #[inline]
    pub fn compute_hessian(  &mut self, iso: &IsoElement, free_local: &[usize] ) {
        let en_free_n = free_local.len();
        for elem!(i, i_local) in mzip!(0..en_free_n, free_local.iter()) {
            iso.compute_hessian(
                *i_local, self.fxx.col_mut(i).slm(), self.fyy.col_mut(i).slm(), self.fxy.col_mut(i).slm(), &self.jac_inv
            );
        }
    }


    #[inline]
    pub fn compute_quad_x( &mut self, iso: &IsoElement ) {
        iso.compute_quad_x(&self.x_co, &mut self.quad_x);
    }


    #[inline]
    pub fn compute_quad_y( &mut self, iso: &IsoElement ) {
        iso.compute_quad_y(&self.y_co, &mut self.quad_y);
    }
}


pub struct Mesh2d {
    // This struct stores all nodes and elements.
    pub n: usize,
    pub dof: usize,
    pub nodes_n: usize,
    pub free_nodes_n: usize, // It is the first index of boundary nodes.
    pub boundary_nodes_n: usize,
    pub elems_n: usize,
    pub quad_n: usize,
    pub groups_n: usize,
    
    pub nodes_xy: Vec<[fsize; 2]>,
    // (nodes_n,). x,y Coordinates of all nodes.
    pub elems_nodes: Arr2<usize>, 
    // (n_nodes_per_elem, elems_n), a column-major matrix. The indices of nodes for each element. Each column corresponds to an element, storing the global ids of element nodes. 
    // For each element, there are 3 types of nodes and they are stored in this order: firstly nodes on the 4 vertices, then nodes on the edges (excluding 4 vertices), finally nodes inside the element.
    //pub elems_groups: Vec<u8>, 
    // (elems_n,). Which group a element belongs to, e.g. plate, ribs...
    pub groups_elems_idx: Vec<[usize; 2]>, // (num_of_groups,), element index range of each group.

    pub en_free_n: Vec<usize>,
    pub en_free_global: Arr2<usize>, 
    pub en_free_local: Arr2<usize>,
}


impl Mesh2d
{
    #[inline]
    pub fn new( 
        free_nodes_n: usize,
        nodes_xy_path: &String,
        elems_nodes_path: &String,
        //elems_groups_path: &String,
        groups_elems_idx_path: &String,
    ) -> Self {
        println!("Reading and processing mesh data...");
        //let pr0 = RVecPrinter::new(12, 3);
        //let pr1 = RMatPrinter::new(12, 3);

        println!("Reading nodes coordinates data at \"{nodes_xy_path}\"...");
        //let nodes_xy: Vec<[fsize; 2]> = unsafe {read_npy_tm::<[fsize; 2]>(nodes_xy_path)};
        let nodes_xy: Vec<[fsize; 2]> = {
            let mut npy = NpyObject::<[fsize; 2]>::new_reader(&nodes_xy_path);
            npy.read_header().unwrap();
            npy.read()
        };
        println!("Finished.\n");

        let nodes_n = nodes_xy.len();
        let dof = dof_reissner_mindlin_plate(free_nodes_n, nodes_n);

        println!("Reading elements nodes indices data at {elems_nodes_path}...");
        let elems_nodes: Arr2<usize> = Arr2::<usize>::read_npy(elems_nodes_path);
        println!("Finished.\n");
        //let elems_groups: Vec<u8> = unsafe {read_npy_tm(elems_groups_path)};
        println!("Reading groups elements indices data at {groups_elems_idx_path}...");
        let groups_elems_idx: Vec<[usize; 2]> = {
            let mut npy = NpyObject::<[usize; 2]>::new_reader(&groups_elems_idx_path);
            npy.read_header().unwrap();
            npy.read()
        };
        println!("Finished.\n");
        let groups_n = groups_elems_idx.len();

        let n = elems_nodes.nrow();
        let elems_n = elems_nodes.ncol();
        assert_eq!(n, IsoElement::N);
        //assert_eq!(elems_n, elems_groups.len());

        //println!("nodes_xy[100..110] = {:?}", &nodes_xy[100..110]);
        //println!("elems_nodes.col(100) = {:?}", elems_nodes.col(100).sl());
        //println!("groups_elems_idx = {:?}", groups_elems_idx);
        
        let boundary_nodes_n = elems_n - free_nodes_n;
        let quad_n = IsoElement::QUAD_N;

        let mut en_free_n: Vec<usize> = vec![0; elems_n];
        let mut en_free_global: Arr2<usize> = Arr2::new(n, elems_n); 
        let mut en_free_local: Arr2<usize> = Arr2::new(n, elems_n); 
        {
            for elem!(i_elem, en_free_n_) in mzip!(0..elems_n, en_free_n.itm()) {
                let mut global = en_free_global.col_mut(i_elem);
                let mut local = en_free_local.col_mut(i_elem);
                let mut it_global = global.itm();
                let mut it_local = local.itm();
                let mut en_free_n_tmp: usize = 0;
                
                for elem!(i_global, i_local) in mzip!(elems_nodes.col(i_elem).it(), 0..n) {
                    if *i_global < free_nodes_n {
                        *it_global.next().unwrap() = *i_global;
                        *it_local.next().unwrap() = i_local;
                        en_free_n_tmp += 1;
                    }
                }
                *en_free_n_ = en_free_n_tmp;
                //println!("en_free_n_ = {en_free_n_}");
                //pr0.print_usize("global", &global);
                //pr0.print_usize("local", &local);
            }
        }
        println!("Finished.\n");

        Self { n, dof, nodes_n, elems_n, free_nodes_n, boundary_nodes_n, quad_n, groups_n,
            nodes_xy, elems_nodes, groups_elems_idx, en_free_n, en_free_global, en_free_local }
    }

    #[inline]
    pub fn compute_elems_center( &self, elems_center_xy: &mut [[fsize; 2]] ) {
        // elems_center_xy: (elems_n,)
        for elem!(i_elem, [x_, y_]) in mzip!(0..self.elems_n, elems_center_xy.iter_mut()) {
            let vertices = self.elems_nodes.subvec2(0, i_elem, 3, i_elem);

            let [x1, y1] = self.nodes_xy[*vertices.idx(0)];
            let [x2, y2] = self.nodes_xy[*vertices.idx(1)];
            let [x3, y3] = self.nodes_xy[*vertices.idx(2)];

            *x_ = (x1+x2+x3) / 3.;
            *y_ = (y1+y2+y3) / 3.;
        }
    }
}


pub struct IsoElement {
    // An isoparametric quadrilateral element defined for x in [-1, 1] and y in [-1, 1].
    // n = (order+1)^2, is also the number of basis functins in an element.
    pub f_pu: Vec<i32>, pub f_pv: Vec<i32>,
    pub fu_pu: Vec<i32>, pub fu_pv: Vec<i32>,
    pub fv_pu: Vec<i32>, pub fv_pv: Vec<i32>,
    pub fuu_pu: Vec<i32>, pub fuu_pv: Vec<i32>,
    pub fvv_pu: Vec<i32>, pub fvv_pv: Vec<i32>,
    pub fuv_pu: Vec<i32>, pub fuv_pv: Vec<i32>,

    pub f_co: Arr2<fsize>, // (n, n) 
    pub fu_co: Arr2<fsize>, // (fu_n, n) 
    pub fv_co: Arr2<fsize>, // (fv_n, n) 
    pub fuu_co: Arr2<fsize>, // (fuu_n, n) 
    pub fvv_co: Arr2<fsize>, // (fvv_n, n) 
    pub fuv_co: Arr2<fsize>, // (fuv_n, n)

    pub fu_idx: Vec<usize>, // wrt f
    pub fv_idx: Vec<usize>, // wrt f
    pub fuu_idx: Vec<usize>, // wrt fu
    pub fvv_idx: Vec<usize>, // wrt fv
    pub fuv_idx: Vec<usize>, // wrt fu

    pub f: Arr2<fsize>, // (quad_n, n)
    pub fu: Arr2<fsize>, // (quad_n, n)
    pub fv: Arr2<fsize>, // (quad_n, n)
    pub fuu: Arr2<fsize>, // (quad_n, n)
    pub fvv: Arr2<fsize>, // (quad_n, n)
    pub fuv: Arr2<fsize>, // (quad_n, n)
}


#[allow(non_snake_case)]
impl IsoElement
{
    pub const ORDER: usize = 3;
    pub const N: usize = 10;

    pub const FU_N: usize = eval_poly2_tr_fx_size!(Self::ORDER, Self::N);
    pub const FV_N: usize = eval_poly2_tr_fx_size!(Self::ORDER, Self::N);
    pub const FUU_N: usize = eval_poly2_tr_fxx_size!(Self::ORDER, Self::N);
    pub const FVV_N: usize = eval_poly2_tr_fxx_size!(Self::ORDER, Self::N);
    pub const FUV_N: usize = eval_poly2_tr_fxy_size!(Self::ORDER, Self::N);

    pub const QUAD_N: usize = Self::N;
    pub const C1: fsize = 0.2763932023;
    pub const C2: fsize = 1.- Self::C1;
    pub const NODES: [[fsize; 2]; Self::N] = [
        [0., 0.], [1., 0.], [0., 1.], 
        [Self::C1, 0.], [Self::C2, 0.], 
        [Self::C2, Self::C1], [Self::C1, Self::C2], 
        [0., Self::C2], [0., Self::C1], 
        [1./3., 1./3.]
    ];
    pub const NODES_U: [fsize; Self::N] = [
        0., 1., 0., Self::C1, Self::C2, Self::C2, Self::C1, 0., 0., 1./3.
    ];
    pub const NODES_V: [fsize; Self::N] = [
        0., 0., 1., 0., 0., Self::C1, Self::C2, Self::C2, Self::C1, 1./3.
    ];
    pub const WEIGHTS: [fsize; 10] = [
        1./120., 1./120., 1./120., 
        1./24., 1./24., 1./24., 1./24., 1./24., 1./24., 
        0.9/4.
    ];

    #[inline]
    pub fn new() -> Self {
        //let pr0 = RVecPrinter::new(12, 3);
        //let pr1 = RMatPrinter::new(12, 3);

        let mut f_pu: Vec<i32> = Vec::with_capacity(Self::N);
        let mut f_pv: Vec<i32> = Vec::with_capacity(Self::N);

        for iu in 0..Self::ORDER+1 {
            for iv in 0..Self::ORDER+1 {
                if iu + iv <= Self::ORDER {
                    let pu = iu as i32;
                    let pv = iv as i32;
                    f_pu.push(pu);
                    f_pv.push(pv);
                }
            }
        }
        
        // Compute polynomial coefficients.
        let mut f_co: Arr2<fsize> = Arr2::new(Self::N, Self::N);
        for elem!(i, pu, pv) in mzip!(0..Self::N, f_pu.iter(), f_pv.iter()) {
            for elem!(s, u, v) in mzip!(f_co.col_mut(i).itm(), Self::NODES_U.iter(), Self::NODES_V.iter()) {
                *s = u.powi(*pu) * v.powi(*pv);
            }
        }
        {
            let n_ = Self::N as BlasInt;
            let mut ipiv: Vec<BlasInt> = vec![0; Self::N];
            unsafe {
                #[cfg(feature="use_32bit_float")] {
                    LAPACKE_sgetrf(COL_MAJ, n_, n_, f_co.ptrm(), n_, ipiv.ptrm());
                    LAPACKE_sgetri(COL_MAJ, n_, f_co.ptrm(), n_, ipiv.ptrm());
                }
                #[cfg(not(feature="use_32bit_float"))] {
                    LAPACKE_dgetrf(COL_MAJ, n_, n_, f_co.ptrm(), n_, ipiv.ptrm());
                    LAPACKE_dgetri(COL_MAJ, n_, f_co.ptrm(), n_, ipiv.ptrm());
                }
            }
        }

        let mut fu_idx: Vec<usize> = vec![0; Self::FU_N];
        let mut fv_idx: Vec<usize> = vec![0; Self::FV_N];
        let mut fuu_idx: Vec<usize> = vec![0; Self::FUU_N];
        let mut fvv_idx: Vec<usize> = vec![0; Self::FVV_N];
        let mut fuv_idx: Vec<usize> = vec![0; Self::FUV_N];

        let mut fu_pu: Vec<i32> = vec![0; Self::FU_N]; let mut fu_pv: Vec<i32> = vec![0; Self::FU_N];
        let mut fv_pu: Vec<i32> = vec![0; Self::FV_N]; let mut fv_pv: Vec<i32> = vec![0; Self::FV_N];
        let mut fuu_pu: Vec<i32> = vec![0; Self::FUU_N]; let mut fuu_pv: Vec<i32> = vec![0; Self::FUU_N];
        let mut fvv_pu: Vec<i32> = vec![0; Self::FVV_N]; let mut fvv_pv: Vec<i32> = vec![0; Self::FVV_N];
        let mut fuv_pu: Vec<i32> = vec![0; Self::FUV_N]; let mut fuv_pv: Vec<i32> = vec![0; Self::FUV_N];

        // Compute fu
        poly2_fx_idx(f_pu.sl(), fu_idx.slm());
        poly2_fx_pow(f_pu.sl(), f_pv.sl(), fu_pu.slm(), fu_pv.slm(), fu_idx.sl());
        // Compute fv
        poly2_fy_idx(f_pv.sl(), fv_idx.slm());
        poly2_fy_pow(f_pu.sl(), f_pv.sl(), fv_pu.slm(), fv_pv.slm(), fv_idx.sl());
        // Compute fuu
        poly2_fx_idx(fu_pu.sl(), fuu_idx.slm());
        poly2_fx_pow(fu_pu.sl(), fu_pv.sl(), fuu_pu.slm(), fuu_pv.slm(), fuu_idx.sl());
        // Compute fvv
        poly2_fy_idx(fv_pv.sl(), fvv_idx.slm());
        poly2_fy_pow(fv_pu.sl(), fv_pv.sl(), fvv_pu.slm(), fvv_pv.slm(), fvv_idx.sl());
        // Compute fuv
        poly2_fy_idx(fu_pv.sl(), fuv_idx.slm());
        poly2_fy_pow(fu_pu.sl(), fu_pv.sl(), fuv_pu.slm(), fuv_pv.slm(), fuv_idx.sl());

        let mut fu_co: Arr2<fsize> = Arr2::new(Self::FU_N, Self::N);
        let mut fv_co: Arr2<fsize> = Arr2::new(Self::FV_N, Self::N);
        let mut fuu_co: Arr2<fsize> = Arr2::new(Self::FUU_N, Self::N);
        let mut fvv_co: Arr2<fsize> = Arr2::new(Self::FVV_N, Self::N);
        let mut fuv_co: Arr2<fsize> = Arr2::new(Self::FUV_N, Self::N);

        for i in 0..Self::N {
            // compute fu
            poly2_fx_coef(f_co.col(i).sl(), fu_co.col_mut(i).slm(), fu_pu.sl(), fu_idx.sl());
            // compute fv
            poly2_fy_coef(f_co.col(i).sl(), fv_co.col_mut(i).slm(), fv_pv.sl(), fv_idx.sl());
            // compute fuu
            poly2_fx_coef(fu_co.col(i).sl(), fuu_co.col_mut(i).slm(), fuu_pu.sl(), fuu_idx.sl());
            // compute fvv
            poly2_fy_coef(fv_co.col(i).sl(), fvv_co.col_mut(i).slm(), fvv_pv.sl(), fvv_idx.sl());
            // compute fuv
            poly2_fy_coef(fu_co.col(i).sl(), fuv_co.col_mut(i).slm(), fuv_pv.sl(), fuv_idx.sl());
        }

        let mut f: Arr2<fsize> = Arr2::new(Self::QUAD_N, Self::N);
        let mut fu: Arr2<fsize> = Arr2::new(Self::QUAD_N, Self::N);
        let mut fv: Arr2<fsize> = Arr2::new(Self::QUAD_N, Self::N);
        let mut fuu: Arr2<fsize> = Arr2::new(Self::QUAD_N, Self::N);
        let mut fvv: Arr2<fsize> = Arr2::new(Self::QUAD_N, Self::N);
        let mut fuv: Arr2<fsize> = Arr2::new(Self::QUAD_N, Self::N);

        for i in 0..Self::N {
            poly2_batch(f_co.col(i).sl(), f_pu.sl(), f_pv.sl(), Self::NODES_U.sl(), Self::NODES_V.sl(), f.col_mut(i).slm());
            poly2_batch(fu_co.col(i).sl(), fu_pu.sl(), fu_pv.sl(), Self::NODES_U.sl(), Self::NODES_V.sl(), fu.col_mut(i).slm());
            poly2_batch(fv_co.col(i).sl(), fv_pu.sl(), fv_pv.sl(), Self::NODES_U.sl(), Self::NODES_V.sl(), fv.col_mut(i).slm());
            poly2_batch(fuu_co.col(i).sl(), fuu_pu.sl(), fuu_pv.sl(), Self::NODES_U.sl(), Self::NODES_V.sl(), fuu.col_mut(i).slm());
            poly2_batch(fvv_co.col(i).sl(), fvv_pu.sl(), fvv_pv.sl(), Self::NODES_U.sl(), Self::NODES_V.sl(), fvv.col_mut(i).slm());
            poly2_batch(fuv_co.col(i).sl(), fuv_pu.sl(), fuv_pv.sl(), Self::NODES_U.sl(), Self::NODES_V.sl(), fuv.col_mut(i).slm());
        }

        /*pr1.print("f_co", &f_co);
        pr1.print("fu_co", &fu_co);
        pr1.print("fv_co", &fv_co);
        pr1.print("fuu_co", &fuu_co);
        pr1.print("fvv_co", &fvv_co);
        pr1.print("fuv_co", &fu_co);

        pr1.print("f", &f);
        pr1.print("fu", &fu);
        pr1.print("fv", &fv);
        pr1.print("fuu", &fuu);
        pr1.print("fvv", &fvv);
        pr1.print("fuv", &fuv);*/

        Self { f_pu, f_pv, fu_pu, fu_pv, fv_pu, fv_pv, 
            fuu_pu, fuu_pv, fvv_pu, fvv_pv, fuv_pu, fuv_pv, 
            fu_idx, fv_idx, fuu_idx, fvv_idx, fuv_idx,
            f_co, fu_co, fv_co, fuu_co, fvv_co, fuv_co, 
            f, fu, fv, fuu, fvv, fuv }
    }


    #[inline]
    pub fn compute_mapping( 
        &self, vertices: &[[fsize; 2]; 3], 
        x_co: &mut [fsize; 3],
        y_co: &mut [fsize; 3],
    ) {
        let [[x0, y0], [x1, y1], [x2, y2]] = *vertices;
        // u, v are the natural coordinates, x, y are the physical coordinates.
        // x = p0 * u + p1 * v + p2
        x_co[0] = -x0 + x1;
        x_co[1] = -x0 + x2;
        x_co[2] = x0;
        // y = q0 * u + q1 * v + q2
        y_co[0] = -y0 + y1;
        y_co[1] = -y0 + y2;
        y_co[2] = y0;
    }


    #[inline]
    pub fn compute_jacobian( 
        &self, 
        x_co: &[fsize; 3], // [p0, p1, p2]
        y_co: &[fsize; 3], // [q0, q1, q2]
        jac_inv: &mut [fsize; 4], // (n_quad,)
        jac_det: &mut fsize // (n_quad,)
    ) {
        // iso_jac is a vector of 2*2 column-major matrix.
        // x = p0 * u + p1 * v + p2
        // y = q0 * u + q1 * v + q2
        let [a_11, a_12, _] = *x_co;
        let [a_21, a_22, _] = *y_co;
        let det = a_11 * a_22 - a_21 * a_12;

        jac_inv[0] = a_22 / det;
        jac_inv[1] = -a_21 / det;
        jac_inv[2] = -a_12 / det;
        jac_inv[3] = a_11 / det;
        *jac_det = det;
    }


    #[inline]
    pub fn f_at_points( &self, i: usize, u: &[fsize], v: &[fsize], f: &mut [fsize] ) {
        // f should have been initialized to zeros.
        poly2_batch(self.f_co.col(i).sl(), self.f_pu.sl(), self.f_pv.sl(), u, v, f);
    }


    #[inline]
    pub fn f_at_point( &self, i: usize, u: fsize, v: fsize ) -> fsize {
        poly2(self.f_co.col(i).sl(), self.f_pu.sl(), self.f_pv.sl(), u, v)
    }


    #[inline]
    pub fn compute_gradient( 
        &self, i: usize, // index of basis function
        fx: &mut [fsize], // (quad_n,)
        fy: &mut [fsize], // (quad_n,)
        jac_inv: &[fsize; 4], // (quad_n,), inverse jacobian matrix.
    ) {
        let [a, b, c, d] = *jac_inv;
        for elem!(fx_, fy_, fu_, fv_) in mzip!(
            fx.iter_mut(), fy.iter_mut(), self.fu.col(i).it(), self.fv.col(i).it()
        ) {
            *fx_ = a * fu_ + b * fv_;
            *fy_ = c * fu_ + d * fv_;
        }
    }


    #[inline]
    pub fn compute_hessian( 
        &self, i: usize, // index of basis function
        fxx: &mut [fsize], // (quad_n,)
        fyy: &mut [fsize], // (quad_n,)
        fxy: &mut [fsize], // (quad_n,)
        jac_inv: &[fsize; 4], // (quad_n,), inverse jacobian matrix.
    ) {
        let [a, b, c, d] = *jac_inv;
        for elem!(fxx_, fyy_, fxy_, fuu_, fvv_, fuv_) in mzip!(
            fxx.iter_mut(), fyy.iter_mut(), fxy.iter_mut(), 
            self.fuu.col(i).it(), self.fvv.col(i).it(), self.fuv.col(i).it()
        ) {
            *fxx_ = a.powi(2) * fuu_ + b.powi(2) * fvv_ + 2.*a*b * fuv_;
            *fyy_ = c.powi(2) * fuu_ + d.powi(2) * fvv_ + 2.*c*d * fuv_;
            *fxy_ = a*c * fuu_ + b*d * fvv_ + (a*d+b*c) * fuv_;
        }
    }


    #[inline]
    pub fn compute_quad_x(
        &self,
        x_co: &[fsize; 3], // (n,)
        quad_x: &mut [fsize], // (quad_n,)
    ) {
        // x = p0 * u + p1 * v + p2
        for elem!(x, u, v) in mzip!(quad_x.iter_mut(), Self::NODES_U.iter(), Self::NODES_V.iter()) {
            *x = x_co[0]*u + x_co[1]*v + x_co[2];
        }
    }

    #[inline]
    pub fn compute_quad_y(
        &self,
        y_co: &[fsize; 3], // (n,)
        quad_y: &mut [fsize], // (quad_n,)
    ) {
        // y = q0 * u + q1 * v + q2
        for elem!(y, u, v) in mzip!(quad_y.iter_mut(), Self::NODES_U.iter(), Self::NODES_V.iter()) {
            *y = y_co[0]*u + y_co[1]*v + y_co[2];
        }
    }


    #[inline]
    pub fn compute_inv_mapping( 
        &self,  
        x_co: &[fsize; 3],
        y_co: &[fsize; 3],
        u_co: &mut [fsize; 3],
        v_co: &mut [fsize; 3],
    ) {
        let [p0, p1, p2] = *x_co;
        let [q0, q1, q2] = *y_co;
        let det = p0 * q1 - p1 * q0;

        u_co[0] = q1 / det;
        u_co[1] = -p1 / det;
        u_co[2] = (p1 * q2 - p2 * q1) / det;

        v_co[0] = -q0 / det;
        v_co[1] = p0 / det;
        v_co[2] = (-p0 * q2 + p2 * q0) / det;
    }


    #[inline]
    pub fn xy_to_uv(
        &self,
        xy: &[[fsize; 2]], // (quad_n,)
        uv: &mut [[fsize; 2]], // (quad_n,)
        u_co: &[fsize; 3], // (n,)
        v_co: &[fsize; 3], // (n,)
    ) {
        for elem!([x, y], [u, v]) in mzip!(xy.iter(), uv.iter_mut()) {
            *u = u_co[0]*x + u_co[1]*y + u_co[2];
            *v = v_co[0]*x + v_co[1]*y + v_co[2];
        }
    }


    #[inline]
    pub fn quad( 
        &self, 
        f: &[fsize],  // (quad_n,)
        jac_det: fsize, // (quad_n,)
    ) -> fsize {
        let mut s: fsize = 0.;
        for elem!(f_, w_) in mzip!(f.iter(), Self::WEIGHTS.iter()) {
            s += f_ * w_ * jac_det;
        }
        s
    }

    #[inline]
    pub fn quad2( 
        &self, 
        f: &[fsize],  // (quad_n,)
        g: &[fsize],  // (quad_n,)
        jac_det: fsize, // (quad_n,)
    ) -> fsize {
        let mut s: fsize = 0.;
        for elem!(f_, g_, w_) in mzip!(f.iter(), g.iter(), Self::WEIGHTS.iter()) {
            s += f_ * g_ * w_ * jac_det;
        }
        s
    }
}
