use fan455_arrf64::*;
use fan455_math_array::*;
use fan455_util::{assert_multi_eq, convert_hashmap_keys, elem, mzip, read_npy_vec};
use super::piano_io::*;
use std::collections::HashMap;


// Node kinds
pub const FREE_NODE: u8 = 0;
pub const BOUNDARY_NODE: u8 = 1;

pub const CORNER_NODE: u8 = 0;
pub const EDGE_NODE: u8 = 1;
pub const INNER_NODE: u8 = 2;


pub struct PianoSoundboardMesh {
    // This struct stores all nodes and elements.
    pub order: [usize; 5],
    pub dof: usize, // Total dof
    /*pub dofs: [usize; 5], 
    // 0: dof for w (transverse displacement)
    // 1: dof for alpha (rotation x)
    // 2: dof for beta (rotation y)
    // 3: dof for p (in-plane shear displacement x)
    // 4: dof for q (in-plane shear displacement y)*/
    pub edof_max: usize,
    pub edofs_max: [usize; 5],
    pub edof_max_unique: usize,
    pub edofs_max_unique: [usize; 3],

    pub iso: [TrElemC0; 3],
    // 0: Isoparametric element for w
    // 1: Isoparametric element for alpha, beta
    // 2: Isoparametric element for p, q

    pub enn: usize,
    pub nodes_n: usize,
    pub elems_n: usize,
    pub quad_n: usize,
    
    pub nodes_kinds: Vec<[u8; 2]>, // (nodes_n,), [corner/edge/inner, is_on_boundary]
    pub nodes_xy: Vec<[f64; 2]>, // (nodes_n,), x,y Coordinates of all nodes.
    pub elems_nodes: Arr2<usize>, // (enn, elems_n)
    pub elems_groups: Vec<usize>, // (elems_n,), which group every element belongs to.
    pub groups_ribs: HashMap<usize, [usize; 2]>, // (groups_n,), maps a group index to ribs index.
    pub groups_bridges: HashMap<usize, usize>, // (groups_n,), maps a group index to bridges index.
    pub nodes_dof: Vec<[usize; 2]>, // (nodes_n,), the DOFs that each node corresponds to.
    pub dof_kinds: Vec<usize>, // (dof,), the kinds that each DOF belongs to.
}


impl PianoSoundboardMesh
{
    pub const ORDER: [usize; 5] = [2, 1, 1, 1, 1];
    pub const ORDER_UNIQUE: [usize; 3] = [2, 1, 1];
    pub const EDOFS_MAX: [usize; 5] = [6, 3, 3, 3, 3];
    pub const EDOFS_MAX_UNIQUE: [usize; 3] = [6, 3, 3];
    pub const EDOF_MAX: usize = 18;
    pub const EDOF_MAX_UNIQUE: usize = 12;
    pub const ENN: usize = 6;

    pub const DISP_Z: usize = 0; // Transverse displacement.
    pub const ROT_X : usize = 1; // Shear rotation in x direction.
    pub const ROT_Y : usize = 2; // Shear rotation in y direction.
    pub const DISP_X: usize = 3; // In-plane shear displacement in x direction.
    pub const DISP_Y: usize = 4; // In-plane shear displacement in y direction.

    #[inline]
    pub fn compute_dof_data( nodes_kinds: &Vec<[u8; 2]> ) -> (Vec<[usize; 2]>, Vec<usize>) {
        let nodes_n = nodes_kinds.len();
        let mut nodes_dof: Vec<[usize; 2]> = Vec::with_capacity(nodes_n);
        let mut dof_kinds: Vec<usize> = Vec::with_capacity(nodes_n*5);
        {
            let dof_corner_free = [Self::DISP_Z, Self::ROT_X, Self::ROT_Y, Self::DISP_X, Self::DISP_Y];
            let dof_corner_boundary = [Self::ROT_X, Self::ROT_Y];
            let [mut i_dof_lb, mut i_dof_ub]: [usize; 2] = [0, 0];

            for kind_ in nodes_kinds.iter() {
                let [node_position, node_boundary] = *kind_;

                match (node_position, node_boundary) {

                    (CORNER_NODE, FREE_NODE) => {
                        dof_kinds.extend(dof_corner_free.iter());
                        i_dof_ub += 5;
                        nodes_dof.push([i_dof_lb, i_dof_ub]);
                    },

                    (CORNER_NODE, BOUNDARY_NODE) => {
                        dof_kinds.extend(dof_corner_boundary.iter());
                        i_dof_ub += 2;
                        nodes_dof.push([i_dof_lb, i_dof_ub]);
                    },

                    (EDGE_NODE, FREE_NODE) => {
                        dof_kinds.push(Self::DISP_Z);
                        i_dof_ub += 1;
                        nodes_dof.push([i_dof_lb, i_dof_ub]);
                    },

                    (EDGE_NODE, BOUNDARY_NODE) => {
                        nodes_dof.push([i_dof_lb, i_dof_ub]);
                    },

                    _ => {
                        panic!("Unknown (node_position, node_boundary) pair: ({node_position}, {node_boundary})");
                    },
                }
                i_dof_lb = i_dof_ub;
            }
        }
        dof_kinds.shrink_to_fit();
        (nodes_dof, dof_kinds)
    }

    #[inline]
    pub fn new( data: &PianoMeshParamsIn, is_preprocess: bool ) -> Self {
        println!("Reading and processing mesh data...");

        let nodes_kinds: Vec<[u8; 2]> = read_npy_vec(&data.nodes_kinds_path);
        let nodes_n = nodes_kinds.len();
        let nodes_xy: Vec<[f64; 2]>;
        if is_preprocess {
            nodes_xy = Vec::new();
        } else {
            nodes_xy = read_npy_vec(&data.nodes_xy_path);
            assert_eq!(nodes_xy.len(), nodes_n);
        }
        let enn: usize = Self::ENN;
        let order = Self::ORDER;
        let edof_max = Self::EDOF_MAX;
        let edofs_max = Self::EDOFS_MAX;
        let edof_max_unique = Self::EDOF_MAX_UNIQUE;
        let edofs_max_unique = Self::EDOFS_MAX_UNIQUE;

        let elems_nodes: Arr2<usize> = Arr2::<usize>::read_npy(&data.elems_nodes_path);
        let elems_n = elems_nodes.ncol();
        assert_eq!(elems_nodes.nrow(), enn);

        let elems_groups: Vec<usize> = read_npy_vec(&data.elems_groups_path);
        assert_eq!(elems_n, elems_groups.len());

        let groups_ribs: HashMap<usize, [usize; 2]> = convert_hashmap_keys(&data.groups_ribs);
        let groups_bridges: HashMap<usize, usize> = convert_hashmap_keys(&data.groups_bridges);

        let iso: [TrElemC0; 3];
        let quad_n: usize;
        let nodes_dof: Vec<[usize; 2]>;
        let dof_kinds: Vec<usize>;

        if is_preprocess {
            iso = [TrElemC0::default(), TrElemC0::default(), TrElemC0::default()];
            quad_n = 0;
            (nodes_dof, dof_kinds) = Self::compute_dof_data(&nodes_kinds);
        } else {
            iso = [
                TrElemC0::new(Self::ORDER_UNIQUE[0], &data.quad_points_path, &data.quad_weights_path),
                TrElemC0::new(Self::ORDER_UNIQUE[1], &data.quad_points_path, &data.quad_weights_path),
                TrElemC0::new(Self::ORDER_UNIQUE[2], &data.quad_points_path, &data.quad_weights_path),
            ];
            quad_n = iso[0].quad_n;
            assert_multi_eq!(quad_n, iso[1].quad_n, iso[2].quad_n);

            nodes_dof = read_npy_vec(&data.nodes_dof_path);
            dof_kinds = read_npy_vec(&data.dof_kinds_path);
        }
        let dof = dof_kinds.len();
        println!("dof = {dof}");
        println!("Finished.\n");

        Self { order, dof, edof_max, edofs_max, edof_max_unique, edofs_max_unique, 
            iso, enn, nodes_n, elems_n, quad_n,
            nodes_kinds, nodes_xy, elems_nodes, elems_groups, groups_bridges, groups_ribs, nodes_dof, dof_kinds }
    }

    #[inline]
    pub fn output_params( &self ) -> PianoMeshParamsOut {
        PianoMeshParamsOut {
            order: self.order,
            dof: self.dof,
            edof_max: self.edof_max,
            edofs_max: self.edofs_max,
            edof_max_unique: self.edof_max_unique,
            edofs_max_unique: self.edofs_max_unique,
            enn: self.enn,
            nodes_n: self.nodes_n,
            elems_n: self.elems_n,
            quad_n: self.quad_n,
        }
    }

    #[inline]
    pub fn compute_elems_center( &self, elems_center_xy: &mut [[f64; 2]] ) {
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

    #[inline]
    pub fn create_edof_idx_buf( &self ) -> Vec<[usize; 3]> {
        vec![[0, 0, 0]; self.edof_max]
    }

    #[inline]
    pub fn get_edof( &self, i_elem: usize, edof_idx: &mut [[usize; 3]] ) -> usize {
        // edof_idx: (edof_max,); stores (kind, i_dof_global, i_dof_local)
        let mut edof: usize = 0;
        let mut edof_idx_itm = edof_idx.iter_mut();
        for elem!(i_node_global, i_node_local) in mzip!(self.elems_nodes.col(i_elem).it(), 0..self.enn) {
            let [i_dof_lb, i_dof_ub] = self.nodes_dof[*i_node_global];

            if i_dof_ub > i_dof_lb {
                edof += i_dof_ub - i_dof_lb;

                for i_dof in i_dof_lb..i_dof_ub {
                    let [kind_, i_global_, i_local_] = edof_idx_itm.next().unwrap();
                    *kind_ = self.dof_kinds[i_dof];
                    *i_global_ = i_dof;
                    *i_local_ = i_node_local;
                }
            }
        }
        edof
    }
}


pub struct PianoSoundboardMeshBuf {
    pub map: TrElemMapBuf,
    pub edof: usize,
    pub edof_idx: Vec<[usize; 3]>, // (edof_max,); stores (kind, i_dof_global, i_dof_local)

    pub b0: TrElemQuadBuf,
    pub b1: TrElemQuadBuf,
    pub b2: TrElemQuadBuf,
}

impl PianoSoundboardMeshBuf
{
    #[inline]
    pub fn new( mesh: &PianoSoundboardMesh ) -> Self {
        let map = TrElemMapBuf::new(mesh.enn);
        let edof = mesh.edof_max;
        let edof_idx = mesh.create_edof_idx_buf();

        let b0 = TrElemQuadBuf::new(&mesh.iso[0]);
        let b1 = TrElemQuadBuf::new(&mesh.iso[1]);
        let b2 = TrElemQuadBuf::new(&mesh.iso[2]);

        Self {map, edof, edof_idx, b0, b1, b2}
    }

    #[inline]
    pub fn process_elem( &mut self, i_elem: usize, mesh: &PianoSoundboardMesh ) {
        index_vec(&mesh.nodes_xy, mesh.elems_nodes.col(i_elem).sl(), &mut self.map.en_xy);
        self.edof = mesh.get_edof(i_elem, &mut self.edof_idx);
    }

    #[inline]
    pub fn iter_edof_idx( &self ) -> std::slice::Iter<[usize; 3]> {
        self.edof_idx[..self.edof].iter()
    }

    #[inline]
    pub fn compute_mapping( &mut self ) {
        self.map.compute_mapping();
    }

    #[inline]
    pub fn compute_jacobian( &mut self ) {
        self.map.compute_jacobian();
    }

    #[inline]
    pub fn compute_inv_mapping( &mut self ) {
        self.map.compute_inv_mapping();
    }

    #[inline]
    pub fn compute_gradient( &mut self, mesh: &PianoSoundboardMesh ) {
        self.b0.compute_gradient(&mesh.iso[0], &self.map);
        self.b1.compute_gradient(&mesh.iso[1], &self.map);
        self.b2.compute_gradient(&mesh.iso[2], &self.map);
    }
}


pub struct TrElemMapBuf {
    // The below data is buffer, updated for each element, so storage will be efficient.
    pub en_xy: Vec<[f64; 2]>, // (n_nodes_per_elem,), element nodes x,y coordinates, indexed from nodes_xy using elems_nodes.

    pub x_co: [f64; 3], // coefficients of isoparametric transform from (u, v) to x.
    pub y_co: [f64; 3], // coefficients of isoparametric transform from (u, v) to y.

    pub u_co: [f64; 3], // coefficients of isoparametric transform from (x, y) to u.
    pub v_co: [f64; 3], // coefficients of isoparametric transform from (x, y) to v.

    pub jac_det: f64, // jacobian determinant
    pub jac_inv: [f64; 4], // inverse jacobian
}

impl TrElemMapBuf
{
    #[inline]
    pub fn new( enn: usize ) -> Self {
        println!("Initializing TrElemMapBuf (triangle element mapping buffer)...");

        let en_xy: Vec<[f64; 2]> = vec![[0., 0.]; enn];

        let x_co: [f64; 3] = [0.; 3];
        let y_co: [f64; 3] = [0.; 3];

        let u_co: [f64; 3] = [0.; 3];
        let v_co: [f64; 3] = [0.; 3];

        let jac_det: f64 = 0.;
        let jac_inv: [f64; 4] = [0.; 4];

        println!("Finished.\n");

        Self { en_xy, x_co, y_co, u_co, v_co, jac_det, jac_inv }
    }

    #[inline]
    pub fn compute_mapping( &mut self ) {
        let vertices: [[f64; 2]; 3] = [ self.en_xy[0], self.en_xy[1], self.en_xy[2] ];
        TrElemC0::compute_mapping(&vertices, &mut self.x_co, &mut self.y_co);
    }

    #[inline]
    pub fn compute_jacobian( &mut self ) {
        TrElemC0::compute_jacobian(&self.x_co, &self.y_co, &mut self.jac_inv, &mut self.jac_det);
    }

    #[inline]
    pub fn compute_inv_mapping( &mut self ) {
        TrElemC0::compute_inv_mapping(&self.x_co, &self.y_co, &mut self.u_co, &mut self.v_co);
    }
}


pub struct TrElemQuadBuf {
    // The below data is buffer, updated for each element, so storage will be efficient.
    pub f: Arr2<f64>, // (quad_n, edof)
    pub fx: Arr2<f64>, // (quad_n, edof)
    pub fy: Arr2<f64>, // (quad_n, edof)

    pub quad_xy: Vec<[f64; 2]>, // (quad_n,), x,y coordinates
}


impl TrElemQuadBuf
{
    #[inline]
    pub fn new( iso: &TrElemC0 ) -> Self {
        println!("Initializing mesh buffer...");

        let f = iso.f.clone();
        let fx = iso.fu.clone();
        let fy = iso.fv.clone();

        let quad_xy = iso.quad_uv.clone();
        println!("Finished.\n");

        Self { f, fx, fy, quad_xy }
    }

    #[inline]
    pub fn compute_gradient(  &mut self, iso: &TrElemC0, map: &TrElemMapBuf ) {
        for i in 0..iso.edof {
            iso.compute_gradient(i, self.fx.col_mut(i).slm(), self.fy.col_mut(i).slm(), &map.jac_inv);
        }
    }

    #[inline]
    pub fn compute_quad_xy( &mut self, iso: &TrElemC0, map: &TrElemMapBuf ) {
        iso.compute_quad_xy(&map.x_co, &map.y_co, &mut self.quad_xy);
    }

    #[inline]
    pub fn edof( &self ) -> usize {
        self.f.ncol()
    }

    #[inline]
    pub fn quad_n( &self ) -> usize {
        self.quad_xy.len()
    }
}


/*pub enum PianoSbMeshInit {
    ComputeDof,
    ReadDof,
}*/


#[derive(Default)]
pub struct TrElemC0 {
    // An isoparametric triangle element with C0 continuity.
    pub order: usize, // Polynomial order.
    pub enn: usize, // Number of nodes in an element.
    pub edof: usize, // Degrees of freedom in an element.
    pub quad_n: usize, // Number of integration points.
    pub poly: Poly2, // Two dimensional polynomial functions.

    pub nodes_uv: Vec<[f64; 2]>, // (enn,)
    pub quad_uv: Vec<[f64; 2]>, // (quad_n,)
    pub quad_weights: Vec<f64>, // (quad_n,)

    pub f_co: Arr2<f64>, // (edof, edof)
    pub f: Arr2<f64>, // (quad_n, edof)
    pub fu: Arr2<f64>, // (quad_n, edof)
    pub fv: Arr2<f64>, // (quad_n, edof)
}


#[allow(non_snake_case)]
impl TrElemC0
{
    #[inline]
    pub fn new( order: usize, quad_points_path: &str, quad_weights_path: &str ) -> Self {
        let (enn, edof): (usize, usize);
        let mut nodes_uv: Vec<[f64; 2]> = vec![[0., 0.], [1., 0.], [0., 1.]];
        match order {
            1 => {
                [enn, edof] = [3, 3];
            },
            2 => {
                [enn, edof] = [6, 6];
                nodes_uv.reserve(enn-3);
                nodes_uv.extend([ [0.5, 0.], [0.5, 0.5], [0., 0.5,] ].iter());
            },
            3 => {
                [enn, edof] = [10, 10];
                nodes_uv.reserve(enn-3);
                nodes_uv.extend([
                    [1./3., 0.], [2./3., 0.], [2./3., 1./3.], [1./3., 2./3.], [0., 2./3.], [0., 1./3.], [1./3., 1./3.], 
                ].iter());
            },
            _ => panic!("Not supporting C0 triangle element with order {order}."),
        };
        assert_eq!(nodes_uv.len(), enn);
        let poly = Poly2::new_tr(order);
        assert_eq!(edof, poly.n);

        let quad_uv: Vec<[f64; 2]> = read_npy_vec(quad_points_path);
        let quad_weights: Vec<f64> = read_npy_vec(quad_weights_path);
        let quad_n = quad_uv.len();
        assert_eq!(quad_n, quad_weights.len());
        
        let mut f_co: Arr2<f64> = Arr2::new(edof, edof);
        poly.fit(&nodes_uv, &mut f_co);

        let mut f: Arr2<f64> = Arr2::new(quad_n, edof);
        let mut fu: Arr2<f64> = Arr2::new(quad_n, edof);
        let mut fv: Arr2<f64> = Arr2::new(quad_n, edof);

        for i in 0..edof {
            poly.f(&quad_uv, f_co.col(i).sl(), f.col_mut(i).slm());
            poly.fx(&quad_uv, f_co.col(i).sl(), fu.col_mut(i).slm());
            poly.fy(&quad_uv, f_co.col(i).sl(), fv.col_mut(i).slm());
        }     

        Self { order, enn, edof, quad_n, poly, 
            nodes_uv, quad_uv, quad_weights, f_co, f, fu, fv, 
        }
    }


    #[inline]
    pub fn compute_mapping( 
        vertices: &[[f64; 2]; 3], 
        x_co: &mut [f64; 3],
        y_co: &mut [f64; 3],
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
        x_co: &[f64; 3], // [p0, p1, p2]
        y_co: &[f64; 3], // [q0, q1, q2]
        jac_inv: &mut [f64; 4], // (n_quad,)
        jac_det: &mut f64 // (n_quad,)
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
    pub fn f_at_points( &self, i: usize, uv: &[[f64; 2]], f: &mut [f64] ) {
        // i: index of element dof.
        self.poly.f(uv, self.f_co.col(i).sl(), f);
    }


    #[inline]
    pub fn f_at_point( &self, i: usize, uv: &[f64; 2] ) -> f64 {
        // i: index of element dof.
        self.poly.f_single(uv, self.f_co.col(i).sl())
    }


    #[inline]
    pub fn compute_gradient( 
        &self, i: usize, // index of basis function
        fx: &mut [f64], // (quad_n,)
        fy: &mut [f64], // (quad_n,)
        jac_inv: &[f64; 4], // (quad_n,), inverse jacobian matrix.
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
    pub fn compute_quad_xy(
        &self,
        x_co: &[f64; 3], // (n,)
        y_co: &[f64; 3], // (n,)
        quad_xy: &mut [[f64; 2]], // (quad_n,)
    ) {
        // x = p0 * u + p1 * v + p2
        // y = q0 * u + q1 * v + q2
        for elem!([x, y], [u, v]) in mzip!(quad_xy.iter_mut(), self.quad_uv.iter()) {
            *x = x_co[0]*u + x_co[1]*v + x_co[2];
            *y = y_co[0]*u + y_co[1]*v + y_co[2];
        }
    }


    #[inline]
    pub fn compute_inv_mapping( 
        x_co: &[f64; 3],
        y_co: &[f64; 3],
        u_co: &mut [f64; 3], // [p0, p1, p2]
        v_co: &mut [f64; 3], // [q0, q1, q2]
    ) {
        // u = p0 * x + p1 * y + p2
        // v = q0 * x + q1 * y + q2
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
    pub fn xy_to_uv_batch(
        xy: &[[f64; 2]], // (quad_n,)
        uv: &mut [[f64; 2]], // (quad_n,)
        u_co: &[f64; 3], // (n,)
        v_co: &[f64; 3], // (n,)
    ) {
        for elem!([x, y], [u, v]) in mzip!(xy.iter(), uv.iter_mut()) {
            *u = u_co[0]*x + u_co[1]*y + u_co[2];
            *v = v_co[0]*x + v_co[1]*y + v_co[2];
        }
    }


    #[inline]
    pub fn xy_to_uv(
        xy: &[f64; 2], // (quad_n,)
        u_co: &[f64; 3], // (n,)
        v_co: &[f64; 3], // (n,)
    ) -> [f64; 2] {
        let [x, y] = *xy;
        [ u_co[0]*x + u_co[1]*y + u_co[2], v_co[0]*x + v_co[1]*y + v_co[2] ]
    }


    #[inline]
    pub fn quad( 
        &self, 
        f: &[f64],  // (quad_n,)
        jac_det: f64, // scalar
    ) -> f64 {
        let mut s: f64 = 0.;
        for elem!(f_, w_) in mzip!(f.iter(), self.quad_weights.iter()) {
            s += f_ * w_ * jac_det;
        }
        s
    }

    
    #[inline]
    pub fn quad2( 
        &self, 
        f1: &[f64], // (quad_n,)
        f2: &[f64], // (quad_n,)
        jac_det: f64, // scalar
    ) -> f64 {
        let mut s: f64 = 0.;
        for elem!(f1_, f2_, w_) in mzip!(f1.iter(), f2.iter(), self.quad_weights.iter()) {
            s += f1_ * f2_ * w_ * jac_det;
        }
        s
    }

    #[inline]
    pub fn quad3( 
        &self, 
        f1: &[f64], // (quad_n,)
        f2: &[f64], // (quad_n,)
        f3: &[f64], // (quad_n,)
        jac_det: f64, // scalar
    ) -> f64 {
        let mut s: f64 = 0.;
        for elem!(f1_, f2_, f3_, w_) in mzip!(f1.iter(), f2.iter(), f3.iter(), self.quad_weights.iter()) {
            s += f1_ * f2_ * f3_ * w_ * jac_det;
        }
        s
    }
}
