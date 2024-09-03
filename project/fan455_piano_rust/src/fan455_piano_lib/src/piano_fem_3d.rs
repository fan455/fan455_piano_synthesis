use fan455_arrf64::*;
use fan455_math_array::*;
use fan455_util::*;
use super::piano_io::*;
use super::piano_fem_basic::*;
use std::collections::HashMap;


pub const CORNER_NODE_LO_1: usize = 11;
pub const CORNER_NODE_LO_2: usize = 12;
pub const CORNER_NODE_UP_1: usize = 21;
pub const CORNER_NODE_UP_2: usize = 22;

pub const RIBS_BEG: usize = 0;
pub const RIBS_MID: usize = 1;
pub const RIBS_END: usize = 2;

pub const TET_MAP_N: usize = 4;
pub const TPR_MAP_N: usize = 6;


pub struct PianoAirMesh {
    // A 3D mesh of the air surrounding the piano, using tetrahedron elements
    // Use 3 variables u, v, w (acoustic velocity in 3 directions).
    pub order: [usize; 3], 
    pub iso: [TetElem; 3],
    // 0: Isoparametric element for u
    // 1: Isoparametric element for v
    // 2: Isoparametric element for w
    pub dofs_n: usize, // Total dof
    pub edofs_max: usize,
    pub enodes_n: usize,
    pub nodes_n: usize,
    pub elems_n: usize,
    pub quad_n: usize,
    
    pub dofs: Vec<Dof>, // (dofs_n,)
    pub nodes: Vec<Node3>, // (nodes_n,)
    pub elems: Vec<Elem3>, // (elems_n,)
}

impl PianoAirMesh
{
    pub const ORDER: [usize; 3] = [1, 1, 1];
    pub const ENODES_N: usize = 4;
    pub const EDOFS_MAX: usize = 4;

    pub const WALL_NODE: isize = -2;
    pub const SOUNDBOARD_NODE_LO: isize = -3;
    pub const SOUNDBOARD_NODE_UP: isize = -4;

    #[inline]
    pub fn new( params: &PianoParamsIn ) -> Self {
        let args = &params.mesh.air;
        let order = Self::ORDER;
        let enodes_n = Self::ENODES_N;
        let edofs_max = Self::EDOFS_MAX;

        let iso = [
            TetElem::new(order[0], &args.quad_points_path, &args.quad_weights_path),
            TetElem::new(order[1], &args.quad_points_path, &args.quad_weights_path),
            TetElem::new(order[2], &args.quad_points_path, &args.quad_weights_path),
        ];
        let dofs = Dof::read_bin_vec(&args.dofs_path);
        let nodes = Node3::read_bin_vec(&args.nodes_path);
        let elems = Elem3::read_bin_vec(&args.elems_path);

        let dofs_n = dofs.len();
        let nodes_n = nodes.len();
        let elems_n = elems.len();
        let quad_n = iso[0].quad_n;

        Self {order, dofs_n, edofs_max, iso, enodes_n, nodes_n, elems_n, quad_n, dofs, nodes, elems}
    }

    #[inline]
    pub fn create_edofs_buf( &self ) -> Vec<[usize; 3]> {
        // kind, i_global, i_local
        vec![[0, 0, 0]; self.edofs_max]
    }
}


pub struct PianoAirMeshBuf {
    pub map: TetElemMapBuf,
    pub edofs_n: usize,
    pub edofs: Vec<[usize; 3]>, // (edof_max,); stores (kind, i_dof_global, i_dof_local)

    pub b0: Elem3QuadBuf,
    pub b1: Elem3QuadBuf,
    pub b2: Elem3QuadBuf,
}

impl PianoAirMeshBuf
{
    #[inline]
    pub fn new( mesh: &PianoAirMesh ) -> Self {
        let map = TetElemMapBuf::new(mesh.enodes_n);
        let edofs_n = mesh.edofs_max;
        let edofs = mesh.create_edofs_buf();

        let b0 = Elem3QuadBuf::new(&mesh.iso[0]);
        let b1 = Elem3QuadBuf::new(&mesh.iso[1]);
        let b2 = Elem3QuadBuf::new(&mesh.iso[2]);

        Self {map, edofs_n, edofs, b0, b1, b2}
    }

    #[inline]
    pub fn process_elem( &mut self, i_elem: usize, mesh: &PianoAirMesh ) {
        mesh3_get_enodes(i_elem, &mesh.elems, &mesh.nodes, &mut self.map);
        self.edofs_n = mesh3_get_edofs(i_elem, &mut self.edofs, &mesh.elems, &mesh.nodes, &mesh.dofs);
    }

    #[inline]
    pub fn iter_edofs( &self ) -> std::slice::Iter<[usize; 3]> {
        self.edofs[..self.edofs_n].iter()
    }

    #[inline]
    pub fn compute_mapping( &mut self, mesh: &PianoAirMesh ) {
        self.map.compute_mapping(&mesh.iso[0]);
    }

    #[inline]
    pub fn compute_jacobian( &mut self, mesh: &PianoAirMesh ) {
        self.map.compute_jacobian(&mesh.iso[0]);
    }

    #[inline]
    pub fn compute_gradient( &mut self, mesh: &PianoAirMesh ) {
        self.map.compute_gradient(&mesh.iso[0], &mut self.b0);
        self.map.compute_gradient(&mesh.iso[1], &mut self.b1);
        self.map.compute_gradient(&mesh.iso[2], &mut self.b2);
    }
}


pub struct PianoSoundboardMesh {
    // A 3D mesh of the piano soundboard, using triangular prism elements, with 3 displacement variables. 
    pub order: [[usize; 2]; 3], // each is [order_xy, order_z]
    pub iso: [TprElem; 3],
    // 0: Isoparametric element for u
    // 1: Isoparametric element for v
    // 2: Isoparametric element for w
    pub dofs_n: usize, // Total dof
    pub edofs_max: usize,
    pub enodes_n: usize,
    pub nodes_n: usize,
    pub elems_n: usize,
    pub quad_n: usize,
    
    pub dofs: Vec<Dof>, // (dofs_n,)
    pub nodes: Vec<Node3>, // (nodes_n,)
    pub elems: Vec<Elem3>, // (elems_n,)
    pub groups_ribs: HashMap<usize, [usize; 2]>, // (groups_n,), maps a group index to ribs index (i_rib, i_part).
    pub groups_bridges: HashMap<usize, usize>, // (groups_n,), maps a group index to bridges index.
}

impl PianoSoundboardMesh
{
    pub const ORDER: [[usize; 2]; 3] = [[1, 1], [1, 1], [2, 1]];
    pub const ENODES_N: usize = 12;
    pub const EDOFS_MAX: usize = 24;

    #[inline]
    pub fn new( params: &PianoParamsIn ) -> Self {
        let args = &params.mesh.sb;
        let order = Self::ORDER;
        let enodes_n = Self::ENODES_N;
        let edofs_max = Self::EDOFS_MAX;

        let iso = [
            TprElem::new(order[0], &args.quad_points_path, &args.quad_weights_path),
            TprElem::new(order[1], &args.quad_points_path, &args.quad_weights_path),
            TprElem::new(order[2], &args.quad_points_path, &args.quad_weights_path),
        ];
        let dofs = Dof::read_bin_vec(&args.dofs_path);
        let nodes = Node3::read_bin_vec(&args.nodes_path);
        let elems = Elem3::read_bin_vec(&args.elems_path);

        let dofs_n = dofs.len();
        let nodes_n = nodes.len();
        let elems_n = elems.len();
        let quad_n = iso[0].quad_n;

        let groups_ribs: HashMap<usize, [usize; 2]> = convert_hashmap_keys(&args.groups_ribs);
        let groups_bridges: HashMap<usize, usize> = convert_hashmap_keys(&args.groups_bridges);

        Self {order, dofs_n, edofs_max, iso, enodes_n, nodes_n, elems_n, quad_n, dofs, nodes, elems, groups_ribs, groups_bridges}
    }

    #[inline]
    pub fn create_edofs_buf( &self ) -> Vec<[usize; 3]> {
        // kind, i_global, i_local
        vec![[0, 0, 0]; self.edofs_max]
    }
}


pub struct PianoSoundboardMeshBuf {
    pub map: TprElemMapBuf,
    pub edofs_n: usize,
    pub edofs: Vec<[usize; 3]>, // (edof_max,); stores (kind, i_dof_global, i_dof_local)

    pub b0: Elem3QuadBuf,
    pub b1: Elem3QuadBuf,
    pub b2: Elem3QuadBuf,
}

impl PianoSoundboardMeshBuf
{
    #[inline]
    pub fn new( mesh: &PianoSoundboardMesh ) -> Self {
        let map = TprElemMapBuf::new(mesh.enodes_n, mesh.quad_n);
        let edofs_n = mesh.edofs_max;
        let edofs = mesh.create_edofs_buf();

        let b0 = Elem3QuadBuf::new(&mesh.iso[0]);
        let b1 = Elem3QuadBuf::new(&mesh.iso[1]);
        let b2 = Elem3QuadBuf::new(&mesh.iso[2]);

        Self {map, edofs_n, edofs, b0, b1, b2}
    }

    #[inline]
    pub fn process_elem( &mut self, i_elem: usize, mesh: &PianoSoundboardMesh ) {
        mesh3_get_enodes(i_elem, &mesh.elems, &mesh.nodes, &mut self.map);
        self.edofs_n = mesh3_get_edofs(i_elem, &mut self.edofs, &mesh.elems, &mesh.nodes, &mesh.dofs);
    }

    #[inline]
    pub fn iter_edofs( &self ) -> std::slice::Iter<[usize; 3]> {
        self.edofs[..self.edofs_n].iter()
    }

    #[inline]
    pub fn compute_mapping( &mut self, mesh: &PianoSoundboardMesh ) {
        self.map.compute_mapping(&mesh.iso[0]);
    }

    #[inline]
    pub fn compute_jacobian( &mut self, mesh: &PianoSoundboardMesh ) {
        self.map.compute_jacobian(&mesh.iso[0]);
    }

    #[inline]
    pub fn compute_gradient( &mut self, mesh: &PianoSoundboardMesh ) {
        self.map.compute_gradient(&mesh.iso[0], &mut self.b0);
        self.map.compute_gradient(&mesh.iso[1], &mut self.b1);
        self.map.compute_gradient(&mesh.iso[2], &mut self.b2);
    }
}


pub struct Elem3MapBuf<const MAP_N: usize, J1, J2, J3> 
{
    // The below data is buffer, updated for each element, so storage will be efficient.
    pub en_coord: Vec<[f64; 3]>, // (enodes_n,), element nodes x,y,z coordinates.
    pub corner_x: [f64; MAP_N],
    pub corner_y: [f64; MAP_N],
    pub corner_z: [f64; MAP_N],

    pub x_co: [f64; MAP_N], // coefficients of isoparametric transform from (u, v, w) to x.
    pub y_co: [f64; MAP_N], // coefficients of isoparametric transform from (u, v, w) to y.
    pub z_co: [f64; MAP_N], // coefficients of isoparametric transform from (u, v, w) to z.

    pub jac: J1, // (quad_n,), jacobian matrix in column major.
    pub jac_det: J2, // (quad_n,), jacobian determinant.
    pub jac_inv: J3, // (quad_n,), inverse jacobian.
}

pub type TetElemMapBuf = Elem3MapBuf<TET_MAP_N, Jac3, f64, JacInv3>;
pub type TprElemMapBuf = Elem3MapBuf<TPR_MAP_N, Vec<Jac3>, Vec<f64>, Vec<JacInv3>>;

impl<'a, const MAP_N: usize, J1, J2, J3> Elem3MapBuf<MAP_N, J1, J2, J3>
where J1: ScalarOrArray<'a, Jac3>, J2: ScalarOrArray<'a, f64>, J3: ScalarOrArray<'a, JacInv3>,
{
    #[inline]
    pub fn compute_mapping( &mut self, iso: &IsoElem3<MAP_N> ) {
        iso.compute_mapping(
            &self.corner_x, &self.corner_y, &self.corner_z, 
            &mut self.x_co, &mut self.y_co, &mut self.z_co,
        );
    }

    #[inline]
    pub fn compute_gradient( &'a self, iso: &IsoElem3<MAP_N>, q: &mut Elem3QuadBuf ) {
        for i in 0..iso.edofs_n {
            iso.compute_gradient(
                i, q.fx.col_mut(i).slm(), q.fy.col_mut(i).slm(), q.fz.col_mut(i).slm(), &self.jac_inv
            );
        }
    }
}

impl TetElemMapBuf
{
    #[inline]
    pub fn new( enodes_n: usize ) -> Self {
        println!("Initializing TprElemMapBuf (triangular prism element mapping buffer)...");
        let en_coord: Vec<[f64; 3]> = vec![[0.; 3]; enodes_n];
        let corner_x = [0.; TET_MAP_N];
        let corner_y = [0.; TET_MAP_N];
        let corner_z = [0.; TET_MAP_N];

        let x_co = [0.; TET_MAP_N];
        let y_co = [0.; TET_MAP_N];
        let z_co = [0.; TET_MAP_N];

        let jac = Jac3::new();
        let jac_det = 0.;
        let jac_inv = JacInv3::new();
        
        println!("Finished.\n");
        Self { en_coord, corner_x, corner_y, corner_z, x_co, y_co, z_co, jac, jac_det, jac_inv }
    }

    #[inline]
    pub fn compute_jacobian( &mut self, iso: &IsoElem3<TET_MAP_N> ) {
        iso.compute_jacobian(&self.x_co, &self.y_co, &self.z_co, &mut self.jac, &mut self.jac_det, &mut self.jac_inv);
    }
}

impl TprElemMapBuf
{
    #[inline]
    pub fn new( enodes_n: usize, quad_n: usize ) -> Self {
        println!("Initializing TprElemMapBuf (triangular prism element mapping buffer)...");
        let en_coord: Vec<[f64; 3]> = vec![[0.; 3]; enodes_n];
        let corner_x = [0.; TPR_MAP_N];
        let corner_y = [0.; TPR_MAP_N];
        let corner_z = [0.; TPR_MAP_N];

        let x_co = [0.; TPR_MAP_N];
        let y_co = [0.; TPR_MAP_N];
        let z_co = [0.; TPR_MAP_N];

        let jac = vec![Jac3::new(); quad_n];
        let jac_det = vec![0.; quad_n];
        let jac_inv = vec![JacInv3::new(); quad_n];
        
        println!("Finished.\n");
        Self { en_coord, corner_x, corner_y, corner_z, x_co, y_co, z_co, jac, jac_det, jac_inv }
    }

    #[inline]
    pub fn compute_jacobian( &mut self, iso: &IsoElem3<TPR_MAP_N> ) {
        iso.compute_jacobian(&self.x_co, &self.y_co, &self.z_co, &mut self.jac, &mut self.jac_det, &mut self.jac_inv);
    }
}


pub struct Elem3QuadBuf {
    // The below data is buffer, updated for each element, so storage will be efficient.
    pub f: Arr2<f64>, // (quad_n, edofs_n)
    pub fx: Arr2<f64>, // (quad_n, edofs_n)
    pub fy: Arr2<f64>, // (quad_n, edofs_n)
    pub fz: Arr2<f64>, // (quad_n, edofs_n)
    pub quad_coord: Vec<[f64; 3]>, // (quad_n,), x,y coordinates
}

impl Elem3QuadBuf
{
    #[inline]
    pub fn new<const MAP_N: usize>( iso: &IsoElem3<MAP_N> ) -> Self {
        println!("Initializing mesh buffer...");

        let f = iso.f.clone();
        let fx = iso.fu.clone();
        let fy = iso.fv.clone();
        let fz = iso.fw.clone();
        let quad_coord = iso.quad_coord.clone();
        println!("Finished.\n");
        Self { f, fx, fy, fz, quad_coord }
    }
}


#[derive(Copy, Clone)]
pub enum IsoElem3Kind {
    Tetrahedron(usize), // Tetrahedron.
    TriPrismZ([usize; 2]), // Triangular prism in z direction.
}


pub struct IsoElem3<const MAP_N: usize> {
    // 3D isoparametric element with C0 continuity.
    pub kind: IsoElem3Kind, // order_xy, order_z
    pub enodes_n: usize, // Number of nodes in an element.
    pub edofs_n: usize, // Degrees of freedom in an element.
    pub quad_n: usize, // Number of integration points.
    pub poly: Poly3, // 3 dimensional polynomial functions.
    pub poly_map: Poly3, // 3 dimensional polynomial functions for isoparametric mapping.

    pub nodes_coord: Vec<[f64; 3]>, // (enodes_n,)
    pub quad_coord: Vec<[f64; 3]>, // (quad_n,)
    pub quad_weights: Vec<f64>, // (quad_n,)

    pub map_co: Arr2<f64>, // (6, 6)
    pub f_co: Arr2<f64>, // (edof, edof)
    pub f: Arr2<f64>, // (quad_n, edof)
    pub fu: Arr2<f64>, // (quad_n, edof)
    pub fv: Arr2<f64>, // (quad_n, edof)
    pub fw: Arr2<f64>, // (quad_n, edof)
}

pub type TetElem = IsoElem3<TET_MAP_N>; // Tetrahedron.
pub type TprElem = IsoElem3<TPR_MAP_N>; // Triangular prism in z direction.

impl<const MAP_N: usize> IsoElem3<MAP_N>
{
    #[inline]
    fn new_general( kind: IsoElem3Kind, quad_points_path: &str, quad_weights_path: &str ) -> Self {
        let (enodes_n, edofs_n): (usize, usize);
        let mut nodes_coord = Vec::<[f64; 3]>::new();
        let poly: Poly3;
        let poly_map: Poly3;
        let mut poly_fit_is_done: bool = false;
        let mut map_co = Arr2::<f64>::new(MAP_N, MAP_N);
        let mut f_co = Arr2::<f64>::new_empty();

        match kind {
            IsoElem3Kind::Tetrahedron(order) => {
                nodes_coord.extend([
                    [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], 
                ]);
                poly_map = Poly3::new(Poly3Kind::Tetrahedron(1));
                poly_map.fit(&nodes_coord, &mut map_co);
                poly = Poly3::new(Poly3Kind::Tetrahedron(order));

                match order {
                    1 => {
                        [enodes_n, edofs_n] = [4, 4];
                    },
                    2 => {
                        [enodes_n, edofs_n] = [10, 10];
                        nodes_coord.extend([ 
                            [0.5, 0., 0.], [0.5, 0.5, 0.], [0., 0.5, 0.], 
                            [0., 0., 0.5], [0.5, 0., 0.5], [0., 0.5, 0.5],
                        ]);
                    },
                    _ => {panic!("Tetrahedron element does not support order = {order}.");},
                };
            },

            IsoElem3Kind::TriPrismZ([order_xy, order_z]) => {
                nodes_coord.extend([
                    [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], 
                    [0., 0., 1.], [1., 0., 1.], [0., 1., 1.], 
                ]);
                poly_map = Poly3::new(Poly3Kind::TriPrismZ([1, 1]));
                poly_map.fit(&nodes_coord, &mut map_co);
                poly = Poly3::new(Poly3Kind::TriPrismZ([order_xy, order_z]));

                match order_xy {
                    1 => {
                        match order_z {
                            0 => {
                                [enodes_n, edofs_n] = [3, 3];
                                f_co.resize(edofs_n, edofs_n, 0.);
                                poly.fit(&nodes_coord[..3], &mut f_co);
                                poly_fit_is_done = true;
                            },
                            1 => {
                                [enodes_n, edofs_n] = [6, 6];
                            },
                            2 => {
                                [enodes_n, edofs_n] = [9, 9];
                                nodes_coord.extend([
                                    [0., 0., 0.5], [1., 0., 0.5], [0., 1., 0.5]
                                ]);
                            },
                            _ => {panic!("TriPrismZ element does not support order_xy = {order_xy} and order_z = {order_z}.");},
                        }
                    },
                    2 => {
                        nodes_coord.extend([ 
                            [0.5, 0., 0.], [0.5, 0.5, 0.], [0., 0.5, 0.], 
                            [0.5, 0., 1.], [0.5, 0.5, 1.], [0., 0.5, 1.] 
                        ]);

                        match order_z {
                            0 => {
                                [enodes_n, edofs_n] = [6, 6];
                                f_co.resize(edofs_n, edofs_n, 0.);
                                let mut coord = Vec::<[f64; 3]>::with_capacity(6);
                                coord.copy_from_slice(&nodes_coord[0..3]);
                                coord.copy_from_slice(&nodes_coord[6..9]);
                                poly.fit(&coord, &mut f_co);
                                poly_fit_is_done = true;
                            },
                            1 => {
                                [enodes_n, edofs_n] = [12, 12];
                            },
                            2 => {
                                [enodes_n, edofs_n] = [18, 18];
                                nodes_coord.extend([ 
                                    [0., 0., 0.5], [1., 0., 0.5], [0., 1., 0.5], 
                                    [0.5, 0., 0.5], [0.5, 0.5, 0.5], [0., 0.5, 0.5]
                                ]);
                            },
                            _ => {panic!("TriPrismZ element does not support order_z = {order_z}.");},
                        }
                    },
                    _ => {panic!("TriPrismZ element does not support order order_xy = {order_xy}.");},
                };
            },
        };
        nodes_coord.shrink_to_fit();
        assert_eq!(nodes_coord.len(), enodes_n);
        assert_eq!(edofs_n, poly.n);

        let quad_coord: Vec<[f64; 3]> = read_npy_vec(quad_points_path);
        let quad_weights: Vec<f64> = read_npy_vec(quad_weights_path);
        let quad_n = quad_coord.len();
        assert_eq!(quad_n, quad_weights.len());
        
        if !poly_fit_is_done {
            f_co.resize(edofs_n, edofs_n, 0.);
            poly.fit(&nodes_coord, &mut f_co);
        }
        
        let mut f: Arr2<f64> = Arr2::new(quad_n, edofs_n);
        let mut fu: Arr2<f64> = Arr2::new(quad_n, edofs_n);
        let mut fv: Arr2<f64> = Arr2::new(quad_n, edofs_n);
        let mut fw: Arr2<f64> = Arr2::new(quad_n, edofs_n);

        for i in 0..edofs_n {
            poly.f (&quad_coord, f_co.col(i).sl(), f .col_mut(i).slm());
            poly.fx(&quad_coord, f_co.col(i).sl(), fu.col_mut(i).slm());
            poly.fy(&quad_coord, f_co.col(i).sl(), fv.col_mut(i).slm());
            poly.fz(&quad_coord, f_co.col(i).sl(), fw.col_mut(i).slm());
        }     

        Self { kind, enodes_n, edofs_n, quad_n, poly, poly_map,
            nodes_coord, quad_coord, quad_weights, map_co, f_co, f, fu, fv, fw }
    }

    #[inline]
    pub fn compute_mapping( &self,
        corner_x: &[f64; MAP_N], corner_y: &[f64; MAP_N], corner_z: &[f64; MAP_N],
        x_co: &mut [f64; MAP_N], y_co: &mut [f64; MAP_N], z_co: &mut [f64; MAP_N],
    ) {
        dgemv(1., &self.map_co, corner_x, 1., x_co, NO_TRANS);
        dgemv(1., &self.map_co, corner_y, 1., y_co, NO_TRANS);
        dgemv(1., &self.map_co, corner_z, 1., z_co, NO_TRANS);
    }

    #[inline]
    pub fn f_at_points( &self, i: usize, uvw: &[[f64; 3]], f: &mut [f64] ) {
        // i: index of element dof.
        self.poly.f(uvw, self.f_co.col(i).sl(), f);
    }

    #[inline]
    pub fn f_at_point( &self, i: usize, uvw: &[f64; 3] ) -> f64 {
        // i: index of element dof.
        self.poly.f_single(uvw, self.f_co.col(i).sl())
    }

    #[inline]
    pub fn compute_gradient<'a, I: ScalarOrArray<'a, JacInv3>>( 
        &self, i: usize, // index of basis function
        fx: &mut [f64], fy: &mut [f64], fz: &mut [f64], // (quad_n,)
        jac_inv: &'a I, // (quad_n,), inverse jacobian matrix.
    ) {
        for elem!(jac_inv_, fx_, fy_, fz_, fu_, fv_, fw_) in mzip!(
            jac_inv.it_scalar_or_array(), fx.iter_mut(), fy.iter_mut(), fz.iter_mut(), 
            self.fu.col(i).it(), self.fv.col(i).it(), self.fw.col(i).it()
        ) {
            [*fx_, *fy_, *fz_] = Self::_compute_gradient(&jac_inv_.data, fu_, fv_, fw_);
        }
    }

    #[inline]
    fn _compute_gradient( a: &[f64; 9], fu: &f64, fv: &f64, fw: &f64 ) -> [f64; 3] {
        [
            a[0]*fu + a[1]*fv + a[2]*fw,
            a[3]*fu + a[4]*fv + a[5]*fw,
            a[6]*fu + a[7]*fv + a[8]*fw,
        ]
    }

    #[inline]
    pub fn quad<'a, I: ScalarOrArray<'a, f64>>( &self, 
        f: &[f64],  // (quad_n,)
        jac_det: &'a I, // (quad_n,)
    ) -> f64 {
        let mut s: f64 = 0.;
        for elem!(f_, w_, det_) in mzip!(
            f.iter(), self.quad_weights.iter(), jac_det.it_scalar_or_array()
        ) {
            s += f_ * w_ * det_;
        }
        s
    }

    #[inline]
    pub fn quad2<'a, I: ScalarOrArray<'a, f64>>( &self, 
        f1: &[f64], f2: &[f64], // (quad_n,)
        jac_det: &'a I, // (quad_n,)
    ) -> f64 {
        let mut s: f64 = 0.;
        for elem!(f1_, f2_, w_, det_) in mzip!(
            f1.iter(), f2.iter(), self.quad_weights.iter(), jac_det.it_scalar_or_array()
        ) {
            s += f1_ * f2_ * w_ * det_;
        }
        s
    }

    #[inline]
    pub fn quad3<'a, I: ScalarOrArray<'a, f64>>( &self, 
        f1: &[f64], f2: &[f64], f3: &[f64], // (quad_n,)
        jac_det: &'a I, // (quad_n,)
    ) -> f64 {
        let mut s: f64 = 0.;
        for elem!(f1_, f2_, f3_, w_, det_) in mzip!(
            f1.iter(), f2.iter(), f3.iter(), self.quad_weights.iter(), jac_det.it_scalar_or_array()
        ) {
            s += f1_ * f2_ * f3_ * w_ * det_;
        }
        s
    }
}

impl TetElem
{
    #[inline]
    pub fn new( order: usize, quad_points_path: &str, quad_weights_path: &str ) -> Self {
        let kind = IsoElem3Kind::Tetrahedron(order);
        Self::new_general(kind, quad_points_path, quad_weights_path)
    }

    #[inline]
    pub fn compute_jacobian( &self,
        x_co: &[f64; 4], y_co: &[f64; 4], z_co: &[f64; 4], // mapping (u, v, w) to (x, y, z)
        jac: &mut Jac3, jac_det: &mut f64, jac_inv: &mut JacInv3, // (quad_n,)
    ) {
        jac.data[0] = x_co[1]; // du
        jac.data[1] = y_co[1]; // du
        jac.data[2] = z_co[1]; // du

        jac.data[3] = x_co[2]; // dv
        jac.data[4] = y_co[2]; // dv
        jac.data[5] = z_co[2]; // dv

        jac.data[6] = x_co[3]; // dw
        jac.data[7] = y_co[3]; // dw
        jac.data[8] = z_co[3]; // dw

        *jac_det = mat_det_inv_3(&jac.data, &mut jac_inv.data);
    }
}

impl TprElem
{
    #[inline]
    pub fn new( order: [usize; 2], quad_points_path: &str, quad_weights_path: &str ) -> Self {
        let kind = IsoElem3Kind::TriPrismZ(order);
        Self::new_general(kind, quad_points_path, quad_weights_path)
    }

    #[inline]
    pub fn compute_jacobian( &self,
        x_co: &[f64; 6], y_co: &[f64; 6], z_co: &[f64; 6], // [q0, q1, q2]
        jac: &mut [Jac3], jac_det: &mut [f64], jac_inv: &mut [JacInv3], // (quad_n,)
    ) {
        for elem!(jac_, det_, inv_, uvw) in mzip!(
            jac.iter_mut(), jac_det.iter_mut(), jac_inv.iter_mut(), self.quad_coord.iter()
        ) {
            jac_.data[0] = self.poly_map.fx_single(uvw, x_co);
            jac_.data[1] = self.poly_map.fx_single(uvw, y_co);
            jac_.data[2] = self.poly_map.fx_single(uvw, z_co);

            jac_.data[3] = self.poly_map.fy_single(uvw, x_co);
            jac_.data[4] = self.poly_map.fy_single(uvw, y_co);
            jac_.data[5] = self.poly_map.fy_single(uvw, z_co);

            jac_.data[6] = self.poly_map.fz_single(uvw, x_co);
            jac_.data[7] = self.poly_map.fz_single(uvw, y_co);
            jac_.data[8] = self.poly_map.fz_single(uvw, z_co);

            *det_ = mat_det_inv_3(&jac_.data, &mut inv_.data);
        }
    }
}


#[inline]
fn mesh3_get_edofs(  
    i_elem: usize, edofs: &mut [[usize; 3]], 
    elems: &Vec<Elem3>, nodes: &Vec<Node3>, dofs: &Vec<Dof>, 
) -> usize {
    // edofs: (edofs_max,); stores (kind, i_dof_global, i_dof_local); return edofs_n
    let mut edofs_itm = edofs.iter_mut();
    let mut edofs_n: usize = 0;
    let mut i_dof_local_u: usize = 0;
    let mut i_dof_local_v: usize = 0;
    let mut i_dof_local_w: usize = 0;

    for i_node_global in elems[i_elem].nodes.iter() {
        let node = &nodes[*i_node_global];
        edofs_n += node.dofs.len();

        for i_dof_global in node.dofs.iter() {
            let [kind_, i_global_, i_local_] = edofs_itm.next().unwrap();
            let dof_kind = dofs[*i_dof_global].kind;

            *kind_ = dof_kind;
            *i_global_ = *i_dof_global;
            *i_local_ = match dof_kind {
                DOF_U => i_dof_local_u,
                DOF_V => i_dof_local_v,
                DOF_W => i_dof_local_w,
                _ => panic!("Not support dof_kind = {dof_kind}."),
            };
        }

        for dof_kind in node.dofs_kinds.iter() {
            match *dof_kind {
                DOF_U => {i_dof_local_u += 1;},
                DOF_V => {i_dof_local_v += 1;},
                DOF_W => {i_dof_local_w += 1;},
                _ => panic!("Not support dof_kind = {dof_kind}."),
            }
        };
    }
    edofs_n
}

#[inline]
fn mesh3_get_enodes<const MAP_N: usize, J1, J2, J3>( 
    i_elem: usize, elems: &Vec<Elem3>, nodes: &Vec<Node3>, map: &mut Elem3MapBuf<MAP_N, J1, J2, J3>,
) {
    let mut it_x = map.corner_x.iter_mut();
    let mut it_y = map.corner_y.iter_mut();
    let mut it_z = map.corner_z.iter_mut();
    let mut it_coord = map.en_coord.iter_mut();

    for i_node in elems[i_elem].nodes.iter() {
        let [x, y, z] = nodes[*i_node].coord;
        if let Some(x_) = it_x.next() {
            *x_ = x;
        }
        if let Some(y_) = it_y.next() {
            *y_ = y;
        }
        if let Some(z_) = it_z.next() {
            *z_ = z;
        }
        if let Some(coord_) = it_coord.next() {
            *coord_ = [x, y, z];
        }
    }
}