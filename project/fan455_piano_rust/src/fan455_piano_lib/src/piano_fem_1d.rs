use fan455_arrf64::*;
use fan455_math_array::*;
use fan455_math_scalar::div_round;
use fan455_util::{elem, mzip, read_npy_vec};
use super::piano_io::*;
use super::piano_fem_basic::*;


pub struct PianoStringMesh
{
    // This struct stores all nodes and elements.
    pub order: [usize; 5],
    pub dofs_n: usize, // Total dof
    /*pub dofs: [usize; 5], 
    // 0: dof for u (longitudinal displacement)
    // 1: dof for v (horizontal displacement)
    // 2: dof for w (vertical displacement)
    // 3: dof for alpha (horizontal rotation)
    // 4: dof for beta (vertical rotation)*/
    pub edofs_max: usize,
    pub iso: (LiElem_1, LiElem_2, LiElem_2, LiElem_1, LiElem_1),

    pub enodes_n: usize,
    pub nodes_n: usize,
    pub elems_n: usize,
    pub quad_n: usize,

    pub dofs: Vec<Dof>,
    pub nodes: Vec<Node1D>,
    pub elems: Vec<Elem1>,

    pub hammer_i_elem: [usize; 2],
    pub hammer_i_node: usize,
    pub hammer_coord: f64,
}


impl PianoStringMesh
{
    pub const ORDER: [usize; 5] = [1, 2, 2, 1, 1];
    pub const EDOFS_MAX: usize = 12;
    pub const ENODES_N: usize = 3;

    pub const FREE_NODE: isize = 1;
    pub const AGRAFFE_NODE: isize = -2;
    pub const BRIDGE_NODE: isize = -3;

    pub const CORNER_NODE: usize = 0;
    pub const EDGE_NODE: usize = 1;
    pub const SURFACE_NODE: usize = 2;
    pub const INNER_NODE: usize = 3;

    pub const DOF_U: usize = 0; // Transverse displacement.
    pub const DOF_V : usize = 1; // Shear rotation in x direction.
    pub const DOF_W : usize = 2; // Shear rotation in y direction.
    pub const DOF_A: usize = 3; // In-plane shear displacement in x direction.
    pub const DOF_B: usize = 4; // In-plane shear displacement in y direction.


    #[inline]
    pub fn compute_dofs_n( nodes_n: usize ) -> usize {
        assert!(nodes_n % 2 == 1);
        let inner_n = nodes_n / 2;
        let corner_n = inner_n + 1;
        assert!(inner_n + corner_n == nodes_n);
        corner_n*5 - 6 + inner_n*2
    }

    #[inline]
    pub fn new( data: &PianoStringMeshParamsIn ) -> Self {
        let l = data.length;
        let dl = data.delta_length;
        let hammer_pos_rel = data.hammer_pos_rel;
        assert!(hammer_pos_rel > 0. && hammer_pos_rel < 1.);
        let l1 = l * hammer_pos_rel;
        let l2 = l - l1;
        let elems_n1 = div_round(l1, dl);
        let elems_n2 = div_round(l2, dl);
        let elems_n = elems_n1 + elems_n2;
        let dx1 = 0.5 * l1 / elems_n1 as f64;
        let dx2 = 0.5 * l2 / elems_n2 as f64;
        let nodes_n = elems_n*2 + 1;

        let enodes_n = Self::ENODES_N;
        let order = Self::ORDER;
        let dofs_n = Self::compute_dofs_n(nodes_n);
        let edofs_max = Self::EDOFS_MAX;

        let mut dofs = Vec::<Dof>::with_capacity(dofs_n);
        let mut nodes = Vec::<Node1D>::with_capacity(nodes_n);
        let mut elems = Vec::<Elem1>::with_capacity(elems_n);

        {
            // corner, boundary
            let dofs_corner_boundary = vec![Dof::new(Self::DOF_A), Dof::new(Self::DOF_B)];
            // corner, free
            let dofs_corner_free = vec![
                Dof::new(Self::DOF_U), Dof::new(Self::DOF_V), Dof::new(Self::DOF_W), 
                Dof::new(Self::DOF_A), Dof::new(Self::DOF_B)
            ];
            // inner, free
            let dofs_inner_free = vec![Dof::new(Self::DOF_V), Dof::new(Self::DOF_W)];

            // dofs kinds corner
            let kinds_corner = vec![Self::DOF_U, Self::DOF_V, Self::DOF_W, Self::DOF_A, Self::DOF_B];
            // dofs kinds inner
            let kinds_inner = vec![Self::DOF_V, Self::DOF_W];
    
            let [mut x1, mut x2]: [f64; 2] = [dx1, 2.*dx1];
            let mut dof_beg: usize = 0;
            let mut dof_end: usize;
            let mut node_beg: usize = 0;
            let mut node_end: usize;

            dofs.extend(dofs_corner_boundary.clone());
            dof_end = dofs.len();
            nodes.push(Node1D::new(Self::CORNER_NODE, Self::AGRAFFE_NODE, 0., kinds_corner.clone(), (dof_beg..dof_end).collect()));
            dof_beg = dof_end;

            dofs.extend(dofs_inner_free.clone());
            dof_end = dofs.len();
            nodes.push(Node1D::new(Self::INNER_NODE, Self::FREE_NODE, x1, kinds_inner.clone(), (dof_beg..dof_end).collect()));
            dof_beg = dof_end;

            dofs.extend(dofs_corner_free.clone());
            dof_end = dofs.len();
            nodes.push(Node1D::new(Self::CORNER_NODE, Self::FREE_NODE, x2, kinds_corner.clone(), (dof_beg..dof_end).collect()));
            dof_beg = dof_end;

            node_end = nodes.len();
            elems.push(Elem1::new(0, 0, (node_beg..node_end).collect()));
            node_beg = node_end;
    
            for i_elem in 1..elems_n-1 {
                if i_elem < elems_n1 {
                    x1 = x2 + dx1;
                    x2 = x1 + dx1;
                } else {
                    x1 = x2 + dx2;
                    x2 = x1 + dx2;
                }   
                dofs.extend(dofs_inner_free.clone());
                dof_end = dofs.len();
                nodes.push(Node1D::new(Self::INNER_NODE, Self::FREE_NODE, x1, kinds_inner.clone(), (dof_beg..dof_end).collect()));
                dof_beg = dof_end;
    
                dofs.extend(dofs_corner_free.clone());
                dof_end = dofs.len();
                nodes.push(Node1D::new(Self::CORNER_NODE, Self::FREE_NODE, x2, kinds_corner.clone(), (dof_beg..dof_end).collect()));
                dof_beg = dof_end;
    
                node_end = nodes.len();
                elems.push(Elem1::new(0, 0, (node_beg..node_end).collect()));
                node_beg = node_end;
            }
    
            x1 = x2 + dx2;
            x2 = x1 + dx2;
    
            dofs.extend(dofs_inner_free.clone());
            dof_end = dofs.len();
            nodes.push(Node1D::new(Self::INNER_NODE, Self::FREE_NODE, x1, kinds_inner.clone(), (dof_beg..dof_end).collect()));
            dof_beg = dof_end;

            dofs.extend(dofs_corner_boundary.clone());
            dof_end = dofs.len();
            nodes.push(Node1D::new(Self::CORNER_NODE, Self::BRIDGE_NODE, x2, kinds_corner.clone(), (dof_beg..dof_end).collect()));
            //dof_beg = dof_end;

            node_end = nodes.len();
            elems.push(Elem1::new(0, 0, (node_beg..node_end).collect()));
            //node_beg = node_end;
        }
        assert_eq!(dofs.len(), dofs_n);
        assert_eq!(nodes.len(), nodes_n);
        assert_eq!(elems.len(), elems_n);

        let hammer_i_elem = [elems_n1-1, elems_n1];
        let hammer_i_node = elems[hammer_i_elem[1]].nodes[0];
        let hammer_coord = nodes[hammer_i_node].coord;

        let iso = (
            LiElem_1::new(&data.quad_points_path, &data.quad_weights_path), 
            LiElem_2::new(&data.quad_points_path, &data.quad_weights_path), 
            LiElem_2::new(&data.quad_points_path, &data.quad_weights_path), 
            LiElem_1::new(&data.quad_points_path, &data.quad_weights_path), 
            LiElem_1::new(&data.quad_points_path, &data.quad_weights_path),
        );
        let quad_n = iso.0.quad_n;

        Self { order, dofs_n, edofs_max, iso, enodes_n, nodes_n, elems_n, quad_n, dofs, nodes, elems, hammer_i_elem, hammer_i_node, hammer_coord }
    }

    #[inline]
    pub fn get_edofs( &self, i_elem: usize, edofs: &mut [[usize; 3]] ) -> usize {
        // edofs: (edofs_max,); stores (kind, i_dof_global, i_dof_local)
        let mut edofs_itm = edofs.iter_mut();
        let mut edofs_n: usize = 0;
        let mut i_dof_local_u: usize = 0;
        let mut i_dof_local_v: usize = 0;
        let mut i_dof_local_w: usize = 0;
        let mut i_dof_local_a: usize = 0;
        let mut i_dof_local_b: usize = 0;

        for i_node_global in self.elems[i_elem].nodes.iter() {
            let node = &self.nodes[*i_node_global];
            edofs_n += node.dofs.len();

            for i_dof_global in node.dofs.iter() {
                let [kind_, i_global_, i_local_] = edofs_itm.next().unwrap();
                let dof_kind = self.dofs[*i_dof_global].kind;

                *kind_ = dof_kind;
                *i_global_ = *i_dof_global;
                *i_local_ = match dof_kind {
                    Self::DOF_U => i_dof_local_u,
                    Self::DOF_V => i_dof_local_v,
                    Self::DOF_W => i_dof_local_w,
                    Self::DOF_A => i_dof_local_a,
                    Self::DOF_B => i_dof_local_b,
                    _ => panic!("Not support dof_kind = {dof_kind}."),
                };
            }

            for dof_kind in node.dofs_kinds.iter() {
                match *dof_kind {
                    Self::DOF_U => {i_dof_local_u += 1;},
                    Self::DOF_V => {i_dof_local_v += 1;},
                    Self::DOF_W => {i_dof_local_w += 1;},
                    Self::DOF_A => {i_dof_local_a += 1;},
                    Self::DOF_B => {i_dof_local_b += 1;},
                    _ => panic!("Not support dof_kind = {dof_kind}."),
                }
            };
        }
        edofs_n
    }

    #[inline]
    pub fn create_edofs_buf( &self ) -> Vec<[usize; 3]> {
        // kind, i_global, i_local
        vec![[0, 0, 0]; self.edofs_max]
    }
}


pub struct PianoStringMeshBuf {
    pub map: LiElemMapBuf,
    pub edofs_n: usize,
    pub edofs: Vec<[usize; 3]>, // (edof_max,); stores (kind, i_dof_global, i_dof_local)

    pub b0: LiElemQuadBuf,
    pub b1: LiElemQuadBuf,
    pub b2: LiElemQuadBuf,
    pub b3: LiElemQuadBuf,
    pub b4: LiElemQuadBuf,
}

impl PianoStringMeshBuf
{
    #[inline]
    pub fn new( mesh: &PianoStringMesh ) -> Self {
        let map = LiElemMapBuf::new();
        let edofs_n = mesh.edofs_max;
        let edofs = mesh.create_edofs_buf();

        let b0 = LiElemQuadBuf::new(&mesh.iso.0);
        let b1 = LiElemQuadBuf::new(&mesh.iso.1);
        let b2 = LiElemQuadBuf::new(&mesh.iso.2);
        let b3 = LiElemQuadBuf::new(&mesh.iso.3);
        let b4 = LiElemQuadBuf::new(&mesh.iso.4);

        Self {map, edofs, edofs_n, b0, b1, b2, b3, b4}
    }

    #[inline]
    pub fn process_elem( &mut self, i_elem: usize, mesh: &PianoStringMesh ) {
        let mut it_coord = self.map.en_coord.iter_mut();
        for i_node in mesh.elems[i_elem].nodes.iter() {
            if let Some(coord_) = it_coord.next() {
                *coord_ = mesh.nodes[*i_node].coord;
            }
        }
        self.edofs_n = mesh.get_edofs(i_elem, &mut self.edofs);
    }

    #[inline]
    pub fn iter_edofs( &self ) -> std::slice::Iter<[usize; 3]> {
        self.edofs[..self.edofs_n].iter()
    }

    #[inline]
    pub fn compute_mapping( &mut self ) {
        self.map.compute_mapping();
    }

    #[inline]
    pub fn compute_inv_mapping( &mut self ) {
        self.map.compute_inv_mapping();
    }

    #[inline]
    pub fn compute_gradient( &mut self, mesh: &PianoStringMesh ) {
        self.map.compute_gradient(&mesh.iso.0, &mut self.b0);
        self.map.compute_gradient(&mesh.iso.1, &mut self.b1);
        self.map.compute_gradient(&mesh.iso.2, &mut self.b2);
        self.map.compute_gradient(&mesh.iso.3, &mut self.b3);
        self.map.compute_gradient(&mesh.iso.4, &mut self.b4);
    }
}


pub struct LiElemMapBuf {
    // The below data is buffer, updated for each element, so storage will be efficient.
    pub en_coord: [f64; PianoStringMesh::ENODES_N], // (n_nodes_per_elem,)
    pub x_co: [f64; 2], // coefficients of isoparametric transform from u to x.
    pub u_co: [f64; 2], // coefficients of isoparametric transform from x to u.
}

impl LiElemMapBuf
{
    #[inline]
    pub fn new() -> Self {
        println!("Initializing TrElemMapBuf (triangle element mapping buffer)...");
        let en_coord = [0.; PianoStringMesh::ENODES_N];
        let x_co: [f64; 2] = [0.; 2];
        let u_co: [f64; 2] = [0.; 2];

        println!("Finished.\n");
        Self { en_coord, x_co, u_co }
    }

    #[inline]
    pub fn compute_mapping( &mut self ) {
        self.x_co = LiElem_1::compute_mapping(self.en_coord[0], self.en_coord[1]);
    }

    #[inline]
    pub fn compute_inv_mapping( &mut self ) {
        self.u_co = LiElem_1::compute_mapping(self.en_coord[0], self.en_coord[1]);
    }

    #[inline]
    pub fn compute_gradient<const ORDER: usize, const N: usize>( 
        &mut self, iso: &LiElem<ORDER, N>, q: &mut LiElemQuadBuf 
    ) where LiElem<ORDER, N>: LiElemTrait<ORDER, N>, PolyStatic<ORDER>: PolyStaticCall<N>,
    {
        for i in 0..LiElem::<ORDER, N>::EDOF {
            iso.compute_gradient(i, self.x_co[0], q.fx.col_mut(i).slm());
        }
    }
}


pub struct LiElemQuadBuf {
    // The below data is buffer, updated for each element, so storage will be efficient.
    pub f: Arr2<f64>, // (quad_n, edof)
    pub fx: Arr2<f64>, // (quad_n, edof)
}

impl LiElemQuadBuf
{
    #[inline]
    pub fn new<const ORDER: usize, const N: usize>( 
        iso: &LiElem<ORDER, N> 
    ) -> Self where LiElem<ORDER, N>: LiElemTrait<ORDER, N>, PolyStatic<ORDER>: PolyStaticCall<N>,
    {
        println!("Initializing mesh buffer...");
        let f = iso.f.clone();
        let fx = iso.fu.clone();

        println!("Finished.\n");
        Self { f, fx }
    }
}


pub trait LiElemTrait<const ORDER: usize, const N: usize>
where PolyStatic<ORDER>: PolyStaticCall<N>,
{
    // An isoparametric line element with C0 continuity.
    const EDOF: usize = N;
    const ENODES_N: usize = N;
    const NODES_U: [f64; N]; // vertices u = -1 and u = 1 come first.
    const CO: [[f64; N]; N];

    #[inline]
    fn get_f( i: usize, quad_u: &[f64], f: &mut [f64] ) {
        PolyStatic::f(quad_u, &Self::CO[i], f);
    }

    #[inline]
    fn get_fu( i: usize, quad_u: &[f64], fu: &mut [f64] ) {
        PolyStatic::fx(quad_u, &Self::CO[i], fu);
    }

    #[inline]
    fn compute_mapping( x0: f64, x1: f64 ) -> [f64; 2] {
        // return x_co: [a, b], x = a*u + b
        [(x1-x0)/2., (x0+x1)/2.]
    }

    #[inline]
    fn compute_inv_mapping( x0: f64, x1: f64 ) -> [f64; 2] {
        // return u_co: [c, d], u = c*x + b
        [2./(x1-x0), (x0+x1)/(x0-x1)]
    }
}


pub struct LiElem<const ORDER: usize, const N: usize> {
    quad_n: usize,
    _quad_u: Vec<f64>,
    quad_weights: Vec<f64>,
    f: Arr2<f64>,
    fu: Arr2<f64>,
}

impl<const ORDER: usize, const N: usize> LiElem<ORDER, N>
where Self: LiElemTrait<ORDER, N>, PolyStatic<ORDER>: PolyStaticCall<N>,
{
    pub const ENODES_N: usize = N;

    #[inline]
    pub fn new( quad_points_path: &str, quad_weights_path: &str ) -> Self {
        let quad_u: Vec<f64> = read_npy_vec(quad_points_path);
        let quad_weights: Vec<f64> = read_npy_vec(quad_weights_path);
        let quad_n = quad_u.len();
        assert_eq!(quad_n, quad_weights.len());

        let mut f = Arr2::<f64>::new(quad_n, N);
        let mut fu = Arr2::<f64>::new(quad_n, N);

        for i in 0..N {
            Self::get_f(i, &quad_u, f.col_mut(i).slm());
            Self::get_fu(i, &quad_u, fu.col_mut(i).slm());
        }

        Self {quad_n, _quad_u: quad_u, quad_weights, f, fu}
    }

    #[inline]
    pub fn compute_gradient( &self, i: usize, x_co_0: f64, fx: &mut [f64] ) {
        for elem!(fu_, fx_) in mzip!(self.fu.col(i).it(), fx.iter_mut()) {
            *fx_ = fu_ / x_co_0;
        }
    }

    #[inline]
    pub fn quad( 
        &self, 
        f: &[f64],  // (quad_n,)
        x_co_0: f64, // scalar
    ) -> f64 {
        let mut s: f64 = 0.;
        for elem!(f_, w_) in mzip!(f.iter(), self.quad_weights.iter()) {
            s += f_ * w_ * x_co_0;
        }
        s
    }

    #[inline]
    pub fn quad2( 
        &self, 
        f1: &[f64],  // (quad_n,)
        f2: &[f64],  // (quad_n,)
        x_co_0: f64, // scalar
    ) -> f64 {
        let mut s: f64 = 0.;
        for elem!(f1_, f2_, w_) in mzip!(f1.iter(), f2.iter(), self.quad_weights.iter()) {
            s += f1_ * f2_ * w_ * x_co_0;
        }
        s
    }

    #[inline]
    pub fn quad3( 
        &self, 
        f1: &[f64],  // (quad_n,)
        f2: &[f64],  // (quad_n,)
        f3: &[f64],  // (quad_n,)
        x_co_0: f64, // scalar
    ) -> f64 {
        let mut s: f64 = 0.;
        for elem!(f1_, f2_, f3_, w_) in mzip!(f1.iter(), f2.iter(), f3.iter(), self.quad_weights.iter()) {
            s += f1_ * f2_ * f3_ * w_ * x_co_0;
        }
        s
    }
}


#[allow(non_camel_case_types)] pub type LiElem_1 = LiElem<1, 2>;
#[allow(non_camel_case_types)] pub type LiElem_2 = LiElem<2, 3>;

impl LiElemTrait<1, 2> for LiElem_1
{
    const NODES_U: [f64; 2] = [-1., 1.];
    const CO: [[f64; 2]; 2] = [[1./2., -1./2.], [1./2., 1./2.]];
}

impl LiElemTrait<2, 3> for LiElem_2
{
    const NODES_U: [f64; 3] = [-1., 1., 0.];
    const CO: [[f64; 3]; 3] = [[0., -1./2., 1./2.], [0., 1./2., 1./2.], [1., 0., -1.]];
}


