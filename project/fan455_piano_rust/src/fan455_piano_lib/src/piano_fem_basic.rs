//use fan455_arrf64::*;
//use fan455_math_array::*;
use fan455_util::*;
use fan455_util_macro::*;


pub const FREE_NODE: isize = 1;
pub const BOUNDARY_NODE: isize = -1;
// -100 to -115 are for RIBS_BEG, -200 to -215 are for RIBS_END
// 100 to 115 are for RIBS_MID_1, 200 to 215 are for RIBS_MID_2

pub const CORNER_NODE: usize = 0;
pub const EDGE_NODE: usize = 1;
pub const SURFACE_NODE: usize = 2;
pub const INNER_NODE: usize = 3;

pub const DOF_U: usize = 0; // displacement/velocity/acceleration in x direction
pub const DOF_V: usize = 1; // displacement/velocity/acceleration in y direction
pub const DOF_W: usize = 2; // displacement/velocity/acceleration in z direction


#[derive(Clone)]
pub struct Jac3 {
    pub data: [f64; 9],
}

#[derive(Clone)]
pub struct JacInv3 {
    pub data: [f64; 9],
}

impl ScalarTrait for Jac3 {}
impl ScalarTrait for JacInv3 {}

impl Jac3 
{
    #[inline]
    pub fn new() -> Self {
        Self {data: [0.; 9]}
    }
}

impl JacInv3 
{
    #[inline]
    pub fn new() -> Self {
        Self {data: [0.; 9]}
    }
}

#[derive(Clone, StaBinStruct)]
pub struct Dof {
    pub kind: usize,
}

#[derive(Clone, DynBinStruct)]
pub struct Node1D {
    pub kind: usize, // corner/inner
    pub boundary: isize, // positive: free node; negative: boundary node
    pub coord: f64, // x
    #[dyn_field] pub dofs_kinds: Vec<usize>, // Include boundary nodes!
    #[dyn_field] pub dofs: Vec<usize>,
}

#[derive(Clone, DynBinStruct)]
pub struct Node3 {
    pub kind: usize, // corner/edge/surface/inner
    pub boundary: isize, // positive: free node; negative: boundary node
    pub coord: [f64; 3], // x, y, z
    #[dyn_field] pub dofs_kinds: Vec<usize>, // Include boundary nodes!
    #[dyn_field] pub dofs: Vec<usize>,
}

#[derive(Clone, DynBinStruct)]
pub struct Elem1 {
    pub kind: usize, // 0: tetrahedron; 1: triangular prism
    pub group: usize,
    #[dyn_field] pub nodes: Vec<usize>,
}

#[derive(Clone, DynBinStruct)]
pub struct Elem3 {
    pub kind: usize, // 0: tetrahedron; 1: triangular prism
    pub group: usize,
    #[dyn_field] pub nodes: Vec<usize>,
}

impl Dof 
{
    #[inline]
    pub fn new( kind: usize ) -> Self {
        Self {kind}
    }
}

impl Node1D 
{
    #[inline]
    pub fn new( kind: usize, boundary: isize, coord: f64, dofs_kinds: Vec<usize>, dofs: Vec<usize> ) -> Self {
        Self {kind, boundary, coord, dofs_kinds, dofs}
    }
}

impl Elem1 
{
    #[inline]
    pub fn new( kind: usize, group: usize, nodes: Vec<usize> ) -> Self {
        Self {kind, group, nodes}
    }
}