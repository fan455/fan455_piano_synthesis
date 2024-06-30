use fan455_arrf64::*;
use fan455_math_scalar::*;
use fan455_util::*;
use fan455_util_macro::*;
use std::collections::HashMap;
use std::cmp::Ordering;


const CORNER_NODE: u8 = 0;
const EDGE_NODE: u8 = 1;
const INNER_NODE: u8 = 2;


#[derive(Default, Clone, Copy, PartialEq, Eq, Hash)]
struct LineKey {
    beg: usize, // corner node tag
    end: usize, // corner node tag
}

#[derive(Default, Clone, Copy)]
struct LineVal {
    mid1: usize, // edge node tag
    mid2: usize, // edge node tag
}


#[derive(Default, Clone, Copy)]
struct NodeVal {
    is_on_boundary: bool,
    coord: [fsize; 2],
    idx: usize,
}

#[derive(Default, Clone, Eq, PartialEq, Hash, Copy)]
struct NodeKey {
    node_type: u8, // 0: corner; 1: edge; 2: inner
    tag: usize, // tag in its node type.
}

#[derive(Default, Clone, Copy)]
struct Node {
    node_type: u8,
    tag: usize,
    is_on_boundary: bool,
    coord: [fsize; 2],
}

impl PartialEq for Node
{
    #[inline]
    fn eq( &self, other: &Self ) -> bool {
        self.node_type == other.node_type && self.tag == other.tag
    }
}

impl Eq for Node {}

impl PartialOrd for Node
{
    #[inline]
    fn partial_cmp( &self, other: &Self ) -> Option<Ordering> {
        match self.is_on_boundary == other.is_on_boundary {
            true => match self.node_type.cmp(&other.node_type) {
                Ordering::Equal => self.tag.partial_cmp(&other.tag),
                Ordering::Less => Some(Ordering::Less),
                Ordering::Greater => Some(Ordering::Greater),
            }
            false => match self.is_on_boundary {
                true => Some(Ordering::Greater),
                false => Some(Ordering::Less),
            }
        }
    }
}

impl Ord for Node
{
    #[inline]
    fn cmp( &self, other: &Self ) -> Ordering {
        match self.is_on_boundary == other.is_on_boundary {
            true => match self.node_type.cmp(&other.node_type) {
                Ordering::Equal => self.tag.cmp(&other.tag),
                Ordering::Less => Ordering::Less,
                Ordering::Greater => Ordering::Greater,
            }
            false => match self.is_on_boundary {
                true => Ordering::Greater,
                false => Ordering::Less,
            }
        }
    }
}


fn main() {

    // Read parameters from toml file.
    parse_cmd_args!();
    // inpute data path
    cmd_arg!(corner_nodes_tag_path, String);
    cmd_arg!(corner_nodes_boundary_status_path, String);
    cmd_arg!(corner_nodes_xy_path, String);
    cmd_arg!(elems_corner_nodes_tag_path, String);

    // output data path
    cmd_arg!(nodes_xy_path, String);
    cmd_arg!(elems_nodes_idx_path, String);
    unknown_cmd_args!();

    println!();

    // Read data.
    let corner_nodes_tag = read_npy_vec::<usize>(&corner_nodes_tag_path);
    let corner_nodes_boundary_status = read_npy_vec::<u8>(&corner_nodes_boundary_status_path);
    let corner_nodes_xy = read_npy_vec::<[fsize; 2]>(&corner_nodes_xy_path);
    let elems_corner_nodes_tag = Arr2::<usize>::read_npy(&elems_corner_nodes_tag_path); // (3, elems_n)

    let corner_nodes_n = corner_nodes_tag.len();
    assert_multi_eq!(corner_nodes_n, corner_nodes_boundary_status.len(), corner_nodes_xy.len());
    let elems_n = elems_corner_nodes_tag.ncol();
    assert_eq!(elems_corner_nodes_tag.nrow(), 3);

    let mut elems_nodes_key = Arr2::<NodeKey>::new(10, elems_n);

    // Create a hashmap of nodes.
    let mut nodes = HashMap::<NodeKey, NodeVal>::with_capacity(elems_n*10);
    for elem!(tag, is_on_boundary, coord) in mzip!(
        corner_nodes_tag.iter(), corner_nodes_boundary_status.iter(), corner_nodes_xy.iter()
    ) {
        nodes.insert(
            NodeKey{node_type: CORNER_NODE, tag: *tag},
            NodeVal{is_on_boundary: *is_on_boundary!=0, coord: *coord, idx: 0},
        );
    }
    assert_eq!(nodes.len(), corner_nodes_n);

    // Create a hashmap of lines.
    let mut lines = HashMap::<LineKey, LineVal>::with_capacity(elems_n*3);

    // Count tags for edge nodes and inner nodes
    let mut tag_edge_node: usize = 0;
    let mut tag_inner_node: usize = 0;

    // Use closure to do three times.
    let mut fn_tmp = |
        nodes: &mut HashMap<NodeKey, NodeVal>, key_0: &NodeKey, val_0: &NodeVal, key_1: &NodeKey, val_1: &NodeVal,
    | -> [NodeKey; 2] {
        
        let [node_key_3, node_key_4]: [NodeKey; 2];
        let i0 = key_0.tag;
        let i1 = key_1.tag;

        if let Some(val) = lines.get(&LineKey{beg: i0, end: i1}) {
            node_key_3 = NodeKey{node_type: EDGE_NODE, tag: val.mid1};
            node_key_4 = NodeKey{node_type: EDGE_NODE, tag: val.mid2};

        } else if let Some(val) = lines.get(&LineKey{beg: i1, end: i0}) {
            node_key_3 = NodeKey{node_type: EDGE_NODE, tag: val.mid2};
            node_key_4 = NodeKey{node_type: EDGE_NODE, tag: val.mid1}; 

        } else {
            let [x0, y0] = val_0.coord;
            let [x1, y1] = val_1.coord;
            let is_on_boundary = val_0.is_on_boundary && val_1.is_on_boundary;

            let [x3, y3] = point_between_two(x0, y0, x1, y1, 0.2763932023);
            let [x4, y4] = point_between_two(x0, y0, x1, y1, 0.7236067977);

            let i3 = tag_edge_node;
            let i4 = tag_edge_node + 1;
            tag_edge_node += 2;

            node_key_3 = NodeKey{node_type: EDGE_NODE, tag: i3};
            node_key_4 = NodeKey{node_type: EDGE_NODE, tag: i4};

            nodes.insert(node_key_3, NodeVal{is_on_boundary, coord: [x3, y3], idx: 0});
            nodes.insert(node_key_4, NodeVal{is_on_boundary, coord: [x4, y4], idx: 0});
            lines.insert(LineKey{beg: i0, end: i1}, LineVal{mid1: i3, mid2: i4}); 
        }
        [node_key_3, node_key_4]
    };


    for i_elem in 0..elems_n {

        let corner_tags_ = elems_corner_nodes_tag.col(i_elem);
        let i0 = *corner_tags_.idx(0);
        let i1 = *corner_tags_.idx(1);
        let i2 = *corner_tags_.idx(2);

        let key_0 = NodeKey{node_type: CORNER_NODE, tag: i0};
        let mut key_1 = NodeKey{node_type: CORNER_NODE, tag: i1};
        let mut key_2 = NodeKey{node_type: CORNER_NODE, tag: i2};

        let val_0 = *nodes.get(&key_0).unwrap();
        let mut val_1 = *nodes.get(&key_1).unwrap();
        let mut val_2 = *nodes.get(&key_2).unwrap();

        let [x0, y0] = val_0.coord;
        let [mut x1, mut y1] = val_1.coord;
        let [mut x2, mut y2] = val_2.coord;

        let swapped = ensure_three_points_counterclock(&x0, &y0, &mut x1, &mut y1, &mut x2, &mut y2);
        if swapped {
            std::mem::swap(&mut key_1, &mut key_2);
            std::mem::swap(&mut val_1, &mut val_2);
        }

        let [key_3, key_4] = fn_tmp(&mut nodes, &key_0, &val_0, &key_1, &val_1);
        let [key_5, key_6] = fn_tmp(&mut nodes, &key_1, &val_1, &key_2, &val_2);
        let [key_7, key_8] = fn_tmp(&mut nodes, &key_2, &val_2, &key_0, &val_0);
        
        let i9 = tag_inner_node;
        tag_inner_node += 1;
        let [x9, y9] = center_of_triangle(x0, y0, x1, y1, x2, y2);
        let key_9 = NodeKey{node_type: INNER_NODE, tag: i9};
        nodes.insert(key_9, NodeVal{is_on_boundary: false, coord: [x9, y9], idx: 0});

        elems_nodes_key.col_mut(i_elem).copy_sl(
            &[key_0, key_1, key_2, key_3, key_4, key_5, key_6, key_7, key_8, key_9]
        );
    }

    nodes.shrink_to_fit();
    lines.shrink_to_fit();
    let nodes_n = nodes.len();
    let inner_nodes_n = elems_n;
    let edge_nodes_n = nodes_n - corner_nodes_n - inner_nodes_n;

    let mut nodes_vec = Vec::<Node>::with_capacity(nodes_n);
    for (key, val) in nodes.iter() {
        nodes_vec.push(Node{node_type: key.node_type, tag: key.tag, is_on_boundary: val.is_on_boundary, coord: val.coord});
    };
    nodes_vec.sort();
    
    let mut free_nodes_n: usize = usize::MAX;
    'outer: for elem!(i, node) in mzip!(0..nodes_n, nodes_vec.iter()) {
        if node.is_on_boundary {
            free_nodes_n = i;
            break 'outer;
        }
    } 
    
    let mut nodes_xy = Vec::<[fsize; 2]>::with_capacity(nodes_n);
    for elem!(i, node) in mzip!(0..nodes_n, nodes_vec.iter()) {
        nodes.get_mut(&NodeKey{node_type: node.node_type, tag: node.tag}).unwrap().idx = i;
        nodes_xy.push(node.coord);
    }
    assert_eq!(nodes_n, nodes_xy.len());

    let mut elems_nodes_idx = Arr2::<usize>::new(10, elems_n);
    for elem!(key, idx) in mzip!(elems_nodes_key.it(), elems_nodes_idx.itm()) {
        *idx = nodes.get(key).unwrap().idx;
    }

    println!("elems_n = {elems_n}");
    println!("nodes_n = {nodes_n}");
    println!("corner_nodes_n = {corner_nodes_n}");
    println!("edge_nodes_n = {edge_nodes_n}");
    println!("inner_nodes_n = {inner_nodes_n}");
    println!("free_nodes_n = {free_nodes_n}");

    // Save npy files.
    write_npy_vec(&nodes_xy_path, &nodes_xy);
    elems_nodes_idx.write_npy(&elems_nodes_idx_path);

    println!("\nProgram ended successfully.");
}