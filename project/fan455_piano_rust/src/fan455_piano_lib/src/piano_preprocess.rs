use fan455_arrf64::*;
use fan455_math_scalar::*;
use fan455_util::*;
use core::panic;
use std::collections::HashMap;
use super::piano_io::*;
use super::piano_fem_basic::*;
use super::piano_fem_3d::*;


#[derive(Default, Clone, Copy, PartialEq, Eq, Hash)]
struct LineKey {
    beg: usize, // corner node tag
    end: usize, // corner node tag
}

#[derive(Default, Clone, Copy)]
struct LineVal {
    mid: usize, // edge node tag
}


#[derive(Default, Clone, Copy)]
struct NodeVal {
    boundary: isize, // positive: free node; negative: boundary node
    coord: [f64; 3],
    idx: usize,
}

#[derive(Default, Clone, Eq, PartialEq, Hash, Copy)]
struct NodeKey {
    kind: usize, // 0: corner; 1: edge; 2: surface; 3: inner
    tag: usize, // tag in its node type.
}

struct PreElem3 {
    kind: usize,
    group: usize,
    nodes: Vec<NodeKey>,
}


pub fn preprocess_soundboard_mesh( params: &PianoParamsIn ) {
    // Convert 2d soundboard mesh (triangle) to 3d (triangle prism).
    println!("Preprocessing piano soundboard mesh data...");
    let args = &params.mesh.sb;
    let nodes_per_elem: usize = 12;
    let edge_nodes_per_elem: usize = 6;
    let max_dof_per_node: usize = 3;

    // Read input data.
    let points_tag = read_npy_vec::<usize>(&args.points_tag_path);
    let points_kind = read_npy_vec::<isize>(&args.points_kind_path);
    let points_coord = read_npy_vec::<[f64; 2]>(&args.points_coord_path);
    let elems_points_tag = Arr2::<usize>::read_npy(&args.points_tag_path); // (3, elems_n)
    let elems_groups = read_npy_vec::<usize>(&args.elems_groups_path);

    let points_n = points_tag.len();
    assert_multi_eq!(points_n, points_kind.len(), points_coord.len());
    let elems_n1 = elems_points_tag.ncol();
    assert_eq!(elems_points_tag.nrow(), 3);
    let elems_capacity = 2*elems_n1;

    let ribs_beg_map: HashMap<isize, usize> = convert_hashmap_keys(&args.ribs_beg_map);
    let ribs_end_map: HashMap<isize, usize> = convert_hashmap_keys(&args.ribs_end_map);
    let ribs_mid1_map: HashMap<isize, usize> = convert_hashmap_keys(&args.ribs_mid1_map);
    let ribs_mid2_map: HashMap<isize, usize> = convert_hashmap_keys(&args.ribs_mid2_map);
    let groups_ribs: HashMap<usize, [usize; 2]> = convert_hashmap_keys(&args.groups_ribs);
    let groups_bridges: HashMap<usize, usize> = convert_hashmap_keys(&args.groups_bridges);
    
    let ribs_n = params.ribs.num;
    let mut ribs_points: Vec::<[[f64; 2]; 4]> = vec![[[0.; 2]; 4]; ribs_n];

    let h_sb = params.sb.thickness;
    let [h_ribs_mid, h_ribs_side] = params.ribs.height;
    let h_bridges = params.bridges.height;
    let z_lo = -h_sb/2.;
    let z_up = h_sb/2.;
    let z_bridges = z_up + h_bridges;
    let z_ribs_mid = z_lo - h_ribs_mid;

    // Create a hashmap of nodes.
    let mut nodes = HashMap::<NodeKey, NodeVal>::with_capacity(elems_capacity*nodes_per_elem);

    // Compute the corner nodes.
    for elem!(tag, kind, coord) in mzip!(points_tag.iter(), points_kind.iter(), points_coord.iter()) {
        if let Some(i_rib) = ribs_beg_map.get(kind) {
            ribs_points[*i_rib][0] = *coord;

        } else if let Some(i_rib) = ribs_mid1_map.get(kind) {
            ribs_points[*i_rib][1] = *coord;

        } else if let Some(i_rib) = ribs_mid2_map.get(kind) {
            ribs_points[*i_rib][2] = *coord;

        } else if let Some(i_rib) = ribs_end_map.get(kind) {
            ribs_points[*i_rib][3] = *coord;
        }

        let [x_, y_] = *coord;
        nodes.insert(
            NodeKey{kind: CORNER_NODE_LO_1, tag: *tag},
            NodeVal{boundary: *kind, coord: [x_, y_, z_lo], idx: 0},
        );
        nodes.insert(
            NodeKey{kind: CORNER_NODE_UP_1, tag: *tag},
            NodeVal{boundary: *kind, coord: [x_, y_, z_up], idx: 0},
        );
    }

    // Compute the height decay coefficients of ribs.
    let mut ribs_decay = Vec::<[f64; 2]>::with_capacity(ribs_n);
    for points in ribs_points.iter() {
        let [x0, y0] = points[0];
        let [x1, y1] = points[1];
        let [x2, y2] = points[2];
        let [x3, y3] = points[3];
        let decay_beg = (h_ribs_mid / h_ribs_side).ln() / distance_2d(x0, y0, x1, y1);
        let decay_end = (h_ribs_mid / h_ribs_side).ln() / distance_2d(x2, y2, x3, y3);
        ribs_decay.push([decay_beg, decay_end]);
    }

    // Compute the corner nodes.
    let mut elems = Vec::<PreElem3>::with_capacity(elems_capacity);
    for elem!(i_elem, group) in mzip!(0..elems_n1, elems_groups.iter()) {
        // Compute the corner nodes for soundboard main part.
        let mut elem_0 = PreElem3 {kind: 1, group: *group, nodes: Vec::with_capacity(nodes_per_elem)};
        let tags_ = elems_points_tag.col(i_elem);

        let k0 = NodeKey{kind: CORNER_NODE_LO_1, tag: tags_[0]};
        let mut k1 = NodeKey{kind: CORNER_NODE_LO_1, tag: tags_[1]};
        let mut k2 = NodeKey{kind: CORNER_NODE_LO_1, tag: tags_[2]};

        let [x0, y0, _] = nodes.get(&k0).unwrap().coord;
        let [mut x1, mut y1, _] = nodes.get(&k1).unwrap().coord;
        let [mut x2, mut y2, _] = nodes.get(&k2).unwrap().coord;

        let swapped = ensure_three_points_counterclock(&x0, &y0, &mut x1, &mut y1, &mut x2, &mut y2);
        if swapped {
            std::mem::swap(&mut k1, &mut k2);
        }

        let k3 = NodeKey{kind: CORNER_NODE_UP_1, tag: k0.tag};
        let k4 = NodeKey{kind: CORNER_NODE_UP_1, tag: k1.tag};
        let k5 = NodeKey{kind: CORNER_NODE_UP_1, tag: k2.tag};

        elem_0.nodes.extend([k0, k1, k2, k3, k4, k5]);
        elems.push(elem_0);
        
        // Compute the corner nodes for bridges.
        if let Some(_) = groups_bridges.get(group) {
            let k3_ = NodeKey{kind: CORNER_NODE_UP_2, tag: k0.tag};
            let k4_ = NodeKey{kind: CORNER_NODE_UP_2, tag: k1.tag};
            let k5_ = NodeKey{kind: CORNER_NODE_UP_2, tag: k2.tag};
            
            let mut v3_ = *nodes.get(&k3).unwrap();
            let mut v4_ = *nodes.get(&k4).unwrap();
            let mut v5_ = *nodes.get(&k5).unwrap();

            v3_.coord[2] = z_bridges;
            v4_.coord[2] = z_bridges;
            v5_.coord[2] = z_bridges;

            nodes.insert(k3_, v3_);
            nodes.insert(k4_, v4_);
            nodes.insert(k5_, v5_);

            let mut elem_1 = PreElem3 {kind: 1, group: *group, nodes: Vec::with_capacity(nodes_per_elem)};
            elem_1.nodes.extend([k3, k4, k5, k3_, k4_, k5_]);
            elems.push(elem_1);
        }

        // Compute the corner nodes for ribs.
        if let Some([i_rib, i_part]) = groups_ribs.get(group) {
            let k3_ = NodeKey{kind: CORNER_NODE_LO_2, tag: k0.tag};
            let k4_ = NodeKey{kind: CORNER_NODE_LO_2, tag: k1.tag};
            let k5_ = NodeKey{kind: CORNER_NODE_LO_2, tag: k2.tag};
            
            let mut v3_ = *nodes.get(&k3).unwrap();
            let mut v4_ = *nodes.get(&k4).unwrap();
            let mut v5_ = *nodes.get(&k5).unwrap();

            if *i_part == RIBS_MID {
                v3_.coord[2] = z_ribs_mid;
                v4_.coord[2] = z_ribs_mid;
                v5_.coord[2] = z_ribs_mid;

            } else {
                let (decay, [x_mid, y_mid]) = match *i_part {
                    RIBS_BEG => (ribs_decay[*i_rib][0], ribs_points[*i_rib][1]),
                    RIBS_END => (ribs_decay[*i_rib][1], ribs_points[*i_rib][2]),
                    _ => panic!("Not support i_part = {i_part}"),
                };
                let [x3_, y3_, _] = v3_.coord;
                let [x4_, y4_, _] = v4_.coord;
                let [x5_, y5_, _] = v5_.coord;
                v3_.coord[2] = z_lo - compute_ribs_height(h_ribs_mid, decay, x_mid, y_mid, x3_, y3_);
                v4_.coord[2] = z_lo - compute_ribs_height(h_ribs_mid, decay, x_mid, y_mid, x4_, y4_);
                v5_.coord[2] = z_lo - compute_ribs_height(h_ribs_mid, decay, x_mid, y_mid, x5_, y5_);
            }
            nodes.insert(k3_, v3_);
            nodes.insert(k4_, v4_);
            nodes.insert(k5_, v5_);

            let mut elem_1 = PreElem3 {kind: 1, group: *group, nodes: Vec::with_capacity(nodes_per_elem)};
            elem_1.nodes.extend([k3_, k4_, k5_, k3, k4, k5]);
            elems.push(elem_1);
        }
    }
    elems.shrink_to_fit();
    let elems_n = elems.len();

    // Create a hashmap of lines.
    let mut lines = HashMap::<LineKey, LineVal>::with_capacity(elems_capacity*edge_nodes_per_elem);

    // Count tags for edge nodes and inner nodes
    let mut tag_edge_node: usize = 0;

    // Use closure to do multiple times.
    let mut fn_tmp = |
        nodes: &mut HashMap<NodeKey, NodeVal>, k0: &NodeKey, v0: &NodeVal, k1: &NodeKey, v1: &NodeVal,
    | -> NodeKey {
        
        let k3: NodeKey;
        let i0 = k0.tag;
        let i1 = k1.tag;

        if let Some(val) = lines.get(&LineKey{beg: i0, end: i1}) {
            k3 = NodeKey{kind: EDGE_NODE, tag: val.mid};

        } else if let Some(val) = lines.get(&LineKey{beg: i1, end: i0}) {
            k3 = NodeKey{kind: EDGE_NODE, tag: val.mid};

        } else {
            let [x0, y0, z0] = v0.coord;
            let [x1, y1, z1] = v1.coord;
            let boundary = { if v0.boundary < 0 && v1.boundary < 0 {-1_isize} else {1_isize} };

            let [x3, y3, z3] = mid_point_3d(x0, y0, z0, x1, y1, z1, 0.5);
            let i3 = tag_edge_node;
            tag_edge_node += 1;
            k3 = NodeKey{kind: EDGE_NODE, tag: i3};
            let v3 = NodeVal{boundary, coord: [x3, y3, z3], idx: 0};

            nodes.insert(k3, v3);
            lines.insert(LineKey{beg: i0, end: i1}, LineVal{mid: i3}); 
        }
        k3
    };

    // Compute the edge nodes.
    for elem in elems.iter_mut() {

        let k0 = elem.nodes[0];
        let k1 = elem.nodes[1];
        let k2 = elem.nodes[2];
        let k3 = elem.nodes[3];
        let k4 = elem.nodes[4];
        let k5 = elem.nodes[5];

        let v0 = *nodes.get(&k0).unwrap();
        let v1 = *nodes.get(&k1).unwrap();
        let v2 = *nodes.get(&k2).unwrap();
        let v3 = *nodes.get(&k3).unwrap();
        let v4 = *nodes.get(&k4).unwrap();
        let v5 = *nodes.get(&k5).unwrap();

        let k6 = fn_tmp(&mut nodes, &k0, &v0, &k1, &v1);
        let k7 = fn_tmp(&mut nodes, &k1, &v1, &k2, &v2);
        let k8 = fn_tmp(&mut nodes, &k2, &v2, &k0, &v0);
        let k9 = fn_tmp(&mut nodes, &k3, &v3, &k4, &v4);
        let k10 = fn_tmp(&mut nodes, &k4, &v4, &k5, &v5);
        let k11 = fn_tmp(&mut nodes, &k5, &v5, &k3, &v3);

        elem.nodes.extend([k6, k7, k8, k9, k10, k11]);
    }
    nodes.shrink_to_fit();
    lines.shrink_to_fit();
    let nodes_n = nodes.len();
    
    // Allocate the arrays of nodes, elements, DOFs.
    let mut dofs_vec = Vec::<Dof>::with_capacity(nodes_n*max_dof_per_node);
    let mut nodes_vec = Vec::<Node3>::with_capacity(nodes_n);
    let mut elems_vec = Vec::<Elem3>::with_capacity(elems_n);

    // Compute the array of nodes.
    for elem!(i, (node_key, node_val)) in mzip!(0..nodes_n, nodes.iter_mut()) {
        node_val.idx = i;
        let kind = match node_key.kind {
            CORNER_NODE | CORNER_NODE_LO_1 | CORNER_NODE_LO_2 | CORNER_NODE_UP_1 | CORNER_NODE_UP_2 => CORNER_NODE,
            EDGE_NODE => EDGE_NODE,
            _ => panic!("Not support node_key.kind = {}", node_key.kind),
        };
        nodes_vec.push(Node3 {
            kind, boundary: node_val.boundary, coord: node_val.coord, 
            dofs_kinds: Vec::with_capacity(max_dof_per_node),
            dofs: Vec::with_capacity(max_dof_per_node),
        });
    }
    
    // Compute the array of elements.
    for elem in elems.iter() {
        elems_vec.push(Elem3 {kind: elem.kind, group: elem.group, nodes: {
            let mut vec = Vec::<usize>::with_capacity(nodes_per_elem);
            for node_key in elem.nodes.iter() {
                vec.push(nodes.get(node_key).unwrap().idx);
            }
            vec
        }})
    }

    // Compute the array of DOFs.
    let mut i_dof: usize = 0;
    for node in nodes_vec.iter_mut() {
        let kind = node.kind;
        let boundary = node.boundary;

        match kind {
            CORNER_NODE => {
                node.dofs_kinds.extend([DOF_U, DOF_V, DOF_W]);
                match boundary {
                    FREE_NODE => {
                        dofs_vec.extend([Dof{kind: DOF_U}, Dof{kind: DOF_V}, Dof{kind: DOF_W}]);
                        node.dofs.extend([i_dof, i_dof+1, i_dof+2]);
                        i_dof += 3;
                    },
                    BOUNDARY_NODE => {},
                    _ => panic!("Not support boundary = {boundary}."),
                };
            },
            EDGE_NODE => {
                node.dofs_kinds.extend([DOF_W]);
                match boundary {
                    FREE_NODE => {
                        dofs_vec.push(Dof{kind: DOF_W});
                        node.dofs.push(i_dof);
                        i_dof += 1;
                    },
                    BOUNDARY_NODE => {},
                    _ => panic!("Not support boundary = {boundary}."),
                };
            }
            _ => panic!("Not support kind = {kind}."),
        };
    }
    dofs_vec.shrink_to_fit();
    let dofs_n = dofs_vec.len();

    // Save data.
    Dof::write_bin_vec(&args.dofs_path, &dofs_vec);
    Node3::write_bin_vec(&args.nodes_path, &nodes_vec);
    Elem3::write_bin_vec(&args.elems_path, &elems_vec);
    
    println!("\nelems_n = {elems_n}");
    println!("nodes_n = {nodes_n}");
    println!("dofs_n = {dofs_n}\n");

    println!("Finished.\n");

}


#[inline]
fn compute_ribs_height( h_ribs_mid: f64, decay: f64, x_mid: f64, y_mid: f64, x: f64, y: f64 ) -> f64 {
    h_ribs_mid * (-decay * distance_2d(x_mid, y_mid, x, y)).exp()
}
