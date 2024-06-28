use fan455_arrf64::*;
use fan455_math_scalar::*;
use fan455_util::*;
use fan455_util_macro::*;


const NULL_INDEX: usize = usize::MAX;


#[derive(Default, Clone)]
struct NodesLine {
    beg: usize,
    end: usize,
    mid1: usize,
    mid2: usize,
}


fn main() {

    // Read parameters from toml file.
    parse_cmd_args!();
    cmd_arg!(corner_nodes_xy_path, String);
    cmd_arg!(elems_corner_nodes_old_path, String);
    cmd_arg!(free_corner_nodes_n, usize);

    cmd_arg!(edge_nodes_xy_path, String);
    cmd_arg!(edge_nodes_on_boundary_path, String);
    cmd_arg!(inner_nodes_xy_path, String);
    cmd_arg!(elems_corner_nodes_path, String);
    cmd_arg!(elems_edge_nodes_path, String);
    unknown_cmd_args!();

    println!();

    let corner_nodes_xy: Vec<[fsize; 2]> = {
        let mut npy = NpyObject::<[fsize; 2]>::new_reader(&corner_nodes_xy_path);
        npy.read_header().unwrap();
        npy.read()
    }; // (nodes_n,), vertice nodes
    let elems_corner_nodes_old = Arr2::<usize>::read_npy(&elems_corner_nodes_old_path); // (3, elems_n)

    let corner_nodes_n = corner_nodes_xy.len();
    let elems_n = elems_corner_nodes_old.ncol();
    assert_eq!(elems_corner_nodes_old.nrow(), 3);

    let mut elems_corner_nodes = Arr2::<usize>::new(3, elems_n);
    let mut elems_edge_nodes = Arr2::<usize>::new(6, elems_n);

    let mut edge_nodes_xy: Vec<[fsize; 2]> = Vec::with_capacity(elems_n*6);
    let mut edge_nodes_on_boundary: Vec<u8> = Vec::with_capacity(elems_n*6);

    let mut inner_nodes_xy: Vec<[fsize; 2]> = Vec::with_capacity(elems_n);

    let mut lines: Vec<NodesLine> = Vec::with_capacity(elems_n*3);

    let mut i_edge_node: usize = 0;

    // Use closure to do three times.
    let mut fn_tmp = |
        i0: usize, i1: usize, x0: fsize, y0: fsize, x1: fsize, y1: fsize,
        i3: &mut usize, i4: &mut usize, 
    | {
        let mut line_0_idx = NULL_INDEX;
        let lines_len = lines.len();

        for elem!(i_line, line) in mzip!(0..lines_len, lines.iter_mut()) {
            if i0 == line.beg {
                if i1 == line.end {
                    if line_0_idx != NULL_INDEX {
                        panic!("Matched a line more than once, which should be impossible!");
                    }
                    line_0_idx = i_line;
                }
            } else if i0 == line.end {
                if i1 == line.beg {
                    if line_0_idx != NULL_INDEX {
                        panic!("Matched a line more than once, which should be impossible!");
                    }
                    line_0_idx = i_line;
                    std::mem::swap(&mut line.beg, &mut line.end);
                    std::mem::swap(&mut line.mid1, &mut line.mid2);
                }
            }
        }
        if line_0_idx == NULL_INDEX {
            let [x3, y3] = point_between_two(x0, y0, x1, y1, 0.2763932023);
            let [x4, y4] = point_between_two(x0, y0, x1, y1, 0.7236067977);
            edge_nodes_xy.push([x3, y3]);
            edge_nodes_xy.push([x4, y4]);

            let is_on_boundary: u8 = match (i0 >= free_corner_nodes_n) && (i1 >= free_corner_nodes_n) {
                true => 1,
                false => 0,
            };
            edge_nodes_on_boundary.push(is_on_boundary);
            edge_nodes_on_boundary.push(is_on_boundary);

            *i3 = i_edge_node;
            *i4 = i_edge_node + 1;
            lines.push( NodesLine {beg: i0, end: i1, mid1: *i3, mid2: *i4} );
            i_edge_node += 2;
            
        } else {
            let line = &lines[line_0_idx];
            *i3 = line.mid1;
            *i4 = line.mid2;
        }
    };


    for i_elem in 0..elems_n {

        let elems_corner_nodes_old_ = elems_corner_nodes_old.col(i_elem);
        let i0 = *elems_corner_nodes_old_.idx(0);
        let mut i1 = *elems_corner_nodes_old_.idx(1);
        let mut i2 = *elems_corner_nodes_old_.idx(2);
        let [
            mut i3, mut i4, mut i5, mut i6, mut i7, mut i8
        ]: [usize; 6] = [0; 6];

        let [x0, y0] = corner_nodes_xy[i0];
        let [mut x1, mut y1] = corner_nodes_xy[i1];
        let [mut x2, mut y2] = corner_nodes_xy[i2];

        ensure_three_points_counterclock_with_index(
            &x0, &y0, &mut x1, &mut y1, &mut x2, &mut y2, &i0, &mut i1, &mut i2
        );

        fn_tmp(i0, i1, x0, y0, x1, y1, &mut i3, &mut i4);
        fn_tmp(i1, i2, x1, y1, x2, y2, &mut i5, &mut i6);
        fn_tmp(i2, i0, x2, y2, x0, y0, &mut i7, &mut i8);

        let [x9, y9] = center_of_triangle(x0, y0, x1, y1, x2, y2);
        inner_nodes_xy.push([x9, y9]);

        let mut elems_corner_nodes_ = elems_corner_nodes.col_mut(i_elem);
        let mut elems_edge_nodes_ = elems_edge_nodes.col_mut(i_elem);

        *elems_corner_nodes_.idxm(0) = i0;
        *elems_corner_nodes_.idxm(1) = i1;
        *elems_corner_nodes_.idxm(2) = i2;

        *elems_edge_nodes_.idxm(0) = i3;
        *elems_edge_nodes_.idxm(1) = i4;
        *elems_edge_nodes_.idxm(2) = i5;
        *elems_edge_nodes_.idxm(3) = i6;
        *elems_edge_nodes_.idxm(4) = i7;
        *elems_edge_nodes_.idxm(5) = i8;
    }

    edge_nodes_xy.shrink_to_fit();
    edge_nodes_on_boundary.shrink_to_fit();
    let edge_nodes_n = edge_nodes_xy.len();
    let nodes_n = corner_nodes_n + edge_nodes_n + elems_n;
    assert_eq!(edge_nodes_n, edge_nodes_on_boundary.len());
    assert_eq!(elems_n, inner_nodes_xy.len());

    println!("elems_n = {elems_n}");
    println!("nodes_n = {nodes_n}");
    println!("corner_nodes_n = {corner_nodes_n}");
    println!("edge_nodes_n = {edge_nodes_n}");
    println!("inner_nodes_n = {elems_n}");

    {
        let mut npy = NpyObject::<[fsize; 2]>::new_writer(
            &edge_nodes_xy_path, [1, 0], true, vec![2, edge_nodes_n]
        );
        npy.write_header().unwrap();
        println!("\nedge_nodes_xy, npy.shape = {:?}, npy.size={}", npy.shape, npy.size);
        npy.write(&edge_nodes_xy);
    }
    {
        let mut npy = NpyObject::<u8>::new_writer(
            &edge_nodes_on_boundary_path, [1, 0], true, vec![edge_nodes_n]
        );
        npy.write_header().unwrap();
        println!("\nedge_nodes_on_boundary, npy.shape = {:?}, npy.size={}", npy.shape, npy.size);
        npy.write(&edge_nodes_on_boundary);
    }
    {
        let mut npy = NpyObject::<[fsize; 2]>::new_writer(
            &inner_nodes_xy_path, [1, 0], true, vec![2, elems_n]
        );
        npy.write_header().unwrap();
        println!("\ninner_nodes_xy, npy.shape = {:?}, npy.size={}", npy.shape, npy.size);
        npy.write(&inner_nodes_xy);
    }
    elems_corner_nodes.write_npy(&elems_corner_nodes_path);
    elems_edge_nodes.write_npy(&elems_edge_nodes_path);

    println!("corner_nodes_xy[10..15] = {:?}", &corner_nodes_xy[10..15]);
    println!("edge_nodes_xy[10..15] = {:?}", &edge_nodes_xy[10..15]);
    println!("edge_nodes_on_boundary[10..15] = {:?}", &edge_nodes_on_boundary[10..15]);

    println!("Program ended successfully.");
}