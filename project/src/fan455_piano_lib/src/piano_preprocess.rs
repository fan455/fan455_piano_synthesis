use fan455_math_scalar::*;
use fan455_arrf64::*;
use fan455_util::*;
use std::collections::BTreeSet;
use super::piano_finite_element::dof_reissner_mindlin_plate;


#[inline]
pub fn compute_mass_stiff_mat_sparse_idx(
    elems_nodes_path: &String,
    nodes_n: usize,
    free_nodes_n: usize,

    row_pos_path: &String,
    row_idx_path: &String,
    col_idx_path: &String,
) {
    println!("Computing the sparse indices of mass and stiffness matrix...");
    let elems_nodes = Arr2::<usize>::read_npy(elems_nodes_path); // (10, elems_n)
    assert_eq!(elems_nodes.nrow(), 10);
    let elems_n = elems_nodes.ncol();
    let dof = dof_reissner_mindlin_plate(free_nodes_n, nodes_n);
    let n = dof.pow(2);

    println!("elems_n = {elems_n}");
    println!("nodes_n = {nodes_n}");
    println!("free_nodes_n = {free_nodes_n}");
    println!("degrees of freedom = {dof}\n");

    let mut non_zero: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); dof];

    for i_elem in 0..elems_n {
        let nodes = elems_nodes.col(i_elem);

        for i in nodes.it() {
            for j in nodes.it() {
                let i1 = *i;
                let j1 = *j;

                if i1 >= j1 {   
                    let i_is_free = i1 < free_nodes_n;
                    let j_is_free = j1 < free_nodes_n;

                    #[cfg(not(feature="clamped_plate"))] {
                        let i2 = i1 + free_nodes_n;
                        let i3 = i2 + nodes_n;
                        let j2 = j1 + free_nodes_n;
                        let j3 = j2 + nodes_n;

                        if i_is_free && j_is_free {
                            non_zero[i1].insert(j1);
                        }
                        if j_is_free {
                            non_zero[i2].insert(j1);
                            non_zero[i3].insert(j1);
                        }
                        non_zero[i2].insert(j2);
                        non_zero[i3].insert(j2);
                        non_zero[i3].insert(j3);
                    }
                    #[cfg(feature="clamped_plate")] {
                        let i2 = i1 + free_nodes_n;
                        let i3 = i2 + free_nodes_n;
                        let j2 = j1 + free_nodes_n;
                        let j3 = j2 + free_nodes_n;

                        if i_is_free && j_is_free {
                            non_zero[i1].insert(j1);
                            non_zero[i2].insert(j1);
                            non_zero[i3].insert(j1);
                            non_zero[i2].insert(j2);
                            non_zero[i3].insert(j2);
                            non_zero[i3].insert(j3);
                        }
                    }
                }
            }
        }
    }
    let nnz = {
        let mut s: usize = 0;
        for row in non_zero.iter() {
            s += row.len();
        }
        s
    };
    let mut row_pos: Vec<usize> = Vec::with_capacity(dof);
    let mut row_idx: Vec<usize> = Vec::with_capacity(nnz);
    let mut col_idx: Vec<usize> = Vec::with_capacity(nnz);

    let mut pos: usize = 0;
    row_pos.push(pos);

    for elem!(i, row) in mzip!(0..dof, non_zero.iter()) {
        if row.len() == 0 {
            panic!("In stiff matrix, row {i} are all zeros.")
        }
        pos += row.len();
        row_pos.push(pos);
        for j in row.iter() {
            row_idx.push(i);
            col_idx.push(*j);
        }
    }
    assert_eq!(dof, row_pos.len()-1);
    assert_eq!(nnz, row_pos[dof]);
    assert_multi_eq!(nnz, pos, row_idx.len(), col_idx.len());

    let nz = n - nnz;
    println!("total number of entries = {n}");
    println!("number of zero entries = {nz}");
    println!("number of non-zero entries = {nnz}");
    println!("proportion of non-zero entries = {:.6}", nnz as fsize / n as fsize);

    {
        let mut npy = NpyObject::<usize>::new_writer(
            row_pos_path, [1, 0], true, vec![dof+1]
        );
        npy.write_header().unwrap();
        println!("\nrow_pos, npy.shape = {:?}, npy.size={}", npy.shape, npy.size);
        npy.write(&row_pos);
    }
    {
        let mut npy = NpyObject::<usize>::new_writer(
            row_idx_path, [1, 0], true, vec![nnz]
        );
        npy.write_header().unwrap();
        println!("\nrow_idx, npy.shape = {:?}, npy.size={}", npy.shape, npy.size);
        npy.write(&row_idx);
    }
    {
        let mut npy = NpyObject::<usize>::new_writer(
            col_idx_path, [1, 0], true, vec![nnz]
        );
        npy.write_header().unwrap();
        println!("\ncol_idx, npy.shape = {:?}, npy.size={}", npy.shape, npy.size);
        npy.write(&col_idx);
    }
    println!("Finished.\n");
}
