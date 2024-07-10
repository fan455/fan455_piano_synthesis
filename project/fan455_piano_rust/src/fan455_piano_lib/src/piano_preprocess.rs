use fan455_arrf64::*;
use fan455_util::*;
use std::collections::BTreeSet;
use super::piano_finite_element::*;
use super::piano_io::*;


#[inline]
pub fn compute_mass_stiff_mat_sparse_idx(
    mesh: &PianoSoundboardMesh,
    args: &PianoParamsIn,
) {
    println!("Computing the sparse indices of mass and stiffness matrix...");
    let n = mesh.dof.pow(2);
    let mut edof_idx = mesh.create_edof_idx_buf();
    let mut mass_obj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); mesh.dof];
    let mut stiff_obj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); mesh.dof];

    println!("mesh.elems_n = {}", mesh.elems_n);

    for elem!(i_elem, i_group) in mzip!(0..mesh.elems_n, mesh.elems_groups.it()) {
        let edof = mesh.get_edof(i_elem, &mut edof_idx);
        let is_not_rib_or_bridge: bool = match (mesh.groups_ribs.get(i_group), mesh.groups_bridges.get(i_group)) {
            (None, None) => true,
            _ => false,
        };

        for dof_i in edof_idx[..edof].iter() {
            let [i_kind, i_global, _] = *dof_i;

            for dof_j in edof_idx[..edof].iter() {
                let [j_kind, j_global, _] = *dof_j;

                if i_global >= j_global {

                    let (i_kind_, j_kind_) = {
                        if i_kind >= j_kind { (i_kind, j_kind) }
                        else { (j_kind, i_kind) }
                    };

                    let mass_is_non_zero: bool = match (i_kind_, j_kind_) {
                        (0, 0) | (1, 1) | (2, 2) | (3, 3) | (4, 4) => true,
                        (3, 1) | (4, 2) => { if is_not_rib_or_bridge {false} else {true} },
                        _ => false,
                    };
                    if mass_is_non_zero {
                        mass_obj[i_global].insert(j_global);
                    }

                    let stiff_is_non_zero: bool = match (i_kind_, j_kind_) {
                        (0, 0) | (1, 0) | (2, 0) | (1, 1) | (2, 1) | (2, 2) | (3, 3) | (4, 3) | (4, 4) => true,
                        (3, 1) | (4, 1) | (3, 2) | (4, 2) => { if is_not_rib_or_bridge {false} else {true} },
                        _ => false,
                    };
                    if stiff_is_non_zero {
                        stiff_obj[i_global].insert(j_global);
                    }
                }
            }
        }
    }

    let fn_tmp = |non_zero: &Vec<BTreeSet<usize>>| 
    -> (usize, usize, Vec<usize>, Vec<usize>, Vec<usize>) {

        let nnz = {
            let mut s: usize = 0;
            for row in non_zero.iter() {
                s += row.len();
            }
            s
        };
        let nz = n - nnz;
        let mut row_pos: Vec<usize> = Vec::with_capacity(mesh.dof);
        let mut row_idx: Vec<usize> = Vec::with_capacity(nnz);
        let mut col_idx: Vec<usize> = Vec::with_capacity(nnz);

        let mut pos: usize = 0;
        row_pos.push(pos);
    
        for elem!(i, row) in mzip!(0..mesh.dof, non_zero.iter()) {
            if row.len() == 0 {
                panic!("In mass or stiff matrix, row {i} are all zeros.")
            }
            pos += row.len();
            row_pos.push(pos);
            for j in row.iter() {
                row_idx.push(i);
                col_idx.push(*j);
            }
        }
        assert_eq!(mesh.dof, row_pos.len()-1);
        assert_eq!(nnz, row_pos[mesh.dof]);
        assert_multi_eq!(nnz, pos, row_idx.len(), col_idx.len());

        (nz, nnz, row_pos, row_idx, col_idx)
    };

    println!("Computing non-zero entries of mass matrix...");
    let (
        mass_nz, mass_nnz, mass_row_pos, mass_row_idx, mass_col_idx
    ) = fn_tmp(&mass_obj);
    println!("Computing non-zero entries of stiff matrix...");
    let (
        stiff_nz, stiff_nnz, stiff_row_pos, stiff_row_idx, stiff_col_idx
    ) = fn_tmp(&stiff_obj);

    println!("\nmass matrix info:");
    println!("total number of entries = {n}");
    println!("number of zero entries = {mass_nz}");
    println!("number of non-zero entries = {mass_nnz}");
    println!("proportion of non-zero entries = {:.6}", mass_nnz as f64 / n as f64);

    println!("\nstiff matrix info:");
    println!("total number of entries = {n}");
    println!("number of zero entries = {stiff_nz}");
    println!("number of non-zero entries = {stiff_nnz}");
    println!("proportion of non-zero entries = {:.6}", stiff_nnz as f64 / n as f64);

    write_npy_vec(&args.vib.mass_mat_row_pos_path, &mass_row_pos);
    write_npy_vec(&args.vib.mass_mat_row_idx_path, &mass_row_idx);
    write_npy_vec(&args.vib.mass_mat_col_idx_path, &mass_col_idx);

    write_npy_vec(&args.vib.stiff_mat_row_pos_path, &stiff_row_pos);
    write_npy_vec(&args.vib.stiff_mat_row_idx_path, &stiff_row_idx);
    write_npy_vec(&args.vib.stiff_mat_col_idx_path, &stiff_col_idx);

    println!("Finished.\n");
}
