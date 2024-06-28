import numpy as np


folder = 'D:/self_research/piano_project/data/mesh'

corner_nodes_xy = np.load(f'{folder}/corner_nodes_xy.npy')
# (corner_nodes_n, 2)
edge_nodes_xy = np.load(f'{folder}/edge_nodes_xy.npy')
# (edge_nodes_n, 2)
inner_nodes_xy = np.load(f'{folder}/inner_nodes_xy.npy')
# (inner_nodes_n, 2)
edge_nodes_on_boundary = np.load(f'{folder}/edge_nodes_on_boundary.npy')
# (edge_nodes_n,)
elems_corner_nodes = np.load(f'{folder}/elems_corner_nodes.npy')
# (elems_n, 3)
elems_edge_nodes = np.load(f'{folder}/elems_edge_nodes.npy')
# (elems_n, 6)

if np.isfortran(corner_nodes_xy):
    corner_nodes_xy = np.swapaxes(corner_nodes_xy, 0, 1)

if np.isfortran(edge_nodes_xy):
    edge_nodes_xy = np.swapaxes(edge_nodes_xy, 0, 1)

if np.isfortran(inner_nodes_xy):
    inner_nodes_xy = np.swapaxes(inner_nodes_xy, 0, 1)

if np.isfortran(elems_corner_nodes):
    elems_corner_nodes = np.swapaxes(elems_corner_nodes, 0, 1)
            
if np.isfortran(elems_edge_nodes):
    elems_edge_nodes = np.swapaxes(elems_edge_nodes, 0, 1)


corner_nodes_n = corner_nodes_xy.shape[0]
edge_nodes_n = edge_nodes_xy.shape[0]
inner_nodes_n = inner_nodes_xy.shape[0]
elems_inner_nodes = np.arange(0, inner_nodes_n).reshape((inner_nodes_n, 1))

assert edge_nodes_n == edge_nodes_on_boundary.shape[0]
assert corner_nodes_xy.shape[1] == edge_nodes_xy.shape[1] == inner_nodes_xy.shape[1] == 2
elems_n = elems_corner_nodes.shape[0]
assert elems_n == elems_edge_nodes.shape[0] == inner_nodes_xy.shape[0]
nodes_n = corner_nodes_n + edge_nodes_n + inner_nodes_n
free_corner_nodes_n = 634

edge_nodes_idx = np.arange(0, edge_nodes_n)
tmp1 = np.copy(edge_nodes_idx[edge_nodes_on_boundary==0])
tmp2 = np.copy(edge_nodes_idx[edge_nodes_on_boundary==1])
edge_nodes_idx_old_in_new = np.concatenate([tmp1, tmp2])
free_edge_nodes_n = tmp1.size
del tmp1, tmp2, edge_nodes_idx
assert edge_nodes_idx_old_in_new.size == edge_nodes_n

edge_nodes_xy = edge_nodes_xy[edge_nodes_idx_old_in_new, :]
elems_edge_nodes = np.reshape(elems_edge_nodes, (elems_n*6,))
elems_edge_nodes = np.array(
    list(np.argwhere(edge_nodes_idx_old_in_new==x).item() for x in elems_edge_nodes)
)
elems_edge_nodes = np.reshape(elems_edge_nodes, (elems_n, 6))
del edge_nodes_idx_old_in_new





free_nodes_n = free_corner_nodes_n + free_edge_nodes_n + inner_nodes_n
boundary_corner_nodes_n = corner_nodes_n - free_corner_nodes_n
boundary_edge_nodes_n = edge_nodes_n - free_edge_nodes_n
boundary_nodes_n = boundary_corner_nodes_n + boundary_edge_nodes_n

corner_nodes_idx_new_in_old = np.arange(0, corner_nodes_n)
corner_nodes_idx_new_in_old[free_corner_nodes_n:] += free_nodes_n - free_corner_nodes_n

edge_nodes_idx_new_in_old = np.arange(0, edge_nodes_n)
edge_nodes_idx_new_in_old[:free_edge_nodes_n] += free_corner_nodes_n
edge_nodes_idx_new_in_old[free_edge_nodes_n:] += free_nodes_n + boundary_corner_nodes_n - free_edge_nodes_n

inner_nodes_idx_new_in_old = np.arange(0, inner_nodes_n) + free_corner_nodes_n + free_edge_nodes_n

nodes_idx_new_in_old = np.concatenate([
    corner_nodes_idx_new_in_old,
    edge_nodes_idx_new_in_old,
    inner_nodes_idx_new_in_old
])

elems_corner_nodes = corner_nodes_idx_new_in_old[elems_corner_nodes]
elems_edge_nodes = edge_nodes_idx_new_in_old[elems_edge_nodes]
elems_inner_nodes = inner_nodes_idx_new_in_old[elems_inner_nodes]
elems_nodes = np.concatenate([
    elems_corner_nodes, 
    elems_edge_nodes, 
    elems_inner_nodes
], axis=1)
elems_nodes = np.asarray(elems_nodes, dtype=np.uint64)
assert elems_nodes.shape == (elems_n, 10)

nodes_xy = np.concatenate([
    corner_nodes_xy[:free_corner_nodes_n, :],
    edge_nodes_xy[:free_edge_nodes_n, :],
    inner_nodes_xy,
    corner_nodes_xy[free_corner_nodes_n:, :],
    edge_nodes_xy[free_edge_nodes_n:, :]
], axis=0, dtype=np.float64)


np.save(f'{folder}/elems_nodes.npy', elems_nodes)
np.save(f'{folder}/nodes_xy.npy', nodes_xy)


print(f'elems_n = {elems_n}, nodes_n = {nodes_n}, free_nodes_n = {free_nodes_n}')