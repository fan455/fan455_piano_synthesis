import numpy as np
import gmsh

# Read gmsh (.msh) file, which contains first-order triangular elems. 
# Then, convert to third-order triangular elems.
# Finally, save as npy file.

# physical groups
#boundary_group = gmsh.model.getPhysicalGroups(dim=1)
# boundary_status: 
# 0: not on boundary; 10/11/12: node 0/1/2 is on boundary; 
# 20/21/22: nodes [0,3,4,1]/[1,5,6,2]/[2,7,8,0] are on boundary

mesh_path = 'D:/self_research/piano_project/mesh/piano_mesh_01.msh'
dir_out = 'D:/self_research/piano_project/data/mesh'

gmsh.initialize()
gmsh.open(mesh_path)
print(f'model = {gmsh.model.list()}')

# Remove duplicated nodes
#gmsh.model.mesh.removeDuplicateNodes()

# Get the edges of elems.
"""gmsh.model.mesh.createEdges()
edges_tags, edges_nodes = gmsh.model.mesh.getAllEdges()
edges_n = edges_tags.size
assert edges_n == int(edges_nodes.size/2)
edges_nodes = np.reshape(edges_nodes, (edges_n, 2), order='C')"""


# Get element types.
target_elems_type = 2
nodes_per_element = 3
elems_types = gmsh.model.mesh.getElementTypes(dim=-1, tag=-1)
#assert np.all(elems_types == target_elems_type)


# Create nodes on the edges.
nodes, nodes_coord = gmsh.model.mesh.getNodes(dim=-1, tag=-1, includeBoundary=False, returnParametricCoord=False)[:2]
nodes = np.asarray(nodes, dtype=np.int32)
nodes_n = nodes.size
assert nodes_n == int(nodes_coord.size/3)
nodes_coord = np.asarray(np.reshape(nodes_coord, (nodes_n, 3))[:, :2], dtype=np.float64) # (nodes_n, 2)


# Get the boundary nodes.
boundary_nodes = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)[0] # "plate_boundary" group


# Remove the unnecessary 1d elems.
"""elems_1d = gmsh.model.mesh.getElementsByType(26)[0]
elems_1d_dim, elems_1d_tag = [], []

for x in elems_1d:
    dim, tag = gmsh.model.mesh.getElement(x)[-2:]
    elems_1d_dim.append(dim)
    elems_1d_tag.append(tag)

for x, dim, tag in zip(elems_1d, elems_1d_dim, elems_1d_tag):
    gmsh.model.mesh.removeElements(dim=dim, tag=tag, elementTags=[x])"""

#gmsh.model.mesh.reclassifyNodes()


# Renumber nodes so that free nodes come first then boundary nodes
boundary_nodes_n = boundary_nodes.size
free_nodes_n = nodes_n - boundary_nodes_n
nodes_index_old = np.arange(nodes_n)
is_boundary_node = np.isin(nodes, boundary_nodes)
nodes_new_index_old = np.concatenate([\
    nodes_index_old[~is_boundary_node], nodes_index_old[is_boundary_node]\
])
nodes_new = nodes[nodes_new_index_old]
nodes_coord_new = nodes_coord[nodes_new_index_old, :]
del nodes_index_old, is_boundary_node


# Get the 2d elems.
surface_groups = gmsh.model.getPhysicalGroups(dim=2)
surface_groups = list(x[1] for x in surface_groups)
groups_n = len(surface_groups)

surface_groups_ref = []
for i_group, tag_group in zip(range(0, groups_n), surface_groups):
    assert i_group + 2 == tag_group
    surface_groups_ref.append([i_group, gmsh.model.getPhysicalName(dim=2, tag=tag_group)])

elems_list = []
elems_groups_list = []
elems_nodes_index_list = []
groups_sizes = []

for (i_group, tag_group) in zip(range(0, groups_n), surface_groups): # each physical group
    surfaces_tags = gmsh.model.getEntitiesForPhysicalGroup(2, tag_group)

    groups_sizes_tmp = 0

    for tag_surface in surfaces_tags: # each surface in current physical group
        elems_tags, elems_nodes = gmsh.model.mesh.getElementsByType(target_elems_type, tag=tag_surface)
        assert elems_tags.ndim == 1
        elems_n = elems_tags.size
        assert elems_n == int(elems_nodes.size/nodes_per_element)
        assert elems_nodes.shape == (elems_n*nodes_per_element,)
        groups_sizes_tmp += elems_n 

        elems_nodes = np.asarray(elems_nodes, dtype=np.int32)
        elems_nodes_index = np.array(list(np.argwhere(nodes_new==x).item() for x in elems_nodes), dtype=np.uint64)

        elems_nodes = np.reshape(elems_nodes, ((elems_n, nodes_per_element)))
        elems_nodes_index = np.reshape(elems_nodes_index, ((elems_n, nodes_per_element)))

        elems_list.append(elems_tags)
        elems_groups_list.append(np.full((elems_n,), i_group, dtype=np.uint8))
        elems_nodes_index_list.append(elems_nodes_index)

    groups_sizes.append(groups_sizes_tmp)


elems = np.concatenate(elems_list, dtype=np.int32)
elems_groups = np.concatenate(elems_groups_list, dtype=np.uint8)
elems_nodes_index = np.concatenate(elems_nodes_index_list, dtype=np.uint64, axis=0)
groups_sizes = np.asarray(groups_sizes, dtype=np.uint64)
del elems_groups_list, elems_nodes_index_list

elems_n = elems_groups.size

groups_elems_idx = np.zeros((groups_n, 2), dtype=np.uint64)
groups_elems_idx[:, 1] = np.cumsum(groups_sizes)
groups_elems_idx[1:, 0] = np.cumsum(groups_sizes)[:-1]


# Save data
print(f'number of nodes = {nodes_n}')
print(f'number of free nodes = {free_nodes_n}')
print(f'number of boundary nodes = {boundary_nodes_n}')
print(f'number of elems = {elems_n}\n')
for group in surface_groups_ref:
    print(f'{group}')

nodes_coord_new = np.asarray(nodes_coord_new, dtype=np.float64)
elems_nodes_index = np.asarray(elems_nodes_index, dtype=np.uint64)
groups_elems_idx = np.asarray(groups_elems_idx, dtype=np.uint64)

np.save(f'{dir_out}/corner_nodes_xy.npy', nodes_coord_new) # (nodes_n = free_nodes_n + boundary_nodes_n, 2)
#np.save(f'{dir_out}/elems_groups.npy', elems_groups) # (elems_n,)
np.save(f'{dir_out}/elems_corner_nodes_index_old.npy', elems_nodes_index) # (elems_n, nodes_per_element), corresponding to nodes_coord_new
#np.save(f'{dir_out}/elems_boundary_status.npy', elems_boundary_status) # (elems_n,)
#np.save(f'{dir_out}/elems_groups_sizes.npy', groups_sizes) # (num_of_groups,)
np.save(f'{dir_out}/groups_elems_idx.npy', groups_elems_idx) # (num_of_groups, 2)

gmsh.finalize()

