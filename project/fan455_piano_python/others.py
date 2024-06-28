

def est_mat_storage(n_modes: int):
    size = (n_modes**2)*0.5*8/(1024**2)
    print(f'memory size of matrix for {n_modes} modes: {size:.2f} MB')