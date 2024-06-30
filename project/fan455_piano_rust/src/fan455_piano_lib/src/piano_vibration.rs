use fan455_arrf64::*;
use fan455_math_scalar::*;
use fan455_util::{elem, mzip, NpyObject, NpyTrait};
use super::piano_io::*;


#[derive(Default)]
pub struct Vibration {
    pub dim: usize,
    pub modes_n: usize,
    pub eig_n: usize,
    pub duration: fsize,
    pub sample_rate: usize,
    pub nt: usize,
    pub dt: fsize,
    pub mass_diag: Arr1<fsize>,
    pub stiff_mat: CsrMat<fsize>,
    pub eigvec: Arr2<fsize>,
    pub eigvec_trans: Arr2<fsize>, // (modes_n, n_basis)
    pub eigval: Arr1<fsize>,
    pub modal_freq: Arr1<fsize>,
    pub modal_damp: Arr1<fsize>,
    pub modal_move: Arr2<fsize>, // (modes_n, n_time)
}


impl Vibration
{
    #[inline]
    pub fn new( dim: usize, data: &PianoVibrationParamsIn ) -> Self {
        println!("Allocating memory for solving the vibration system...");
        let mass_diag = Arr1::<fsize>::new(dim);

        let stiff_mat: CsrMat<fsize> = {
            let row_pos: Vec<usize> = {
                let mut npy = NpyObject::<usize>::new_reader(&data.stiff_mat_row_pos_path);
                npy.read_header().unwrap();
                npy.read()
            };
            let row_idx: Vec<usize> = {
                let mut npy = NpyObject::<usize>::new_reader(&data.stiff_mat_row_idx_path);
                npy.read_header().unwrap();
                npy.read()
            };
            let col_idx: Vec<usize> = {
                let mut npy = NpyObject::<usize>::new_reader(&data.stiff_mat_col_idx_path);
                npy.read_header().unwrap();
                npy.read()
            };
            let nnz = row_idx.len();
            assert_eq!(nnz, col_idx.len(), "The length of row_idx and col_idx may be incorrect.");
            let data: Vec<fsize> = vec![0.; nnz];
            CsrMat { nrow: dim, ncol: dim, nnz, data, row_pos, row_idx, col_idx }
        };
        println!("Finished.\n");
        Self { dim, mass_diag, stiff_mat, 
            ..Default::default() }
    }


    #[inline]
    pub fn new_for_modal( dim: usize, params: &PianoVibrationParamsIn, eigval_path: &String ) -> Self {
        println!("Reading eigenvalues data for computing modal frequency, modal damping and modal movement...");
        let eigval = Arr1::<fsize>::read_npy(&eigval_path);
        let eig_n = eigval.size();
        let modes_n = eig_n;
        let modal_damp = Arr1::<fsize>::read_npy(&params.damp_path);
        assert_eq!(eig_n, modal_damp.size());
        println!("Finished.\n");
        Self { dim, eig_n, eigval, modes_n, modal_damp, ..Default::default() }
    }


    #[inline]
    pub fn output_params( &self ) -> PianoVibrationParamsOut {
        PianoVibrationParamsOut {
            eig_n: self.eig_n,
            modes_n: self.modes_n,
            modal_freq_range: [self.modal_freq[0], self.modal_freq[self.modal_freq.size()-1]],
        }
    }


    #[inline]
    pub fn eigval_to_freq( eigval: fsize, damp: fsize ) -> fsize {
        // Loss factor damping.
        let freq = (eigval - damp.powi(2)).sqrt() / (2.*PI);
        if freq.is_nan() {
            panic!("Frequency is nan");
        }
        freq
    }


    #[inline]
    pub fn freq_to_eigval( freq: fsize, damp: fsize ) -> fsize {
        let eigval = (freq*2.*PI).powi(2) + damp.powi(2);
        eigval
    }


    #[inline]
    pub fn compute_eigen( &mut self, params: &PianoVibrationParamsIn ) {
        println!("Computing eigenvalues and eigenvectors of the system equations...");

        // Factorize the mass matrix.
        for s in self.mass_diag.itm() {
            *s = s.sqrt();
        }
        for elem!(s, i, j) in self.stiff_mat.itm() {
            *s /= self.mass_diag[*i] * self.mass_diag[*j];
        }

        // Allocate memory for eigenvalues and eigenvectors.
        let mut eigfreq_lb = params.eigfreq_lb;
        let mut eigfreq_ub = params.eigfreq_ub;
        let mut eigval_lb: fsize;
        let mut eigval_ub: fsize;
        let mut eig_n_guess = params.eig_n_guess;

        let mut eigsol = FeastSparse::<fsize>::new();
        eigsol.set_runtime_print(params.eigsol.runtime_print);
        eigsol.set_tol(params.eigsol.tol);
        eigsol.set_max_loops(params.eigsol.max_loops);
        eigsol.set_stop_type(params.eigsol.stop_type);
        eigsol.set_num_contour_points(params.eigsol.num_contour_points);
        eigsol.set_sparse_mat_check(params.eigsol.sparse_mat_check);
        eigsol.set_positive_mat_check(params.eigsol.positive_mat_check);
        
        self.stiff_mat.change_to_one_based_index();

        'outer: loop {
            eigval_lb = (eigfreq_lb*2.*PI).powi(2);
            eigval_ub = (eigfreq_ub*2.*PI).powi(2);

            self.eigval.resize(eig_n_guess, 0.);
            self.eigvec.resize(self.dim, eig_n_guess, 0.);
            eigsol.init_guess(eigval_lb, eigval_ub, eig_n_guess);
            eigsol.call(&mut self.stiff_mat, &mut self.eigval, &mut self.eigvec);
            eigsol.report();

            if eigsol.info == 0 {
                println!("Execution of MKL eigensolver routine was successful.");
                break 'outer;
            } else {
                println!("Execution of MKL eigensolver routine failed with output info: {}. Please check the MKL reference for more details.", eigsol.info);

                if eigsol.info == 3 {
                    println!("The initial guess of the amount of eigenfrequencies in range [{eigfreq_lb:.2}, {eigfreq_ub:.2}] Hz were too small. Please re-guess.");
                    {
                        println!("Please enter a new lower bound for eigenfrequencies (non-number input means unchanged):");
                        let mut input = String::new();
                        std::io::stdin().read_line(&mut input).expect("Failed to read line.");
                        eigfreq_lb = match input.trim().parse() {
                            Ok(value) => value,
                            Err(_) => eigfreq_lb,
                        };
                    }
                    {
                        println!("Please enter a new upper bound for eigenfrequencies (non-number input means unchanged):");
                        let mut input = String::new();
                        std::io::stdin().read_line(&mut input).expect("Failed to read line.");
                        eigfreq_ub = match input.trim().parse() {
                            Ok(value) => value,
                            Err(_) => eigfreq_ub,
                        };
                    }
                    {
                        println!("Please enter a new amount for eigenfrequencies (non-number input means unchanged):");
                        let mut input = String::new();
                        std::io::stdin().read_line(&mut input).expect("Failed to read line.");
                        eig_n_guess = match input.trim().parse() {
                            Ok(value) => value,
                            Err(_) => eig_n_guess,
                        };
                    }
                    self.eigval.resize(eig_n_guess, 0.);
                    self.eigvec.resize(self.dim, eig_n_guess, 0.);

                } else {
                    panic!("Eigensolver terminated.");
                }
            }
        }
        self.eig_n = eigsol.eig_n;
        self.stiff_mat.change_to_zero_based_index();
        self.eigval.truncate(self.eig_n);
        self.eigvec.truncate(self.dim, self.eig_n);

        // Transform the eigenvectors so that eigvec^T * mass_mat * eigvec = identity_mat
        for j in 0..self.eig_n {
            self.eigvec.col_mut(j).divassign(&self.mass_diag);
        }
        println!("Finished.\n");
    }


    #[inline]
    pub fn compute_modal_freq_damp( &mut self, params: &PianoVibrationParamsIn ) {
        println!("Computing modal frequencies and modal damping...");
        let print_first_freq = params.print_first_freq;

        self.modal_freq.resize(self.eig_n, 0.);
        let mut eigval_ = self.eigval[0];
        println!("self.eigval[0] = {}", self.eigval[0]);
        assert!(eigval_.is_nan() == false, "NaN eigenvalue encountered.");
        assert!(eigval_ >= 0., "Negative eigenvalue encountered.");

        if self.modal_damp.size() == 0 {
            self.modal_damp.resize(self.eig_n, 0.);
        }
        
        for elem!(a, b, c) in mzip!(self.modal_freq.itm(), self.modal_damp.it(), self.eigval.it()) {
            assert!((*c).is_nan() == false, "NaN eigenvalue encountered.");
            assert!(*c >= eigval_, "Descending eigenvalue encountered.");
            *a = Self::eigval_to_freq(*c, *b);
            assert!((*a).is_nan() == false, "NaN modal frequency encountered.");
            eigval_ = *c;
        }
        
        println!("Printing the first {print_first_freq} modal frequencies...");
        for elem!(i, a) in mzip!(0..print_first_freq, self.modal_freq.it()) {
            println!("modal_freq[{}] = {:.3}", i, a);
        }
        println!("Modal frequency range: {:.2} Hz to {:.2} Hz", self.modal_freq[0], self.modal_freq[self.eig_n-1]);
        println!("Finished.\n");
    }


    #[inline]
    pub fn select_modes( &mut self, modal_freq_ub: fsize ) {
        println!("Selecting eigenmodes with frequencies below {modal_freq_ub:.2} Hz...");
        self.modes_n = self.eig_n;
        for elem!(i, a) in mzip!(0..self.dim, self.modal_freq.data.iter()) {
            if *a > modal_freq_ub {
                self.modes_n = i;
                break;
            }
        }
        println!("Finished: modes_n = {}, highest modal frequency = {:.2} Hz\n", self.modes_n, self.modal_freq[self.modes_n-1]);
    }


    #[inline]
    pub fn transpose_eigvec( &mut self, dim_beg: usize, dim_end: usize ) {
        self.eigvec_trans.resize(self.modes_n, dim_end-dim_beg, 0.);

        for i in 0..self.modes_n {
            for elem!(j, eigvec_) in mzip!(dim_beg..dim_end, self.eigvec.subvec2(dim_beg, i, dim_end, i).it()) {
                *self.eigvec_trans.idxm2(i, j) = *eigvec_;
            }
        }
    }


    #[inline]
    pub fn allocate_modal_movement( &mut self, du: fsize, sr: usize ) {
        self.sample_rate = sr;
        self.nt = (du*sr as fsize).ceil() as usize;
        self.dt = (sr as fsize).recip();
        self.duration = self.dt * self.nt as fsize;
        println!("Sample rate: {}", self.sample_rate);
        println!("Duration: {:.4} seconds", self.duration);
        println!("Number of time steps: {}", self.nt);
        println!("Size of one time step: {:.6} seconds", self.dt);

        println!("Allocating memory for modal movement matrix...");
        self.modal_move.resize(self.modes_n, self.nt, 0.);
        assert_eq!(self.modal_move.nrow()*self.modal_move.ncol(), self.modal_move.data.len());
        println!("Finished.\n");
    }

    
    #[inline]
    pub fn compute_modal_movement( &mut self ) {
        println!("Computing modal movement (acceleration) under free vibration...");
        let dt = self.dt;
        // u(t) = exp(-b*t) * sin(a*t)
        for i in 0..self.nt {
            let t = (i as fsize) * dt;

            for elem!(u, freq, b) in mzip!(
                self.modal_move.col_mut(i).itm(),
                self.modal_freq.subvec(0, self.modes_n).it(), 
                self.modal_damp.subvec(0, self.modes_n).it()
            ) {
                let a = 2.*PI*freq;

                *u = (-b*t).exp() * (
                    ( b.powi(2) - a.powi(2) ) * (a*t).sin() - 2.*a*b * (a*t).cos()
                ) / ( a * (a.powi(2) + b.powi(2)) );
            }
        }
        println!("Finished.\n");
    }


    #[inline]
    pub fn truncate_eigvec( &mut self ) {
        self.eigvec.truncate(self.dim, self.modes_n);
    }


    #[inline]
    pub fn truncate_eigval( &mut self ) {
        self.eigval.truncate(self.modes_n);
    }


    #[inline]
    pub fn truncate_modal_freq( &mut self ) {
        self.modal_freq.truncate(self.modes_n);
    }


    #[inline]
    pub fn truncate_modal_damp( &mut self ) {
        self.modal_damp.truncate(self.modes_n);
    }
}