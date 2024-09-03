use fan455_arrf64::*;
use fan455_math_scalar::*;
use fan455_util::*;
use fan455_util_macro::*;
use fan455_piano_lib::*;
use std::{fs::{read_to_string, File}, io::Write};
use std::time::Instant;


fn main() {

    // Read parameters from toml file.
    println!("Reading toml file of piano parameters...");
    parse_cmd_args!();
    cmd_arg!(piano_toml, String);
    unknown_cmd_args!();

    let piano_toml_string = read_to_string(&piano_toml).unwrap();
    let args: PianoParamsIn = toml::from_str(&piano_toml_string).unwrap();
    println!("Finished.\n");

    let mesh = PianoSoundboardMesh::new(&args.mesh, false);
    let mut mbuf = PianoSoundboardMeshBuf::new(&mesh);
    let sb = PianoSoundboard::new(&args.sb);

    // Some basic parameters.
    let dof = mesh.dof;
    let elems_n = mesh.elems_n;
    let normalize = args.normalize;
    let dir = args.data_dir;
    let n_prog = args.n_prog;
    let dt = (args.vib.sample_rate as f64).recip();

    // The program runs differently for stage 0 or 1.
    if args.stage == 0 {
        let ribs = PianoRibs::new(&args.ribs, sb.thickness);
        let bridges = PianoBridges::new(&args.bridges, sb.thickness);
        let mut vib = Vibration::new(dof, &args.vib);

        // Assemble the mass matrix and stiffness matrix.
        sb.compute_mass_stiff(&mut vib.mass_mat, &mut vib.stiff_mat, &mesh, &mut mbuf, normalize, n_prog);
        ribs.compute_mass_stiff(&mut vib.mass_mat, &mut vib.stiff_mat, &mesh, &mut mbuf, normalize, n_prog);
        bridges.compute_mass_stiff(&mut vib.mass_mat, &mut vib.stiff_mat, &mesh, &mut mbuf, normalize, n_prog);
    
        // Solve the generalized eigenvalue problem.
        let time_now = Instant::now();
        vib.compute_eigen(&args.vib);
        {
            let (t_, t__) = modulo(time_now.elapsed().as_secs(), 60);
            println!("Solving the generalized eigenvalue problem took {t_} min {t__} s.", );
        }

        // Compute modal frequencies and damping.
        vib.compute_modal_freq_damp(args.vib.print_first_freq);
        
        // The user selects the highest modal frequency needed.
        let mut modal_freq_ub = args.vib.modal_freq_ub;
        {
            vib.select_modes(modal_freq_ub);
            loop {
                let mut confirm = String::new();
                println!("Confirm the selected vibrational modes? (y/n)");
                std::io::stdin().read_line(&mut confirm).expect("Failed to read line.");
                if confirm.trim().to_lowercase() == "y" {
                    break;
                }
                println!("Please enter a new modal frequency upper bound: ");
                loop {
                    let mut input = String::new();
                    std::io::stdin().read_line(&mut input).expect("Failed to read line.");
                    modal_freq_ub = match input.trim().parse() {
                        Ok(value) => value,
                        Err(_) => { println!("Invalid input value, please re-enter."); continue },
                    };
                    break;
                }
                vib.select_modes(modal_freq_ub);
            }
        }
        let modes_n = vib.modes_n;

        // Transpose eigenvectors.
        vib.transpose_eigvec(0, dof);

        // Save the eigenvalues, modal frequencies and modal damping.
        println!("Saving data (mass diagonal, eigenvalues, eigenvectors, modal frequencies, modal damping) to {dir}...");
        if args.vib.truncate_eigvec { vib.truncate_eigvec(); }
        if args.vib.truncate_eigval { vib.truncate_eigval(); }
        if args.vib.truncate_modal_freq { vib.truncate_modal_freq(); }
        if args.vib.truncate_modal_damp { vib.truncate_modal_damp(); }

        write_npy_vec(&format!("{dir}/soundboard_mass_mat_data.npy"), &vib.mass_mat.data);
        write_npy_vec(&format!("{dir}/soundboard_stiff_mat_data.npy"), &vib.stiff_mat.data);
        vib.eigvec.write_npy(&format!("{dir}/soundboard_eigenvectors.npy"));
        vib.eigvec_trans.write_npy(&format!("{dir}/soundboard_eigenvectors_transposed.npy"));
        vib.eigval.write_npy(&format!("{dir}/soundboard_eigenvalues.npy"));
        vib.modal_freq.write_npy(&format!("{dir}/soundboard_modal_frequency.npy"));
        vib.modal_damp.write_npy(&format!("{dir}/soundboard_modal_damping.npy"));
        println!("Finished.\n");
    
        // Compute the modal quadratures.
        let mut modal_quad: Arr2<f64> = Arr2::new(modes_n, elems_n);
        sb.compute_modal_quad(&mut modal_quad, &vib.eigvec_trans, &mesh, &mut mbuf, normalize, n_prog);

        println!("Saving data (elements modal quadrature) to {dir}...");
        modal_quad.write_npy_tm(&format!("{dir}/soundboard_elements_modal_quadrature.npy"));
        println!("Finished.\n");

        // Save some soundboard parameters.
        println!("Saving data (piano output parameters) to {dir}...");
        let args_out = PianoParamsOut {
            mesh: mesh.output_params(),
            vib: vib.output_params(),
        };
        let piano_toml_out_string = toml::to_string(&args_out).unwrap();
        let mut piano_toml_out_file = File::create(
            &format!("{dir}/piano_params_out.toml")
        ).unwrap();
        piano_toml_out_file.write_all(piano_toml_out_string.as_bytes()).unwrap();
        println!("Finished.\n");

    } else if args.stage == 1 {
        // Compute the modal movement (acceleration). 
        // This is also for adjusting modal damping. The number of modes can not be adjusted.
        let mut vib = Vibration::new_for_modal(dof, &args.vib, &format!("{dir}/soundboard_eigenvalues.npy"));
        vib.compute_modal_freq_damp(args.vib.print_first_freq);

        println!("Saving data (modal frequencies, modal damping) to {dir}...");
        vib.modal_freq.write_npy(&format!("{dir}/soundboard_modal_frequency.npy"));
        vib.modal_damp.write_npy(&format!("{dir}/soundboard_modal_damping.npy"));
        println!("Finished.\n");

        vib.allocate_modal_movement(args.vib.duration, args.vib.sample_rate);
        vib.compute_modal_movement();
        println!("Saving data (modal movement) to {dir}...");
        vib.modal_move.write_npy(&format!("{dir}/soundboard_modal_movement.npy"));
        println!("Finished.\n");
    
    } else if args.stage == 2 {
        println!("Reading precomputed data (modal movement, modal quadrature, eigenvectors)...");
        let modal_move = Arr2::<f64>::read_npy(&format!("{dir}/soundboard_modal_movement.npy"));
        let modal_quad = Arr2::<f64>::read_npy(&format!("{dir}/soundboard_elements_modal_quadrature.npy"));
        let eigvec_trans = Arr2::<f64>::read_npy(&format!("{dir}/soundboard_eigenvectors_transposed.npy"));
        
        let modes_n = modal_move.nrow();
        let nt = modal_move.ncol();
        assert_multi_eq!(modes_n, modal_quad.nrow(), eigvec_trans.nrow());
        assert_eq!(dof, eigvec_trans.ncol());
        assert_eq!(elems_n, modal_quad.ncol());
        println!("Finished.\n");

        let mut response = Arr2::<f64>::new(nt, 5);
        let mut modal_force = Arr2::<f64>::new(modes_n, 5); 
        let mut modal_buf = Arr2::<f64>::new(modes_n, 5); 

        let mut elems_center_xy: Vec<[f64; 2]> = vec![[0., 0.]; elems_n];
        mesh.compute_elems_center(&mut elems_center_xy);

        // Compute the soundboard response for fixed bridge position and listening position.
        let time_now = Instant::now();
        sb.compute_response(
            dt, args.rad.sound_speed, args.rad.bridge_pos, args.rad.listen_pos, &elems_center_xy,
            &mut response, &mut modal_force, &mut modal_buf, &modal_quad, &eigvec_trans, &modal_move,
            &mesh, &mut mbuf, args.normalize, args.n_prog,
        );
        {
            let (t_, t__) = modulo(time_now.elapsed().as_secs(), 60);
            println!("Computing soundboard response took {t_} min {t__} s.", );
        }

        println!("Saving data (soundboard response)...");
        response.write_npy_tm(&format!("{}/soundboard_response_01.npy", args.rad.response_dir));
        println!("Finished.\n");

    } else if args.stage == -1 {
        // Recompute the modal quadrature using computed eigenvalues.
        let eigvec = Arr2::<f64>::read_npy(&format!("{dir}/soundboard_eigenvectors.npy"));
        assert_eq!(dof, eigvec.nrow());
        let modes_n = eigvec.ncol();

        let mut eigvec_trans = Arr2::<f64>::new(modes_n, dof);
        eigvec_trans.get_trans(&eigvec);

        let mut modal_quad: Arr2<f64> = Arr2::new(modes_n, elems_n);
        sb.compute_modal_quad(&mut modal_quad, &eigvec_trans, &mesh, &mut mbuf, normalize, n_prog);

        println!("Saving data (eigenvectors transposed, elements modal quadrature) to {dir}...");
        eigvec_trans.write_npy(&format!("{dir}/soundboard_eigenvectors_transposed.npy"));
        modal_quad.write_npy_tm(&format!("{dir}/soundboard_elements_modal_quadrature.npy"));
        println!("Finished.\n");

    } else {
        panic!("Computation stage {} is not supported.", args.stage);
    }
    println!("Program finished successfully.");
}