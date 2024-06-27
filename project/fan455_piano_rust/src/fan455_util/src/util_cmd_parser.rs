use std::collections::hash_map::HashMap;


pub struct CmdParser {
    pub args: HashMap<String, String>,
}


impl CmdParser {
    pub fn new() -> Self {
        let args_vec: Vec<String> = std::env::args().collect();
        let mut args: HashMap<String, String> = HashMap::with_capacity(args_vec.len()-1);
        let mut pos: usize;
    
        for arg in &args_vec[1..] {
            pos = arg.find('=').expect( format!("Error: argument {} does not have \"=\"", arg).as_str() );
            args.insert( (&arg[0..pos]).to_string(), (&arg[pos+1..]).to_string() );
        }
        Self {args}
    }
}
