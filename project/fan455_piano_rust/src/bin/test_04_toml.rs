//#![allow(warnings)]
//#[allow(dead_code, unused_imports, non_snake_case, non_upper_case_globals, nonstandard_style)]

use fan455_util::*;
use fan455_util_macro::*;
use std::collections::{HashMap, BTreeMap};
use serde::{Serialize, Deserialize};
use std::fs::read_to_string;


#[derive(Serialize, Deserialize, Debug, Default)]
pub struct MyStruct {
    pub names: Vec<String>,
    pub ids: Vec<usize>,
    pub names_groups: HashMap<String, Vec<usize>>,
    pub ids_groups: BTreeMap<String, [usize; 2]>,
}


fn main()  {

    parse_cmd_args!();
    cmd_arg!(mystruct_toml, String);
    unknown_cmd_args!();

    let mystruct_toml_string = read_to_string(&mystruct_toml).unwrap();
    let args: MyStruct = toml::from_str(&mystruct_toml_string).unwrap();
    let ids_groups_2: BTreeMap<usize, [usize; 2]> = convert_btreemap_keys(&args.ids_groups);

    println!("args.names = {:?}", args.names);
    println!("args.ids = {:?}", args.ids);
    println!("args.names_groups = {:?}", args.names_groups);
    println!("args.ids_groups = {:?}", args.ids_groups);
    println!("args.ids_groups_2 = {:?}", ids_groups_2);
}
