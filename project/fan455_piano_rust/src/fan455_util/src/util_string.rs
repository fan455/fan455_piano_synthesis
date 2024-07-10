use std::str::FromStr;
use std::fmt::Debug;
use std::fs::File;
use std::io::Write;


pub struct VecPrinter {
    pub name_width: usize,
    pub width: usize,
    pub prec: usize,
}

impl VecPrinter {

    #[inline]
    pub fn new() -> Self {
        Self { name_width: 15, width: 12, prec: 4 }
    }
    
    #[inline]
    pub fn print<T>( &self, x: &Vec<T>, name: &str ) 
    where T: std::fmt::Display,
    {
        let name_width = self.name_width;
        let width = self.width;
        let prec = self.prec;

        print!("{:<name_width$}:", name);
        for x_ in x {
            print!(" {x_:>width$.prec$}");
        }
        print!("\n");
    }

    #[inline]
    pub fn print_string( &self, x: &Vec<String>, name: &str ) 
    {
        let name_width = self.name_width;
        let width = self.width;

        print!("{:<name_width$}:", name);
        for x_ in x {
            print!(" {x_:>width$}");
        }
        print!("\n");
    }
}

pub struct CsvWriter {
    pub file: File,
    pub name_width: usize,
    pub width: usize,
    pub prec: usize,
}

impl CsvWriter {

    #[inline]
    pub fn new( csv_path: &str ) -> Self {
        let file = File::create(csv_path).unwrap();
        Self { file, name_width: 15, width: 12, prec: 4 }
    }

    #[inline]
    pub fn write_str( &mut self, x: &str ) {
        let s: String = x.to_string();
        self.file.write(&s.into_bytes()[..]).unwrap();
    }

    #[inline]
    pub fn write_comma( &mut self ) {
        self.file.write(&", ".to_string().into_bytes()[..]).unwrap();
    }

    #[inline]
    pub fn write_newline( &mut self ) {
        self.file.write(&'\n'.to_string().into_bytes()[..]).unwrap();
    }

    #[inline]
    pub fn write_var<T>( &mut self, x: &T, name: &str ) 
    where T: std::fmt::Display,
    {
        let name_width = self.name_width;
        let width = self.width;
        let prec = self.prec;

        let mut s: String = format!("{:<name_width$}, ", name);
        s.push_str( format!("{x:>width$.prec$}").as_str() );
        s.push('\n');
        self.file.write(&s.into_bytes()[..]).unwrap();
    }

    #[inline]
    pub fn write_vec<T>( &mut self, x: &Vec<T>, name: &str ) 
    where T: std::fmt::Display,
    {
        let name_width = self.name_width;
        let width = self.width;
        let prec = self.prec;

        let mut s: String = format!("{:<name_width$}, ", name);
        s.push_str(x.iter().map(
            |x| format!("{x:width$.prec$}")
        ).collect::<Vec<String>>().join(", ").as_str());
        s.push('\n');
        self.file.write(&s.into_bytes()[..]).unwrap();
    }

    #[inline]
    pub fn write_vec_string( &mut self, x: &Vec<String>, name: &str ) 
    {
        let name_width = self.name_width;
        let width = self.width;

        let mut s: String = format!("{:<name_width$}, ", name);
        s.push_str(x.iter().map(
            |x| format!("{x:>width$}")
        ).collect::<Vec<String>>().join(", ").as_str());
        s.push('\n');
        self.file.write(&s.into_bytes()[..]).unwrap();
    }

}


// Convert string like "[0,1,2]"" to Vec
#[inline]
pub fn string_to_vec<T>(s: &str) -> Vec<T>
where 
    T: FromStr + std::fmt::Debug, 
    <T as FromStr>::Err: Debug,
{
    let mut v: Vec<T> = Vec::new();
    vec_from_string(&mut v, s);
    v
}


// Convert string like "0,1,2" to Vec.
#[inline]
pub fn string_to_vec_no_bracket<T>(s: &str) -> Vec<T> 
where 
    T: FromStr + std::fmt::Debug, 
    <T as FromStr>::Err: Debug,
{
    let mut v: Vec<T> = Vec::new();
    vec_from_string_no_bracket(&mut v, s);
    v
}


#[inline]
pub fn vec_from_string<T>(v: &mut Vec<T>, s: &str)
where 
    T: FromStr + std::fmt::Debug, 
    <T as FromStr>::Err: Debug,
{
    let mut p1: usize = 1;
    let end: usize = s.len() - 1;
    let mut p2: usize = match (&s[p1..]).find(',') {
        Some(p) => p + p1,
        _ => end,
    };
    while p2 != end {
        v.push( (&s[p1..p2]).parse().unwrap() );
        p1 = p2 + 1;
        p2 = match (&s[p1..]).find(',') {
            Some(p) => p + p1,
            _ => end,
        };
    }
    v.push( (&s[p1..p2]).parse().unwrap() );
}


#[inline]
pub fn vec_from_string_no_bracket<T>(v: &mut Vec<T>, s: &str)
where 
    T: FromStr + std::fmt::Debug, 
    <T as FromStr>::Err: Debug,
{
    let mut p1: usize = 0;
    let end: usize = s.len();
    let mut p2: usize = match (&s[p1..]).find(',') {
        Some(p) => p,
        _ => end,
    };
    while p2 != end {
        v.push( (&s[p1..p2]).parse().unwrap() );
        p1 = p2 + 1;
        p2 = match (&s[p1..]).find(',') {
            Some(p) => p + p1,
            _ => end,
        };
    }
    v.push( (&s[p1..p2]).parse().unwrap() );
}