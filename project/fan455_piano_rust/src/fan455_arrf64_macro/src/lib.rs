extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::parse_macro_input;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::token::Comma;
use syn::Expr;
use std::collections::HashSet;


const NO_TRANS: i8 = 78_i8;
const TRANS: i8 = 84_i8;
const UPPER: i8 = 85_i8;
const LOWER: i8 = 76_i8;
const UNIT: i8 = 85_i8;
const NON_UNIT: i8 = 78_i8;
const LEFT: i8 = 76_i8;
const RIGHT: i8 = 82_i8;

/*#[cfg(not(feature="use_32bit_float"))] #[allow(non_camel_case_types)]
type fsize = f64;

#[cfg(feature="use_32bit_float")] #[allow(non_camel_case_types)]
type fsize = f32;*/


struct BlasArgs<const SEP: usize> {
    all: Punctuated<Expr, Comma>,
    opt: HashSet<String>,
}


impl<const SEP: usize> Parse for BlasArgs<SEP> {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let parsed_input = 
            Punctuated::<Expr, Comma>::parse_terminated(input)?;
        let mut opt: HashSet<String> = HashSet::new();

        if SEP == parsed_input.len() {
        } else if SEP < parsed_input.len() {
            opt.reserve(parsed_input.len() - SEP);
            for i in SEP..parsed_input.len() {
                opt.insert( parsed_input[i].to_token_stream().to_string() );
            }
        } else {
            panic!("Not enough arguments.");
        }
        Ok( BlasArgs{all: parsed_input, opt} )
    }
}



#[proc_macro]
pub fn dposv(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as BlasArgs<2>);

    let a = &args.all[0];
    let b = &args.all[1];
    let mut uplo: i8 = LOWER;

    if args.opt.contains("upper") { uplo = UPPER; }
    
    TokenStream::from(
        quote! {
            dposv(#a, #b, #uplo);
        }
    )
}


#[proc_macro]
pub fn dpotri(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as BlasArgs<1>);

    let a = &args.all[0];
    let mut uplo: i8 = LOWER;

    if args.opt.contains("upper") { uplo = UPPER; }
    
    TokenStream::from(
        quote! {
            dpotri(#a, #uplo);
        }
    )
}


#[proc_macro]
pub fn dpotrf(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as BlasArgs<1>);

    let a = &args.all[0];
    let mut uplo: i8 = LOWER;

    if args.opt.contains("upper") { uplo = UPPER; }
    
    TokenStream::from(
        quote! {
            dpotrf(#a, #uplo);
        }
    )
}

#[proc_macro]
pub fn dtrmm(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as BlasArgs<3>);

    let alpha = &args.all[0];
    let a = &args.all[1];
    let b = &args.all[2];
    let mut side: i8 = LEFT;
    let mut trans: i8 = NO_TRANS;
    let mut uplo: i8 = LOWER;
    let mut diag: i8 = NON_UNIT;

    if args.opt.contains("right") { side = RIGHT; }
    if args.opt.contains("trans") { trans = TRANS; }
    if args.opt.contains("upper") { uplo = UPPER; }
    if args.opt.contains("unit") { diag = UNIT; }

    TokenStream::from(
        quote! {
            dtrmm(#alpha, #a, #b, #side, #trans, #uplo, #diag);
        }
    )
}

#[proc_macro]
pub fn dtrsm(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as BlasArgs<3>);

    let alpha = &args.all[0];
    let a = &args.all[1];
    let b = &args.all[2];
    let mut side: i8 = LEFT;
    let mut trans: i8 = NO_TRANS;
    let mut uplo: i8 = LOWER;
    let mut diag: i8 = NON_UNIT;

    if args.opt.contains("right") { side = RIGHT; }
    if args.opt.contains("trans") { trans = TRANS; }
    if args.opt.contains("upper") { uplo = UPPER; }
    if args.opt.contains("unit") { diag = UNIT; }

    TokenStream::from(
        quote! {
            dtrsm(#alpha, #a, #b, #side, #trans, #uplo, #diag);
        }
    )
}

#[proc_macro]
pub fn dsyrk(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as BlasArgs<4>);

    let alpha = &args.all[0];
    let a = &args.all[1];
    let beta = &args.all[2];
    let c = &args.all[3];
    let mut trans: i8 = NO_TRANS;
    let mut uplo: i8 = LOWER;

    if args.opt.contains("trans") { trans = TRANS; }
    if args.opt.contains("upper") { uplo = UPPER; }
    
    if trans == NO_TRANS {
        TokenStream::from(
            quote! {
                dsyrk_notrans(#alpha, #a, #beta, #c, #trans, #uplo);
            }
        )
    } else {
        TokenStream::from(
            quote! {
                dsyrk_trans(#alpha, #a, #beta, #c, #trans, #uplo);
            }
        )
    }
}

#[proc_macro]
pub fn dsymm(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as BlasArgs<5>);

    let alpha = &args.all[0];
    let a = &args.all[1];
    let b = &args.all[2];
    let beta = &args.all[3];
    let c = &args.all[4];
    let mut side: i8 = LEFT;
    let mut uplo: i8 = LOWER;

    if args.opt.contains("right") { side = RIGHT; }
    if args.opt.contains("upper") { uplo = UPPER; }
    
    TokenStream::from(
        quote! {
            dsymm(#alpha, #a, #b, #beta, #c, #side, #uplo);
        }
    )
}

#[proc_macro]
pub fn dgemm(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as BlasArgs<5>);

    let alpha = &args.all[0];
    let a = &args.all[1];
    let b = &args.all[2];
    let beta = &args.all[3];
    let c = &args.all[4];
    let mut transa: i8 = NO_TRANS;
    let mut transb: i8 = NO_TRANS;

    if args.opt.contains("transa") { transa = TRANS; }
    if args.opt.contains("transb") { transb = TRANS; }
    
    if transa == NO_TRANS {
        TokenStream::from(
            quote! {
                dgemm_notransa(#alpha, #a, #b, #beta, #c, #transa, #transb);
            }
        )
    } else {
        TokenStream::from(
            quote! {
                dgemm_transa(#alpha, #a, #b, #beta, #c, #transa, #transb);
            }
        )
    }
}

#[proc_macro]
pub fn dsyr(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as BlasArgs<3>);

    let alpha = &args.all[0];
    let x = &args.all[1];
    let a = &args.all[2];
    let mut uplo: i8 = LOWER;

    if args.opt.contains("upper") { uplo = UPPER; }
    
    TokenStream::from(
        quote! {
            dsyr(#alpha, #x, #a, #uplo);
        }
    )
}

#[proc_macro]
pub fn dgemv(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as BlasArgs<5>);

    let alpha = &args.all[0];
    let a = &args.all[1];
    let x = &args.all[2];
    let beta = &args.all[3];
    let y = &args.all[4];
    let mut trans: i8 = NO_TRANS;

    if args.opt.contains("trans") { trans = TRANS; }
    
    TokenStream::from(
        quote! {
            dgemv(#alpha, #a, #x, #beta, #y, #trans);
        }
    )
}

#[proc_macro]
pub fn dsymv(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as BlasArgs<5>);

    let alpha = &args.all[0];
    let a = &args.all[1];
    let x = &args.all[2];
    let beta = &args.all[3];
    let y = &args.all[4];
    let mut uplo: i8 = LOWER;

    if args.opt.contains("upper") { uplo = UPPER; }
    
    TokenStream::from(
        quote! {
            dsymv(#alpha, #a, #x, #beta, #y, #uplo);
        }
    )
}

#[proc_macro]
pub fn dtrmv(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as BlasArgs<2>);

    let a = &args.all[0];
    let x = &args.all[1];
    let mut trans: i8 = NO_TRANS;
    let mut uplo: i8 = LOWER;
    let mut diag: i8 = NON_UNIT;

    if args.opt.contains("trans") { trans = TRANS; }
    if args.opt.contains("upper") { uplo = UPPER; }
    if args.opt.contains("unit") { diag = UNIT; }
    
    TokenStream::from(
        quote! {
            dtrmv(#a, #x, #trans, #uplo, #diag);
        }
    )
}