extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::parse_macro_input;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::token::Comma;
use syn::Expr;


#[proc_macro]
pub fn parse_cmd_args(_input: TokenStream) -> TokenStream {
    TokenStream::from(
        quote! {
            let mut cmd_parser = CmdParser::new();
        }
    )
}


#[proc_macro]
pub fn unknown_cmd_args(_input: TokenStream) -> TokenStream {
    TokenStream::from(
        quote! {
            if !cmd_parser.args.is_empty() {
                print!("Warning: unknown cmd arguments: ");
                for key in cmd_parser.args.keys() {
                    print!("'{}', ", key);
                }
                println!();
            }
        }
    )
}

struct CmdArg {
    data: Punctuated<Expr, Comma>,
}


impl Parse for CmdArg {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let parsed_input = 
            Punctuated::<Expr, Comma>::parse_terminated(input)?;
        Ok( CmdArg{data: parsed_input} )
    }
}


#[proc_macro]
pub fn cmd_arg(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as CmdArg);

    if parsed.data.len() == 3 { // immut, opt
        let token_var = &parsed.data[0];
        let token_var_str = token_var.to_token_stream().to_string();
        let token_type = &parsed.data[1];
        let token_val = &parsed.data[2];
        TokenStream::from(
            quote! {
                let #token_var: #token_type = match cmd_parser.args.get(#token_var_str) {
                    None => #token_val,
                    Some(s) => s.parse().unwrap(),
                };
                cmd_parser.args.remove(#token_var_str);
            }
        )
    } else if parsed.data.len() == 2 { // immut, req
        let token_var = &parsed.data[0];
        let token_var_str = token_var.to_token_stream().to_string();
        let token_type = &parsed.data[1];
        TokenStream::from(
            quote! {
                let #token_var: #token_type = cmd_parser.args.get(#token_var_str).expect(
                    format!("Error: variable '{}' should have user input.", #token_var_str).as_str()
                ).parse().unwrap();
                cmd_parser.args.remove(#token_var_str);
            }
        )
    } else  {
        panic!("The number of arguments into cmd_arg! should be 2 or 3")
    }
}


#[proc_macro]
pub fn cmd_arg_mut(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as CmdArg);

    if parsed.data.len() == 3 { // mut, opt
        let token_var = &parsed.data[0];
        let token_var_str = token_var.to_token_stream().to_string();
        let token_type = &parsed.data[1];
        let token_val = &parsed.data[2];
        TokenStream::from(
            quote! {
                let mut #token_var: #token_type = match cmd_parser.args.get(#token_var_str) {
                    None => #token_val,
                    Some(s) => s.parse().unwrap(),
                };
                cmd_parser.args.remove(#token_var_str);
            }
        )
    } else if parsed.data.len() == 2 { // mut, req
        let token_var = &parsed.data[0];
        let token_var_str = token_var.to_token_stream().to_string();
        let token_type = &parsed.data[1];
        TokenStream::from(
            quote! {
                let mut #token_var: #token_type = cmd_parser.args.get(#token_var_str).expect(
                    format!("Error: variable '{}' should have user input.", #token_var_str).as_str()
                ).parse().unwrap();
                cmd_parser.args.remove(#token_var_str);
            }
        )
    } else  {
        panic!("The number of arguments into cmd_arg_mut! should be 2 or 3")
    }
}


#[proc_macro]
pub fn cmd_vec(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as CmdArg);

    if parsed.data.len() == 3 { // immut, opt
        let token_var = &parsed.data[0];
        let token_var_str = token_var.to_token_stream().to_string();
        let token_type = &parsed.data[1];
        let token_val = &parsed.data[2];
        TokenStream::from(
            quote! {
                let #token_var: Vec<#token_type> = match cmd_parser.args.get(#token_var_str) {
                    None => #token_val,
                    Some(s) => string_to_vec::<#token_type>(s),
                };
                cmd_parser.args.remove(#token_var_str);
            }
        )
    } else if parsed.data.len() == 2 { // immut, req
        let token_var = &parsed.data[0];
        let token_var_str = token_var.to_token_stream().to_string();
        let token_type = &parsed.data[1];
        TokenStream::from(
            quote! {
                let #token_var: Vec<#token_type> = string_to_vec::<#token_type>(
                    cmd_parser.args.get(#token_var_str).expect(
                        format!("Error: variable '{}' should have user input.", #token_var_str).as_str()
                    )
                );
                cmd_parser.args.remove(#token_var_str);
            }
        )
    } else  {
        panic!("The number of arguments into cmd_vec! should be 2 or 3")
    }
}


#[proc_macro]
pub fn cmd_vec_mut(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as CmdArg);

    if parsed.data.len() == 3 { // mut, opt
        let token_var = &parsed.data[0];
        let token_var_str = token_var.to_token_stream().to_string();
        let token_type = &parsed.data[1];
        let token_val = &parsed.data[2];
        TokenStream::from(
            quote! {
                let mut #token_var: Vec<#token_type> = match cmd_parser.args.get(#token_var_str) {
                    None => #token_val,
                    Some(s) => string_to_vec::<#token_type>(s),
                };
                cmd_parser.args.remove(#token_var_str);
            }
        )
    } else if parsed.data.len() == 2 { // mut, req
        let token_var = &parsed.data[0];
        let token_var_str = token_var.to_token_stream().to_string();
        let token_type = &parsed.data[1];
        TokenStream::from(
            quote! {
                let mut #token_var: Vec<#token_type> = string_to_vec::<#token_type>(
                    cmd_parser.args.get(#token_var_str).expect(
                        format!("Error: variable '{}' should have user input.", #token_var_str).as_str()
                    )
                );
                cmd_parser.args.remove(#token_var_str);
            }
        )
    } else  {
        panic!("The number of arguments into cmd_vec_mut! should be 2 or 3")
    }
}