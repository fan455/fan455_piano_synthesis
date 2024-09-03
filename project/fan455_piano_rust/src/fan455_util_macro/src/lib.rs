extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::parse_macro_input;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::token::Comma;
use syn::{Expr, DeriveInput};
use std::iter::zip;


#[proc_macro_derive(StaBinStruct)] #[allow(non_snake_case)]
pub fn derive_StaBinStruct(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_StaBinStruct(&ast)
}

#[proc_macro_derive(DynBinStruct, attributes(dyn_field))] #[allow(non_snake_case)]
pub fn derive_DynBinStruct(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_DynBinStruct(&ast)
}


#[allow(non_snake_case)]
fn impl_StaBinStruct(ast: &DeriveInput) -> TokenStream {
    let struct_name = &ast.ident;
    let fields = match &ast.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(fields_) => &fields_.named,
            _ => panic!("Not support unnamed fields."),
        },
        _ => panic!("This is not a struct"),
    };
    let fields_n = fields.len();
    let mut idx = Vec::<usize>::with_capacity(fields_n);
    let mut names = Vec::<syn::Ident>::with_capacity(fields_n);
    let mut types = Vec::<syn::Type>::with_capacity(fields_n);

    for (i, field) in zip(0..fields_n, fields.iter()) {
        idx.push(i);
        types.push(field.ty.clone());
        names.push(field.ident.clone().unwrap());
    }
    quote! {
        impl StaBinStruct<#fields_n> for #struct_name
        {
            const SIZES: [usize; #fields_n] = [#(<#types>::STA_BIN_SIZE),*];
            
            #[inline]
            fn struct_to_bytes( &self, 
                sta_bytes: &mut [u8], sta_pos: &[usize; #fields_n],
            ) {
                #(
                    self.#names.to_bytes_sl( &mut sta_bytes[sta_pos[#idx]..sta_pos[#idx]+Self::SIZES[#idx]] );
                )*
            }
            
            #[inline]
            fn bytes_to_struct( 
                sta_bytes: &[u8], sta_pos: &[usize; #fields_n],
            ) -> Self {
                #(
                    let #names = #types::from_bytes_sl( &sta_bytes[sta_pos[#idx]..sta_pos[#idx]+Self::SIZES[#idx]] );
                )*
                Self {#(#names),*}
            }
        }
    }.into()
}


#[allow(non_snake_case)]
fn impl_DynBinStruct(ast: &DeriveInput) -> TokenStream {
    let struct_name = &ast.ident;
    let fields = match &ast.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(fields_) => &fields_.named,
            _ => panic!("Not support unnamed fields."),
        },
        _ => panic!("This is not a struct"),
    };
    let fields_n = fields.len();
    let mut dyn_n: usize = 0;
    let mut names = Vec::<syn::Ident>::with_capacity(fields_n);
    let mut types = Vec::<syn::Type>::with_capacity(fields_n);

    let mut tk0 = Vec::<proc_macro2::TokenStream>::with_capacity(fields_n);
    let mut tk1 = Vec::<proc_macro2::TokenStream>::with_capacity(fields_n);
    let mut tk2 = Vec::<proc_macro2::TokenStream>::with_capacity(fields_n);

    for (i, field) in zip(0..fields_n, fields.iter()) {

        let name = field.ident.clone().unwrap();
        let ty = field.ty.clone();
        
        if field.attrs.len() > 0 {
            let meta = match &field.ty {
                syn::Type::Path(meta_) => meta_.path.segments[0].clone(),
                _ => panic!("Not support"),
            };
            let dyn_type = meta.ident.to_string();
            if dyn_type == "Vec" {
                let ty_ = match meta.arguments {
                    syn::PathArguments::AngleBracketed(ty__) => match &ty__.args[0] {
                        syn::GenericArgument::Type(ty___) => ty___.clone(),
                        _ => panic!("Not support."),
                    }
                    _ => panic!("Not support."),
                };
                tk0.push(quote!{
                    std::mem::size_of::<#ty_>()*self.#name.len()
                });
                tk1.push(quote! {
                    {
                        let [j0, j1] = dyn_pos[#dyn_n];
                        vec_to_bytes(&self.#name, &mut dyn_bytes[j0..j1]);
                    }
                });
                tk2.push(quote! {
                    let #name = {
                        let [j0, j1] = dyn_pos[#dyn_n];
                        let mut #name: Vec<#ty_> = vec![<#ty_>::default(); (j1-j0)/std::mem::size_of::<#ty_>()];
                        bytes_to_vec(&dyn_bytes[j0..j1], &mut #name);
                        #name
                    };
                });
            } else {
                panic!("Not support this dynamic field type.");
            }
            dyn_n += 1;
        } else {
            tk1.push(quote! { 
                self.#name.to_bytes_sl( &mut sta_bytes[sta_pos[#i]..sta_pos[#i]+Self::SIZES[#i]] ); 
            });
            tk2.push(quote! { 
                let #name = <#ty>::from_bytes_sl( &sta_bytes[sta_pos[#i]..sta_pos[#i]+Self::SIZES[#i]] ); 
            });
        }
        names.push(name);
        types.push(ty);
    }
    quote! {
        impl DynBinStruct<#fields_n, #dyn_n> for #struct_name
        {
            const SIZES: [usize; #fields_n] = [#(<#types>::STA_BIN_SIZE),*];

            #[inline]
            fn dyn_sizes( &self ) -> [usize; #dyn_n] { [#(#tk0),*] }
            
            #[inline]
            fn struct_to_bytes( &self, 
                sta_bytes: &mut [u8], sta_pos: &[usize; #fields_n], 
                dyn_bytes: &mut [u8], dyn_pos: &[[usize; 2]; #dyn_n],
            ) {
                #(#tk1)*
            }
            
            #[inline]
            fn bytes_to_struct( 
                sta_bytes: &[u8], sta_pos: &[usize; #fields_n], 
                dyn_bytes: &[u8], dyn_pos: &[[usize; 2]; #dyn_n],
            ) -> Self {
                #(#tk2)*
                Self {#(#names),*}
            }
        }
    }.into()
}


struct TypesList {
    data: Punctuated<syn::Type, Comma>,
}


impl Parse for TypesList {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let parsed_input = 
            Punctuated::<syn::Type, Comma>::parse_terminated(input)?;
        Ok( TypesList{data: parsed_input} )
    }
}


#[proc_macro] #[allow(non_snake_case)]
pub fn impl_ToBytesSl_for(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as TypesList);
    let ty = parsed.data.iter();
    quote! {
        #(
            impl ToBytesSl for #ty
            {
                #[inline]
                fn to_bytes_sl( &self, bytes: &mut [u8] ) {
                    bytes.copy_from_slice(&self.to_le_bytes())
                }
            }
        )*
    }.into()
}


#[proc_macro] #[allow(non_snake_case)]
pub fn impl_FromBytesSl_for(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as TypesList);
    let ty = parsed.data.iter();
    quote! {
        #(
            impl FromBytesSl for #ty
            {
                #[inline]
                fn from_bytes_sl( bytes: &[u8] ) -> Self {
                    #ty::from_le_bytes(<[u8; std::mem::size_of::<#ty>()]>::try_from(bytes).unwrap())
                }
            }
        )*
    }.into()
}


#[proc_macro] #[allow(non_snake_case)]
pub fn impl_ToBytesSl_for_array(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as TypesList);
    let ty = parsed.data.iter();
    quote! {
        #(
            impl<const N: usize> ToBytesSl for [#ty; N]
            {
                #[inline]
                fn to_bytes_sl( &self, bytes: &mut [u8] ) {
                    for elem!(val_, bytes_) in mzip!(self.iter(), bytes.chunks_exact_mut( std::mem::size_of::<#ty>() )) {
                        bytes_.copy_from_slice(&val_.to_le_bytes());
                    }
                }
            }
        )*
    }.into()
}


#[proc_macro] #[allow(non_snake_case)]
pub fn impl_FromBytesSl_for_array(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as TypesList);
    let ty = parsed.data.iter();
    quote! {
        #(
            impl<const N: usize> FromBytesSl for [#ty; N]
            {           
                #[inline]
                fn from_bytes_sl( bytes: &[u8] ) -> Self {
                    let mut arr: Self = [#ty::default(); N];
                    for elem!(arr_, bytes_) in mzip!(arr.iter_mut(), bytes.chunks_exact( std::mem::size_of::<#ty>() )) {
                        *arr_ = #ty::from_le_bytes(<[u8; std::mem::size_of::<#ty>()]>::try_from(bytes_).unwrap());
                    }
                    arr
                }
            }
        )*
    }.into()
}


struct MatElems {
    data: Punctuated<syn::ExprArray, Comma>,
}

impl Parse for MatElems {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let parsed_input = 
            Punctuated::<syn::ExprArray, Comma>::parse_terminated(input)?;
        Ok( MatElems{data: parsed_input} )
    }
}

#[proc_macro]
pub fn sta_mat(input: TokenStream) -> TokenStream {
    // Create a static-sized column-major matrix (flattened array) using row-major expression, return as array.
    let data = parse_macro_input!(input as MatElems).data;
    let nrow = data.len();
    let ncol = data[0].elems.len(); // Better assert each has equal ncol.
    for i in 1..nrow {
        assert_eq!(data[i].elems.len(), ncol, "ncol not equal.");
    }
    let size = nrow*ncol;
    let mut mat = Vec::<Expr>::with_capacity(size);
    for j in 0..ncol {
        for i in 0..nrow {
            mat.push(data[i].elems[j].clone());
        }
    }
    quote! { [#(#mat),*] }.into()
}

#[proc_macro]
pub fn dyn_mat(input: TokenStream) -> TokenStream {
    // Create a static-sized column-major matrix (flattened array) using row-major expression, return as array.
    let data = parse_macro_input!(input as MatElems).data;
    let nrow = data.len();
    let ncol = data[0].elems.len(); // Better assert each has equal ncol.
    for i in 1..nrow {
        assert_eq!(data[i].elems.len(), ncol, "ncol not equal.");
    }
    let size = nrow*ncol;
    let mut mat = Vec::<Expr>::with_capacity(size);
    for j in 0..ncol {
        for i in 0..nrow {
            mat.push(data[i].elems[j].clone());
        }
    }
    quote! { vec![#(#mat),*] }.into()
}

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