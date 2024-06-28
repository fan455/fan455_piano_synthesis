fn main() {

    #[cfg(all(feature="x86-64-windows", feature="mkl"))]
    {
        println!("cargo:rustc-link-search=native=C:/Program Files (x86)/Intel/oneAPI/mkl/2024.2/lib");// You need to set this path to your mkl lib path.
        println!("cargo:rustc-link-lib=static=mkl_intel_ilp64");
        println!("cargo:rustc-link-lib=static=mkl_sequential");
        println!("cargo:rustc-link-lib=static=mkl_core");
    }
    
    #[cfg(all(feature="x86-64-windows", feature="openblas"))]
    {
        println!("cargo:rustc-link-search=native=D:/sofs/openblas/lib"); // You need to set this path to your openblas lib path.
        println!("cargo:rustc-link-lib=static=libopenblas");
    }
}