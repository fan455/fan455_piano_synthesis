use fan455_arrf64::*;
//use fan455_math_array::*;
use fan455_math_scalar::*;
//use fan455_util::{assert_multi_eq, elem, mzip};
use super::piano_finite_element::*;




pub struct PlateModel {
    pub density: f64,
    pub stiff_co: [f64; 9],
    pub quad_z0: Vec<f64>, // (quad_n,), integrate z^0 in a thickness range; thickness may vary by points.
    pub quad_z1: Vec<f64>, // (quad_n,), integrate z^1 in a thickness range; thickness may vary by points.
    pub quad_z2: Vec<f64>, // (quad_n,), integrate z^2 in a thickness range; thickness may vary by points.

    pub quad_z1_is_zero: bool,
}


impl PlateModel
{
    #[inline]
    pub fn new( density: f64, stiff_co: [f64; 9], quad_n: usize ) -> Self {
        let quad_z0: Vec<f64> = vec![0.; quad_n];
        let quad_z1: Vec<f64> = vec![0.; quad_n];
        let quad_z2: Vec<f64> = vec![0.; quad_n];

        Self {density, stiff_co, quad_z0, quad_z1, quad_z2, quad_z1_is_zero: false}
    }

    #[inline] #[allow(non_snake_case)]
    pub fn compute_stiff_coef_rotated(
        young_modulus: [f64; 2],
        shear_modulus: [f64; 3],
        poisson_ratio: f64,
        shear_correct: [f64; 2],
        angle: f64, // The global axes rotates angle to become the orthotropic axes.
    ) -> [f64; 9] {

        let [E_x, E_y] = young_modulus;
        let [G_xy, G_xz, G_yz] = shear_modulus;
        let nu_xy = poisson_ratio;
        let nu_yx = nu_xy * E_y / E_x;
        let [k_x, k_y] = shear_correct;

        let mut D1 = SArr2::<f64, 3, 3, 9>::new(); // symmetric
        let mut D2 = SArr2::<f64, 2, 2, 4>::new(); // symmetric

        *D1.idxm2(0, 0) = E_x / (1.- nu_xy * nu_yx);
        *D1.idxm2(1, 0) = E_x * nu_yx / (1.- nu_xy * nu_yx);

        *D1.idxm2(0, 1) = E_x * nu_yx / (1.- nu_xy * nu_yx);
        *D1.idxm2(1, 1) = E_y / (1.- nu_xy * nu_yx);

        *D1.idxm2(2, 2) = G_xy;

        *D2.idxm2(0, 0) = k_x * G_xz;
        *D2.idxm2(1, 1) = k_y * G_yz;

        let (s, c) = angle.sin_cos();
        let mut A1 = SArr2::<f64, 3, 3, 9>::new(); // rotation matrix
        let mut A2 = SArr2::<f64, 2, 2, 4>::new(); // rotation matrix

        *A1.idxm2(0, 0) = c*c;
        *A1.idxm2(1, 0) = s*s;
        *A1.idxm2(2, 0) = -2.*c*s;

        *A1.idxm2(0, 1) = s*s;
        *A1.idxm2(1, 1) = c*c;
        *A1.idxm2(2, 1) = 2.*c*s;

        *A1.idxm2(0, 2) = c*s;
        *A1.idxm2(1, 2) = -c*s;
        *A1.idxm2(2, 2) = c*c-s*s;

        *A2.idxm2(0, 0) = c;
        *A2.idxm2(1, 0) = -s;
        
        *A2.idxm2(0, 1) = s;
        *A2.idxm2(1, 1) = c;

        let mut D1_ = D1.clone();
        let mut D2_ = D2.clone();

        dgemm(1., &A1, &D1, 0., &mut D1_, TRANS, NO_TRANS); // A1.T * D1 -> D1_
        dgemm(1., &D1_, &A1, 0., &mut D1, NO_TRANS, NO_TRANS); // D1_ * A1 -> D1

        dgemm(1., &A2, &D2, 0., &mut D2_, TRANS, NO_TRANS); // A2.T * D2 -> D2_
        dgemm(1., &D2_, &A2, 0., &mut D2, NO_TRANS, NO_TRANS); // D2_ * A2 -> D2

        [
            D1[(0, 0)], D1[(1, 0)], D1[(2, 0)], 
            D1[(1, 1)], D1[(2, 1)], D1[(2, 2)], 
            D2[(0, 0)], D2[(1, 0)], D2[(1, 1)],
        ]
    }

    #[inline]
    pub fn compute_quad_z_with_const_thick( 
        &mut self, a: f64, b: f64, symmetric: bool,
    ) { // symmetric should be true if a + b = 0
        self.quad_z0.fill(b-a);
        if symmetric {
            self.quad_z1_is_zero = true;
            self.quad_z1.fill(0.);
        } else {
            self.quad_z1.fill( (b.powi(2)-a.powi(2))/2.);
        }
        self.quad_z2.fill( (b.powi(3)-a.powi(3))/3. );
    }


    #[inline]
    pub fn compute_mass_mat_entry( 
        &self, mut i_kind: usize, mut j_kind: usize, // kinds of test and basis functions
        mut i: usize, mut j: usize, // i and j are local indices of dof
        mesh: &PianoSoundboardMesh, 
        mbuf: &PianoSoundboardMeshBuf,
    ) -> MayBeZero {

        //println!("mass entry:  i_kind = {i_kind}, j_kind = {j_kind}, i = {i}, j = {j}");

        macro_rules! quad {
            ($f1:expr, $f2:expr, $f3:expr) => {
                mesh.iso[0].quad3($f1, $f2, $f3, mbuf.map.jac_det)
            };
        }
        macro_rules! f  { ($i:ident) => { mbuf.b0.f. col($i).sl() }; }
        macro_rules! g  { ($i:ident) => { mbuf.b1.f. col($i).sl() }; }
        macro_rules! h  { ($i:ident) => { mbuf.b2.f. col($i).sl() }; }

        macro_rules! z0 { () => { self.quad_z0.sl() }; }
        macro_rules! z1 { () => { self.quad_z1.sl() }; }
        macro_rules! z2 { () => { self.quad_z2.sl() }; }
    
        let rho = self.density;
        if i_kind < j_kind {
            std::mem::swap(&mut i_kind, &mut j_kind);
            std::mem::swap(&mut i, &mut j);
        }

        match (i_kind, j_kind) {
            (0, 0) => {
                MayBeZero::NonZero( rho * quad!( f![i], f![j], z0![] ) )
            },
            (1, 1) | (2, 2) => {
                MayBeZero::NonZero( rho * quad!( g![i], g![j], z2![] ) )
            },
            (3, 3) | (4, 4) => {
                MayBeZero::NonZero( rho * quad!( h![i], h![j], z0![] ) )
            },
            (3, 1) | (4, 2) => {
                if self.quad_z1_is_zero {
                    MayBeZero::Zero
                } else {
                    MayBeZero::NonZero( -rho * quad!( g![j], h![i], z1![] ) )
                }
            },
            _ => MayBeZero::Zero,
        }
    }


    #[inline] #[allow(non_snake_case)]
    pub fn compute_stiff_mat_entry( 
        &self, mut i_kind: usize, mut j_kind: usize, // kinds of test and basis functions
        mut i: usize, mut j: usize, // i and j are local indices of dof
        mesh: &PianoSoundboardMesh, 
        mbuf: &PianoSoundboardMeshBuf,
    ) -> MayBeZero {

        //println!("stiff entry:  i_kind = {i_kind}, j_kind = {j_kind}, i = {i}, j = {j}");

        macro_rules! quad {
            ($f1:expr, $f2:expr, $f3:expr) => {
                mesh.iso[0].quad3($f1, $f2, $f3, mbuf.map.jac_det)
            };
        }
        //macro_rules! f  { ($i:ident) => { mbuf.b0.f. col($i).sl() }; }
        macro_rules! fx { ($i:ident) => { mbuf.b0.fx.col($i).sl() }; }
        macro_rules! fy { ($i:ident) => { mbuf.b0.fy.col($i).sl() }; }
        macro_rules! g  { ($i:ident) => { mbuf.b1.f. col($i).sl() }; }
        macro_rules! gx { ($i:ident) => { mbuf.b1.fx.col($i).sl() }; }
        macro_rules! gy { ($i:ident) => { mbuf.b1.fy.col($i).sl() }; }
       //macro_rules! h  { ($i:ident) => { mbuf.b2.f. col($i).sl() }; }
        macro_rules! hx { ($i:ident) => { mbuf.b2.fx.col($i).sl() }; }
        macro_rules! hy { ($i:ident) => { mbuf.b2.fy.col($i).sl() }; }

        macro_rules! z0 { () => { self.quad_z0.sl() }; }
        macro_rules! z1 { () => { self.quad_z1.sl() }; }
        macro_rules! z2 { () => { self.quad_z2.sl() }; }

        let [D1, D2, D3, D4, D5, D6, D7, D8, D9] = self.stiff_co;
        if i_kind < j_kind {
            std::mem::swap(&mut i_kind, &mut j_kind);
            std::mem::swap(&mut i, &mut j);
        }

        match (i_kind, j_kind) {
            (0, 0) => {
                MayBeZero::NonZero(
                    D7 * quad!( fx![i], fx![j], z0![] ) + D8 * quad!( fx![i], fy![j], z0![] ) + 
                    D8 * quad!( fx![j], fy![i], z0![] ) + D9 * quad!( fy![i], fy![j], z0![] )
                )
            },
            (1, 0) => {
                MayBeZero::NonZero(
                    -D7 * quad!( fx![j], g![i], z0![] ) - D8 * quad!( fy![j], g![i], z0![] )
                )
            },
            (2, 0) => {
                MayBeZero::NonZero(
                    -D8 * quad!( fx![j], g![i], z0![] ) - D9 * quad!( fy![j], g![i], z0![] )
                )
            },
            (1, 1) => {
                MayBeZero::NonZero(
                    D7 * quad!( g![i], g![j], z0![] ) + 
                    D1 * quad!( gx![i], gx![j], z2![] ) + D3 * quad!( gx![i], gy![j], z2![] ) + 
                    D3 * quad!( gx![j], gy![i], z2![] ) + D6 * quad!( gy![i], gy![j], z2![] )
                )
            },
            (2, 1) => {
                MayBeZero::NonZero(
                    D8 * quad!( g![i], g![j], z0![] ) + 
                    D2 * quad!( gx![j], gy![i], z2![] ) + D3 * quad!( gx![i], gx![j], z2![] ) + 
                    D5 * quad!( gy![i], gy![j], z2![] ) + D6 * quad!( gx![i], gy![j], z2![] )
                )
            },
            (3, 1) => {
                if self.quad_z1_is_zero {
                    MayBeZero::Zero
                } else {
                    MayBeZero::NonZero(
                        -D1 * quad!( gx![j], hx![i], z1![] ) - D3 * quad!( gx![j], hy![i], z1![] )
                        -D3 * quad!( gy![j], hx![i], z1![] ) - D6 * quad!( gy![j], hy![i], z1![] )
                    )
                }
            },
            (4, 1) => {
                if self.quad_z1_is_zero {
                    MayBeZero::Zero
                } else {
                    MayBeZero::NonZero(
                        -D2 * quad!( gx![j], hx![i], z1![] ) - D3 * quad!( gx![j], hy![i], z1![] )
                        -D5 * quad!( gy![j], hy![i], z1![] ) - D6 * quad!( gy![j], hx![i], z1![] )
                    )
                }
            },
            (2, 2) => {
                MayBeZero::NonZero(
                    D9 * quad!( g![i], g![j], z0![] ) + 
                    D4 * quad!( gy![i], gy![j], z2![] ) + D5 * quad!( gx![i], gy![j], z2![] ) + 
                    D5 * quad!( gx![j], gy![i], z2![] ) + D6 * quad!( gx![i], gx![j], z2![] )
                )
            },
            (3, 2) => {
                if self.quad_z1_is_zero {
                    MayBeZero::Zero
                } else {
                    MayBeZero::NonZero(
                        -D2 * quad!( gy![j], hx![i], z1![] ) - D3 * quad!( gx![j], hx![i], z1![] ) 
                        -D5 * quad!( gy![j], hy![i], z1![] ) - D6 * quad!( gx![j], hy![i], z1![] )
                    )
                }
            },
            (4, 2) => {
                if self.quad_z1_is_zero {
                    MayBeZero::Zero
                } else {
                    MayBeZero::NonZero(
                        -D4 * quad!( gy![j], hy![i], z1![] ) - D5 * quad!( gx![j], hy![i], z1![] )
                        -D5 * quad!( gy![j], hx![i], z1![] ) - D6 * quad!( gx![j], hx![i], z1![] )
                    )
                }
            },
            (3, 3) => {
                MayBeZero::NonZero(
                    D1 * quad!( hx![i], hx![j], z0![] ) + D3 * quad!( hx![i], hy![j], z0![] ) + 
                    D3 * quad!( hx![j], hy![i], z0![] ) + D6 * quad!( hy![i], hy![j], z0![] )
                )
            },
            (4, 3) => {
                MayBeZero::NonZero(
                    D2 * quad!( hx![j], hy![i], z0![] ) + D3 * quad!( hx![i], hx![j], z0![] ) + 
                    D5 * quad!( hy![i], hy![j], z0![] ) + D6 * quad!( hx![i], hy![j], z0![] )
                )
            },
            (4, 4) => {
                MayBeZero::NonZero(
                    D4 * quad!( hy![i], hy![j], z0![] ) + D5 * quad!( hx![i], hy![j], z0![] ) + 
                    D5 * quad!( hx![j], hy![i], z0![] ) + D6 * quad!( hx![i], hx![j], z0![] )
                )
            },
            _ => MayBeZero::Zero,
        }
    }
}


