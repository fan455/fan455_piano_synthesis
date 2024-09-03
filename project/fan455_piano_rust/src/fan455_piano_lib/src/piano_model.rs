use fan455_arrf64::*;
use fan455_math_array::*;
use fan455_math_scalar::*;
use fan455_util_macro::{sta_mat, dyn_mat};
//use fan455_util::{assert_multi_eq, elem, mzip};
use super::piano_fem_1d::*;
use super::piano_fem_3d::*;


pub struct WaveModel {
    // A 3D acoustic wave model in the air, with 3 variables u, v, w (acoustic velocity in 3 directions).
    pub mass_co: f64,
}

impl WaveModel
{
    #[inline]
    pub fn new( sound_speed: f64 ) -> Self {
        Self {mass_co: 1./sound_speed.powi(2)}
    }

    #[inline] #[allow(non_snake_case)]
    pub fn compute_mass_mat_entry( 
        &self, mut i_kind: usize, mut j_kind: usize, // kinds of test and basis functions
        mut i: usize, mut j: usize, // i and j are local indices of dof
        mesh: &PianoAirMesh, 
        mbuf: &PianoAirMeshBuf,
    ) -> Option<f64> {

        macro_rules! quad {
            ($f1:expr, $f2:expr) => {
                mesh.iso[0].quad2($f1, $f2, &mbuf.map.jac_det)
            };
        }
        macro_rules! f { ($i:ident) => { mbuf.b0.f.col($i).sl() }; }
        macro_rules! g { ($i:ident) => { mbuf.b1.f.col($i).sl() }; }
        macro_rules! h { ($i:ident) => { mbuf.b2.f.col($i).sl() }; }

        if i_kind < j_kind {
            std::mem::swap(&mut i_kind, &mut j_kind);
            std::mem::swap(&mut i, &mut j);
        }

        match (i_kind, j_kind) {
            (0, 0) => Some( self.mass_co * quad!(f![i], f![j]) ),
            (1, 1) => Some( self.mass_co * quad!(g![i], g![j]) ),
            (2, 2) => Some( self.mass_co * quad!(h![i], h![j]) ),
            _ => None,
        }
    }


    #[inline] #[allow(non_snake_case)]
    pub fn compute_stiff_mat_entry( 
        &self, mut i_kind: usize, mut j_kind: usize, // kinds of test and basis functions
        mut i: usize, mut j: usize, // i and j are local indices of dof
        mesh: &PianoAirMesh, 
        mbuf: &PianoAirMeshBuf,
    ) -> Option<f64> {

        macro_rules! quad {
            ($f1:expr, $f2:expr) => {
                mesh.iso[0].quad2($f1, $f2, &mbuf.map.jac_det)
            };
        }
        macro_rules! fx { ($i:ident) => { mbuf.b0.fx.col($i).sl() }; }
        macro_rules! gy { ($i:ident) => { mbuf.b1.fy.col($i).sl() }; }
        macro_rules! hz { ($i:ident) => { mbuf.b2.fz.col($i).sl() }; }
    
        if i_kind < j_kind {
            std::mem::swap(&mut i_kind, &mut j_kind);
            std::mem::swap(&mut i, &mut j);
        }

        match (i_kind, j_kind) {
            (0, 0) => Some( quad!(fx![i], fx![j]) ),
            (1, 1) => Some( quad!(gy![i], gy![j]) ),
            (2, 2) => Some( quad!(hz![i], hz![j]) ),
            (1, 0) => Some( quad!(fx![j], gy![i]) ),
            (2, 0) => Some( quad!(fx![j], hz![i]) ),
            (2, 1) => Some( quad!(gy![j], hz![i]) ),
            _ => None,
        }
    }
}


pub struct PlateModel {
    // A 3D plate model.
    pub density: f64,
    pub tension_co: [f64; 2],
    pub stiff_co: [f64; 13], // 13 non-zero entries in a 6*6 symmetric matrix.
}

impl PlateModel
{
    #[inline]
    pub fn new( density: f64, tension_co: [f64; 2], stiff_co: [f64; 13] ) -> Self {
        Self {density, tension_co, stiff_co}
    }

    #[inline] #[allow(non_snake_case)]
    pub fn compute_tension_co_rotated(
        tension: [f64; 2], // Tension in the orthotropic axes.
        angle: f64, // The global axes rotates angle to become the orthotropic axes.
    ) -> [f64; 2] {
        let (s, c) = angle.sin_cos();
        mat_vec_2(&sta_mat!(
            [c, -s],
            [s,  c],
        ), &tension)
    }

    #[inline] #[allow(non_snake_case)]
    pub fn compute_stiff_co_rotated(
        young_modulus: [f64; 3],
        shear_modulus: [f64; 3],
        poisson_ratio: [f64; 3],
        shear_correct: [f64; 2],
        angle: f64, // The global axes rotates angle to become the orthotropic axes.
    ) -> [f64; 13] {
        let [E_x, E_y, E_z] = young_modulus;
        let [G_xy, mut G_xz, mut G_yz] = shear_modulus;
        let [nu_xy, nu_xz, nu_yz] = poisson_ratio;
        let [k_xz, k_yz] = shear_correct;
        G_xz *= k_xz;
        G_yz *= k_yz;

        let C1 = sta_mat!(
            [    1./E_x, -nu_xy/E_x, -nu_xz/E_x],
            [-nu_xy/E_x,     1./E_y, -nu_yz/E_y],
            [-nu_xz/E_x, -nu_yz/E_y,     1./E_z],
        );
        let D1 = get_mat_inv_3(&C1);
        let D2 = Arr2::from_vec(dyn_mat!(
            [D1[0], D1[3], D1[6], 0.  , 0.  , 0.  ],
            [D1[1], D1[4], D1[7], 0.  , 0.  , 0.  ],
            [D1[2], D1[5], D1[8], 0.  , 0.  , 0.  ],
            [0.   , 0.   , 0.   , G_xy, 0.  , 0.  ],
            [0.   , 0.   , 0.   , 0.  , G_xz, 0.  ],
            [0.   , 0.   , 0.   , 0.  , 0.  , G_yz],
        ), 6, 6);

        let (s, c) = angle.sin_cos();
        let T = Arr2::from_vec(dyn_mat!(
            [c*c    , s*s   , 0., s*c    , 0., 0.],
            [s*s    , c*c   , 0., -s*c   , 0., 0.],
            [0.     , 0.    , 1., 0.     , 0., 0.],
            [-2.*s*c, 2.*s*c, 0., c*c-s*s, 0., 0.],
            [0.     , 0.    , 0., 0.     , c , s ],
            [0.     , 0.    , 0., 0.     , -s, c ],
        ), 6, 6);

        let mut D = Arr2::<f64>::new(6, 6);
        let mut D_ = Arr2::<f64>::new(6, 6);
        dgemm(1., &T, &D2, 0., &mut D_, TRANS, NO_TRANS);
        dgemm(1., &D_, &T, 0., &mut D, NO_TRANS, NO_TRANS);
        [
            D[(0, 0)], D[(1, 0)], D[(2, 0)], D[(3, 0)], 
            D[(1, 1)], D[(2, 1)], D[(3, 1)], 
            D[(2, 2)], D[(3, 2)], 
            D[(3, 3)], 
            D[(4, 4)], D[(5, 4)], 
            D[(5, 5)],
        ]
    }

    #[inline] #[allow(non_snake_case)]
    pub fn compute_mass_mat_entry( 
        &self, mut i_kind: usize, mut j_kind: usize, // kinds of test and basis functions
        mut i: usize, mut j: usize, // i and j are local indices of dof
        mesh: &PianoSoundboardMesh, 
        mbuf: &PianoSoundboardMeshBuf,
    ) -> Option<f64> {

        macro_rules! quad {
            ($f1:expr, $f2:expr) => {
                self.density * mesh.iso[0].quad2($f1, $f2, &mbuf.map.jac_det)
            };
        }
        macro_rules! f { ($i:ident) => { mbuf.b0.f.col($i).sl() }; }
        macro_rules! g { ($i:ident) => { mbuf.b1.f.col($i).sl() }; }
        macro_rules! h { ($i:ident) => { mbuf.b2.f.col($i).sl() }; }

        if i_kind < j_kind {
            std::mem::swap(&mut i_kind, &mut j_kind);
            std::mem::swap(&mut i, &mut j);
        }

        match (i_kind, j_kind) {
            (0, 0) => Some( quad!(f![i], f![j]) ),
            (1, 1) => Some( quad!(g![i], g![j]) ),
            (2, 2) => Some( quad!(h![i], h![j]) ),
            _ => None,
        }
    }


    #[inline] #[allow(non_snake_case)]
    pub fn compute_stiff_mat_entry( 
        &self, mut i_kind: usize, mut j_kind: usize, // kinds of test and basis functions
        mut i: usize, mut j: usize, // i and j are local indices of dof
        mesh: &PianoSoundboardMesh, 
        mbuf: &PianoSoundboardMeshBuf,
    ) -> Option<f64> {

        let [T_x, T_y] = self.tension_co;
        let [D_11, D_21, D_31, D_41, D_22, D_32, D_42, D_33, D_43, D_44, D_55, D_65, D_66] = self.stiff_co;

        macro_rules! quad {
            ($f1:expr, $f2:expr) => {
                mesh.iso[0].quad2($f1, $f2, &mbuf.map.jac_det)
            };
        }
        macro_rules! fx { ($i:ident) => { mbuf.b0.fx.col($i).sl() }; }
        macro_rules! fy { ($i:ident) => { mbuf.b0.fy.col($i).sl() }; }
        macro_rules! fz { ($i:ident) => { mbuf.b0.fz.col($i).sl() }; }

        macro_rules! gx { ($i:ident) => { mbuf.b1.fx.col($i).sl() }; }
        macro_rules! gy { ($i:ident) => { mbuf.b1.fy.col($i).sl() }; }
        macro_rules! gz { ($i:ident) => { mbuf.b1.fz.col($i).sl() }; }

        macro_rules! hx { ($i:ident) => { mbuf.b2.fx.col($i).sl() }; }
        macro_rules! hy { ($i:ident) => { mbuf.b2.fy.col($i).sl() }; }
        macro_rules! hz { ($i:ident) => { mbuf.b2.fz.col($i).sl() }; }
    
        if i_kind < j_kind {
            std::mem::swap(&mut i_kind, &mut j_kind);
            std::mem::swap(&mut i, &mut j);
        }

        match (i_kind, j_kind) {
            (0, 0) => Some( 
                D_11*quad!(fx![i], fx![j]) + D_41*quad!(fx![i], fy![j]) + D_41*quad!(fx![j], fy![i]) + 
                D_44*quad!(fy![i], fy![j]) + D_55*quad!(fz![i], fz![j]) + 
                T_x *quad!(fx![i], fx![j]) + T_y *quad!(fy![i], fy![j])
            ),
            (1, 1) => Some( 
                D_22*quad!(gy![i], gy![j]) + D_42*quad!(gx![i], gy![j]) + D_42*quad!(gx![j], gy![i]) + 
                D_44*quad!(gx![i], gx![j]) + D_66*quad!(gz![i], gz![j]) + 
                T_x *quad!(gx![i], gx![j]) + T_y *quad!(gy![i], gy![j])
            ),
            (2, 2) => Some( 
                D_33*quad!(hz![i], hz![j]) + D_55*quad!(hx![i], hx![j]) + D_65*quad!(hx![i], hy![j]) + 
                D_65*quad!(hx![j], hy![i]) + D_66*quad!(hy![i], hy![j]) + 
                T_x *quad!(hx![i], hx![j]) + T_y *quad!(hy![i], hy![j])
            ),
            (1, 0) => Some( 
                D_21*quad!(fx![j], gy![i]) + D_41*quad!(fx![j], gx![i]) + D_42*quad!(fy![j], gy![i]) + 
                D_44*quad!(fy![j], gx![i]) + D_65*quad!(fz![j], gz![i])
            ),
            (2, 0) => Some( 
                D_31*quad!(fx![j], hz![i]) + D_43*quad!(fy![j], hz![i]) +  
                D_55*quad!(fz![j], hx![i]) + D_65*quad!(fz![j], hy![i])
            ),
            (2, 1) => Some( 
                D_32*quad!(gy![j], hz![i]) + D_43*quad!(gx![j], hz![i]) +  
                D_65*quad!(gz![j], hx![i]) + D_66*quad!(gz![j], hy![i])
            ),
            _ => None,
        }
    }
}


pub struct StringModel {
    // Displacement field:
    // u(x, y, t) = u(x, t) - y * alpha(x, t) - z * beta(x, t)
    // v(x, y, t) = v(x, t)
    // w(x, y, t) = w(x, t)
    pub density: f64,
    pub tension: f64,
    pub area: f64, // pi * r^2
    pub inertia: f64, // pi * r^4 / 4
    pub stiff_co: [f64; 3], // E_x, G_xy, G_xz
}

impl StringModel
{
    #[inline]
    pub fn new(
        diameter: f64,
        density: f64, 
        tension: f64, 
        stiff_co: [f64; 3], 
    ) -> Self {
        let r = diameter / 2.;
        let area = PI * r.powi(2);
        let inertia = PI * r.powi(4) / 4.;
        Self { density, tension, area, inertia, stiff_co }
    }


    #[inline] #[allow(non_snake_case)]
    pub fn compute_mass_mat_entry( 
        &self, mut i_kind: usize, mut j_kind: usize, // kinds of test and basis functions
        mut i: usize, mut j: usize, // i and j are local indices of dof
        mesh: &PianoStringMesh, 
        mbuf: &PianoStringMeshBuf,
    ) -> Option<f64> {

        let rho = self.density;
        let A = self.area;
        let I = self.inertia;
        let det = mbuf.map.x_co[0];

        macro_rules! quad {
            ($f1:expr, $f2:expr) => {
                mesh.iso.0.quad2($f1, $f2, det)
            };
        }
        macro_rules! f_u { ($i:ident) => { mbuf.b0.f.col($i).sl() }; }
        macro_rules! f_v { ($i:ident) => { mbuf.b1.f.col($i).sl() }; }
        macro_rules! f_w { ($i:ident) => { mbuf.b2.f.col($i).sl() }; }
        macro_rules! f_a { ($i:ident) => { mbuf.b3.f.col($i).sl() }; }
        macro_rules! f_b { ($i:ident) => { mbuf.b4.f.col($i).sl() }; }
    
        if i_kind < j_kind {
            std::mem::swap(&mut i_kind, &mut j_kind);
            std::mem::swap(&mut i, &mut j);
        }

        match (i_kind, j_kind) {
            (0, 0) => Some( rho*A * quad!(f_u![i], f_u![j]) ),
            (1, 1) => Some( rho*A * quad!(f_v![i], f_v![j]) ),
            (2, 2) => Some( rho*A * quad!(f_w![i], f_w![j]) ),
            (3, 3) => Some( rho*I * quad!(f_a![i], f_a![j]) ),
            (4, 4) => Some( rho*I * quad!(f_b![i], f_b![j]) ),
            _ => None,
        }
    }


    #[inline] #[allow(non_snake_case)]
    pub fn compute_stiff_mat_entry( 
        &self, mut i_kind: usize, mut j_kind: usize, // kinds of test and basis functions
        mut i: usize, mut j: usize, // i and j are local indices of dof
        mesh: &PianoStringMesh, 
        mbuf: &PianoStringMeshBuf,
    ) -> Option<f64> {

        let [D1, D2, D3] = self.stiff_co;
        let T = self.tension;
        let A = self.area;
        let I = self.inertia;
        let det = mbuf.map.x_co[0];

        macro_rules! quad {
            ($f1:expr, $f2:expr) => {
                mesh.iso.0.quad2($f1, $f2, det)
            };
        }
        macro_rules! f_a { ($i:ident) => { mbuf.b3.f.col($i).sl() }; }
        macro_rules! f_b { ($i:ident) => { mbuf.b4.f.col($i).sl() }; }

        macro_rules! fx_u { ($i:ident) => { mbuf.b0.fx.col($i).sl() }; }
        macro_rules! fx_v { ($i:ident) => { mbuf.b1.fx.col($i).sl() }; }
        macro_rules! fx_w { ($i:ident) => { mbuf.b2.fx.col($i).sl() }; }
        macro_rules! fx_a { ($i:ident) => { mbuf.b3.fx.col($i).sl() }; }
        macro_rules! fx_b { ($i:ident) => { mbuf.b4.fx.col($i).sl() }; }
    
        if i_kind < j_kind {
            std::mem::swap(&mut i_kind, &mut j_kind);
            std::mem::swap(&mut i, &mut j);
        }

        match (i_kind, j_kind) {
            (0, 0) => Some( 
                D1*A * quad!(fx_u![i], fx_u![j]) + 
                T*A  * quad!(fx_u![i], fx_u![j]) 
            ),
            (1, 1) => Some( 
                D2*A * quad!(fx_v![i], fx_v![j]) + 
                T*A  * quad!(fx_v![i], fx_v![j]) 
            ),
            (2, 2) => Some( 
                D3*A * quad!(fx_w![i], fx_w![j]) + 
                T*A  * quad!(fx_w![i], fx_w![j]) 
            ),
            (3, 3) => Some( 
                D1*I * quad!(fx_a![i], fx_a![j]) + D2*A * quad!(f_a![i], f_a![j]) + 
                T*I  * quad!(fx_a![i], fx_a![j]) 
            ),
            (4, 4) => Some( 
                D1*I * quad!(fx_b![i], fx_b![j]) + D3*A * quad!(f_b![i], f_b![j]) +
                T*I  * quad!(fx_b![i], fx_b![j]) 
            ),
            (3, 1) => Some( 
                -D2*A * quad!(f_a![i], fx_v![j]) 
            ),
            (4, 2) => Some( 
                -D3*A * quad!(f_b![i], fx_w![j]) 
            ),
            _ => None,
        }
    }
}
