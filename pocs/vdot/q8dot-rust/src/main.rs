use llama_q8dot_rust::{
    fill_q4_0, fill_q4_1, fill_q8_0, simple_dot_q4_0, simple_dot_q4_1, BlockQ4_0, BlockQ4_1,
    BlockQ8_0, Rng, Stat, QK4_1,
};
use std::ffi::c_void;
use std::time::Instant;

const K_VEC_SIZE: usize = 1 << 16;
const GGML_TYPE_Q4_0: i32 = 2;
const GGML_TYPE_Q4_1: i32 = 3;

type GgmlVecDot = unsafe extern "C" fn(
    n: i32,
    s: *mut f32,
    bs: usize,
    x: *const c_void,
    bx: usize,
    y: *const c_void,
    by: usize,
    nrc: i32,
);

#[repr(C)]
struct GgmlTypeTraitsCpu {
    from_float: *const c_void,
    vec_dot: GgmlVecDot,
    vec_dot_type: i32,
    nrows: i64,
}

extern "C" {
    fn ggml_get_type_traits_cpu(type_: i32) -> *const GgmlTypeTraitsCpu;
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let nloop = args
        .get(1)
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(10);
    let type_ = args.get(2).and_then(|v| v.parse::<i32>().ok()).unwrap_or(1);

    let mut rng = Rng::new(1234);
    let mut x41 = Vec::<BlockQ4_1>::new();
    let mut x40 = Vec::<BlockQ4_0>::new();
    let mut y = vec![
        BlockQ8_0 {
            d: 0.0,
            s: 0.0,
            qs: [0; 32],
        };
        K_VEC_SIZE
    ];

    if type_ == 0 {
        x40.resize(
            K_VEC_SIZE,
            BlockQ4_0 {
                d: 0.0,
                qs: [0; 16],
            },
        );
    } else {
        x41.resize(
            K_VEC_SIZE,
            BlockQ4_1 {
                d: 0.0,
                m: 1.0,
                qs: [0; 16],
            },
        );
    }

    let ggml_type = if type_ == 0 {
        GGML_TYPE_Q4_0
    } else {
        GGML_TYPE_Q4_1
    };
    let funcs = unsafe { ggml_get_type_traits_cpu(ggml_type) };
    if funcs.is_null() {
        eprintln!("failed to get ggml CPU type traits");
        std::process::exit(1);
    }

    let mut simple = Stat::default();
    let mut ggml = Stat::default();

    for iloop in 0..nloop {
        if type_ == 0 {
            fill_q4_0(&mut x40, &mut rng);
        } else {
            fill_q4_1(&mut x41, &mut rng);
        }
        fill_q8_0(&mut y, &mut rng);

        let t1 = Instant::now();
        let s = if type_ == 0 {
            x40.iter()
                .zip(&y)
                .map(|(x, y)| f64::from(simple_dot_q4_0(x, y)))
                .sum()
        } else {
            x41.iter()
                .zip(&y)
                .map(|(x, y)| f64::from(simple_dot_q4_1(x, y)))
                .sum()
        };
        let t = t1.elapsed().as_nanos() as f64 * 1.0e-3;
        if iloop > 3 {
            simple.add_result(s, t);
        }

        let t1 = Instant::now();
        let mut fs = 0.0_f32;
        unsafe {
            let x_ptr = if type_ == 0 {
                x40.as_ptr() as *const c_void
            } else {
                x41.as_ptr() as *const c_void
            };
            ((*funcs).vec_dot)(
                (K_VEC_SIZE * QK4_1) as i32,
                &mut fs,
                0,
                x_ptr,
                0,
                y.as_ptr() as *const c_void,
                0,
                1,
            );
        }
        let t = t1.elapsed().as_nanos() as f64 * 1.0e-3;
        if iloop > 3 {
            ggml.add_result(f64::from(fs), t);
        }
    }

    simple.report("Simple");
    ggml.report("ggml");
}
