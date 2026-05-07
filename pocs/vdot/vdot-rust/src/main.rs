use std::ffi::c_void;
use std::time::Instant;

use llama_vdot_rust::{
    dot_q4_0_f32, dot_q4_0_q8_0, dot_q4_1_f32, fill_random_gaussian, quantize_row_q8_0_reference,
    BlockQ4_0, BlockQ4_1, BlockQ8_0, BlockQ8_1, Rng, TimingStats, QK4_0, QK4_1, QK8_0,
};

const K_VEC_SIZE: usize = 1 << 18;
const GGML_TYPE_Q4_0: i32 = 2;
const GGML_TYPE_Q4_1: i32 = 3;

type GgmlFromFloat = unsafe extern "C" fn(*const f32, *mut c_void, i64);
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
    from_float: GgmlFromFloat,
    vec_dot: GgmlVecDot,
    vec_dot_type: i32,
    nrows: i64,
}

extern "C" {
    fn ggml_cpu_init();
    fn ggml_get_type_traits_cpu(type_: i32) -> *const GgmlTypeTraitsCpu;
    fn ggml_row_size(type_: i32, ne: i64) -> usize;
}

fn main() {
    unsafe {
        ggml_cpu_init();
    }

    let args = std::env::args().collect::<Vec<_>>();
    let nloop = args
        .get(1)
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(10);
    let scalar = args.get(2).and_then(|v| v.parse::<i32>().ok()).unwrap_or(0) != 0;
    let use_q4_1 = args.get(3).and_then(|v| v.parse::<i32>().ok()).unwrap_or(0) != 0;

    if scalar && use_q4_1 {
        println!("It is not possible to use Q4_1 quantization and scalar implementations");
        std::process::exit(1);
    }

    let ggml_type = if use_q4_1 {
        GGML_TYPE_Q4_1
    } else {
        GGML_TYPE_Q4_0
    };
    let funcs_cpu = unsafe { ggml_get_type_traits_cpu(ggml_type) };
    if funcs_cpu.is_null() {
        eprintln!("failed to get ggml CPU type traits");
        std::process::exit(1);
    }

    let mut rng = Rng::new(1234);
    let mut x1 = vec![0.0_f32; K_VEC_SIZE];
    let mut y1 = vec![0.0_f32; K_VEC_SIZE];
    let n4 = align64(if use_q4_1 {
        K_VEC_SIZE / QK4_1
    } else {
        K_VEC_SIZE / QK4_0
    });
    let n8 = align64(K_VEC_SIZE / QK8_0);
    let mut q40 = vec![
        BlockQ4_0 {
            d: 0,
            qs: [0; QK4_0 / 2]
        };
        if use_q4_1 { 0 } else { n4 }
    ];
    let mut q41 = vec![
        BlockQ4_1 {
            d: 0,
            m: 0,
            qs: [0; QK4_1 / 2]
        };
        if use_q4_1 { n4 } else { 0 }
    ];
    let mut q8_scalar = vec![
        BlockQ8_0 {
            d: 0,
            qs: [0; QK8_0]
        };
        n8
    ];
    let mut q80_ggml = vec![
        BlockQ8_0 {
            d: 0,
            qs: [0; QK8_0]
        };
        if use_q4_1 { 0 } else { n8 }
    ];
    let mut q81_ggml = vec![
        BlockQ8_1 {
            d: 0,
            s: 0,
            qs: [0; QK8_0]
        };
        if use_q4_1 { n8 } else { 0 }
    ];
    let _expected_vec_dot_row_size =
        unsafe { ggml_row_size((*funcs_cpu).vec_dot_type, K_VEC_SIZE as i64) };

    let mut sumt = TimingStats::default();
    let mut sumqt = TimingStats::default();
    let mut sum = 0.0_f64;
    let mut sumq = 0.0_f64;
    let mut exact_sum = 0.0_f64;

    for _ in 0..nloop {
        fill_random_gaussian(&mut x1, &mut rng, 0.0);
        fill_random_gaussian(&mut y1, &mut rng, 0.0);
        exact_sum += x1
            .iter()
            .zip(&y1)
            .map(|(x, y)| f64::from(x * y))
            .sum::<f64>();

        unsafe {
            if use_q4_1 {
                ((*funcs_cpu).from_float)(x1.as_ptr(), q41.as_mut_ptr().cast(), K_VEC_SIZE as i64);
            } else {
                ((*funcs_cpu).from_float)(x1.as_ptr(), q40.as_mut_ptr().cast(), K_VEC_SIZE as i64);
            }
        }

        let t1 = Instant::now();
        if use_q4_1 {
            sum += dot_q4_1_f32(&q41[..K_VEC_SIZE / QK4_1], &y1);
        } else {
            sum += dot_q4_0_f32(&q40[..K_VEC_SIZE / QK4_0], &y1);
        }
        sumt.add(t1.elapsed().as_nanos() as f64 * 1.0e-3);

        let t1 = Instant::now();
        let mut result = 0.0_f32;
        unsafe {
            if scalar {
                quantize_row_q8_0_reference(&y1, &mut q8_scalar);
                result = dot_q4_0_q8_0(K_VEC_SIZE, &q40, &q8_scalar);
            } else {
                let vdot = ggml_get_type_traits_cpu((*funcs_cpu).vec_dot_type);
                let y_ptr = if use_q4_1 {
                    ((*vdot).from_float)(
                        y1.as_ptr(),
                        q81_ggml.as_mut_ptr().cast(),
                        K_VEC_SIZE as i64,
                    );
                    q81_ggml.as_ptr().cast()
                } else {
                    ((*vdot).from_float)(
                        y1.as_ptr(),
                        q80_ggml.as_mut_ptr().cast(),
                        K_VEC_SIZE as i64,
                    );
                    q80_ggml.as_ptr().cast()
                };
                let x_ptr = if use_q4_1 {
                    q41.as_ptr().cast()
                } else {
                    q40.as_ptr().cast()
                };
                ((*funcs_cpu).vec_dot)(K_VEC_SIZE as i32, &mut result, 0, x_ptr, 0, y_ptr, 0, 1);
            }
        }
        sumq += f64::from(result);
        sumqt.add(t1.elapsed().as_nanos() as f64 * 1.0e-3);
    }

    let nloop_f = f64::from(nloop);
    sum /= nloop_f;
    sumq /= nloop_f;
    exact_sum /= nloop_f;
    println!("Exact result: <dot> = {exact_sum}");
    println!("<dot> = {sum}, {sumq}");
    let (mean, stddev) = sumt.mean_stddev(nloop_f);
    println!(
        "time = {mean} +/- {stddev} us. maxt = {max} us",
        max = sumt.max
    );
    let (meanq, stddevq) = sumqt.mean_stddev(nloop_f);
    println!(
        "timeq = {meanq} +/- {stddevq} us. maxt = {maxq} us",
        maxq = sumqt.max
    );
}

fn align64(value: usize) -> usize {
    64 * value.div_ceil(64)
}
