use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::process::ExitCode;
use std::ptr;

use llama_gguf_rust::{tensor_shape, usage, DeterministicRng, GGML_MAX_DIMS};

const GGML_TYPE_F32: c_int = 0;
const GGUF_TYPE_INT16: c_int = 3;
const GGUF_TYPE_FLOAT32: c_int = 6;
const LC_NUMERIC: c_int = 1;

#[repr(C)]
struct ggml_context {
    _private: [u8; 0],
}

#[repr(C)]
struct ggml_backend_buffer {
    _private: [u8; 0],
}

#[repr(C)]
struct gguf_context {
    _private: [u8; 0],
}

#[repr(C)]
struct ggml_init_params {
    mem_size: usize,
    mem_buffer: *mut c_void,
    no_alloc: bool,
}

#[repr(C)]
struct gguf_init_params {
    no_alloc: bool,
    ctx: *mut *mut ggml_context,
}

#[repr(C)]
struct ggml_tensor {
    type_: c_int,
    buffer: *mut ggml_backend_buffer,
    ne: [i64; 4],
    nb: [usize; 4],
    op: c_int,
    op_params: [i32; 16],
    flags: i32,
    src: [*mut ggml_tensor; 10],
    view_src: *mut ggml_tensor,
    view_offs: usize,
    data: *mut c_void,
    name: [c_char; 64],
    extra: *mut c_void,
    padding: [c_char; 8],
}

extern "C" {
    fn setlocale(category: c_int, locale: *const c_char) -> *mut c_char;

    fn ggml_init(params: ggml_init_params) -> *mut ggml_context;
    fn ggml_free(ctx: *mut ggml_context);
    fn ggml_new_tensor(
        ctx: *mut ggml_context,
        type_: c_int,
        n_dims: c_int,
        ne: *const i64,
    ) -> *mut ggml_tensor;
    fn ggml_set_name(tensor: *mut ggml_tensor, name: *const c_char) -> *mut ggml_tensor;
    fn ggml_nelements(tensor: *const ggml_tensor) -> i64;
    fn ggml_get_tensor(ctx: *mut ggml_context, name: *const c_char) -> *mut ggml_tensor;
    fn ggml_n_dims(tensor: *const ggml_tensor) -> c_int;
    fn ggml_get_mem_size(ctx: *const ggml_context) -> usize;
    fn ggml_type_name(type_: c_int) -> *const c_char;
    fn ggml_type_size(type_: c_int) -> usize;

    fn gguf_init_empty() -> *mut gguf_context;
    fn gguf_init_from_file(fname: *const c_char, params: gguf_init_params) -> *mut gguf_context;
    fn gguf_free(ctx: *mut gguf_context);
    fn gguf_get_version(ctx: *const gguf_context) -> u32;
    fn gguf_get_alignment(ctx: *const gguf_context) -> usize;
    fn gguf_get_data_offset(ctx: *const gguf_context) -> usize;
    fn gguf_get_n_kv(ctx: *const gguf_context) -> i64;
    fn gguf_get_key(ctx: *const gguf_context, key_id: i64) -> *const c_char;
    fn gguf_find_key(ctx: *const gguf_context, key: *const c_char) -> i64;
    fn gguf_get_val_str(ctx: *const gguf_context, key_id: i64) -> *const c_char;
    fn gguf_get_n_tensors(ctx: *const gguf_context) -> i64;
    fn gguf_get_tensor_name(ctx: *const gguf_context, tensor_id: i64) -> *const c_char;
    fn gguf_get_tensor_size(ctx: *const gguf_context, tensor_id: i64) -> usize;
    fn gguf_get_tensor_offset(ctx: *const gguf_context, tensor_id: i64) -> usize;
    fn gguf_get_tensor_type(ctx: *const gguf_context, tensor_id: i64) -> c_int;
    fn gguf_set_val_u8(ctx: *mut gguf_context, key: *const c_char, val: u8);
    fn gguf_set_val_i8(ctx: *mut gguf_context, key: *const c_char, val: i8);
    fn gguf_set_val_u16(ctx: *mut gguf_context, key: *const c_char, val: u16);
    fn gguf_set_val_i16(ctx: *mut gguf_context, key: *const c_char, val: i16);
    fn gguf_set_val_u32(ctx: *mut gguf_context, key: *const c_char, val: u32);
    fn gguf_set_val_i32(ctx: *mut gguf_context, key: *const c_char, val: i32);
    fn gguf_set_val_f32(ctx: *mut gguf_context, key: *const c_char, val: f32);
    fn gguf_set_val_u64(ctx: *mut gguf_context, key: *const c_char, val: u64);
    fn gguf_set_val_i64(ctx: *mut gguf_context, key: *const c_char, val: i64);
    fn gguf_set_val_f64(ctx: *mut gguf_context, key: *const c_char, val: f64);
    fn gguf_set_val_bool(ctx: *mut gguf_context, key: *const c_char, val: bool);
    fn gguf_set_val_str(ctx: *mut gguf_context, key: *const c_char, val: *const c_char);
    fn gguf_set_arr_data(
        ctx: *mut gguf_context,
        key: *const c_char,
        type_: c_int,
        data: *const c_void,
        n: usize,
    );
    fn gguf_set_arr_str(
        ctx: *mut gguf_context,
        key: *const c_char,
        data: *const *const c_char,
        n: usize,
    );
    fn gguf_add_tensor(ctx: *mut gguf_context, tensor: *const ggml_tensor);
    fn gguf_write_to_file(ctx: *const gguf_context, fname: *const c_char, only_meta: bool) -> bool;
}

fn main() -> ExitCode {
    unsafe {
        let locale = CString::new("C").unwrap();
        setlocale(LC_NUMERIC, locale.as_ptr());
    }

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() < 3 {
        print!(
            "{}",
            usage(args.first().map(String::as_str).unwrap_or("llama-gguf"))
        );
        return Err("bad arguments".to_string());
    }

    let fname = args[1].clone();
    let mode = args[2].as_str();
    let check_data = args.len() != 4;

    match mode {
        "w" => gguf_ex_write(&fname),
        "r" => {
            gguf_ex_read_0(&fname)?;
            gguf_ex_read_1(&fname, check_data)
        }
        _ => Err("mode must be r or w".to_string()),
    }
}

fn gguf_ex_write(fname: &str) -> Result<(), String> {
    unsafe {
        let ctx = gguf_init_empty();
        if ctx.is_null() {
            return Err("gguf_init_empty failed".to_string());
        }

        set_scalar_kv(ctx)?;

        let arr_i16 = [1_i16, 2, 3, 4];
        gguf_set_arr_data(
            ctx,
            cstring("some.parameter.arr.i16")?.as_ptr(),
            GGUF_TYPE_INT16,
            arr_i16.as_ptr().cast(),
            arr_i16.len(),
        );
        let arr_f32 = [3.145_f32, 2.718, 1.414];
        gguf_set_arr_data(
            ctx,
            cstring("some.parameter.arr.f32")?.as_ptr(),
            GGUF_TYPE_FLOAT32,
            arr_f32.as_ptr().cast(),
            arr_f32.len(),
        );
        let strings = [cstring("hello")?, cstring("world")?, cstring("!")?];
        let string_ptrs = strings.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        gguf_set_arr_str(
            ctx,
            cstring("some.parameter.arr.str")?.as_ptr(),
            string_ptrs.as_ptr(),
            string_ptrs.len(),
        );

        let ctx_data = ggml_init(ggml_init_params {
            mem_size: 128 * 1024 * 1024,
            mem_buffer: ptr::null_mut(),
            no_alloc: false,
        });
        if ctx_data.is_null() {
            gguf_free(ctx);
            return Err("ggml_init failed".to_string());
        }

        let mut rng = DeterministicRng::new(123456);
        for i in 0..10 {
            let name = cstring(&format!("tensor_{i}"))?;
            let (ne, n_dims) = tensor_shape(&mut rng);
            let cur = ggml_new_tensor(ctx_data, GGML_TYPE_F32, n_dims, ne.as_ptr());
            ggml_set_name(cur, name.as_ptr());

            let n_elements = ggml_nelements(cur) as usize;
            let data = std::slice::from_raw_parts_mut((*cur).data.cast::<f32>(), n_elements);
            for value in data {
                *value = 100.0 + i as f32;
            }

            gguf_add_tensor(ctx, cur);
        }

        let c_fname = cstring(fname)?;
        if !gguf_write_to_file(ctx, c_fname.as_ptr(), false) {
            ggml_free(ctx_data);
            gguf_free(ctx);
            return Err(format!("failed to write gguf file {fname}"));
        }
        println!("gguf_ex_write: wrote file '{fname};");

        ggml_free(ctx_data);
        gguf_free(ctx);
        Ok(())
    }
}

unsafe fn set_scalar_kv(ctx: *mut gguf_context) -> Result<(), String> {
    gguf_set_val_u8(ctx, cstring("some.parameter.uint8")?.as_ptr(), 0x12);
    gguf_set_val_i8(ctx, cstring("some.parameter.int8")?.as_ptr(), -0x13);
    gguf_set_val_u16(ctx, cstring("some.parameter.uint16")?.as_ptr(), 0x1234);
    gguf_set_val_i16(ctx, cstring("some.parameter.int16")?.as_ptr(), -0x1235);
    gguf_set_val_u32(ctx, cstring("some.parameter.uint32")?.as_ptr(), 0x12345678);
    gguf_set_val_i32(ctx, cstring("some.parameter.int32")?.as_ptr(), -0x12345679);
    gguf_set_val_f32(
        ctx,
        cstring("some.parameter.float32")?.as_ptr(),
        0.123456789,
    );
    gguf_set_val_u64(
        ctx,
        cstring("some.parameter.uint64")?.as_ptr(),
        0x123456789abcdef0,
    );
    gguf_set_val_i64(
        ctx,
        cstring("some.parameter.int64")?.as_ptr(),
        -0x123456789abcdef1,
    );
    gguf_set_val_f64(
        ctx,
        cstring("some.parameter.float64")?.as_ptr(),
        0.1234567890123456789,
    );
    gguf_set_val_bool(ctx, cstring("some.parameter.bool")?.as_ptr(), true);
    gguf_set_val_str(
        ctx,
        cstring("some.parameter.string")?.as_ptr(),
        cstring("hello world")?.as_ptr(),
    );
    Ok(())
}

fn gguf_ex_read_0(fname: &str) -> Result<(), String> {
    unsafe {
        let c_fname = cstring(fname)?;
        let ctx = gguf_init_from_file(
            c_fname.as_ptr(),
            gguf_init_params {
                no_alloc: false,
                ctx: ptr::null_mut(),
            },
        );
        if ctx.is_null() {
            return Err(format!("gguf_ex_read_0: failed to load '{fname}'"));
        }

        print_header("gguf_ex_read_0", ctx);
        print_kv("gguf_ex_read_0", ctx);
        find_string_kv("gguf_ex_read_0", ctx)?;
        print_tensor_info("gguf_ex_read_0", ctx, false);

        gguf_free(ctx);
        Ok(())
    }
}

fn gguf_ex_read_1(fname: &str, check_data: bool) -> Result<(), String> {
    unsafe {
        let c_fname = cstring(fname)?;
        let mut ctx_data: *mut ggml_context = ptr::null_mut();
        let ctx = gguf_init_from_file(
            c_fname.as_ptr(),
            gguf_init_params {
                no_alloc: false,
                ctx: &mut ctx_data,
            },
        );
        if ctx.is_null() {
            return Err(format!("gguf_ex_read_1: failed to load '{fname}'"));
        }

        print_header("gguf_ex_read_1", ctx);
        print_kv("gguf_ex_read_1", ctx);
        print_tensor_info("gguf_ex_read_1", ctx, true);
        print_tensor_data("gguf_ex_read_1", ctx, ctx_data, check_data)?;

        println!(
            "gguf_ex_read_1: ctx_data size: {}",
            ggml_get_mem_size(ctx_data)
        );
        ggml_free(ctx_data);
        gguf_free(ctx);
        Ok(())
    }
}

unsafe fn print_header(prefix: &str, ctx: *const gguf_context) {
    println!("{prefix}: version:      {}", gguf_get_version(ctx));
    println!("{prefix}: alignment:   {}", gguf_get_alignment(ctx));
    println!("{prefix}: data offset: {}", gguf_get_data_offset(ctx));
}

unsafe fn print_kv(prefix: &str, ctx: *const gguf_context) {
    let n_kv = gguf_get_n_kv(ctx);
    println!("{prefix}: n_kv: {n_kv}");
    for i in 0..n_kv {
        println!("{prefix}: kv[{i}]: key = {}", cstr(gguf_get_key(ctx, i)));
    }
}

unsafe fn find_string_kv(prefix: &str, ctx: *const gguf_context) -> Result<(), String> {
    let findkey = cstring("some.parameter.string")?;
    let keyidx = gguf_find_key(ctx, findkey.as_ptr());
    if keyidx == -1 {
        println!("{prefix}: find key: some.parameter.string not found.");
    } else {
        println!(
            "{prefix}: find key: some.parameter.string found, kv[{keyidx}] value = {}",
            cstr(gguf_get_val_str(ctx, keyidx))
        );
    }
    Ok(())
}

unsafe fn print_tensor_info(prefix: &str, ctx: *const gguf_context, with_type: bool) {
    let n_tensors = gguf_get_n_tensors(ctx);
    println!("{prefix}: n_tensors: {n_tensors}");
    for i in 0..n_tensors {
        let name = cstr(gguf_get_tensor_name(ctx, i));
        let size = gguf_get_tensor_size(ctx, i);
        let offset = gguf_get_tensor_offset(ctx, i);
        if with_type {
            let type_ = gguf_get_tensor_type(ctx, i);
            let type_name = cstr(ggml_type_name(type_));
            let type_size = ggml_type_size(type_);
            let n_elements = size / type_size;
            println!("{prefix}: tensor[{i}]: name = {name}, size = {size}, offset = {offset}, type = {type_name}, n_elts = {n_elements}");
        } else {
            println!("{prefix}: tensor[{i}]: name = {name}, size = {size}, offset = {offset}");
        }
    }
}

unsafe fn print_tensor_data(
    prefix: &str,
    ctx: *const gguf_context,
    ctx_data: *mut ggml_context,
    check_data: bool,
) -> Result<(), String> {
    let n_tensors = gguf_get_n_tensors(ctx);
    for i in 0..n_tensors {
        println!("{prefix}: reading tensor {i} data");
        let name_ptr = gguf_get_tensor_name(ctx, i);
        let name = cstr(name_ptr);
        let cur = ggml_get_tensor(ctx_data, name_ptr);
        if cur.is_null() {
            return Err(format!("{prefix}: tensor {name} was not loaded"));
        }
        let tensor = &*cur;
        println!(
            "{prefix}: tensor[{i}]: n_dims = {}, ne = ({}, {}, {}, {}), name = {}, data = {:?}",
            ggml_n_dims(cur),
            tensor.ne[0],
            tensor.ne[1],
            tensor.ne[2],
            tensor.ne[3],
            cstr(tensor.name.as_ptr()),
            tensor.data
        );

        let n_elements = ggml_nelements(cur) as usize;
        let data = std::slice::from_raw_parts(tensor.data.cast::<f32>(), n_elements);
        print!("{name} data[:10] : ");
        for value in data.iter().take(10) {
            print!("{value:.6} ");
        }
        println!("\n");

        if check_data {
            for (j, value) in data.iter().enumerate() {
                let expected = 100.0 + i as f32;
                if *value != expected {
                    gguf_free(ctx as *mut gguf_context);
                    return Err(format!(
                        "{prefix}: tensor[{i}], data[{j}]: found {value}, expected {expected}"
                    ));
                }
            }
        }
    }
    Ok(())
}

fn cstring(value: &str) -> Result<CString, String> {
    CString::new(value).map_err(|err| err.to_string())
}

unsafe fn cstr(value: *const c_char) -> String {
    if value.is_null() {
        return "<null>".to_string();
    }
    CStr::from_ptr(value).to_string_lossy().into_owned()
}

#[allow(dead_code)]
fn _assert_tensor_dims_shape() {
    let _ = [0_i64; GGML_MAX_DIMS];
}
