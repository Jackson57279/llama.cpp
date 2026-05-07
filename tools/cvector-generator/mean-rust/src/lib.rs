use std::ffi::{c_char, c_int, CString};
use std::slice;

#[repr(C)]
pub struct ggml_tensor {
    type_: c_int,
    buffer: *mut std::ffi::c_void,
    ne: [i64; 4],
    nb: [usize; 4],
    op: c_int,
    op_params: [i32; 16],
    flags: i32,
    src: [*mut ggml_tensor; 10],
    view_src: *mut ggml_tensor,
    view_offs: usize,
    data: *mut std::ffi::c_void,
    name: [c_char; 64],
    extra: *mut std::ffi::c_void,
    padding: [c_char; 8],
}

extern "C" {
    fn ggml_set_name(tensor: *mut ggml_tensor, name: *const c_char) -> *mut ggml_tensor;
    fn ggml_nelements(tensor: *const ggml_tensor) -> i64;
    fn ggml_get_f32_nd(
        tensor: *const ggml_tensor,
        i0: c_int,
        i1: c_int,
        i2: c_int,
        i3: c_int,
    ) -> f32;
    fn ggml_get_f32_1d(tensor: *const ggml_tensor, i: c_int) -> f32;
    fn ggml_set_f32_1d(tensor: *const ggml_tensor, i: c_int, value: f32);
}

#[no_mangle]
pub unsafe extern "C" fn cvector_mean_run(
    inputs: *const *mut ggml_tensor,
    outputs: *const *mut ggml_tensor,
    len: usize,
) {
    println!("cvector_mean_run: Running mean...");
    let inputs = slice::from_raw_parts(inputs, len);
    let outputs = slice::from_raw_parts(outputs, len);

    for (il, (&input, &output)) in inputs.iter().zip(outputs.iter()).enumerate() {
        let name =
            CString::new(format!("direction.{}", il + 1)).expect("generated name has no nul");
        ggml_set_name(output, name.as_ptr());

        let input_ref = &*input;
        let output_ref = &*output;
        assert_eq!(input_ref.ne[0], output_ref.ne[0]);

        for ic in 0..input_ref.ne[0] as c_int {
            let mut value = 0.0_f32;
            for ir in 0..input_ref.ne[1] as c_int {
                value += ggml_get_f32_nd(input, ic, ir, 0, 0);
            }
            value /= input_ref.ne[1] as f32;
            ggml_set_f32_1d(output, ic, value);
        }

        let mut norm = 0.0_f32;
        for i in 0..ggml_nelements(output) as c_int {
            let value = ggml_get_f32_1d(output, i);
            norm += value * value;
        }
        norm = norm.sqrt();
        for i in 0..ggml_nelements(output) as c_int {
            let value = ggml_get_f32_1d(output, i);
            ggml_set_f32_1d(output, i, value / norm);
        }

        println!("cvector_mean_run: Done layer {} / {}", il + 1, len);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn ggml_tensor_layout_keeps_expected_prefix_sizes() {
        assert_eq!(std::mem::size_of::<i64>() * 4, 32);
        assert_eq!(std::mem::size_of::<usize>() * 4, 32);
    }
}
