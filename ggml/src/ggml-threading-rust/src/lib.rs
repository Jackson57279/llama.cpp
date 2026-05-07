use std::sync::{Mutex, MutexGuard};
use std::ffi::c_void;

const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

static CRITICAL_SECTION: Mutex<()> = Mutex::new(());
static mut CRITICAL_SECTION_GUARD: Option<MutexGuard<'static, ()>> = None;

#[repr(C)]
struct GgmlHashSetRust {
    size: usize,
    used: *mut u32,
    keys: *mut *mut c_void,
}

#[no_mangle]
pub extern "C" fn ggml_critical_section_start() {
    let guard = CRITICAL_SECTION
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    unsafe {
        CRITICAL_SECTION_GUARD = Some(guard);
    }
}

#[no_mangle]
pub extern "C" fn ggml_critical_section_end() {
    unsafe {
        CRITICAL_SECTION_GUARD = None;
    }
}

#[no_mangle]
pub extern "C" fn ggml_op_is_empty_rust(op: i32) -> bool {
    matches!(op, 0 | 36 | 37 | 38 | 39)
}

#[no_mangle]
pub extern "C" fn ggml_is_view_op_rust(op: i32) -> bool {
    matches!(op, 36 | 37 | 38 | 39)
}

#[no_mangle]
pub extern "C" fn ggml_bitset_size_rust(n: usize) -> usize {
    (n + 31) >> 5
}

#[no_mangle]
pub extern "C" fn ggml_aligned_offset_rust(buffer: *const std::ffi::c_void, offset: usize, alignment: usize) -> usize {
    debug_assert!(alignment != 0 && alignment.is_power_of_two());
    let address = buffer as usize + offset;
    let align = (alignment - (address % alignment)) % alignment;
    offset + align
}

#[no_mangle]
pub unsafe extern "C" fn ggml_get_node_buffer_id_rust(node_buffer_ids: *const i32, index: i32) -> i32 {
    if node_buffer_ids.is_null() {
        return 0;
    }
    unsafe { *node_buffer_ids.add(index as usize) }
}

#[no_mangle]
pub extern "C" fn ggml_buffer_address_less_rust(
    a_chunk: i32,
    a_offset: usize,
    b_chunk: i32,
    b_offset: usize,
) -> bool {
    if a_chunk != b_chunk {
        return a_chunk < b_chunk;
    }
    a_offset < b_offset
}

#[no_mangle]
pub extern "C" fn ggml_isinf_fp16_rust(value: u16) -> bool {
    (value & 0x7c00) == 0x7c00 && (value & 0x03ff) == 0
}

#[no_mangle]
pub extern "C" fn ggml_isnan_fp16_rust(value: u16) -> bool {
    (value & 0x7c00) == 0x7c00 && (value & 0x03ff) != 0
}

#[no_mangle]
pub extern "C" fn ggml_is_invalid_e8m0_rust(value: u8) -> bool {
    value == 0xff
}

#[no_mangle]
pub extern "C" fn ggml_float_validation_kind_rust(value: f32) -> i32 {
    if value.is_infinite() {
        return 1;
    }
    if value.is_nan() {
        return 2;
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn ggml_int_pair_compare_rust(left: *const c_void, right: *const c_void) -> i32 {
    let left = unsafe { std::slice::from_raw_parts(left.cast::<i32>(), 2) };
    let right = unsafe { std::slice::from_raw_parts(right.cast::<i32>(), 2) };
    let first = compare_i32(left[0], right[0]);
    if first != 0 {
        first
    } else {
        compare_i32(left[1], right[1])
    }
}

#[no_mangle]
pub unsafe extern "C" fn ggml_float_pair_compare_rust(left: *const c_void, right: *const c_void) -> i32 {
    let left = unsafe { *left.cast::<f32>() };
    let right = unsafe { *right.cast::<f32>() };
    compare_f32(left, right)
}

#[no_mangle]
pub extern "C" fn ggml_graph_nbytes_rust(
    size: usize,
    grads: bool,
    hash_size: usize,
    cgraph_size: usize,
    tensor_ptr_size: usize,
    int32_size: usize,
    bitset_size: usize,
    bitset_t_size: usize,
) -> usize {
    let mut cursor = 0usize;
    incr_ptr_aligned(&mut cursor, cgraph_size, 1);
    incr_ptr_aligned(&mut cursor, size * tensor_ptr_size, tensor_ptr_size);
    incr_ptr_aligned(&mut cursor, size * tensor_ptr_size, tensor_ptr_size);
    incr_ptr_aligned(&mut cursor, hash_size * int32_size, int32_size);
    incr_ptr_aligned(&mut cursor, hash_size * tensor_ptr_size, tensor_ptr_size);
    if grads {
        incr_ptr_aligned(&mut cursor, hash_size * tensor_ptr_size, tensor_ptr_size);
        incr_ptr_aligned(&mut cursor, hash_size * tensor_ptr_size, tensor_ptr_size);
    }
    incr_ptr_aligned(&mut cursor, bitset_size * bitset_t_size, bitset_t_size);
    cursor
}

#[no_mangle]
pub unsafe extern "C" fn ggml_iq2_find_best_neighbour_rust(
    neighbours: *const u16,
    grid: *const u64,
    xval: *const f32,
    weight: *const f32,
    scale: f32,
    out_l: *mut i8,
) -> i32 {
    find_best_neighbour(neighbours, grid.cast(), xval, weight, scale, out_l, 8, 1)
}

#[no_mangle]
pub unsafe extern "C" fn ggml_iq3_find_best_neighbour_rust(
    neighbours: *const u16,
    grid: *const u32,
    xval: *const f32,
    weight: *const f32,
    scale: f32,
    out_l: *mut i8,
) -> i32 {
    find_best_neighbour(neighbours, grid.cast(), xval, weight, scale, out_l, 4, 1)
}

#[no_mangle]
pub unsafe extern "C" fn ggml_iq1_find_best_neighbour2_rust(
    neighbours: *const u16,
    grid: *const u64,
    xval: *const f32,
    weight: *const f32,
    scale: f32,
    xg: *const f32,
    out_l: *mut i8,
    ngrid: i32,
) -> i32 {
    let num_neighbors = unsafe { *neighbours };
    if num_neighbors == 0 {
        return -1;
    }

    let xval = unsafe { std::slice::from_raw_parts(xval, 8) };
    let weight = unsafe { std::slice::from_raw_parts(weight, 8) };
    let xg = unsafe { std::slice::from_raw_parts(xg, 3) };
    let grid = grid.cast::<i8>();
    let mut best_score = f32::MAX;
    let mut grid_index = -1i32;

    for j in 1..=num_neighbors as usize {
        let candidate = unsafe { *neighbours.add(j) } as usize;
        let pg = unsafe { std::slice::from_raw_parts(grid.add(candidate * 8), 8) };
        let mut d2 = 0.0f32;
        for i in 0..8 {
            let q = xg[((pg[i] - 1) / 2) as usize];
            let diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best_score {
            best_score = d2;
            grid_index = candidate as i32;
        }
    }

    if grid_index < 0 {
        for i in 0..ngrid as usize {
            let grid_i = unsafe { std::slice::from_raw_parts(grid.add(i * 8), 8) };
            let mut d2 = 0.0f32;
            for j in 0..8 {
                let q = xg[((grid_i[j] - 1) / 2) as usize];
                let diff = scale * q - xval[i];
                d2 += weight[j] * diff * diff;
            }
            if d2 < best_score {
                best_score = d2;
                grid_index = i as i32;
            }
        }
    }

    if grid_index >= 0 {
        let pg = unsafe { std::slice::from_raw_parts(grid.add(grid_index as usize * 8), 8) };
        for i in 0..8 {
            unsafe {
                *out_l.add(i) = (pg[i] - 1) / 2;
            }
        }
    }
    grid_index
}

#[no_mangle]
pub extern "C" fn ggml_calc_conv_output_size_rust(ins: i64, ks: i64, s: i32, p: i32, d: i32) -> i64 {
    (ins + 2 * p as i64 - d as i64 * (ks - 1) - 1) / s as i64 + 1
}

#[no_mangle]
pub extern "C" fn ggml_calc_conv_transpose_1d_output_size_rust(
    ins: i64,
    ks: i64,
    s: i32,
    p: i32,
    d: i32,
) -> i64 {
    (ins - 1) * s as i64 - 2 * p as i64 + d as i64 * (ks - 1) + 1
}

#[no_mangle]
pub extern "C" fn ggml_calc_conv_transpose_output_size_rust(
    ins: i64,
    ks: i64,
    s: i32,
    p: i32,
) -> i64 {
    (ins - 1) * s as i64 - 2 * p as i64 + ks
}

#[no_mangle]
pub extern "C" fn ggml_calc_pool_output_size_rust(ins: i64, ks: i32, s: i32, p: f32) -> i64 {
    ((ins as f32 + 2.0 * p - ks as f32) / s as f32 + 1.0) as i64
}

#[no_mangle]
pub extern "C" fn ggml_rope_yarn_corr_dim_rust(
    n_dims: i32,
    n_ctx_orig: i32,
    n_rot: f32,
    base: f32,
) -> f32 {
    n_dims as f32 * (n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

#[no_mangle]
pub extern "C" fn ggml_nearest_int_rust(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4_194_303.0);
    let bits = (fval + 12_582_912.0).to_bits() as i32;
    (bits & 0x007f_ffff) - 0x0040_0000
}

#[no_mangle]
pub extern "C" fn ggml_iq2_data_index_rust(ggml_type: i32) -> i32 {
    match ggml_type {
        16 => 0,
        17 => 1,
        19 | 29 => 2,
        22 => 3,
        _ => -1,
    }
}

#[no_mangle]
pub extern "C" fn ggml_iq2_grid_size_rust(ggml_type: i32) -> i32 {
    match ggml_type {
        16 => 256,
        17 => 512,
        19 | 29 => 2048,
        22 => 1024,
        _ => -1,
    }
}

#[no_mangle]
pub extern "C" fn ggml_iq3_data_index_rust(grid_size: i32) -> i32 {
    match grid_size {
        256 => 0,
        512 => 1,
        _ => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn ggml_best_index_int8_rust(n: i32, val: *const i8, x: f32) -> i32 {
    if n <= 0 {
        return -1;
    }
    let n = n as usize;
    let val = unsafe { std::slice::from_raw_parts(val, n) };
    if x <= val[0] as f32 {
        return 0;
    }
    if x >= val[n - 1] as f32 {
        return (n - 1) as i32;
    }
    let mut ml = 0usize;
    let mut mu = n - 1;
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < val[mav] as f32 {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    if x - (val[mu - 1] as f32) < val[mu] as f32 - x {
        (mu - 1) as i32
    } else {
        mu as i32
    }
}

#[no_mangle]
pub extern "C" fn ggml_best_index_mxfp4_rust(x: f32, e: f32) -> i32 {
    let mut best_index = 0i32;
    let mut best_err = (KVALUES_MXFP4[0] as f32 * e - x).abs();
    for (i, value) in KVALUES_MXFP4.iter().enumerate().skip(1) {
        let err = (*value as f32 * e - x).abs();
        if err < best_err {
            best_index = i as i32;
            best_err = err;
        }
    }
    best_index
}

#[no_mangle]
pub extern "C" fn ggml_up32_rust(n: i32) -> i32 {
    (n + 31) & !31
}

#[no_mangle]
pub extern "C" fn ggml_up_rust(n: i32, m: i32) -> i32 {
    debug_assert!(m > 0 && (m & (m - 1)) == 0);
    (n + m - 1) & !(m - 1)
}

#[no_mangle]
pub extern "C" fn ggml_compute_softplus_f32_rust(input: f32) -> f32 {
    if input > 20.0 {
        input
    } else {
        (1.0 + input.exp()).ln()
    }
}

#[no_mangle]
pub unsafe extern "C" fn ggml_are_same_layout_rust(
    a_type: i32,
    b_type: i32,
    a_ne: *const i64,
    b_ne: *const i64,
    a_nb: *const usize,
    b_nb: *const usize,
    n_dims: usize,
) -> bool {
    if a_type != b_type {
        return false;
    }
    if (a_ne.is_null() || b_ne.is_null() || a_nb.is_null() || b_nb.is_null()) && n_dims != 0 {
        return false;
    }

    let a_ne = unsafe { std::slice::from_raw_parts(a_ne, n_dims) };
    let b_ne = unsafe { std::slice::from_raw_parts(b_ne, n_dims) };
    let a_nb = unsafe { std::slice::from_raw_parts(a_nb, n_dims) };
    let b_nb = unsafe { std::slice::from_raw_parts(b_nb, n_dims) };

    a_ne == b_ne && a_nb == b_nb
}

#[no_mangle]
pub unsafe extern "C" fn ggml_get_op_params_i32_rust(op_params: *const i32, index: u32) -> i32 {
    unsafe { *op_params.add(index as usize) }
}

#[no_mangle]
pub unsafe extern "C" fn ggml_get_op_params_f32_rust(op_params: *const i32, index: u32) -> f32 {
    unsafe { *op_params.cast::<f32>().add(index as usize) }
}

#[no_mangle]
pub unsafe extern "C" fn ggml_set_op_params_i32_rust(op_params: *mut i32, index: u32, value: i32) {
    unsafe {
        *op_params.add(index as usize) = value;
    }
}

#[no_mangle]
pub unsafe extern "C" fn ggml_set_op_params_f32_rust(op_params: *mut i32, index: u32, value: f32) {
    unsafe {
        *op_params.cast::<f32>().add(index as usize) = value;
    }
}

#[no_mangle]
pub unsafe extern "C" fn ggml_bitset_get_rust(bitset: *const u32, i: usize) -> bool {
    let word = unsafe { *bitset.add(i >> 5) };
    (word & (1u32 << (i & 31))) != 0
}

#[no_mangle]
pub unsafe extern "C" fn ggml_bitset_set_rust(bitset: *mut u32, i: usize) {
    let word = unsafe { bitset.add(i >> 5) };
    unsafe {
        *word |= 1u32 << (i & 31);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ggml_bitset_clear_rust(bitset: *mut u32, i: usize) {
    let word = unsafe { bitset.add(i >> 5) };
    unsafe {
        *word &= !(1u32 << (i & 31));
    }
}

#[no_mangle]
pub extern "C" fn ggml_hash_rust(ptr: *const c_void) -> usize {
    ptr as usize >> 4
}

#[no_mangle]
pub extern "C" fn ggml_impl_is_view_rust(view_src: *const c_void) -> bool {
    !view_src.is_null()
}

#[no_mangle]
pub unsafe extern "C" fn ggml_hash_find_rust(hash_set: *const c_void, key: *const c_void) -> usize {
    let hash_set = unsafe { &*hash_set.cast::<GgmlHashSetRust>() };
    let h = ggml_hash_rust(key) % hash_set.size;
    let mut i = h;
    while unsafe { ggml_bitset_get_rust(hash_set.used, i) }
        && unsafe { *hash_set.keys.add(i) } != key.cast_mut()
    {
        i = (i + 1) % hash_set.size;
        if i == h {
            return usize::MAX;
        }
    }
    i
}

#[no_mangle]
pub unsafe extern "C" fn ggml_hash_contains_rust(hash_set: *const c_void, key: *mut c_void) -> bool {
    let hash_set_ref = unsafe { &*hash_set.cast::<GgmlHashSetRust>() };
    let i = unsafe { ggml_hash_find_rust(hash_set, key.cast_const()) };
    i != usize::MAX && unsafe { ggml_bitset_get_rust(hash_set_ref.used, i) }
}

#[no_mangle]
pub unsafe extern "C" fn ggml_hash_insert_rust(hash_set: *mut c_void, key: *mut c_void) -> usize {
    let hash_set = unsafe { &mut *hash_set.cast::<GgmlHashSetRust>() };
    let h = ggml_hash_rust(key.cast_const()) % hash_set.size;
    let mut i = h;
    loop {
        if !unsafe { ggml_bitset_get_rust(hash_set.used, i) } {
            unsafe {
                ggml_bitset_set_rust(hash_set.used, i);
                *hash_set.keys.add(i) = key;
            }
            return i;
        }
        if unsafe { *hash_set.keys.add(i) } == key {
            return usize::MAX - 1;
        }
        i = (i + 1) % hash_set.size;
        if i == h {
            std::process::abort();
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ggml_hash_find_or_insert_rust(hash_set: *mut c_void, key: *mut c_void) -> usize {
    let hash_set = unsafe { &mut *hash_set.cast::<GgmlHashSetRust>() };
    let h = ggml_hash_rust(key.cast_const()) % hash_set.size;
    let mut i = h;
    loop {
        if !unsafe { ggml_bitset_get_rust(hash_set.used, i) } {
            unsafe {
                ggml_bitset_set_rust(hash_set.used, i);
                *hash_set.keys.add(i) = key;
            }
            return i;
        }
        if unsafe { *hash_set.keys.add(i) } == key {
            return i;
        }
        i = (i + 1) % hash_set.size;
        if i == h {
            std::process::abort();
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ggml_node_get_use_count_rust(
    node: *const c_void,
    hash_set: *const c_void,
    use_counts: *const i32,
) -> i32 {
    if node.is_null() || hash_set.is_null() || use_counts.is_null() {
        return 0;
    }
    let hash_pos = unsafe { ggml_hash_find_rust(hash_set, node) };
    let hash_set = unsafe { &*hash_set.cast::<GgmlHashSetRust>() };
    if !unsafe { ggml_bitset_get_rust(hash_set.used, hash_pos) } {
        return 0;
    }
    unsafe { *use_counts.add(hash_pos) }
}

#[no_mangle]
pub extern "C" fn ggml_node_has_n_uses_rust(
    use_count: i32,
    n_uses: i32,
    view_src: *const c_void,
    flags: i32,
    output_flag: i32,
) -> bool {
    use_count == n_uses && view_src.is_null() && (flags & output_flag) == 0
}

#[no_mangle]
pub extern "C" fn ggml_e8m0_to_fp32_rust(x: u8) -> f32 {
    let bits = if x == 0 {
        0x0040_0000
    } else {
        (x as u32) << 23
    };
    f32::from_bits(bits)
}

#[no_mangle]
pub extern "C" fn ggml_e8m0_to_fp32_half_rust(x: u8) -> f32 {
    let bits = if x < 2 {
        0x0020_0000u32 << x
    } else {
        ((x as u32) - 1) << 23
    };
    f32::from_bits(bits)
}

#[no_mangle]
pub extern "C" fn ggml_ue4m3_to_fp32_rust(x: u8) -> f32 {
    if x == 0 || x == 0x7f {
        return 0.0;
    }
    let exp = ((x >> 3) & 0x0f) as i32;
    let man = (x & 0x07) as i32;
    let raw = if exp == 0 {
        (man as f32) * 2.0f32.powi(-9)
    } else {
        (1.0 + man as f32 / 8.0) * 2.0f32.powi(exp - 7)
    };
    raw * 0.5
}

#[no_mangle]
pub extern "C" fn ggml_fp32_to_ue4m3_rust(mut x: f32) -> u8 {
    if x <= 0.0 || x.is_nan() {
        return 0;
    }
    if x > 448.0 {
        x = 448.0;
    }
    let bits = x.to_bits();
    let fp32_exp = ((bits >> 23) & 0xff) as i32 - 127;
    let fp32_man = ((bits >> 20) & 0x07) as i32;
    let mut ue4m3_exp = fp32_exp + 7;
    if ue4m3_exp <= 0 {
        let mut man = (x * 512.0 + 0.5) as i32;
        if man > 7 {
            man = 7;
        }
        if man < 1 {
            return 0;
        }
        return man as u8;
    }
    if ue4m3_exp >= 15 {
        return 0x7e;
    }
    let round_bit = ((bits >> 19) & 1) as i32;
    let mut ue4m3_man = fp32_man + round_bit;
    if ue4m3_man > 7 {
        ue4m3_man = 0;
        ue4m3_exp += 1;
        if ue4m3_exp >= 15 {
            return 0x7e;
        }
    }
    ((ue4m3_exp << 3) | ue4m3_man) as u8
}

fn compare_i32(left: i32, right: i32) -> i32 {
    if left < right {
        -1
    } else if left > right {
        1
    } else {
        0
    }
}

fn compare_f32(left: f32, right: f32) -> i32 {
    if left < right {
        -1
    } else if left > right {
        1
    } else {
        0
    }
}

fn incr_ptr_aligned(cursor: &mut usize, size: usize, align: usize) {
    let aligned = (*cursor + align - 1) & !(align - 1);
    *cursor = aligned + size;
}

unsafe fn find_best_neighbour(
    neighbours: *const u16,
    grid: *const i8,
    xval: *const f32,
    weight: *const f32,
    scale: f32,
    out_l: *mut i8,
    dims: usize,
    l_bias: i8,
) -> i32 {
    let num_neighbors = unsafe { *neighbours };
    if num_neighbors == 0 {
        return -1;
    }

    let xval = unsafe { std::slice::from_raw_parts(xval, dims) };
    let weight = unsafe { std::slice::from_raw_parts(weight, dims) };
    let mut best_d2 = f32::MAX;
    let mut grid_index = -1i32;

    for j in 1..=num_neighbors as usize {
        let candidate = unsafe { *neighbours.add(j) } as usize;
        let pg = unsafe { std::slice::from_raw_parts(grid.add(candidate * dims), dims) };
        let mut d2 = 0.0f32;
        for i in 0..dims {
            let q = pg[i] as f32;
            let diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            grid_index = candidate as i32;
        }
    }

    if grid_index >= 0 {
        let pg = unsafe { std::slice::from_raw_parts(grid.add(grid_index as usize * dims), dims) };
        for i in 0..dims {
            unsafe {
                *out_l.add(i) = (pg[i] - l_bias) / 2;
            }
        }
    }
    grid_index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_and_unlocks() {
        ggml_critical_section_start();
        ggml_critical_section_end();
    }

    #[test]
    fn identifies_empty_ops() {
        assert!(ggml_op_is_empty_rust(0));
        assert!(ggml_op_is_empty_rust(36));
        assert!(ggml_op_is_empty_rust(37));
        assert!(ggml_op_is_empty_rust(38));
        assert!(ggml_op_is_empty_rust(39));
        assert!(!ggml_op_is_empty_rust(1));
    }

    #[test]
    fn identifies_view_ops_without_none() {
        assert!(!ggml_is_view_op_rust(0));
        assert!(!ggml_is_view_op_rust(35));
        assert!(ggml_is_view_op_rust(36));
        assert!(ggml_is_view_op_rust(37));
        assert!(ggml_is_view_op_rust(38));
        assert!(ggml_is_view_op_rust(39));
        assert!(!ggml_is_view_op_rust(40));
    }

    #[test]
    fn computes_bitset_word_count() {
        assert_eq!(ggml_bitset_size_rust(0), 0);
        assert_eq!(ggml_bitset_size_rust(1), 1);
        assert_eq!(ggml_bitset_size_rust(32), 1);
        assert_eq!(ggml_bitset_size_rust(33), 2);
    }

    #[test]
    fn computes_aligned_offsets() {
        assert_eq!(ggml_aligned_offset_rust(std::ptr::null(), 0, 32), 0);
        assert_eq!(ggml_aligned_offset_rust(std::ptr::null(), 1, 32), 32);
        assert_eq!(ggml_aligned_offset_rust(std::ptr::null(), 64, 32), 64);

        let data = [0u8; 64];
        let base = data.as_ptr() as usize;
        let expected = (16 - ((base + 3) % 16)) % 16 + 3;
        assert_eq!(ggml_aligned_offset_rust(data.as_ptr().cast(), 3, 16), expected);
    }

    #[test]
    fn reads_node_buffer_ids_with_default() {
        let ids = [7, 11, 13];
        unsafe {
            assert_eq!(ggml_get_node_buffer_id_rust(std::ptr::null(), 2), 0);
            assert_eq!(ggml_get_node_buffer_id_rust(ids.as_ptr(), 0), 7);
            assert_eq!(ggml_get_node_buffer_id_rust(ids.as_ptr(), 2), 13);
        }
    }

    #[test]
    fn compares_buffer_addresses() {
        assert!(ggml_buffer_address_less_rust(0, 99, 1, 0));
        assert!(ggml_buffer_address_less_rust(1, 4, 1, 5));
        assert!(!ggml_buffer_address_less_rust(2, 0, 1, 99));
        assert!(!ggml_buffer_address_less_rust(1, 5, 1, 5));
    }

    #[test]
    fn classifies_fp16_and_e8m0_values() {
        assert!(ggml_isinf_fp16_rust(0x7c00));
        assert!(ggml_isinf_fp16_rust(0xfc00));
        assert!(!ggml_isinf_fp16_rust(0x7c01));
        assert!(!ggml_isinf_fp16_rust(0x3c00));

        assert!(ggml_isnan_fp16_rust(0x7c01));
        assert!(ggml_isnan_fp16_rust(0x7fff));
        assert!(!ggml_isnan_fp16_rust(0x7c00));
        assert!(!ggml_isnan_fp16_rust(0x3c00));

        assert!(ggml_is_invalid_e8m0_rust(0xff));
        assert!(!ggml_is_invalid_e8m0_rust(0xfe));
    }

    #[test]
    fn classifies_float_validation_values() {
        assert_eq!(ggml_float_validation_kind_rust(1.0), 0);
        assert_eq!(ggml_float_validation_kind_rust(f32::INFINITY), 1);
        assert_eq!(ggml_float_validation_kind_rust(f32::NEG_INFINITY), 1);
        assert_eq!(ggml_float_validation_kind_rust(f32::NAN), 2);
    }

    #[test]
    fn compares_pairs_for_qsort() {
        let left = [1, 9];
        let right = [2, 0];
        let same_first = [1, 10];
        unsafe {
            assert_eq!(ggml_int_pair_compare_rust(left.as_ptr().cast(), right.as_ptr().cast()), -1);
            assert_eq!(ggml_int_pair_compare_rust(right.as_ptr().cast(), left.as_ptr().cast()), 1);
            assert_eq!(ggml_int_pair_compare_rust(left.as_ptr().cast(), same_first.as_ptr().cast()), -1);
            assert_eq!(ggml_int_pair_compare_rust(left.as_ptr().cast(), left.as_ptr().cast()), 0);
        }
    }

    #[test]
    fn compares_float_pairs_by_first_value() {
        let left = [1.0f32, 99.0];
        let right = [2.0f32, 0.0];
        let same = [1.0f32, 0.0];
        unsafe {
            assert_eq!(ggml_float_pair_compare_rust(left.as_ptr().cast(), right.as_ptr().cast()), -1);
            assert_eq!(ggml_float_pair_compare_rust(right.as_ptr().cast(), left.as_ptr().cast()), 1);
            assert_eq!(ggml_float_pair_compare_rust(left.as_ptr().cast(), same.as_ptr().cast()), 0);
        }
    }

    #[test]
    fn computes_graph_nbytes_from_c_layout_inputs() {
        assert_eq!(
            ggml_graph_nbytes_rust(2, false, 5, 3, 8, 4, 1, 4),
            108
        );
        assert_eq!(
            ggml_graph_nbytes_rust(2, true, 5, 3, 8, 4, 1, 4),
            188
        );
    }

    #[test]
    fn finds_iq2_best_neighbour() {
        let grid = [
            u64::from_ne_bytes([1, 1, 1, 1, 1, 1, 1, 1]),
            u64::from_ne_bytes([3, 3, 3, 3, 3, 3, 3, 3]),
        ];
        let neighbours = [2u16, 0, 1];
        let xval = [3.0f32; 8];
        let weight = [1.0f32; 8];
        let mut out = [0i8; 8];
        let idx = unsafe {
            ggml_iq2_find_best_neighbour_rust(
                neighbours.as_ptr(),
                grid.as_ptr(),
                xval.as_ptr(),
                weight.as_ptr(),
                1.0,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(idx, 1);
        assert_eq!(out, [1; 8]);
    }

    #[test]
    fn finds_iq3_best_neighbour() {
        let grid = [
            u32::from_ne_bytes([1, 1, 1, 1]),
            u32::from_ne_bytes([5, 5, 5, 5]),
        ];
        let neighbours = [2u16, 0, 1];
        let xval = [5.0f32; 4];
        let weight = [1.0f32; 4];
        let mut out = [0i8; 4];
        let idx = unsafe {
            ggml_iq3_find_best_neighbour_rust(
                neighbours.as_ptr(),
                grid.as_ptr(),
                xval.as_ptr(),
                weight.as_ptr(),
                1.0,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(idx, 1);
        assert_eq!(out, [2; 4]);
    }

    #[test]
    fn finds_iq1_best_neighbour2() {
        let grid = [
            u64::from_ne_bytes([1, 1, 1, 1, 1, 1, 1, 1]),
            u64::from_ne_bytes([5, 5, 5, 5, 5, 5, 5, 5]),
        ];
        let neighbours = [2u16, 0, 1];
        let xval = [1.25f32; 8];
        let weight = [1.0f32; 8];
        let xg = [-1.25f32, 0.0, 1.25];
        let mut out = [0i8; 8];
        let idx = unsafe {
            ggml_iq1_find_best_neighbour2_rust(
                neighbours.as_ptr(),
                grid.as_ptr(),
                xval.as_ptr(),
                weight.as_ptr(),
                1.0,
                xg.as_ptr(),
                out.as_mut_ptr(),
                2,
            )
        };
        assert_eq!(idx, 1);
        assert_eq!(out, [2; 8]);
    }

    #[test]
    fn computes_conv_and_pool_output_sizes() {
        assert_eq!(ggml_calc_conv_output_size_rust(32, 3, 1, 1, 1), 32);
        assert_eq!(ggml_calc_conv_output_size_rust(31, 5, 2, 2, 1), 16);
        assert_eq!(ggml_calc_conv_output_size_rust(15, 3, 2, 0, 2), 6);

        assert_eq!(ggml_calc_conv_transpose_1d_output_size_rust(8, 3, 2, 0, 1), 17);
        assert_eq!(ggml_calc_conv_transpose_1d_output_size_rust(8, 3, 2, 1, 2), 17);
        assert_eq!(ggml_calc_conv_transpose_output_size_rust(8, 3, 2, 0), 17);
        assert_eq!(ggml_calc_conv_transpose_output_size_rust(8, 3, 2, 1), 15);

        assert_eq!(ggml_calc_pool_output_size_rust(32, 2, 2, 0.0), 16);
        assert_eq!(ggml_calc_pool_output_size_rust(31, 3, 2, 1.0), 16);
    }

    #[test]
    fn computes_rope_yarn_correction_dimension() {
        let actual = ggml_rope_yarn_corr_dim_rust(128, 4096, 32.0, 10000.0);
        let expected = 20.94448f32;
        assert!((actual - expected).abs() < 0.0001);
    }

    #[test]
    fn rounds_with_nearest_int_bit_trick() {
        assert_eq!(ggml_nearest_int_rust(0.0), 0);
        assert_eq!(ggml_nearest_int_rust(0.49), 0);
        assert_eq!(ggml_nearest_int_rust(0.5), 0);
        assert_eq!(ggml_nearest_int_rust(0.51), 1);
        assert_eq!(ggml_nearest_int_rust(1.5), 2);
        assert_eq!(ggml_nearest_int_rust(-0.49), 0);
        assert_eq!(ggml_nearest_int_rust(-0.5), 0);
        assert_eq!(ggml_nearest_int_rust(-0.51), -1);
        assert_eq!(ggml_nearest_int_rust(-1.5), -2);
    }

    #[test]
    fn maps_iq_grid_table_indexes() {
        assert_eq!(ggml_iq2_data_index_rust(16), 0);
        assert_eq!(ggml_iq2_data_index_rust(17), 1);
        assert_eq!(ggml_iq2_data_index_rust(19), 2);
        assert_eq!(ggml_iq2_data_index_rust(29), 2);
        assert_eq!(ggml_iq2_data_index_rust(22), 3);
        assert_eq!(ggml_iq2_data_index_rust(18), -1);

        assert_eq!(ggml_iq2_grid_size_rust(16), 256);
        assert_eq!(ggml_iq2_grid_size_rust(17), 512);
        assert_eq!(ggml_iq2_grid_size_rust(19), 2048);
        assert_eq!(ggml_iq2_grid_size_rust(29), 2048);
        assert_eq!(ggml_iq2_grid_size_rust(22), 1024);
        assert_eq!(ggml_iq2_grid_size_rust(18), -1);

        assert_eq!(ggml_iq3_data_index_rust(256), 0);
        assert_eq!(ggml_iq3_data_index_rust(512), 1);
        assert_eq!(ggml_iq3_data_index_rust(1024), -1);
    }

    #[test]
    fn finds_best_int8_table_index() {
        let values = [-32i8, -8, 0, 7, 31];
        unsafe {
            assert_eq!(ggml_best_index_int8_rust(values.len() as i32, values.as_ptr(), -40.0), 0);
            assert_eq!(ggml_best_index_int8_rust(values.len() as i32, values.as_ptr(), 40.0), 4);
            assert_eq!(ggml_best_index_int8_rust(values.len() as i32, values.as_ptr(), -7.0), 1);
            assert_eq!(ggml_best_index_int8_rust(values.len() as i32, values.as_ptr(), -3.5), 2);
            assert_eq!(ggml_best_index_int8_rust(values.len() as i32, values.as_ptr(), 20.0), 4);
        }
    }

    #[test]
    fn finds_best_mxfp4_index() {
        assert_eq!(ggml_best_index_mxfp4_rust(0.0, 2.0), 0);
        assert_eq!(ggml_best_index_mxfp4_rust(11.0, 1.0), 7);
        assert_eq!(ggml_best_index_mxfp4_rust(-11.0, 1.0), 15);
        assert_eq!(ggml_best_index_mxfp4_rust(5.2, 1.0), 5);
        assert_eq!(ggml_best_index_mxfp4_rust(-5.2, 1.0), 13);
    }

    #[test]
    fn rounds_up_to_power_of_two_multiple() {
        assert_eq!(ggml_up32_rust(0), 0);
        assert_eq!(ggml_up32_rust(1), 32);
        assert_eq!(ggml_up32_rust(32), 32);
        assert_eq!(ggml_up32_rust(33), 64);

        assert_eq!(ggml_up_rust(0, 8), 0);
        assert_eq!(ggml_up_rust(1, 8), 8);
        assert_eq!(ggml_up_rust(16, 8), 16);
        assert_eq!(ggml_up_rust(17, 8), 24);
    }

    #[test]
    fn computes_softplus_like_c_helper() {
        assert_eq!(ggml_compute_softplus_f32_rust(21.0), 21.0);
        let zero = ggml_compute_softplus_f32_rust(0.0);
        assert!((zero - std::f32::consts::LN_2).abs() < 0.000001);
        let negative = ggml_compute_softplus_f32_rust(-2.0);
        assert!((negative - 0.12692805).abs() < 0.000001);
    }

    #[test]
    fn compares_tensor_layout_fields() {
        unsafe {
            let ne = [1_i64, 2, 3, 4];
            let nb = [4_usize, 8, 16, 32];
            assert!(ggml_are_same_layout_rust(
                0,
                0,
                ne.as_ptr(),
                ne.as_ptr(),
                nb.as_ptr(),
                nb.as_ptr(),
                4,
            ));

            let other_ne = [1_i64, 2, 5, 4];
            assert!(!ggml_are_same_layout_rust(
                0,
                0,
                ne.as_ptr(),
                other_ne.as_ptr(),
                nb.as_ptr(),
                nb.as_ptr(),
                4,
            ));

            let other_nb = [4_usize, 8, 64, 32];
            assert!(!ggml_are_same_layout_rust(
                0,
                0,
                ne.as_ptr(),
                ne.as_ptr(),
                nb.as_ptr(),
                other_nb.as_ptr(),
                4,
            ));
            assert!(!ggml_are_same_layout_rust(
                0,
                1,
                ne.as_ptr(),
                ne.as_ptr(),
                nb.as_ptr(),
                nb.as_ptr(),
                4,
            ));
            assert!(!ggml_are_same_layout_rust(
                0,
                0,
                std::ptr::null(),
                ne.as_ptr(),
                nb.as_ptr(),
                nb.as_ptr(),
                4,
            ));
        }
    }

    #[test]
    fn reads_and_writes_op_params() {
        unsafe {
            let mut params = [0_i32; 16];
            ggml_set_op_params_i32_rust(params.as_mut_ptr(), 0, -123);
            assert_eq!(ggml_get_op_params_i32_rust(params.as_ptr(), 0), -123);

            ggml_set_op_params_f32_rust(params.as_mut_ptr(), 1, 1.5);
            assert_eq!(ggml_get_op_params_f32_rust(params.as_ptr(), 1), 1.5);
            assert_eq!(params[1], 1.5_f32.to_bits() as i32);

            params[2] = (-2.25_f32).to_bits() as i32;
            assert_eq!(ggml_get_op_params_f32_rust(params.as_ptr(), 2), -2.25);
        }
    }

    #[test]
    fn reads_bitset_words() {
        let bitset = [0b1010u32, 1u32 << 3];
        unsafe {
            assert!(!ggml_bitset_get_rust(bitset.as_ptr(), 0));
            assert!(ggml_bitset_get_rust(bitset.as_ptr(), 1));
            assert!(!ggml_bitset_get_rust(bitset.as_ptr(), 2));
            assert!(ggml_bitset_get_rust(bitset.as_ptr(), 3));
            assert!(ggml_bitset_get_rust(bitset.as_ptr(), 35));
        }
    }

    #[test]
    fn mutates_bitset_words() {
        let mut bitset = [0u32; 2];
        unsafe {
            ggml_bitset_set_rust(bitset.as_mut_ptr(), 0);
            ggml_bitset_set_rust(bitset.as_mut_ptr(), 35);
            assert_eq!(bitset, [1, 1 << 3]);

            ggml_bitset_clear_rust(bitset.as_mut_ptr(), 0);
            assert_eq!(bitset, [0, 1 << 3]);
            ggml_bitset_clear_rust(bitset.as_mut_ptr(), 35);
            assert_eq!(bitset, [0, 0]);
        }
    }

    #[test]
    fn hashes_aligned_pointers_like_c_helper() {
        let ptr = 0x12340usize as *const c_void;
        assert_eq!(ggml_hash_rust(ptr), 0x1234);
    }

    #[test]
    fn detects_view_source_nullability() {
        assert!(!ggml_impl_is_view_rust(std::ptr::null()));
        assert!(ggml_impl_is_view_rust(1usize as *const c_void));
    }

    #[test]
    fn probes_and_mutates_hash_sets() {
        let key_a = 0x10usize as *mut c_void;
        let key_b = 0x30usize as *mut c_void;
        let mut used = [0u32; 1];
        let mut keys = [std::ptr::null_mut::<c_void>(); 4];
        let mut hash_set = GgmlHashSetRust {
            size: 4,
            used: used.as_mut_ptr(),
            keys: keys.as_mut_ptr(),
        };

        unsafe {
            assert_eq!(ggml_hash_find_rust((&hash_set as *const GgmlHashSetRust).cast(), key_a), 1);
            assert!(!ggml_hash_contains_rust((&hash_set as *const GgmlHashSetRust).cast(), key_a));

            assert_eq!(ggml_hash_insert_rust((&mut hash_set as *mut GgmlHashSetRust).cast(), key_a), 1);
            assert!(ggml_hash_contains_rust((&hash_set as *const GgmlHashSetRust).cast(), key_a));
            assert_eq!(keys[1], key_a);
            assert_eq!(
                ggml_hash_insert_rust((&mut hash_set as *mut GgmlHashSetRust).cast(), key_a),
                usize::MAX - 1
            );

            assert_eq!(ggml_hash_find_or_insert_rust((&mut hash_set as *mut GgmlHashSetRust).cast(), key_b), 3);
            assert_eq!(ggml_hash_find_or_insert_rust((&mut hash_set as *mut GgmlHashSetRust).cast(), key_b), 3);
        }
    }

    #[test]
    fn reads_node_use_counts_from_hash_set() {
        let key_a = 0x10usize as *mut c_void;
        let key_b = 0x30usize as *mut c_void;
        let mut used = [0u32; 1];
        let mut keys = [std::ptr::null_mut::<c_void>(); 4];
        let mut hash_set = GgmlHashSetRust {
            size: 4,
            used: used.as_mut_ptr(),
            keys: keys.as_mut_ptr(),
        };
        let mut use_counts = [0i32; 4];

        unsafe {
            assert_eq!(
                ggml_node_get_use_count_rust(
                    key_a,
                    (&hash_set as *const GgmlHashSetRust).cast(),
                    use_counts.as_ptr(),
                ),
                0
            );

            let index = ggml_hash_insert_rust((&mut hash_set as *mut GgmlHashSetRust).cast(), key_b);
            use_counts[index] = 7;
            assert_eq!(
                ggml_node_get_use_count_rust(
                    key_b,
                    (&hash_set as *const GgmlHashSetRust).cast(),
                    use_counts.as_ptr(),
                ),
                7
            );
            assert_eq!(
                ggml_node_get_use_count_rust(
                    std::ptr::null(),
                    (&hash_set as *const GgmlHashSetRust).cast(),
                    use_counts.as_ptr(),
                ),
                0
            );
        }
    }

    #[test]
    fn classifies_nodes_with_expected_use_count() {
        assert!(ggml_node_has_n_uses_rust(1, 1, std::ptr::null(), 0, 1));
        assert!(!ggml_node_has_n_uses_rust(2, 1, std::ptr::null(), 0, 1));
        assert!(!ggml_node_has_n_uses_rust(1, 1, 1usize as *const c_void, 0, 1));
        assert!(!ggml_node_has_n_uses_rust(1, 1, std::ptr::null(), 1, 1));
    }

    #[test]
    fn converts_e8m0_values_to_float() {
        assert_eq!(ggml_e8m0_to_fp32_rust(0), f32::from_bits(0x0040_0000));
        assert_eq!(ggml_e8m0_to_fp32_rust(127), 1.0);
        assert_eq!(ggml_e8m0_to_fp32_rust(128), 2.0);

        assert_eq!(ggml_e8m0_to_fp32_half_rust(0), f32::from_bits(0x0020_0000));
        assert_eq!(ggml_e8m0_to_fp32_half_rust(1), f32::from_bits(0x0040_0000));
        assert_eq!(ggml_e8m0_to_fp32_half_rust(127), 0.5);
        assert_eq!(ggml_e8m0_to_fp32_half_rust(128), 1.0);
    }

    #[test]
    fn converts_ue4m3_values() {
        assert_eq!(ggml_ue4m3_to_fp32_rust(0), 0.0);
        assert_eq!(ggml_ue4m3_to_fp32_rust(0x7f), 0.0);
        assert_eq!(ggml_ue4m3_to_fp32_rust(0x38), 0.5);
        assert_eq!(ggml_fp32_to_ue4m3_rust(0.0), 0);
        assert_eq!(ggml_fp32_to_ue4m3_rust(-1.0), 0);
        assert_eq!(ggml_fp32_to_ue4m3_rust(448.0), 0x7e);
        assert_eq!(ggml_fp32_to_ue4m3_rust(999.0), 0x7e);
        assert_eq!(ggml_fp32_to_ue4m3_rust(1.0), 0x38);
    }
}
