extern "C" {
    fn ggml_backend_sycl_print_sycl_devices();
}

fn main() {
    let _ = llama_ls_sycl_device_rust::binary_name();

    unsafe {
        ggml_backend_sycl_print_sycl_devices();
    }
}
