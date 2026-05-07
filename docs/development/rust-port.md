# Rust port status

This private-fork migration is using a measurable rule: the port is not complete while any C/C++ source or header files remain.

Check the current count:

```sh
scripts/rust-port-status.sh
```

Use the hard gate when the migration is expected to be complete:

```sh
scripts/rust-port-status.sh enforce
```

Current first Rust-owned artifact:

- `examples/simple-rust`: Rust port of `examples/simple/simple.cpp` over the llama.cpp C API, with unit tests for argument parsing.

The final target is stricter than adding Rust bindings: all `.c`, `.cc`, `.cpp`, `.cxx`, `.h`, `.hh`, `.hpp`, and `.hxx` files must be removed or replaced.
