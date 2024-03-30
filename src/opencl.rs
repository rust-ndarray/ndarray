pub(crate) fn rust_type_to_c_name<T>() -> Option<&'static str> {
    match std::any::type_name::<T>() {
        "f32" => Some("float"),
        "f64" => Some("double"),
        "i8" => Some("int8_t"),
        "i16" => Some("int16_t"),
        "i32" => Some("int32_t"),
        "i64" => Some("int64_t"),
        "u8" => Some("uint8_t"),
        "u16" => Some("uint16_t"),
        "u32" => Some("uint32_t"),
        "u64" | "usize" => Some("uint64_t"),
        _ => None,
    }
}

pub(crate) fn gen_contiguous_linear_kernel_3(kernel_name: &str, typename: &str, op: &str) -> String {
    format!(
        r#"
        #ifndef NDARRAY_INCLUDE_STDINT
        #define NDARRAY_INCLUDE_STDINT

        // We should probably verify that these are, in fact, correct
        typedef char int8_t;
        typedef short int16_t;
        typedef int int32_t;
        typedef long int64_t;
        typedef unsigned char uint8_t;
        typedef unsigned short uint16_t;
        typedef unsigned int uint32_t;
        typedef unsigned long uint64_t;
        #endif // NDARRAY_INCLUDE_STDINT

        __kernel void {kernel_name}(__global const {typename} *a, __global const {typename} *b, __global {typename} *c) {{
            // Get id as 64-bit integer to avoid overflow
            uint64_t i = get_global_id(0);
            c[i] = a[i] {op} b[i];
        }}
        "#,
        kernel_name = kernel_name,
        typename = typename,
        op = op,
    )
}
