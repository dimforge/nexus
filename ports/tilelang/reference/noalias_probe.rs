// Compiled to LLVM IR before every fixture export. Never executed with aliases.
#![crate_type = "lib"]

#[unsafe(no_mangle)]
pub fn mutable_reference_probe(x: &mut u32, y: &mut u32) -> u32 {
    *x = 1;
    *y = 2;
    *x
}

#[unsafe(no_mangle)]
pub fn shared_reference_probe(x: &u32) -> u32 {
    *x
}
