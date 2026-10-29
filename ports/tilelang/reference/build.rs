fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    assert_eq!(
        std::env::var("CARGO_ENCODED_RUSTFLAGS").as_deref(),
        Ok("-Zmutable-noalias=no"),
        "The khal CPU oracle requires mutable-reference noalias disabled. Use python ports/tilelang/cpu_oracle.py <cargo subcommand> [arguments]."
    );
}
