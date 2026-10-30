fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    if std::env::var_os("CARGO_FEATURE_SCENE_UPLOAD").is_some() {
        assert_eq!(
            std::env::var("CARGO_ENCODED_RUSTFLAGS").as_deref(),
            Ok("-Zmutable-noalias=no"),
            "CPU scene export requires the pinned cpu_oracle environment with mutable noalias disabled"
        );
    }
}
