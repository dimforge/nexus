#! /bin/bash

tmp=`mktemp -d`

echo $tmp

cp -r crates $tmp/.
cp Cargo.toml $tmp/.

cp -r src $tmp/crates/nexus_rbd2d/.
cp -r LICENSE $tmp/crates/nexus_rbd2d/.
cp -r README.md $tmp/crates/nexus_rbd2d/.
cp -r shaders $tmp/crates/nexus_rbd2d/.

cp -r src $tmp/crates/nexus_rbd3d/.
cp -r LICENSE $tmp/crates/nexus_rbd3d/.
cp -r README.md $tmp/crates/nexus_rbd3d/.
cp -r shaders $tmp/crates/nexus_rbd3d/.

# Publish nexus_rbd2d
cd $tmp/crates/nexus_rbd2d
ls
sed 's#\.\./\.\./src_rbd#src#g' ./Cargo.toml > ./Cargo.toml.new
mv Cargo.toml.new Cargo.toml
sed 's#\.\./\.\./shaders#shaders#g' ./src_rbd/lib.rs > ./src/lib.rs.new
mv src/lib.rs.new src/lib.rs
cargo publish --features runtime

# Publish nexus_rbd3d
cd ../nexus_rbd3d
sed 's#\.\./\.\./src_rbd#src#g' ./Cargo.toml > ./Cargo.toml.new
mv Cargo.toml.new Cargo.toml
sed 's#\.\./\.\./shaders#shaders#g' ./src_rbd/lib.rs > ./src/lib.rs.new
mv src/lib.rs.new src/lib.rs
cargo publish --features runtime

# Cleanup
rm -rf $tmp

