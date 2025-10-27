#! /bin/bash

tmp=`mktemp -d`

echo $tmp

cp -r crates $tmp/.
cp Cargo.toml $tmp/.

cp -r src $tmp/crates/nexus2d/.
cp -r LICENSE $tmp/crates/nexus2d/.
cp -r README.md $tmp/crates/nexus2d/.
cp -r shaders $tmp/crates/nexus2d/.

cp -r src $tmp/crates/nexus3d/.
cp -r LICENSE $tmp/crates/nexus3d/.
cp -r README.md $tmp/crates/nexus3d/.
cp -r shaders $tmp/crates/nexus3d/.

# Publish nexus2d
cd $tmp/crates/nexus2d
ls
sed 's#\.\./\.\./src#src#g' ./Cargo.toml > ./Cargo.toml.new
mv Cargo.toml.new Cargo.toml
sed 's#\.\./\.\./shaders#shaders#g' ./src/lib.rs > ./src/lib.rs.new
mv src/lib.rs.new src/lib.rs
cargo publish

# Publish nexus3d
cd ../nexus3d
sed 's#\.\./\.\./src#src#g' ./Cargo.toml > ./Cargo.toml.new
mv Cargo.toml.new Cargo.toml
sed 's#\.\./\.\./shaders#shaders#g' ./src/lib.rs > ./src/lib.rs.new
mv src/lib.rs.new src/lib.rs
cargo publish

# Cleanup
rm -rf $tmp

