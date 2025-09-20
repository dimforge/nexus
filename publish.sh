#! /bin/bash

tmp=`mktemp -d`

echo $tmp

cp -r src $tmp/.
cp -r LICENSE README.md $tmp/.

### Publish the 2D version.
sed 's#\.\./\.\./src#src#g' crates/nexus2d/Cargo.toml > $tmp/Cargo.toml
rm -rf $tmp/examples
cp -r crates/nexus2d/examples $tmp/examples
currdir=`pwd`
cd $tmp && cargo publish
cd $currdir


### Publish the 3D version.
sed 's#\.\./\.\./src#src#g' crates/nexus3d/Cargo.toml > $tmp/Cargo.toml
rm -rf $tmp/examples
cp -r crates/nexus3d/examples $tmp/examples
cp -r LICENSE README.md $tmp/.
cd $tmp && cargo publish
cd $currdir

rm -rf $tmp

