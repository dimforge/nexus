{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShell = with pkgs;
          mkShell {
            buildInputs = [
              rustup
            ];
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath([
              libX11
              libXcursor
              libXi
              libXrandr
              libxkbcommon
              vulkan-loader
              wayland
            ]);
          };
      });
}
