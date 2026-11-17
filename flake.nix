{
  description = "particle-filter dev environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pkgs.python3 ];

        # Runtime libs for pip wheels:
        #  - numpy: libstdc++, zlib
        #  - pygame (bundled SDL2 dlopens these): GL, X11, Wayland, audio
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
          stdenv.cc.cc.lib
          zlib
          libGL
          libxkbcommon
          wayland
          xorg.libX11
          xorg.libXext
          xorg.libXcursor
          xorg.libXrandr
          xorg.libXi
          xorg.libXrender
          xorg.libXfixes
          alsa-lib
          libpulseaudio
        ]);
      };
    };
}
