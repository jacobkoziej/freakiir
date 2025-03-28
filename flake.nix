{
  description = "Frequency Response Enhanced by All-pass estimations of Kth order Infinite Impulse Responses";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    inputs:

    inputs.flake-utils.lib.eachDefaultSystem (
      system:

      let
        pkgs = import inputs.nixpkgs {
          inherit system;

          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        inherit (pkgs) lib;
        inherit (pkgs) python3;

        python3-pkgs = python3.withPackages (
          python-pkgs: with python-pkgs; [
            einops
            ipython
            numpy
            pytest
            pytorch-lightning
            scipy
            torch
          ]
        );

      in
      {
        devShells.default = pkgs.mkShellNoCC (
          let
            pre-commit-bin = "${lib.getBin pkgs.pre-commit}/bin/pre-commit";

          in
          {
            packages = with pkgs; [
              black
              mdformat
              pre-commit
              python3-pkgs
              ruff
              shfmt
              statix
              toml-sort
              treefmt2
              yamlfmt
              yamllint
            ];

            shellHook = ''
              ${pre-commit-bin} install --allow-missing-config > /dev/null
            '';
          }
        );

        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
