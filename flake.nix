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
          python-pkgs:
          let
            freakiir-dependencies =
              let
                inherit (lib) importTOML;
                inherit (lib.attrsets) attrVals;

                pyproject = importTOML ./pyproject.toml;

                inherit (pyproject.project) dependencies;
                inherit (pyproject.project.optional-dependencies) dev;

              in
              attrVals (dependencies ++ dev) python-pkgs;

          in
          with python-pkgs;
          [
            ipython
          ]
          ++ freakiir-dependencies
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
              treefmt
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
