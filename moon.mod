name = "Kaida-Amethyst/openblas"

version = "0.1.3"

readme = "README.mbt.md"

repository = "https://github.com/moonbit-community/openblas.mbt"

license = "Apache-2.0"

keywords = [ "OpenBlas", "Math", "Linear Algebra", "BLAS", "LAPACK" ]

description = "Use OpenBlas in MoonBit"

source = ""

preferred_target = "native"

options(
  "--moonbit-unstable-prebuild": "build.py",
)
