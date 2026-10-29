# Source this file before native builds and tests.
# Both OpenBLAS and LAPACKE are required; Homebrew OpenBLAS includes LAPACKE.
if [ "$(uname -s)" = "Darwin" ]; then
  OPENBLAS_PATH=$(brew --prefix openblas) || return 1
  export CC="${CC:-clang}"
  export CC_FLAGS="-I$OPENBLAS_PATH/include"
  export CC_LINK_FLAGS="-L$OPENBLAS_PATH/lib -lopenblas"
else
  pkg-config --exists openblas lapacke || return 1
  export CC="${CC:-cc}"
  export CC_FLAGS="$(pkg-config --cflags openblas lapacke)"
  export CC_LINK_FLAGS="$(pkg-config --libs openblas lapacke)"
fi
