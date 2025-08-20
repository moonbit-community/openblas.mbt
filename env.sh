OPENBLAS_PATH=$(brew --prefix openblas)

export CC=gcc
export CC_FLAGS="-I$OPENBLAS_PATH/include"
export CC_LINK_FLAGS="-L$OPENBLAS_PATH/lib -lopenblas"
export C_INCLUDE_PATH="$C_INCLUDE_PATH:/opt/homebrew/include:$OPENBLAS_PATH/include"
