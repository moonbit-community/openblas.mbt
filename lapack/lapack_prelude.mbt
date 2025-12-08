///|
pub type Lapack_int = Int

///|
pub type Lapack_complex_float = (Float, Float)

///|
pub type Lapack_complex_double = (Double, Double)

///|
pub type Lapack_logical = Lapack_int

///|
pub type Lapack_s_select2 = FuncRef[
  (FixedArray[Float], FixedArray[Float]) -> Lapack_logical,
]

///|
pub type Lapack_s_select3 = FuncRef[
  (FixedArray[Float], FixedArray[Float], FixedArray[Float]) -> Lapack_logical,
]

///|
pub type Lapack_d_select2 = FuncRef[
  (FixedArray[Double], FixedArray[Double]) -> Lapack_logical,
]

///|
pub type Lapack_d_select3 = FuncRef[
  (FixedArray[Double], FixedArray[Double], FixedArray[Double]) -> Lapack_logical,
]

///|
pub type Lapack_c_select1 = FuncRef[(FixedArray[Float]) -> Lapack_logical]

///|
pub type Lapack_c_select2 = FuncRef[
  (FixedArray[Float], FixedArray[Float]) -> Lapack_logical,
]

///|
pub type Lapack_z_select1 = FuncRef[(FixedArray[Double]) -> Lapack_logical]

///|
pub type Lapack_z_select2 = FuncRef[
  (FixedArray[Double], FixedArray[Double]) -> Lapack_logical,
]
