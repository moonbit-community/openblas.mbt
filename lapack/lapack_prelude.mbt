///|
pub type Lapack_int = Int

///|

///|
#valtype
pub(all) struct Lapack_complex_float {
  re : Float
  im : Float
}

///|

///|
#valtype
pub(all) struct Lapack_complex_double {
  re : Double
  im : Double
}

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
