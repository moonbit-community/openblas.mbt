///|
let lapack_col_major = 102

///|
test "lapacke_dgesv" {
  let n = 2
  let nrhs = 1
  // A = [[3, 1], [1, 2]] in column-major order.
  let a : FixedArray[Double] = [3.0, 1.0, 1.0, 2.0]
  let b : FixedArray[Double] = [4.0, 3.0]
  let ipiv : FixedArray[Int] = [0, 0]
  let info = lapacke_dgesv(lapack_col_major, n, nrhs, a, n, ipiv, b, n)
  assert_eq(info, 0)
  assert_eq((b[0] - 1.0).abs() < 0.001, true)
  assert_eq((b[1] - 1.0).abs() < 0.001, true)
}

///|
test "lapacke_dgetrf" {
  let n = 2
  let a : FixedArray[Double] = [1.0, 0.0, 0.0, 1.0]
  let ipiv : FixedArray[Int] = [0, 0]
  let info = lapacke_dgetrf(lapack_col_major, n, n, a, n, ipiv)
  assert_eq(info, 0)
  assert_eq(ipiv[0], 1)
  assert_eq(ipiv[1], 2)
}

///|
test "lapacke_dgeqrf" {
  let n = 2
  let a : FixedArray[Double] = [1.0, 0.0, 0.0, 1.0]
  let tau : FixedArray[Double] = [0.0, 0.0]
  let info = lapacke_dgeqrf(lapack_col_major, n, n, a, n, tau)
  assert_eq(info, 0)
  assert_eq((a[0].abs() - 1.0).abs() < 0.001, true)
  assert_eq((a[3].abs() - 1.0).abs() < 0.001, true)
  assert_eq(a[1].abs() < 0.001, true)
  assert_eq(a[2].abs() < 0.001, true)
}

///|
test "lapacke_dgesvd" {
  let n = 2
  let a : FixedArray[Double] = [3.0, 0.0, 0.0, 2.0]
  let s : FixedArray[Double] = [0.0, 0.0]
  let u : FixedArray[Double] = [0.0, 0.0, 0.0, 0.0]
  let vt : FixedArray[Double] = [0.0, 0.0, 0.0, 0.0]
  let superb : FixedArray[Double] = [0.0]
  let info = lapacke_dgesvd(
    lapack_col_major,
    'S'.to_int().to_byte(),
    'S'.to_int().to_byte(),
    n,
    n,
    a,
    n,
    s,
    u,
    n,
    vt,
    n,
    superb,
  )
  assert_eq(info, 0)
  assert_eq((s[0] - 3.0).abs() < 0.001, true)
  assert_eq((s[1] - 2.0).abs() < 0.001, true)
}
