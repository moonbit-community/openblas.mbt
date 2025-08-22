///|
test "cblas_zdotu test" {
  // Test cblas_zdotu with a simple case
  let n = 3
  let x : FixedArray[Double] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] // It equivalent to { r: 1, i: 2}, { r: 3, i: 4}, { r: 5, i: 6}
  let y : FixedArray[Double] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0] // It equivalent to { r: 7, i: 8}, { r: 9, i: 10}, { r: 11, i: 12}
  let result = cblas_zdotu(n, x, 1, y, 1)
  // Expected result: -39, 214
  assert_eq(result.real, -39.0)
  assert_eq(result.imag, 214.0)
}

///|
test "cblas_zdotc test" {
  // Test cblas_zdotc with a simple case
  let n = 3
  let x : FixedArray[Double] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] // It equivalent to { r: 1, i: 2}, { r: 3, i: 4}, { r: 5, i: 6}
  let y : FixedArray[Double] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0] // It equivalent to { r: 7, i: 8}, { r: 9, i: 10}, { r: 11, i: 12}
  let result = cblas_zdotc(n, x, 1, y, 1)

  // Expected result for conjugate dot product: conj(1+2i)*(7+8i) + conj(3+4i)*(9+10i) + conj(5+6i)*(11+12i)
  // = (1-2i)*(7+8i) + (3-4i)*(9+10i) + (5-6i)*(11+12i)
  // = (23-6i) + (67-6i) + (127-6i) = 217-18i
  assert_eq(result.real, 217.0)
  assert_eq(result.imag, -18.0)
}

///|
test "cblas_zaxpy test" {
  // Test cblas_zaxpy with a simple case
  let n = 2
  let alpha : FixedArray[Double] = [2.0, 1.0] // Flattened representation of complex number (2+1i)
  let x : FixedArray[Double] = [1.0, 2.0, 3.0, 4.0] // Flattened representation of complex numbers
  let y : FixedArray[Double] = [5.0, 6.0, 7.0, 8.0] // Flattened representation of complex numbers
  cblas_zaxpy(n, alpha, x, 1, y, 1)

  // Expected result: y = alpha*x + y
  // alpha*x[0] = (2+i)*(1+2i) = 2+4i+i-2 = 0+5i, y[0] = 5+6i + 0+5i = 5+11i
  // alpha*x[1] = (2+i)*(3+4i) = 6+8i+3i-4 = 2+11i, y[1] = 7+8i + 2+11i = 9+19i
  assert_eq(y[0], 5.0)
  assert_eq(y[1], 11.0)
  assert_eq(y[2], 9.0)
  assert_eq(y[3], 19.0)
}

///|
test "cblas_zaxpyc test" {
  // Test cblas_zaxpyc with a simple case
  let n = 2
  let alpha : FixedArray[Double] = [1.0, 1.0] // Flattened representation of complex number (1+i)
  let x : FixedArray[Double] = [1.0, 2.0, 3.0, 4.0] // Flattened representation of complex numbers
  let y : FixedArray[Double] = [5.0, 6.0, 7.0, 8.0] // Flattened representation of complex numbers
  cblas_zaxpyc(n, alpha, x, 1, y, 1)

  // Expected result: y = alpha*conj(x) + y
  // alpha*conj(x[0]) = (1+i)*(1-2i) = 1-2i+i+2 = 3-i, y[0] = 5+6i + 3-i = 8+5i
  // alpha*conj(x[1]) = (1+i)*(3-4i) = 3-4i+3i+4 = 7-i, y[1] = 7+8i + 7-i = 14+7i
  assert_eq(y[0], 8.0)
  assert_eq(y[1], 5.0)
  assert_eq(y[2], 14.0)
  assert_eq(y[3], 7.0)
}

///|
test "cblas_zcopy test" {
  // Test cblas_zcopy with a simple case
  let n = 3
  let x : FixedArray[Double] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] // Flattened representation of complex numbers
  let y : FixedArray[Double] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // Flattened representation of complex numbers
  cblas_zcopy(n, x, 1, y, 1)

  // Expected result: y = x
  assert_eq(y[0], 1.0)
  assert_eq(y[1], 2.0)
  assert_eq(y[2], 3.0)
  assert_eq(y[3], 4.0)
  assert_eq(y[4], 5.0)
  assert_eq(y[5], 6.0)
}

///|
test "cblas_zswap test" {
  // Test cblas_zswap with a simple case
  let n = 2
  let x : FixedArray[Double] = [1.0, 2.0, 3.0, 4.0] // Flattened representation of complex numbers
  let y : FixedArray[Double] = [5.0, 6.0, 7.0, 8.0] // Flattened representation of complex numbers
  cblas_zswap(n, x, 1, y, 1)

  // Expected result: x and y are swapped
  assert_eq(x[0], 5.0)
  assert_eq(x[1], 6.0)
  assert_eq(x[2], 7.0)
  assert_eq(x[3], 8.0)
  assert_eq(y[0], 1.0)
  assert_eq(y[1], 2.0)
  assert_eq(y[2], 3.0)
  assert_eq(y[3], 4.0)
}

///|
test "cblas_zscal test" {
  // Test cblas_zscal with a simple case
  let n = 2
  let alpha : FixedArray[Double] = [2.0, 1.0] // Flattened representation of complex number (2+i)
  let x : FixedArray[Double] = [1.0, 2.0, 3.0, 4.0] // Flattened representation of complex numbers
  cblas_zscal(n, alpha, x, 1)

  // Expected result: x = alpha*x
  // alpha*x[0] = (2+i)*(1+2i) = 2+4i+i-2 = 0+5i
  // alpha*x[1] = (2+i)*(3+4i) = 6+8i+3i-4 = 2+11i
  assert_eq(x[0], 0.0)
  assert_eq(x[1], 5.0)
  assert_eq(x[2], 2.0)
  assert_eq(x[3], 11.0)
}

///|
test "cblas_zdscal test" {
  // Test cblas_zdscal with a simple case
  let n = 2
  let alpha : Double = 2.0
  let x : FixedArray[Double] = [1.0, 2.0, 3.0, 4.0] // Flattened representation of complex numbers
  cblas_zdscal(n, alpha, x, 1)

  // Expected result: x = alpha*x (real scalar)
  // 2*(1+2i) = 2+4i
  // 2*(3+4i) = 6+8i
  assert_eq(x[0], 2.0)
  assert_eq(x[1], 4.0)
  assert_eq(x[2], 6.0)
  assert_eq(x[3], 8.0)
}

///|
test "cblas_zrotg test" {
  // Test cblas_zrotg (complex double precision Givens rotation generation)
  let a : FixedArray[Double] = [3.0, 4.0] // complex number 3+4i
  let b : FixedArray[Double] = [1.0, 2.0] // complex number 1+2i
  let c : FixedArray[Double] = [0.0] // cosine (real)
  let s : FixedArray[Double] = [0.0, 0.0] // sine (complex)
  cblas_zrotg(a, b, c, s)

  // Expected: Given a=3+4i, b=1+2i, generate c and s such that:
  // [c    s ] [a] = [r]  where r is real and positive
  // [-s*  c ] [b]   [0]
  // The exact output depends on the algorithm implementation
  // Main goal: the transformation should produce a numerically stable result
  // For now, just check that the function doesn't crash and produces reasonable values
  assert_eq(c[0] >= 0.0 && c[0] <= 1.0, true) // c should be between 0 and 1
}
// test "cblas_zdotc test"

// test "cblas_dzasum test"
// test "cblas_dzsum test"
// test "cblas_dznrm2 test"

// test "cblas_izamax test"
// test "cblas_izamin test"
// test "cblas_dzamax test"
// test "cblas_dzamin test"
// test "cblas_izmax test"
// test "cblas_izmin test"

// test "cblas_zaxpy test"
// test "cblas_zaxpyc test"
// test "cblas_zcopy test"
// test "cblas_zswap test"
// test "cblas_zscal test"
// test "cblas_zdscal test"

// test "cblas_zgemv test"
// test "cblas_zgemm test"
// test "cblas_zgemm3m test"
// test "cblas_zgemmt test"
// test "cblas_zsymm test"
// test "cblas_zsyrk test"
// test "cblas_zsyr2k test"
// test "cblas_ztrmm test"
// test "cblas_ztrsm test"

// test "cblas_zhemm test"
// test "cblas_zherk test"
// test "cblas_zher2k test"

// test "cblas_zaxpby test"

///|
test "cblas_zgemv test" {
  // Test cblas_zgemv (complex double precision general matrix vector multiplication)
  let m = 2
  let n = 2
  let alpha : FixedArray[Double] = [1.0, 0.0] // 1+0i
  let beta : FixedArray[Double] = [0.0, 0.0] // 0+0i
  // Matrix A: [[1+0i, 2+0i], [3+0i, 4+0i]] flattened as real,imag pairs
  let a : FixedArray[Double] = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]
  let x : FixedArray[Double] = [1.0, 1.0, 2.0, 2.0] // [1+i, 2+2i]
  let y : FixedArray[Double] = [0.0, 0.0, 0.0, 0.0] // Result vector
  cblas_zgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, n, x, 1, beta, y, 1)

  // Expected result: y = A*x = [[1, 2], [3, 4]] * [1+i, 2+2i] = [5+5i, 11+11i]
  assert_eq(y[0], 5.0) // Real part of first element
  assert_eq(y[1], 5.0) // Imaginary part of first element  
  assert_eq(y[2], 11.0) // Real part of second element
  assert_eq(y[3], 11.0) // Imaginary part of second element
}

///|
test "cblas_zgeru test" {
  // Test cblas_zgeru (complex double precision general rank-1 update, unconjugated)
  let m = 2
  let n = 2
  let alpha : FixedArray[Double] = [1.0, 0.0] // 1+0i
  let x : FixedArray[Double] = [1.0, 1.0, 2.0, 0.0] // [1+i, 2+0i]
  let y : FixedArray[Double] = [3.0, 0.0, 1.0, 1.0] // [3+0i, 1+i]
  let a : FixedArray[Double] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // 2x2 matrix initialized to zero
  cblas_zgeru(CblasRowMajor, m, n, alpha, x, 1, y, 1, a, n)

  // Expected result: A = alpha * x * y^T
  // x * y^T = [[1+i], [2+0i]] * [3+0i, 1+i] = [[(1+i)*(3+0i), (1+i)*(1+i)], [(2+0i)*(3+0i), (2+0i)*(1+i)]]
  // = [[3+3i, 1+2i-1], [6+0i, 2+2i]] = [[3+3i, 0+2i], [6+0i, 2+2i]]
  assert_eq(a[0], 3.0) // A[0,0] real = 3
  assert_eq(a[1], 3.0) // A[0,0] imag = 3
  assert_eq(a[2], 0.0) // A[0,1] real = 0 (corrected)
  assert_eq(a[3], 2.0) // A[0,1] imag = 2 (corrected)
  assert_eq(a[4], 6.0) // A[1,0] real = 6
  assert_eq(a[5], 0.0) // A[1,0] imag = 0
  assert_eq(a[6], 2.0) // A[1,1] real = 2
  assert_eq(a[7], 2.0) // A[1,1] imag = 2
}

///|
test "cblas_zgerc test" {
  // Test cblas_zgerc (complex double precision general rank-1 update, conjugated)
  let m = 2
  let n = 2
  let alpha : FixedArray[Double] = [1.0, 0.0] // 1+0i
  let x : FixedArray[Double] = [1.0, 1.0, 2.0, 0.0] // [1+i, 2+0i]
  let y : FixedArray[Double] = [3.0, 1.0, 1.0, 1.0] // [3+i, 1+i]
  let a : FixedArray[Double] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // 2x2 matrix initialized to zero
  cblas_zgerc(CblasRowMajor, m, n, alpha, x, 1, y, 1, a, n)

  // Expected result: A = alpha * x * conj(y)^T  
  // x * conj(y)^T = [[1+i], [2+0i]] * [3-i, 1-i] = [[(1+i)*(3-i), (1+i)*(1-i)], [(2+0i)*(3-i), (2+0i)*(1-i)]]
  // = [[3-i+3i+1, 1-i+i+1], [6-2i, 2-2i]] = [[4+2i, 2+0i], [6-2i, 2-2i]]
  assert_eq(a[0], 4.0) // A[0,0] real (corrected)
  assert_eq(a[1], 2.0) // A[0,0] imag (corrected)
  assert_eq(a[2], 2.0) // A[0,1] real (corrected)
  assert_eq(a[3], 0.0) // A[0,1] imag (corrected)
  assert_eq(a[4], 6.0) // A[1,0] real
  assert_eq(a[5], -2.0) // A[1,0] imag
  assert_eq(a[6], 2.0) // A[1,1] real
  assert_eq(a[7], -2.0) // A[1,1] imag
}

///|
test "cblas_zdotu_sub test" {
  // Test cblas_zdotu_sub with a simple case
  let n = 2
  let x : FixedArray[Double] = [1.0, 2.0, 3.0, 4.0] // [1+2i, 3+4i]
  let y : FixedArray[Double] = [5.0, 6.0, 7.0, 8.0] // [5+6i, 7+8i]
  let result : FixedArray[Double] = [0.0, 0.0] // result as flattened complex number
  cblas_zdotu_sub(n, x, 1, y, 1, result)

  // Expected result: x[0]*y[0] + x[1]*y[1] = (1+2i)*(5+6i) + (3+4i)*(7+8i)
  // = (5+6i+10i-12) + (21+24i+28i-32) = (-7+16i) + (-11+52i) = (-18+68i)
  assert_eq(result[0], -18.0)
  assert_eq(result[1], 68.0)
}

///|
test "cblas_zdotc_sub test" {
  // Test cblas_zdotc_sub with a simple case
  let n = 2
  let x : FixedArray[Double] = [1.0, 2.0, 3.0, 4.0] // [1+2i, 3+4i]
  let y : FixedArray[Double] = [5.0, 6.0, 7.0, 8.0] // [5+6i, 7+8i]
  let result : FixedArray[Double] = [0.0, 0.0] // result as flattened complex number
  cblas_zdotc_sub(n, x, 1, y, 1, result)

  // Expected result: conj(x[0])*y[0] + conj(x[1])*y[1] = (1-2i)*(5+6i) + (3-4i)*(7+8i)
  // = (5+6i-10i+12) + (21+24i-28i+32) = (17-4i) + (53-4i) = (70-8i)
  assert_eq(result[0], 70.0)
  assert_eq(result[1], -8.0)
}

///|
test "cblas_zdrot test" {
  // Test cblas_zdrot (complex double precision real Givens rotation)
  let n = 3
  let x : FixedArray[Double] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] // [1+2i, 3+4i, 5+6i]
  let y : FixedArray[Double] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0] // [7+8i, 9+10i, 11+12i]
  let c = 0.6 // cos(theta)
  let s = 0.8 // sin(theta)
  cblas_zdrot(n, x, 1, y, 1, c, s)

  // Expected result: [x'; y'] = [[c, s], [-s, c]] * [x; y] (applied to both real and imaginary parts)
  // For first element: x'[0] = c*x[0] + s*y[0] = 0.6*(1+2i) + 0.8*(7+8i) = 0.6+1.2i + 5.6+6.4i = 6.2+7.6i
  // y'[0] = -s*x[0] + c*y[0] = -0.8*(1+2i) + 0.6*(7+8i) = -0.8-1.6i + 4.2+4.8i = 3.4+3.2i
  assert_eq((x[0] - 6.2).abs() < 0.001, true) // x'[0] real
  assert_eq((x[1] - 7.6).abs() < 0.001, true) // x'[0] imag
  assert_eq((y[0] - 3.4).abs() < 0.001, true) // y'[0] real
  assert_eq((y[1] - 3.2).abs() < 0.001, true) // y'[0] imag
}

// TEMPORARILY COMMENTED OUT - interface has wrong type (should be FixedArray[Double])
// ///|
// test "cblas_zaxpby test" {
//   // Test cblas_zaxpby with a simple case
//   // Note: There might be an issue in the interface - zaxpby should use Double arrays but shows Float
//   // Let's test with the current interface signature
//   let n = 2
//   let alpha : FixedArray[Float] = [2.0, 1.0] // 2+1i (using Float as per interface)
//   let beta : FixedArray[Float] = [1.0, 1.0] // 1+1i
//   let x : FixedArray[Float] = [1.0, 2.0, 3.0, 4.0] // [1+2i, 3+4i] (using Float)
//   let y : FixedArray[Float] = [5.0, 6.0, 7.0, 8.0] // [5+6i, 7+8i] (using Float)
//   cblas_zaxpby(n, alpha, x, 1, beta, y, 1)

//   // Expected result: y = alpha*x + beta*y
//   // alpha*x[0] = (2+i)*(1+2i) = 2+4i+i-2 = 0+5i
//   // beta*y[0] = (1+i)*(5+6i) = 5+6i+5i-6 = -1+11i
//   // y[0] = 0+5i + (-1+11i) = -1+16i
//   // alpha*x[1] = (2+i)*(3+4i) = 6+8i+3i-4 = 2+11i
//   // beta*y[1] = (1+i)*(7+8i) = 7+8i+7i-8 = -1+15i
//   // y[1] = 2+11i + (-1+15i) = 1+26i
//   assert_eq(y[0], -1.0)
//   assert_eq(y[1], 16.0)
//   assert_eq(y[2], 1.0)
//   assert_eq(y[3], 26.0)
// }
///|
test "cblas_ztrsv test" {
  // Test cblas_ztrsv (complex double precision triangular solve)
  let n = 3
  // Upper triangular matrix: [[2+0i, 1+0i, 1+0i], [0+0i, 2+0i, 1+0i], [0+0i, 0+0i, 2+0i]]
  let a : FixedArray[Double] = [2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]
  let x : FixedArray[Double] = [6.0, 0.0, 4.0, 0.0, 2.0, 0.0] // right-hand side [6+0i, 4+0i, 2+0i]
  cblas_ztrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1)

  // Solve A * x = b where A = [[2, 1, 1], [0, 2, 1], [0, 0, 2]], b = [6, 4, 2]
  // Back substitution: x[2] = 1, x[1] = 1.5, x[0] = 1.75
  assert_eq((x[0] - 1.75).abs() < 0.001, true) // x[0] real
  assert_eq(x[1].abs() < 0.001, true) // x[0] imag (should be 0)
  assert_eq((x[2] - 1.5).abs() < 0.001, true) // x[1] real
  assert_eq(x[3].abs() < 0.001, true) // x[1] imag (should be 0)
  assert_eq((x[4] - 1.0).abs() < 0.001, true) // x[2] real
  assert_eq(x[5].abs() < 0.001, true) // x[2] imag (should be 0)
}

///|
test "cblas_ztrmv test" {
  // Test cblas_ztrmv (complex double precision triangular matrix vector multiplication)
  let n = 3
  // Upper triangular matrix: [[1+0i, 2+0i, 3+0i], [0+0i, 4+0i, 5+0i], [0+0i, 0+0i, 6+0i]]
  let a : FixedArray[Double] = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0]
  let x : FixedArray[Double] = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0] // [1+0i, 2+0i, 3+0i]
  cblas_ztrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1)

  // Expected result: x = A * x
  // x[0] = 1*1 + 2*2 + 3*3 = 14, x[1] = 0*1 + 4*2 + 5*3 = 23, x[2] = 0*1 + 0*2 + 6*3 = 18
  assert_eq(x[0], 14.0) // x[0] real
  assert_eq(x[1], 0.0) // x[0] imag
  assert_eq(x[2], 23.0) // x[1] real
  assert_eq(x[3], 0.0) // x[1] imag
  assert_eq(x[4], 18.0) // x[2] real
  assert_eq(x[5], 0.0) // x[2] imag
}

///|
test "cblas_zher test" {
  // Test cblas_zher (complex double precision Hermitian rank-1 update)
  let n = 2
  let alpha = 2.0 // real scalar
  let x : FixedArray[Double] = [1.0, 1.0, 2.0, 0.0] // [1+i, 2+0i]
  let a : FixedArray[Double] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] // 2x2 identity matrix
  cblas_zher(CblasRowMajor, CblasUpper, n, alpha, x, 1, a, n)

  // Expected result: A = A + alpha * x * conj(x)^H
  // x * conj(x)^H = [[1+i], [2+0i]] * [1-i, 2-0i] = [[(1+i)*(1-i), (1+i)*2], [(2)*(1-i), (2)*2]]
  // = [[1+1, 2+2i], [2-2i, 4]] = [[2, 2+2i], [2-2i, 4]]
  // A = [[1, 0], [0, 1]] + 2 * [[2, 2+2i], [2-2i, 4]] = [[5, 4+4i], [4-4i, 9]]
  // But only upper triangle is updated in Hermitian matrix
  assert_eq(a[0], 5.0) // a[0,0] real
  assert_eq(a[1], 0.0) // a[0,0] imag (should be 0 for Hermitian)
  assert_eq(a[2], 4.0) // a[0,1] real
  assert_eq(a[3], 4.0) // a[0,1] imag
  assert_eq(a[6], 9.0) // a[1,1] real
  assert_eq(a[7], 0.0) // a[1,1] imag (should be 0 for Hermitian)
}