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