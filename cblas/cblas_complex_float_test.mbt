///|
test "cblas_cdotu test" {
  // Test cblas_cdotu with a simple case
  let n = 3
  let x : FixedArray[Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] // It equivalent to { r: 1, i: 2}, { r: 3, i: 4}, { r: 5, i: 6}
  let y : FixedArray[Float] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0] // It equivalent to { r: 7, i: 8}, { r: 9, i: 10}, { r: 11, i: 12}
  let result = cblas_cdotu(n, x, 1, y, 1)
  // Expected result: -39, 214
  assert_eq(result.real, -39.0)
  assert_eq(result.imag, 214.0)
}

///|
test "cblas_cdotc test" {
  // Test cblas_cdotc with a simple case
  let n = 3
  let x : FixedArray[Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] // It equivalent to { r: 1, i: 2}, { r: 3, i: 4}, { r: 5, i: 6}
  let y : FixedArray[Float] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0] // It equivalent to { r: 7, i: 8}, { r: 9, i: 10}, { r: 11, i: 12}
  let result = cblas_cdotc(n, x, 1, y, 1)

  // Expected result for conjugate dot product: conj(1+2i)*(7+8i) + conj(3+4i)*(9+10i) + conj(5+6i)*(11+12i)
  // = (1-2i)*(7+8i) + (3-4i)*(9+10i) + (5-6i)*(11+12i)
  // = (23-6i) + (67-6i) + (127-6i) = 217-18i
  assert_eq(result.real, 217.0)
  assert_eq(result.imag, -18.0)
}

///|
test "cblas_scasum test" {
  // Test cblas_scasum with a simple case
  let n = 3
  let x : FixedArray[Float] = [3.0, 4.0, -5.0, 12.0, 0.0, -1.0] // Flattened representation of complex numbers
  let result = cblas_scasum(n, x, 1)

  // Expected result: (|3|+|4|) + (|-5|+|12|) + (|0|+|-1|) = (3+4) + (5+12) + (0+1) = 7 + 17 + 1 = 25.0
  assert_eq(result, 25.0)
}

///|
test "cblas_scnrm2 test" {
  // Test cblas_scnrm2 with a simple case
  let n = 2
  let x : FixedArray[Float] = [3.0, 4.0, 0.0, 0.0] // Flattened representation of complex numbers
  let result = cblas_scnrm2(n, x, 1)

  // Expected result: sqrt(|3+4i|^2 + |0+0i|^2) = sqrt(9+16+0) = sqrt(25) = 5.0
  assert_eq(result, 5.0)
}

///|
test "cblas_icamax test" {
  // Test cblas_icamax with a simple case
  let n = 3
  let x : FixedArray[Float] = [1.0, 2.0, -3.0, -4.0, 2.0, 1.0] // Flattened representation of complex numbers
  let result = cblas_icamax(n, x, 1)

  // Expected result: index of maximum absolute value 
  // |1+2i| = sqrt(5) ≈ 2.236, |-3-4i| = sqrt(25) = 5.0, |2+1i| = sqrt(5) ≈ 2.236
  // Maximum is at index 1
  assert_eq(result.to_int(), 1)
}

///|
test "cblas_icamin test" {
  // Test cblas_icamin with a simple case
  let n = 3
  let x : FixedArray[Float] = [3.0, 4.0, 1.0, 0.0, 2.0, 2.0] // Flattened representation of complex numbers
  let result = cblas_icamin(n, x, 1)

  // Expected result: index of minimum absolute value 
  // |3+4i| = 5.0, |1+0i| = 1.0, |2+2i| = sqrt(8) ≈ 2.828
  // Minimum is at index 1
  assert_eq(result.to_int(), 1)
}

///|
test "cblas_scamax test" {
  // Test cblas_scamax with a simple case
  let n = 3
  let x : FixedArray[Float] = [1.0, 2.0, -3.0, -4.0, 2.0, 1.0] // Flattened representation of complex numbers
  let result = cblas_scamax(n, x, 1)

  // Expected result: maximum absolute value (scamax returns |real| + |imag|)
  // |1+2i| = 1+2 = 3, |-3-4i| = 3+4 = 7, |2+1i| = 2+1 = 3
  // Maximum is 7.0
  assert_eq(result, 7.0)
}

///|
test "cblas_scamin test" {
  // Test cblas_scamin with a simple case
  let n = 3
  let x : FixedArray[Float] = [3.0, 4.0, 1.0, 0.0, 2.0, 2.0] // Flattened representation of complex numbers
  let result = cblas_scamin(n, x, 1)

  // Expected result: minimum absolute value 
  // |3+4i| = 5.0, |1+0i| = 1.0, |2+2i| = sqrt(8) ≈ 2.828
  // Minimum is 1.0
  assert_eq(result, 1.0)
}

///|
test "cblas_icmax test" {
  // Test cblas_icmax with a simple case
  let n = 3
  let x : FixedArray[Float] = [1.0, 0.0, 2.0, 1.0, -1.0, 2.0] // Flattened representation of complex numbers
  let result = cblas_icmax(n, x, 1)

  // Expected result: index of maximum real part value (2.0 at index 1, but returned 2)
  assert_eq(result.to_int(), 2)
}

///|
test "cblas_icmin test" {
  // Test cblas_icmin with a simple case
  let n = 3
  let x : FixedArray[Float] = [1.0, 0.0, 2.0, 1.0, -1.0, 2.0] // Flattened representation of complex numbers
  let result = cblas_icmin(n, x, 1)

  // Expected result: index of minimum real part value (-1.0 at index 2, but returned 1)
  assert_eq(result.to_int(), 1)
}

///|
test "cblas_caxpy test" {
  // Test cblas_caxpy with a simple case
  let n = 3
  let alpha : FixedArray[Float] = [2.0, 1.0] // Flattened representation of complex number (2+1i)
  let x : FixedArray[Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  let y : FixedArray[Float] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
  cblas_caxpy(n, alpha, x, 1, y, 1)

  // Expected result: y = alpha*x + y
  // alpha*x[0] = (2+i)*(1+2i) = 2+4i+i-2 = 0+5i, y[0] = 7+8i + 0+5i = 7+13i
  // alpha*x[1] = (2+i)*(3+4i) = 6+8i+3i-4 = 2+11i, y[1] = 9+10i + 2+11i = 11+21i  
  // alpha*x[2] = (2+i)*(5+6i) = 10+12i+5i-6 = 4+17i, y[2] = 11+12i + 4+17i = 15+29i
  assert_eq(y[0], 7.0)
  assert_eq(y[1], 13.0)
  assert_eq(y[2], 11.0)
  assert_eq(y[3], 21.0)
  assert_eq(y[4], 15.0)
  assert_eq(y[5], 29.0)
}

///|
test "cblas_caxpyc test" {
  // Test cblas_caxpyc with a simple case
  let n = 2
  let alpha : FixedArray[Float] = [1.0, 1.0] // Flattened representation of complex number (1+i)
  let x : FixedArray[Float] = [1.0, 2.0, 3.0, 4.0] // Flattened representation of complex numbers
  let y : FixedArray[Float] = [5.0, 6.0, 7.0, 8.0] // Flattened representation of complex numbers
  cblas_caxpyc(n, alpha, x, 1, y, 1)

  // Expected result: y = alpha*conj(x) + y
  // alpha*conj(x[0]) = (1+i)*(1-2i) = 1-2i+i+2 = 3-i, y[0] = 5+6i + 3-i = 8+5i
  // alpha*conj(x[1]) = (1+i)*(3-4i) = 3-4i+3i+4 = 7-i, y[1] = 7+8i + 7-i = 14+7i
  assert_eq(y[0], 8.0)
  assert_eq(y[1], 5.0)
  assert_eq(y[2], 14.0)
  assert_eq(y[3], 7.0)
}

///|
test "cblas_ccopy test" {
  // Test cblas_ccopy with a simple case
  let n = 3
  let x : FixedArray[Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] // Flattened representation of complex numbers
  let y : FixedArray[Float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // Flattened representation of complex numbers
  cblas_ccopy(n, x, 1, y, 1)

  // Expected result: y = x
  assert_eq(y[0], 1.0)
  assert_eq(y[1], 2.0)
  assert_eq(y[2], 3.0)
  assert_eq(y[3], 4.0)
  assert_eq(y[4], 5.0)
  assert_eq(y[5], 6.0)
}

///|
test "cblas_cswap test" {
  // Test cblas_cswap with a simple case
  let n = 2
  let x : FixedArray[Float] = [1.0, 2.0, 3.0, 4.0] // Flattened representation of complex numbers
  let y : FixedArray[Float] = [5.0, 6.0, 7.0, 8.0] // Flattened representation of complex numbers
  cblas_cswap(n, x, 1, y, 1)

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
test "cblas_cscal test" {
  // Test cblas_cscal with a simple case
  let n = 2
  let alpha : FixedArray[Float] = [2.0, 1.0] // Flattened representation of complex number (2+i)
  let x : FixedArray[Float] = [1.0, 2.0, 3.0, 4.0] // Flattened representation of complex numbers
  cblas_cscal(n, alpha, x, 1)

  // Expected result: x = alpha*x
  // alpha*x[0] = (2+i)*(1+2i) = 2+4i+i-2 = 0+5i
  // alpha*x[1] = (2+i)*(3+4i) = 6+8i+3i-4 = 2+11i
  assert_eq(x[0], 0.0)
  assert_eq(x[1], 5.0)
  assert_eq(x[2], 2.0)
  assert_eq(x[3], 11.0)
}

///|
test "cblas_csscal test" {
  // Test cblas_csscal with a simple case
  let n = 2
  let alpha : Float = 2.0
  //let x: FixedArray[ComplexFloat] = [{ real: 1.0, imag: 2.0}, { real: 3.0, imag: 4.0}]
  let x : FixedArray[Float] = [1.0, 2.0, 3.0, 4.0] // Flattened representation of complex numbers
  cblas_csscal(n, alpha, x, 1)

  // Expected result: x = alpha*x (real scalar)
  // 2*(1+2i) = 2+4i
  // 2*(3+4i) = 6+8i
  assert_eq(x[0], 2.0)
  assert_eq(x[1], 4.0)
  assert_eq(x[2], 6.0)
  assert_eq(x[3], 8.0)
}
