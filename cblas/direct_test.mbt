
test "cblas_sdsdot test" {
  // Test cblas_sdsdot with a simple case
  let n = 3
  let x: FixedArray[Float] = [1.0, 2.0, 3.0]
  let y: FixedArray[Float] = [4.0, 5.0, 6.0]
  let result = cblas_sdsdot(n, 1.0, x, 1, y, 1)
  
  // Expected result: 1.0 * (1*4 + 2*5 + 3*6) = 1.0 * (4 + 10 + 18) = 32.0
  assert_eq(result, 33.0)
}

test "cblas_dsdot test" {
  // Test cblas_dsdot with a simple case
  let n = 3
  let x: FixedArray[Float] = [1.0, 2.0, 3.0]
  let y: FixedArray[Float] = [4.0, 5.0, 6.0]
  let result = cblas_dsdot(n, x, 1, y, 1)
  
  // Expected result: (1*4 + 2*5 + 3*6) = (4 + 10 + 18) = 32.0
  assert_eq(result, 32.0)
}

test "cblas_sdot test" {
  // Test cblas_sdot with a simple case
  let n = 3
  let x: FixedArray[Float] = [1.0, 2.0, 3.0]
  let y: FixedArray[Float] = [4.0, 5.0, 6.0]
  let result = cblas_sdot(n, x, 1, y, 1)
  
  // Expected result: (1*4 + 2*5 + 3*6) = (4 + 10 + 18) = 32.0
  assert_eq(result, 32.0)
}

test "cblas_ddot test" {
  // Test cblas_ddot with a simple case
  let n = 3
  let x: FixedArray[Double] = [1.0, 2.0, 3.0]
  let y: FixedArray[Double] = [4.0, 5.0, 6.0]
  let result = cblas_ddot(n, x, 1, y, 1)
  
  // Expected result: (1*4 + 2*5 + 3*6) = (4 + 10 + 18) = 32.0
  assert_eq(result, 32.0)
}

// test "cblas_cdotu test" { ... }
// test "cblas_cdotc test" { ... }
// test "cblas_zdotu test" { ... }
// test "cblas_zdotc test" { ... }
// test "cblas_cdotu_sub test" { ... }
// test "cblas_cdotc_sub test" { ... }
// test "cblas_zdotu_sub test" { ... }
// test "cblas_zdotc_sub test" { ... }


test "cblas_sasum test" {
  // Test cblas_sasum with a simple case
  let n = 3
  let x: FixedArray[Float] = [1.0, -2.0, 3.0]
  let result = cblas_sasum(n, x, 1)
  
  // Expected result: |1.0| + |-2.0| + |3.0| = 1.0 + 2.0 + 3.0 = 6.0
  assert_eq(result, 6.0)
}

test "cblas_dasum test" {
  // Test cblas_dasum with a simple case
  let n = 3
  let x: FixedArray[Double] = [1.0, -2.0, 3.0]
  let result = cblas_dasum(n, x, 1)
  
  // Expected result: |1.0| + |-2.0| + |3.0| = 1.0 + 2.0 + 3.0 = 6.0
  assert_eq(result, 6.0)
}

// test "cblas_scasum test" { ... }
// test "cblas_dzasum test" { ... }

test "cblas_ssum test" {
  // Test cblas_sasum with a simple case
  let n = 3
  let x: FixedArray[Float] = [1.0, -2.0, 3.0]
  let result = cblas_ssum(n, x, 1)
  
  // Expected result: |1.0| + |-2.0| + |3.0| = 1.0 + 2.0 + 3.0 = 6.0
  assert_eq(result, 2.0)
}

test "cblas_dsum test" {
  // Test cblas_dasum with a simple case
  let n = 3
  let x: FixedArray[Double] = [1.0, -2.0, 3.0]
  let result = cblas_dsum(n, x, 1)
  
  // Expected result: |1.0| + |-2.0| + |3.0| = 1.0 + 2.0 + 3.0 = 6.0
  assert_eq(result, 2.0)
}

test "cblas_scsum test" {
  // Test cblas_sasum with a simple case
  let n = 3
  let x: FixedArray[Float] = [1.0, -2.0, 3.0]
  let result = cblas_sasum(n, x, 1)
  
  // Expected result: |1.0| + |-2.0| + |3.0| = 1.0 + 2.0 + 3.0 = 6.0
  assert_eq(result, 6.0)
}
