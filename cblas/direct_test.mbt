
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
