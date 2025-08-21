test "cblas_zdotu test" {
  // Test cblas_cdotu with a simple case
  let n = 3
  let x : FixedArray[Double] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] // It equivalent to { r: 1, i: 2}, { r: 3, i: 4}, { r: 5, i: 6}
  let y : FixedArray[Double] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0] // It equivalent to { r: 7, i: 8}, { r: 9, i: 10}, { r: 11, i: 12}
  let result = cblas_zdotu(n, x, 1, y, 1)
  // Expected result: -39, 214
  assert_eq(result.real, -39.0)
  assert_eq(result.imag, 214.0)
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
