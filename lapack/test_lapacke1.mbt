///|
let lapack_col_major = 102

///|
let float_buf : FixedArray[Float] = [0.0]

///|
let float_stat_buf : FixedArray[Float] = [0.0, 0.0, 0.0, 0.0, 0.0]

///|
let double_buf : FixedArray[Double] = [0.0]

///|
let complex_float_buf : FixedArray[Float] = [0.0, 0.0]

///|
let complex_double_buf : FixedArray[Double] = [0.0, 0.0]

///|
let int_buf : FixedArray[Int] = [0]

///|
let int_stat_buf : FixedArray[Int] = [0, 0, 0]

///|
let byte_buf : FixedArray[Byte] = [0]

///|
test "test_lapack_make_complex_float" {
  let v = lapack_make_complex_float(1.0, 2.0)
  assert_eq(v.re, 1.0)
  assert_eq(v.im, 2.0)
}

///|
test "test_lapack_make_complex_double" {
  let v = lapack_make_complex_double(1.0, 2.0)
  assert_eq(v.re, 1.0)
  assert_eq(v.im, 2.0)
}

///|
test "test_lapacke_sbdsdc" {
  let info = lapacke_sbdsdc(
    lapack_col_major,
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    float_buf,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
    float_buf,
    int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dbdsdc" {
  let info = lapacke_dbdsdc(
    lapack_col_major,
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    double_buf,
    double_buf,
    double_buf,
    1,
    double_buf,
    1,
    double_buf,
    int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sbdsqr" {
  let info = lapacke_sbdsqr(
    lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    float_buf,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
    float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dbdsqr" {
  let info = lapacke_dbdsqr(
    lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    double_buf,
    double_buf,
    double_buf,
    1,
    double_buf,
    1,
    double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cbdsqr" {
  let info = lapacke_cbdsqr(
    lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zbdsqr" {
  let info = lapacke_zbdsqr(
    lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sbdsvdx" {
  let info = lapacke_sbdsvdx(
    lapack_col_major,
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    0,
    float_buf,
    float_buf,
    0.0,
    0.0,
    1,
    1,
    int_buf,
    float_buf,
    float_buf,
    1,
    int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dbdsvdx" {
  let info = lapacke_dbdsvdx(
    lapack_col_major,
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    0,
    double_buf,
    double_buf,
    0.0,
    0.0,
    1,
    1,
    int_buf,
    double_buf,
    double_buf,
    1,
    int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sdisna" {
  let info = lapacke_sdisna('E'.to_int().to_byte(), 0, 0, float_buf, float_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_ddisna" {
  let info = lapacke_ddisna(
    'E'.to_int().to_byte(),
    0,
    0,
    double_buf,
    double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgbbrd" {
  let info = lapacke_sgbbrd(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    0,
    float_buf,
    1,
    float_buf,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
    float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgbbrd" {
  let info = lapacke_dgbbrd(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    0,
    double_buf,
    1,
    double_buf,
    double_buf,
    double_buf,
    1,
    double_buf,
    1,
    double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgbbrd" {
  let info = lapacke_cgbbrd(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    0,
    complex_float_buf,
    1,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgbbrd" {
  let info = lapacke_zgbbrd(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    0,
    complex_double_buf,
    1,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgbcon" {
  let info = lapacke_sgbcon(
    lapack_col_major,
    '1'.to_int().to_byte(),
    0,
    0,
    0,
    float_buf,
    1,
    int_buf,
    0.0,
    float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgbcon" {
  let info = lapacke_dgbcon(
    lapack_col_major,
    '1'.to_int().to_byte(),
    0,
    0,
    0,
    double_buf,
    1,
    int_buf,
    0.0,
    double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgbcon" {
  let info = lapacke_cgbcon(
    lapack_col_major,
    '1'.to_int().to_byte(),
    0,
    0,
    0,
    complex_float_buf,
    1,
    int_buf,
    0.0,
    complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgbcon" {
  let info = lapacke_zgbcon(
    lapack_col_major,
    '1'.to_int().to_byte(),
    0,
    0,
    0,
    complex_double_buf,
    1,
    int_buf,
    0.0,
    complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgbequ" {
  let info = lapacke_sgbequ(
    lapack_col_major, 0, 0, 0, 0, float_buf, 1, float_buf, float_buf, float_buf,
    float_buf, float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgbequ" {
  let info = lapacke_dgbequ(
    lapack_col_major, 0, 0, 0, 0, double_buf, 1, double_buf, double_buf, double_buf,
    double_buf, double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgbequ" {
  let info = lapacke_cgbequ(
    lapack_col_major, 0, 0, 0, 0, complex_float_buf, 1, complex_float_buf, complex_float_buf,
    complex_float_buf, complex_float_buf, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgbequ" {
  let info = lapacke_zgbequ(
    lapack_col_major, 0, 0, 0, 0, complex_double_buf, 1, complex_double_buf, complex_double_buf,
    complex_double_buf, complex_double_buf, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgbequb" {
  let info = lapacke_sgbequb(
    lapack_col_major, 0, 0, 0, 0, float_buf, 1, float_buf, float_buf, float_buf,
    float_buf, float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgbequb" {
  let info = lapacke_dgbequb(
    lapack_col_major, 0, 0, 0, 0, double_buf, 1, double_buf, double_buf, double_buf,
    double_buf, double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgbequb" {
  let info = lapacke_cgbequb(
    lapack_col_major, 0, 0, 0, 0, complex_float_buf, 1, complex_float_buf, complex_float_buf,
    complex_float_buf, complex_float_buf, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgbequb" {
  let info = lapacke_zgbequb(
    lapack_col_major, 0, 0, 0, 0, complex_double_buf, 1, complex_double_buf, complex_double_buf,
    complex_double_buf, complex_double_buf, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgbrfs" {
  let info = lapacke_sgbrfs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    float_buf,
    1,
    float_buf,
    1,
    int_buf,
    float_buf,
    1,
    float_buf,
    1,
    float_buf,
    float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgbrfs" {
  let info = lapacke_dgbrfs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    double_buf,
    1,
    double_buf,
    1,
    int_buf,
    double_buf,
    1,
    double_buf,
    1,
    double_buf,
    double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgbrfs" {
  let info = lapacke_cgbrfs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    int_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgbrfs" {
  let info = lapacke_zgbrfs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    int_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
    complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgbsv" {
  let info = lapacke_sgbsv(
    lapack_col_major, 0, 0, 0, 0, float_buf, 1, int_buf, float_buf, 1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgbsv" {
  let info = lapacke_dgbsv(
    lapack_col_major, 0, 0, 0, 0, double_buf, 1, int_buf, double_buf, 1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgbsv" {
  let info = lapacke_cgbsv(
    lapack_col_major, 0, 0, 0, 0, complex_float_buf, 1, int_buf, complex_float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgbsv" {
  let info = lapacke_zgbsv(
    lapack_col_major, 0, 0, 0, 0, complex_double_buf, 1, int_buf, complex_double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgbsvx" {
  let info = lapacke_sgbsvx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    float_buf,
    1,
    float_buf,
    1,
    int_buf,
    byte_buf,
    float_buf,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
    float_buf,
    float_buf,
    float_buf,
    float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgbsvx" {
  let info = lapacke_dgbsvx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    double_buf,
    1,
    double_buf,
    1,
    int_buf,
    byte_buf,
    double_buf,
    double_buf,
    double_buf,
    1,
    double_buf,
    1,
    double_buf,
    double_buf,
    double_buf,
    double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgbsvx" {
  let info = lapacke_cgbsvx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    int_buf,
    byte_buf,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgbsvx" {
  let info = lapacke_zgbsvx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    int_buf,
    byte_buf,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgbtrf" {
  let info = lapacke_sgbtrf(lapack_col_major, 0, 0, 0, 0, float_buf, 1, int_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgbtrf" {
  let info = lapacke_dgbtrf(
    lapack_col_major, 0, 0, 0, 0, double_buf, 1, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgbtrf" {
  let info = lapacke_cgbtrf(
    lapack_col_major, 0, 0, 0, 0, complex_float_buf, 1, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgbtrf" {
  let info = lapacke_zgbtrf(
    lapack_col_major, 0, 0, 0, 0, complex_double_buf, 1, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgbtrs" {
  let info = lapacke_sgbtrs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    float_buf,
    1,
    int_buf,
    float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgbtrs" {
  let info = lapacke_dgbtrs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    double_buf,
    1,
    int_buf,
    double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgbtrs" {
  let info = lapacke_cgbtrs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    complex_float_buf,
    1,
    int_buf,
    complex_float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgbtrs" {
  let info = lapacke_zgbtrs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    0,
    complex_double_buf,
    1,
    int_buf,
    complex_double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgebak" {
  let info = lapacke_sgebak(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'R'.to_int().to_byte(),
    1,
    1,
    1,
    float_buf,
    1,
    float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgebak" {
  let info = lapacke_dgebak(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'R'.to_int().to_byte(),
    1,
    1,
    1,
    double_buf,
    1,
    double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgebak" {
  let info = lapacke_cgebak(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'R'.to_int().to_byte(),
    1,
    1,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgebak" {
  let info = lapacke_zgebak(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'R'.to_int().to_byte(),
    1,
    1,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgebal" {
  let info = lapacke_sgebal(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    float_buf,
    1,
    int_buf,
    int_buf,
    float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgebal" {
  let info = lapacke_dgebal(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    double_buf,
    1,
    int_buf,
    int_buf,
    double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgebal" {
  let info = lapacke_cgebal(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    complex_float_buf,
    1,
    int_buf,
    int_buf,
    complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgebal" {
  let info = lapacke_zgebal(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    complex_double_buf,
    1,
    int_buf,
    int_buf,
    complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgebrd" {
  let info = lapacke_sgebrd(
    lapack_col_major, 0, 0, float_buf, 1, float_buf, float_buf, float_buf, float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgebrd" {
  let info = lapacke_dgebrd(
    lapack_col_major, 0, 0, double_buf, 1, double_buf, double_buf, double_buf, double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgebrd" {
  let info = lapacke_cgebrd(
    lapack_col_major, 0, 0, complex_float_buf, 1, complex_float_buf, complex_float_buf,
    complex_float_buf, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgebrd" {
  let info = lapacke_zgebrd(
    lapack_col_major, 0, 0, complex_double_buf, 1, complex_double_buf, complex_double_buf,
    complex_double_buf, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgecon" {
  let info = lapacke_sgecon(
    lapack_col_major,
    '1'.to_int().to_byte(),
    0,
    float_buf,
    1,
    0.0,
    float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgecon" {
  let info = lapacke_dgecon(
    lapack_col_major,
    '1'.to_int().to_byte(),
    0,
    double_buf,
    1,
    0.0,
    double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgecon" {
  let info = lapacke_cgecon(
    lapack_col_major,
    '1'.to_int().to_byte(),
    0,
    complex_float_buf,
    1,
    0.0,
    complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgecon" {
  let info = lapacke_zgecon(
    lapack_col_major,
    '1'.to_int().to_byte(),
    0,
    complex_double_buf,
    1,
    0.0,
    complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgeequ" {
  let info = lapacke_sgeequ(
    lapack_col_major, 0, 0, float_buf, 1, float_buf, float_buf, float_buf, float_buf,
    float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgeequ" {
  let info = lapacke_dgeequ(
    lapack_col_major, 0, 0, double_buf, 1, double_buf, double_buf, double_buf, double_buf,
    double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgeequ" {
  let info = lapacke_cgeequ(
    lapack_col_major, 0, 0, complex_float_buf, 1, complex_float_buf, complex_float_buf,
    complex_float_buf, complex_float_buf, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgeequ" {
  let info = lapacke_zgeequ(
    lapack_col_major, 0, 0, complex_double_buf, 1, complex_double_buf, complex_double_buf,
    complex_double_buf, complex_double_buf, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgeequb" {
  let info = lapacke_sgeequb(
    lapack_col_major, 0, 0, float_buf, 1, float_buf, float_buf, float_buf, float_buf,
    float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgeequb" {
  let info = lapacke_dgeequb(
    lapack_col_major, 0, 0, double_buf, 1, double_buf, double_buf, double_buf, double_buf,
    double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgeequb" {
  let info = lapacke_cgeequb(
    lapack_col_major, 0, 0, complex_float_buf, 1, complex_float_buf, complex_float_buf,
    complex_float_buf, complex_float_buf, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgeequb" {
  let info = lapacke_zgeequb(
    lapack_col_major, 0, 0, complex_double_buf, 1, complex_double_buf, complex_double_buf,
    complex_double_buf, complex_double_buf, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgees" {
  let select : Lapack_s_select2 = (wr, wi) => {
    let v = wr[0] + wi[0]
    if v < 0.0 {
      return 1
    }
    0
  }
  let info = lapacke_sgees(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'S'.to_int().to_byte(),
    select,
    1,
    float_buf,
    1,
    int_buf,
    float_buf,
    float_buf,
    float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgeev" {
  let info = lapacke_sgeev(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    float_buf,
    1,
    float_buf,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgeev" {
  let info = lapacke_dgeev(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    double_buf,
    1,
    double_buf,
    double_buf,
    double_buf,
    1,
    double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgeev" {
  let info = lapacke_cgeev(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    complex_float_buf,
    1,
    complex_float_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgeev" {
  let info = lapacke_zgeev(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    complex_double_buf,
    1,
    complex_double_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgeevx" {
  let info = lapacke_sgeevx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    float_buf,
    1,
    float_buf,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
    int_buf,
    int_buf,
    float_buf,
    float_buf,
    float_buf,
    float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgeevx" {
  let info = lapacke_dgeevx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    double_buf,
    1,
    double_buf,
    double_buf,
    double_buf,
    1,
    double_buf,
    1,
    int_buf,
    int_buf,
    double_buf,
    double_buf,
    double_buf,
    double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgeevx" {
  let info = lapacke_cgeevx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    complex_float_buf,
    1,
    complex_float_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    int_buf,
    int_buf,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgeevx" {
  let info = lapacke_zgeevx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    complex_double_buf,
    1,
    complex_double_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    int_buf,
    int_buf,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgehrd" {
  let info = lapacke_sgehrd(lapack_col_major, 1, 1, 1, float_buf, 1, float_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgehrd" {
  let info = lapacke_dgehrd(
    lapack_col_major, 1, 1, 1, double_buf, 1, double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgehrd" {
  let info = lapacke_cgehrd(
    lapack_col_major, 1, 1, 1, complex_float_buf, 1, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgehrd" {
  let info = lapacke_zgehrd(
    lapack_col_major, 1, 1, 1, complex_double_buf, 1, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgejsv" {
  let info = lapacke_sgejsv(
    lapack_col_major,
    'A'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    float_buf,
    1,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
    float_stat_buf,
    int_stat_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgejsv" {
  let info = lapacke_dgejsv(
    lapack_col_major,
    'A'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    double_buf,
    1,
    double_buf,
    double_buf,
    1,
    double_buf,
    1,
    double_buf,
    int_stat_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgejsv" {
  let info = lapacke_cgejsv(
    lapack_col_major,
    'A'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    complex_float_buf,
    1,
    float_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    float_stat_buf,
    int_stat_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgejsv" {
  let info = lapacke_zgejsv(
    lapack_col_major,
    'A'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    complex_double_buf,
    1,
    double_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
    int_stat_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgelq2" {
  let info = lapacke_sgelq2(lapack_col_major, 0, 0, float_buf, 1, float_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgelq2" {
  let info = lapacke_dgelq2(lapack_col_major, 0, 0, double_buf, 1, double_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgelq2" {
  let info = lapacke_cgelq2(
    lapack_col_major, 0, 0, complex_float_buf, 1, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgelq2" {
  let info = lapacke_zgelq2(
    lapack_col_major, 0, 0, complex_double_buf, 1, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgelqf" {
  let info = lapacke_sgelqf(lapack_col_major, 0, 0, float_buf, 1, float_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgelqf" {
  let info = lapacke_dgelqf(lapack_col_major, 0, 0, double_buf, 1, double_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgelqf" {
  let info = lapacke_cgelqf(
    lapack_col_major, 0, 0, complex_float_buf, 1, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgelqf" {
  let info = lapacke_zgelqf(
    lapack_col_major, 0, 0, complex_double_buf, 1, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgels" {
  let info = lapacke_sgels(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    float_buf,
    1,
    float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgels" {
  let info = lapacke_dgels(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    double_buf,
    1,
    double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgels" {
  let info = lapacke_cgels(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgels" {
  let info = lapacke_zgels(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgelsd" {
  let info = lapacke_sgelsd(
    lapack_col_major, 0, 0, 0, float_buf, 1, float_buf, 1, float_buf, 0.0, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgelsd" {
  let info = lapacke_dgelsd(
    lapack_col_major, 0, 0, 0, double_buf, 1, double_buf, 1, double_buf, 0.0, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgelsd" {
  let info = lapacke_cgelsd(
    lapack_col_major, 0, 0, 0, complex_float_buf, 1, complex_float_buf, 1, float_buf,
    0.0, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgelsd" {
  let info = lapacke_zgelsd(
    lapack_col_major, 0, 0, 0, complex_double_buf, 1, complex_double_buf, 1, double_buf,
    0.0, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgelss" {
  let info = lapacke_sgelss(
    lapack_col_major, 0, 0, 0, float_buf, 1, float_buf, 1, float_buf, 0.0, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgelss" {
  let info = lapacke_dgelss(
    lapack_col_major, 0, 0, 0, double_buf, 1, double_buf, 1, double_buf, 0.0, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgelss" {
  let info = lapacke_cgelss(
    lapack_col_major, 0, 0, 0, complex_float_buf, 1, complex_float_buf, 1, float_buf,
    0.0, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgelss" {
  let info = lapacke_zgelss(
    lapack_col_major, 0, 0, 0, complex_double_buf, 1, complex_double_buf, 1, double_buf,
    0.0, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgelsy" {
  let info = lapacke_sgelsy(
    lapack_col_major, 0, 0, 0, float_buf, 1, float_buf, 1, int_buf, 0.0, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgelsy" {
  let info = lapacke_dgelsy(
    lapack_col_major, 0, 0, 0, double_buf, 1, double_buf, 1, int_buf, 0.0, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgelsy" {
  let info = lapacke_cgelsy(
    lapack_col_major, 0, 0, 0, complex_float_buf, 1, complex_float_buf, 1, int_buf,
    0.0, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgelsy" {
  let info = lapacke_zgelsy(
    lapack_col_major, 0, 0, 0, complex_double_buf, 1, complex_double_buf, 1, int_buf,
    0.0, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgeqlf" {
  let info = lapacke_sgeqlf(lapack_col_major, 0, 0, float_buf, 1, float_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgeqlf" {
  let info = lapacke_dgeqlf(lapack_col_major, 0, 0, double_buf, 1, double_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgeqlf" {
  let info = lapacke_cgeqlf(
    lapack_col_major, 0, 0, complex_float_buf, 1, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgeqlf" {
  let info = lapacke_zgeqlf(
    lapack_col_major, 0, 0, complex_double_buf, 1, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgeqp3" {
  let info = lapacke_sgeqp3(
    lapack_col_major, 0, 0, float_buf, 1, int_buf, float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgeqp3" {
  let info = lapacke_dgeqp3(
    lapack_col_major, 0, 0, double_buf, 1, int_buf, double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgeqp3" {
  let info = lapacke_cgeqp3(
    lapack_col_major, 0, 0, complex_float_buf, 1, int_buf, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgeqp3" {
  let info = lapacke_zgeqp3(
    lapack_col_major, 0, 0, complex_double_buf, 1, int_buf, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgeqpf" {
  let info = lapacke_sgeqpf(
    lapack_col_major, 0, 0, float_buf, 1, int_buf, float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgeqpf" {
  let info = lapacke_dgeqpf(
    lapack_col_major, 0, 0, double_buf, 1, int_buf, double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgeqpf" {
  let info = lapacke_cgeqpf(
    lapack_col_major, 0, 0, complex_float_buf, 1, int_buf, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgeqpf" {
  let info = lapacke_zgeqpf(
    lapack_col_major, 0, 0, complex_double_buf, 1, int_buf, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgeqr2" {
  let info = lapacke_sgeqr2(lapack_col_major, 0, 0, float_buf, 1, float_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgeqr2" {
  let info = lapacke_dgeqr2(lapack_col_major, 0, 0, double_buf, 1, double_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgeqr2" {
  let info = lapacke_cgeqr2(
    lapack_col_major, 0, 0, complex_float_buf, 1, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgeqr2" {
  let info = lapacke_zgeqr2(
    lapack_col_major, 0, 0, complex_double_buf, 1, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgeqrf" {
  let info = lapacke_sgeqrf(lapack_col_major, 0, 0, float_buf, 1, float_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgeqrf" {
  let info = lapacke_cgeqrf(
    lapack_col_major, 0, 0, complex_float_buf, 1, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgeqrf" {
  let info = lapacke_zgeqrf(
    lapack_col_major, 0, 0, complex_double_buf, 1, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgeqrfp" {
  let info = lapacke_sgeqrfp(lapack_col_major, 0, 0, float_buf, 1, float_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgeqrfp" {
  let info = lapacke_dgeqrfp(lapack_col_major, 0, 0, double_buf, 1, double_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgeqrfp" {
  let info = lapacke_cgeqrfp(
    lapack_col_major, 0, 0, complex_float_buf, 1, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgeqrfp" {
  let info = lapacke_zgeqrfp(
    lapack_col_major, 0, 0, complex_double_buf, 1, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgerfs" {
  let info = lapacke_sgerfs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    float_buf,
    1,
    float_buf,
    1,
    int_buf,
    float_buf,
    1,
    float_buf,
    1,
    float_buf,
    float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgerfs" {
  let info = lapacke_dgerfs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    double_buf,
    1,
    double_buf,
    1,
    int_buf,
    double_buf,
    1,
    double_buf,
    1,
    double_buf,
    double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgerfs" {
  let info = lapacke_cgerfs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    int_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgerfs" {
  let info = lapacke_zgerfs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    int_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
    complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgerqf" {
  let info = lapacke_sgerqf(lapack_col_major, 0, 0, float_buf, 1, float_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgerqf" {
  let info = lapacke_dgerqf(lapack_col_major, 0, 0, double_buf, 1, double_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgerqf" {
  let info = lapacke_cgerqf(
    lapack_col_major, 0, 0, complex_float_buf, 1, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgerqf" {
  let info = lapacke_zgerqf(
    lapack_col_major, 0, 0, complex_double_buf, 1, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgesdd" {
  let info = lapacke_sgesdd(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    float_buf,
    1,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgesdd" {
  let info = lapacke_dgesdd(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    double_buf,
    1,
    double_buf,
    double_buf,
    1,
    double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgesdd" {
  let info = lapacke_cgesdd(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    complex_float_buf,
    1,
    float_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgesdd" {
  let info = lapacke_zgesdd(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    complex_double_buf,
    1,
    double_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgesv" {
  let info = lapacke_sgesv(
    lapack_col_major, 0, 0, float_buf, 1, int_buf, float_buf, 1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgesv" {
  let info = lapacke_cgesv(
    lapack_col_major, 0, 0, complex_float_buf, 1, int_buf, complex_float_buf, 1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgesv" {
  let info = lapacke_zgesv(
    lapack_col_major, 0, 0, complex_double_buf, 1, int_buf, complex_double_buf, 1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dsgesv" {
  let info = lapacke_dsgesv(
    lapack_col_major, 0, 0, double_buf, 1, int_buf, double_buf, 1, double_buf, 1,
    int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zcgesv" {
  let info = lapacke_zcgesv(
    lapack_col_major, 0, 0, complex_double_buf, 1, int_buf, complex_double_buf, 1,
    complex_double_buf, 1, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgesvd" {
  let info = lapacke_sgesvd(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    float_buf,
    1,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
    float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgesvd" {
  let info = lapacke_cgesvd(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    complex_float_buf,
    1,
    float_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgesvd" {
  let info = lapacke_zgesvd(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    complex_double_buf,
    1,
    double_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgesvdx" {
  let info = lapacke_sgesvdx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    0,
    0,
    float_buf,
    1,
    0.0,
    0.0,
    1,
    1,
    int_buf,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
    int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgesvdx" {
  let info = lapacke_dgesvdx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    0,
    0,
    double_buf,
    1,
    0.0,
    0.0,
    1,
    1,
    int_buf,
    double_buf,
    double_buf,
    1,
    double_buf,
    1,
    int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgesvdx" {
  let info = lapacke_cgesvdx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    0,
    0,
    complex_float_buf,
    1,
    0.0,
    0.0,
    1,
    1,
    int_buf,
    float_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgesvdx" {
  let info = lapacke_zgesvdx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    0,
    0,
    complex_double_buf,
    1,
    0.0,
    0.0,
    1,
    1,
    int_buf,
    double_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgesvdq" {
  let info = lapacke_sgesvdq(
    lapack_col_major,
    'A'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    float_buf,
    1,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
    int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgesvdq" {
  let info = lapacke_dgesvdq(
    lapack_col_major,
    'A'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    double_buf,
    1,
    double_buf,
    double_buf,
    1,
    double_buf,
    1,
    int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgesvdq" {
  let info = lapacke_cgesvdq(
    lapack_col_major,
    'A'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    complex_float_buf,
    1,
    float_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgesvdq" {
  let info = lapacke_zgesvdq(
    lapack_col_major,
    'A'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    complex_double_buf,
    1,
    double_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    int_buf,
  )
  assert_eq(info, -10)
}

///|
test "test_lapacke_sgesvj" {
  let info = lapacke_sgesvj(
    lapack_col_major,
    'G'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    float_buf,
    1,
    float_buf,
    0,
    float_buf,
    1,
    float_stat_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgesvj" {
  let info = lapacke_dgesvj(
    lapack_col_major,
    'G'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    double_buf,
    1,
    double_buf,
    0,
    double_buf,
    1,
    double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgesvj" {
  let info = lapacke_cgesvj(
    lapack_col_major,
    'G'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    complex_float_buf,
    1,
    float_buf,
    1,
    complex_float_buf,
    1,
    float_stat_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgesvj" {
  let info = lapacke_zgesvj(
    lapack_col_major,
    'G'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    complex_double_buf,
    1,
    double_buf,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgesvx" {
  let info = lapacke_sgesvx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    float_buf,
    1,
    float_buf,
    1,
    int_buf,
    byte_buf,
    float_buf,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
    float_buf,
    float_buf,
    float_buf,
    float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgesvx" {
  let info = lapacke_dgesvx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    double_buf,
    1,
    double_buf,
    1,
    int_buf,
    byte_buf,
    double_buf,
    double_buf,
    double_buf,
    1,
    double_buf,
    1,
    double_buf,
    double_buf,
    double_buf,
    double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgesvx" {
  let info = lapacke_cgesvx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    int_buf,
    byte_buf,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgesvx" {
  let info = lapacke_zgesvx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    int_buf,
    byte_buf,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgetf2" {
  let info = lapacke_sgetf2(lapack_col_major, 0, 0, float_buf, 1, int_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgetf2" {
  let info = lapacke_dgetf2(lapack_col_major, 0, 0, double_buf, 1, int_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgetf2" {
  let info = lapacke_cgetf2(
    lapack_col_major, 0, 0, complex_float_buf, 1, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgetf2" {
  let info = lapacke_zgetf2(
    lapack_col_major, 0, 0, complex_double_buf, 1, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgetrf" {
  let info = lapacke_sgetrf(lapack_col_major, 0, 0, float_buf, 1, int_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgetrf" {
  let info = lapacke_cgetrf(
    lapack_col_major, 0, 0, complex_float_buf, 1, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgetrf" {
  let info = lapacke_zgetrf(
    lapack_col_major, 0, 0, complex_double_buf, 1, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgetrf2" {
  let info = lapacke_sgetrf2(lapack_col_major, 0, 0, float_buf, 1, int_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgetrf2" {
  let info = lapacke_dgetrf2(lapack_col_major, 0, 0, double_buf, 1, int_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgetrf2" {
  let info = lapacke_cgetrf2(
    lapack_col_major, 0, 0, complex_float_buf, 1, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgetrf2" {
  let info = lapacke_zgetrf2(
    lapack_col_major, 0, 0, complex_double_buf, 1, int_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgetri" {
  let info = lapacke_sgetri(lapack_col_major, 0, float_buf, 1, int_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgetri" {
  let info = lapacke_dgetri(lapack_col_major, 0, double_buf, 1, int_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgetri" {
  let info = lapacke_cgetri(lapack_col_major, 0, complex_float_buf, 1, int_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgetri" {
  let info = lapacke_zgetri(lapack_col_major, 0, complex_double_buf, 1, int_buf)
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgetrs" {
  let info = lapacke_sgetrs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    float_buf,
    1,
    int_buf,
    float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgetrs" {
  let info = lapacke_dgetrs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    double_buf,
    1,
    int_buf,
    double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgetrs" {
  let info = lapacke_cgetrs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    complex_float_buf,
    1,
    int_buf,
    complex_float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgetrs" {
  let info = lapacke_zgetrs(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    complex_double_buf,
    1,
    int_buf,
    complex_double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sggbak" {
  let info = lapacke_sggbak(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'R'.to_int().to_byte(),
    1,
    1,
    1,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dggbak" {
  let info = lapacke_dggbak(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'R'.to_int().to_byte(),
    1,
    1,
    1,
    double_buf,
    double_buf,
    1,
    double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cggbak" {
  let info = lapacke_cggbak(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'R'.to_int().to_byte(),
    1,
    1,
    1,
    complex_float_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zggbak" {
  let info = lapacke_zggbak(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'R'.to_int().to_byte(),
    1,
    1,
    1,
    complex_double_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sggbal" {
  let info = lapacke_sggbal(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    float_buf,
    1,
    float_buf,
    1,
    int_buf,
    int_buf,
    float_buf,
    float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dggbal" {
  let info = lapacke_dggbal(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    double_buf,
    1,
    double_buf,
    1,
    int_buf,
    int_buf,
    double_buf,
    double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cggbal" {
  let info = lapacke_cggbal(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    int_buf,
    int_buf,
    complex_float_buf,
    complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zggbal" {
  let info = lapacke_zggbal(
    lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    int_buf,
    int_buf,
    complex_double_buf,
    complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sggev" {
  let info = lapacke_sggev(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    float_buf,
    1,
    float_buf,
    1,
    float_buf,
    float_buf,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dggev" {
  let info = lapacke_dggev(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    double_buf,
    1,
    double_buf,
    1,
    double_buf,
    double_buf,
    double_buf,
    double_buf,
    1,
    double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cggev" {
  let info = lapacke_cggev(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zggev" {
  let info = lapacke_zggev(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sggev3" {
  let info = lapacke_sggev3(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    float_buf,
    1,
    float_buf,
    1,
    float_buf,
    float_buf,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dggev3" {
  let info = lapacke_dggev3(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    double_buf,
    1,
    double_buf,
    1,
    double_buf,
    double_buf,
    double_buf,
    double_buf,
    1,
    double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cggev3" {
  let info = lapacke_cggev3(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zggev3" {
  let info = lapacke_zggev3(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sggevx" {
  let info = lapacke_sggevx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    float_buf,
    1,
    float_buf,
    1,
    float_buf,
    float_buf,
    float_buf,
    float_buf,
    1,
    float_buf,
    1,
    int_buf,
    int_buf,
    float_buf,
    float_buf,
    float_buf,
    float_buf,
    float_buf,
    float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dggevx" {
  let info = lapacke_dggevx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    double_buf,
    1,
    double_buf,
    1,
    double_buf,
    double_buf,
    double_buf,
    double_buf,
    1,
    double_buf,
    1,
    int_buf,
    int_buf,
    double_buf,
    double_buf,
    double_buf,
    double_buf,
    double_buf,
    double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cggevx" {
  let info = lapacke_cggevx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    int_buf,
    int_buf,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
    complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zggevx" {
  let info = lapacke_zggevx(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    int_buf,
    int_buf,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
    complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sggglm" {
  let info = lapacke_sggglm(
    lapack_col_major, 0, 0, 0, float_buf, 1, float_buf, 1, float_buf, float_buf,
    float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dggglm" {
  let info = lapacke_dggglm(
    lapack_col_major, 0, 0, 0, double_buf, 1, double_buf, 1, double_buf, double_buf,
    double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cggglm" {
  let info = lapacke_cggglm(
    lapack_col_major, 0, 0, 0, complex_float_buf, 1, complex_float_buf, 1, complex_float_buf,
    complex_float_buf, complex_float_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zggglm" {
  let info = lapacke_zggglm(
    lapack_col_major, 0, 0, 0, complex_double_buf, 1, complex_double_buf, 1, complex_double_buf,
    complex_double_buf, complex_double_buf,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgghrd" {
  let info = lapacke_sgghrd(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    float_buf,
    1,
    float_buf,
    1,
    float_buf,
    1,
    float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgghrd" {
  let info = lapacke_dgghrd(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    double_buf,
    1,
    double_buf,
    1,
    double_buf,
    1,
    double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgghrd" {
  let info = lapacke_cgghrd(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_zgghrd" {
  let info = lapacke_zgghrd(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
    complex_double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_sgghd3" {
  let info = lapacke_sgghd3(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    float_buf,
    1,
    float_buf,
    1,
    float_buf,
    1,
    float_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_dgghd3" {
  let info = lapacke_dgghd3(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    double_buf,
    1,
    double_buf,
    1,
    double_buf,
    1,
    double_buf,
    1,
  )
  assert_eq(info, 0)
}

///|
test "test_lapacke_cgghd3" {
  let info = lapacke_cgghd3(
    lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
    complex_float_buf,
    1,
  )
  assert_eq(info, 0)
}
