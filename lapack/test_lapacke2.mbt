///|
let e2_lapack_col_major = 102

///|
let e2_float_buf : FixedArray[Float] = [
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
]

///|
let e2_float_buf2 : FixedArray[Float] = [
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
]

///|
let e2_double_buf : FixedArray[Double] = [
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
]

///|
let e2_double_buf2 : FixedArray[Double] = [
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
]

///|
let e2_complex_float_buf : FixedArray[Float] = [
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
]

///|
let e2_complex_float_buf2 : FixedArray[Float] = [
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
]

///|
let e2_complex_double_buf : FixedArray[Double] = [
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
]

///|
let e2_complex_double_buf2 : FixedArray[Double] = [
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
]

///|
let e2_int_buf : FixedArray[Int] = [0, 0, 0, 0, 0, 0, 0, 0]

///|
let e2_int_buf2 : FixedArray[Int] = [0, 0, 0, 0, 0, 0, 0, 0]

///|
let e2_iseed_buf : FixedArray[Int] = [1, 2, 3, 4]

///|
let e2_complex_float_zero : Lapack_complex_float = lapack_make_complex_float(
  0.0,
  0.0,
)

///|
let e2_complex_double_zero : Lapack_complex_double = lapack_make_complex_double(
  0.0,
  0.0,
)

///|
test "test_lapacke2_gg" {
  let info_sggqrf = lapacke_sggqrf(
    e2_lapack_col_major,
    0,
    0,
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
    e2_float_buf,
    1,
    e2_float_buf2,
  )
  assert_eq(info_sggqrf, 0)

  let info_dggqrf = lapacke_dggqrf(
    e2_lapack_col_major,
    0,
    0,
    1,
    e2_double_buf,
    1,
    e2_double_buf2,
    e2_double_buf,
    1,
    e2_double_buf2,
  )
  assert_eq(info_dggqrf, 0)

  let info_cggqrf = lapacke_cggqrf(
    e2_lapack_col_major,
    0,
    0,
    1,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
  )
  assert_eq(info_cggqrf, 0)

  let info_zggqrf = lapacke_zggqrf(
    e2_lapack_col_major,
    1,
    1,
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
  )
  assert_eq(info_zggqrf, 0)

  let info_sggrqf = lapacke_sggrqf(
    e2_lapack_col_major,
    0,
    0,
    0,
    e2_float_buf,
    1,
    e2_float_buf2,
    e2_float_buf,
    1,
    e2_float_buf2,
  )
  assert_eq(info_sggrqf, 0)

  let info_dggrqf = lapacke_dggrqf(
    e2_lapack_col_major,
    0,
    0,
    0,
    e2_double_buf,
    1,
    e2_double_buf2,
    e2_double_buf,
    1,
    e2_double_buf2,
  )
  assert_eq(info_dggrqf, 0)

  let info_cggrqf = lapacke_cggrqf(
    e2_lapack_col_major,
    0,
    0,
    0,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
  )
  assert_eq(info_cggrqf, 0)

  let info_zggrqf = lapacke_zggrqf(
    e2_lapack_col_major,
    0,
    0,
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
  )
  assert_eq(info_zggrqf, 0)

  let info_sggsvd = lapacke_sggsvd(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    e2_int_buf,
    e2_int_buf2,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
    e2_float_buf,
    1,
    e2_int_buf,
  )
  assert_eq(info_sggsvd, 0)

  let info_dggsvd = lapacke_dggsvd(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    e2_int_buf,
    e2_int_buf2,
    e2_double_buf,
    1,
    e2_double_buf2,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    1,
    e2_double_buf2,
    1,
    e2_double_buf,
    1,
    e2_int_buf,
  )
  assert_eq(info_dggsvd, 0)

  let info_cggsvd = lapacke_cggsvd(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    e2_int_buf,
    e2_int_buf2,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_complex_float_buf,
    1,
    e2_int_buf,
  )
  assert_eq(info_cggsvd, 0)

  let info_zggsvd = lapacke_zggsvd(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    e2_int_buf,
    e2_int_buf2,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_complex_double_buf,
    1,
    e2_int_buf,
  )
  assert_eq(info_zggsvd, 0)

  let info_sggsvd3 = lapacke_sggsvd3(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    e2_int_buf,
    e2_int_buf2,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
    e2_float_buf,
    1,
    e2_int_buf,
  )
  assert_eq(info_sggsvd3, 0)

  let info_dggsvd3 = lapacke_dggsvd3(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    e2_int_buf,
    e2_int_buf2,
    e2_double_buf,
    1,
    e2_double_buf2,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    1,
    e2_double_buf2,
    1,
    e2_double_buf,
    1,
    e2_int_buf,
  )
  assert_eq(info_dggsvd3, 0)

  let info_cggsvd3 = lapacke_cggsvd3(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    e2_int_buf,
    e2_int_buf2,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_complex_float_buf,
    1,
    e2_int_buf,
  )
  assert_eq(info_cggsvd3, 0)

  let info_zggsvd3 = lapacke_zggsvd3(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    0,
    e2_int_buf,
    e2_int_buf2,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_complex_double_buf,
    1,
    e2_int_buf,
  )
  assert_eq(info_zggsvd3, 0)

  let info_sggsvp = lapacke_sggsvp(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
    0.0,
    0.0,
    e2_int_buf,
    e2_int_buf2,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
    e2_float_buf,
    1,
  )
  assert_eq(info_sggsvp, 0)

  let info_dggsvp = lapacke_dggsvp(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_double_buf,
    1,
    e2_double_buf2,
    1,
    0.0,
    0.0,
    e2_int_buf,
    e2_int_buf2,
    e2_double_buf,
    1,
    e2_double_buf2,
    1,
    e2_double_buf,
    1,
  )
  assert_eq(info_dggsvp, 0)

  let info_cggsvp = lapacke_cggsvp(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    0.0,
    0.0,
    e2_int_buf,
    e2_int_buf2,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_cggsvp, 0)

  let info_zggsvp = lapacke_zggsvp(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    0.0,
    0.0,
    e2_int_buf,
    e2_int_buf2,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zggsvp, 0)

  let info_sggsvp3 = lapacke_sggsvp3(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
    0.0,
    0.0,
    e2_int_buf,
    e2_int_buf2,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
    e2_float_buf,
    1,
  )
  assert_eq(info_sggsvp3, 0)

  let info_dggsvp3 = lapacke_dggsvp3(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_double_buf,
    1,
    e2_double_buf2,
    1,
    0.0,
    0.0,
    e2_int_buf,
    e2_int_buf2,
    e2_double_buf,
    1,
    e2_double_buf2,
    1,
    e2_double_buf,
    1,
  )
  assert_eq(info_dggsvp3, 0)

  let info_cggsvp3 = lapacke_cggsvp3(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    0.0,
    0.0,
    e2_int_buf,
    e2_int_buf2,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_cggsvp3, 0)

  let info_zggsvp3 = lapacke_zggsvp3(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    0.0,
    0.0,
    e2_int_buf,
    e2_int_buf2,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zggsvp3, 0)

  let info_sgglse = lapacke_sgglse(
    e2_lapack_col_major,
    0,
    0,
    0,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
  )
  assert_eq(info_sgglse, 0)

  let info_dgglse = lapacke_dgglse(
    e2_lapack_col_major,
    0,
    0,
    0,
    e2_double_buf,
    1,
    e2_double_buf2,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
  )
  assert_eq(info_dgglse, 0)

  let info_cgglse = lapacke_cgglse(
    e2_lapack_col_major,
    0,
    0,
    0,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_complex_float_buf,
  )
  assert_eq(info_cgglse, 0)

  let info_zgglse = lapacke_zgglse(
    e2_lapack_col_major,
    0,
    0,
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_complex_double_buf,
  )
  assert_eq(info_zgglse, 0)

  let info_zgghd3 = lapacke_zgghd3(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zgghd3, 0)
}

///|
test "test_lapacke2_gt" {
  let info_sgtcon = lapacke_sgtcon(
    '1'.to_int().to_byte(),
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    e2_float_buf2,
    e2_int_buf,
    0.0,
    e2_float_buf,
  )
  assert_eq(info_sgtcon, 0)

  let info_dgtcon = lapacke_dgtcon(
    '1'.to_int().to_byte(),
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    e2_double_buf2,
    e2_int_buf,
    0.0,
    e2_double_buf,
  )
  assert_eq(info_dgtcon, 0)

  let info_cgtcon = lapacke_cgtcon(
    '1'.to_int().to_byte(),
    1,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_int_buf,
    0.0,
    e2_float_buf,
  )
  assert_eq(info_cgtcon, 0)

  let info_zgtcon = lapacke_zgtcon(
    '1'.to_int().to_byte(),
    1,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_int_buf,
    0.0,
    e2_double_buf,
  )
  assert_eq(info_zgtcon, 0)

  let info_sgtrfs = lapacke_sgtrfs(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    e2_int_buf,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
    e2_float_buf,
    e2_float_buf2,
  )
  assert_eq(info_sgtrfs, 0)

  let info_dgtrfs = lapacke_dgtrfs(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    e2_int_buf,
    e2_double_buf,
    1,
    e2_double_buf2,
    1,
    e2_double_buf,
    e2_double_buf2,
  )
  assert_eq(info_dgtrfs, 0)

  let info_cgtrfs = lapacke_cgtrfs(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    1,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    e2_int_buf,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_float_buf,
    e2_float_buf2,
  )
  assert_eq(info_cgtrfs, 0)

  let info_zgtrfs = lapacke_zgtrfs(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    1,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    e2_int_buf,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_double_buf,
    e2_double_buf2,
  )
  assert_eq(info_zgtrfs, 0)

  let info_sgtsv = lapacke_sgtsv(
    e2_lapack_col_major,
    0,
    0,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    e2_float_buf2,
    1,
  )
  assert_eq(info_sgtsv, 0)

  let info_dgtsv = lapacke_dgtsv(
    e2_lapack_col_major,
    0,
    0,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    e2_double_buf2,
    1,
  )
  assert_eq(info_dgtsv, 0)

  let info_cgtsv = lapacke_cgtsv(
    e2_lapack_col_major,
    0,
    0,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    1,
  )
  assert_eq(info_cgtsv, 0)

  let info_zgtsv = lapacke_zgtsv(
    e2_lapack_col_major,
    0,
    0,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zgtsv, 0)

  let info_sgtsvx = lapacke_sgtsvx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    e2_int_buf,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
  )
  assert_eq(info_sgtsvx, 0)

  let info_dgtsvx = lapacke_dgtsvx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    e2_int_buf,
    e2_double_buf,
    1,
    e2_double_buf2,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
  )
  assert_eq(info_dgtsvx, 0)

  let info_cgtsvx = lapacke_cgtsvx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    e2_int_buf,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
  )
  assert_eq(info_cgtsvx, 0)

  let info_zgtsvx = lapacke_zgtsvx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    e2_int_buf,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
  )
  assert_eq(info_zgtsvx, 0)

  let info_sgttrf = lapacke_sgttrf(
    0,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    e2_float_buf2,
    e2_int_buf,
  )
  assert_eq(info_sgttrf, 0)

  let info_dgttrf = lapacke_dgttrf(
    0,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    e2_double_buf2,
    e2_int_buf,
  )
  assert_eq(info_dgttrf, 0)

  let info_cgttrf = lapacke_cgttrf(
    0,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_int_buf,
  )
  assert_eq(info_cgttrf, 0)

  let info_zgttrf = lapacke_zgttrf(
    0,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_int_buf,
  )
  assert_eq(info_zgttrf, 0)

  let info_sgttrs = lapacke_sgttrs(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    e2_float_buf2,
    e2_int_buf,
    e2_float_buf,
    1,
  )
  assert_eq(info_sgttrs, 0)

  let info_dgttrs = lapacke_dgttrs(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    e2_double_buf2,
    e2_int_buf,
    e2_double_buf,
    1,
  )
  assert_eq(info_dgttrs, 0)

  let info_cgttrs = lapacke_cgttrs(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_int_buf,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_cgttrs, 0)

  let info_zgttrs = lapacke_zgttrs(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    0,
    0,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_int_buf,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zgttrs, 0)
}

///|
test "test_lapacke2_hb" {
  let info_chbev = lapacke_chbev(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e2_complex_float_buf,
    1,
    e2_float_buf,
    e2_complex_float_buf2,
    1,
  )
  assert_eq(info_chbev, 0)

  let info_zhbev = lapacke_zhbev(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e2_complex_double_buf,
    1,
    e2_double_buf,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zhbev, 0)

  let info_chbevd = lapacke_chbevd(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e2_complex_float_buf,
    1,
    e2_float_buf,
    e2_complex_float_buf2,
    1,
  )
  assert_eq(info_chbevd, 0)

  let info_zhbevd = lapacke_zhbevd(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e2_complex_double_buf,
    1,
    e2_double_buf,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zhbevd, 0)

  let info_chbevx = lapacke_chbevx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    0.0,
    0.0,
    1,
    1,
    0.0,
    e2_int_buf,
    e2_float_buf,
    e2_complex_float_buf2,
    1,
    e2_int_buf2,
  )
  assert_eq(info_chbevx, 0)

  let info_zhbevx = lapacke_zhbevx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    0.0,
    0.0,
    1,
    1,
    0.0,
    e2_int_buf,
    e2_double_buf,
    e2_complex_double_buf2,
    1,
    e2_int_buf2,
  )
  assert_eq(info_zhbevx, 0)

  let info_chbgst = lapacke_chbgst(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    0,
    0,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_chbgst, 0)

  let info_zhbgst = lapacke_zhbgst(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    0,
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zhbgst, 0)

  let info_chbgv = lapacke_chbgv(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    0,
    0,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_float_buf,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_chbgv, 0)

  let info_zhbgv = lapacke_zhbgv(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    0,
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_double_buf,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zhbgv, 0)

  let info_chbgvd = lapacke_chbgvd(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    0,
    0,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_float_buf,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_chbgvd, 0)

  let info_zhbgvd = lapacke_zhbgvd(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    0,
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_double_buf,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zhbgvd, 0)

  let info_chbgvx = lapacke_chbgvx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    0,
    0,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_complex_float_buf,
    1,
    0.0,
    0.0,
    0,
    0,
    0.0,
    e2_int_buf,
    e2_float_buf,
    e2_complex_float_buf2,
    1,
    e2_int_buf2,
  )
  assert_eq(info_chbgvx, 0)

  let info_zhbgvx = lapacke_zhbgvx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    0,
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_complex_double_buf,
    1,
    0.0,
    0.0,
    0,
    0,
    0.0,
    e2_int_buf,
    e2_double_buf,
    e2_complex_double_buf2,
    1,
    e2_int_buf2,
  )
  assert_eq(info_zhbgvx, 0)

  let info_chbtrd = lapacke_chbtrd(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e2_complex_float_buf,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_complex_float_buf2,
    1,
  )
  assert_eq(info_chbtrd, 0)

  let info_zhbtrd = lapacke_zhbtrd(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e2_complex_double_buf,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zhbtrd, 0)
}

///|
test "test_lapacke2_he" {
  let info_checon = lapacke_checon(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_complex_float_buf,
    1,
    e2_int_buf,
    0.0,
    e2_float_buf,
  )
  assert_eq(info_checon, 0)

  let info_zhecon = lapacke_zhecon(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_complex_double_buf,
    1,
    e2_int_buf,
    0.0,
    e2_double_buf,
  )
  assert_eq(info_zhecon, 0)

  let info_cheequb = lapacke_cheequb(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_complex_float_buf,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
  )
  assert_eq(info_cheequb, 0)

  let info_zheequb = lapacke_zheequb(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_complex_double_buf,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
  )
  assert_eq(info_zheequb, 0)

  let info_cheev = lapacke_cheev(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e2_complex_float_buf,
    1,
    e2_float_buf,
  )
  assert_eq(info_cheev, 0)

  let info_zheev = lapacke_zheev(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e2_complex_double_buf,
    1,
    e2_double_buf,
  )
  assert_eq(info_zheev, 0)

  let info_cheevd = lapacke_cheevd(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e2_complex_float_buf,
    1,
    e2_float_buf,
  )
  assert_eq(info_cheevd, 0)

  let info_zheevd = lapacke_zheevd(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e2_complex_double_buf,
    1,
    e2_double_buf,
  )
  assert_eq(info_zheevd, 0)

  let info_cheevr = lapacke_cheevr(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e2_complex_float_buf,
    1,
    0.0,
    0.0,
    1,
    1,
    0.0,
    e2_int_buf,
    e2_float_buf,
    e2_complex_float_buf2,
    1,
    e2_int_buf2,
  )
  assert_eq(info_cheevr, 0)

  let info_zheevr = lapacke_zheevr(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e2_complex_double_buf,
    1,
    0.0,
    0.0,
    1,
    1,
    0.0,
    e2_int_buf,
    e2_double_buf,
    e2_complex_double_buf2,
    1,
    e2_int_buf2,
  )
  assert_eq(info_zheevr, 0)

  let info_cheevx = lapacke_cheevx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e2_complex_float_buf,
    1,
    0.0,
    0.0,
    1,
    1,
    0.0,
    e2_int_buf,
    e2_float_buf,
    e2_complex_float_buf2,
    1,
    e2_int_buf2,
  )
  assert_eq(info_cheevx, 0)

  let info_zheevx = lapacke_zheevx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e2_complex_double_buf,
    1,
    0.0,
    0.0,
    1,
    1,
    0.0,
    e2_int_buf,
    e2_double_buf,
    e2_complex_double_buf2,
    1,
    e2_int_buf2,
  )
  assert_eq(info_zheevx, 0)

  let info_chegst = lapacke_chegst(
    e2_lapack_col_major,
    1,
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
  )
  assert_eq(info_chegst, 0)

  let info_zhegst = lapacke_zhegst(
    e2_lapack_col_major,
    1,
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zhegst, 0)

  let info_chegv = lapacke_chegv(
    e2_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_float_buf,
  )
  assert_eq(info_chegv, 0)

  let info_zhegv = lapacke_zhegv(
    e2_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_double_buf,
  )
  assert_eq(info_zhegv, 0)

  let info_chegvd = lapacke_chegvd(
    e2_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_float_buf,
  )
  assert_eq(info_chegvd, 0)

  let info_zhegvd = lapacke_zhegvd(
    e2_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_double_buf,
  )
  assert_eq(info_zhegvd, 0)

  let info_chegvx = lapacke_chegvx(
    e2_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    0.0,
    0.0,
    0,
    0,
    0.0,
    e2_int_buf,
    e2_float_buf,
    e2_complex_float_buf2,
    1,
    e2_int_buf2,
  )
  assert_eq(info_chegvx, 0)

  let info_zhegvx = lapacke_zhegvx(
    e2_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    0.0,
    0.0,
    0,
    0,
    0.0,
    e2_int_buf,
    e2_double_buf,
    e2_complex_double_buf2,
    1,
    e2_int_buf2,
  )
  assert_eq(info_zhegvx, 0)

  let info_cherfs = lapacke_cherfs(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_int_buf,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_float_buf,
    e2_float_buf2,
  )
  assert_eq(info_cherfs, 0)

  let info_zherfs = lapacke_zherfs(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_int_buf,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_double_buf,
    e2_double_buf2,
  )
  assert_eq(info_zherfs, 0)


  let info_chesv = lapacke_chesv(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_float_buf,
    1,
    e2_int_buf,
    e2_complex_float_buf2,
    1,
  )
  assert_eq(info_chesv, 0)

  let info_zhesv = lapacke_zhesv(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_double_buf,
    1,
    e2_int_buf,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zhesv, 0)

  let info_chesvx = lapacke_chesvx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_int_buf,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
  )
  assert_eq(info_chesvx, 0)

  let info_zhesvx = lapacke_zhesvx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_int_buf,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
  )
  assert_eq(info_zhesvx, 0)


  let info_chetrd = lapacke_chetrd(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_complex_float_buf,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_complex_float_buf2,
  )
  assert_eq(info_chetrd, 0)

  let info_zhetrd = lapacke_zhetrd(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_complex_double_buf,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_complex_double_buf2,
  )
  assert_eq(info_zhetrd, 0)

  let info_chetrf = lapacke_chetrf(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    1,
    e2_int_buf,
  )
  assert_eq(info_chetrf, 0)

  let info_zhetrf = lapacke_zhetrf(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    1,
    e2_int_buf,
  )
  assert_eq(info_zhetrf, 0)

  let info_chetri = lapacke_chetri(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_complex_float_buf,
    1,
    e2_int_buf,
  )
  assert_eq(info_chetri, 0)

  let info_zhetri = lapacke_zhetri(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_complex_double_buf,
    1,
    e2_int_buf,
  )
  assert_eq(info_zhetri, 0)

  let info_chetrs = lapacke_chetrs(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_float_buf,
    1,
    e2_int_buf,
    e2_complex_float_buf2,
    1,
  )
  assert_eq(info_chetrs, 0)

  let info_zhetrs = lapacke_zhetrs(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_double_buf,
    1,
    e2_int_buf,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zhetrs, 0)

  let info_chfrk = lapacke_chfrk(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    0.0,
    e2_complex_float_buf,
    1,
    0.0,
    e2_complex_float_buf2,
  )
  assert_eq(info_chfrk, 0)

  let info_zhfrk = lapacke_zhfrk(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    0.0,
    e2_complex_double_buf,
    1,
    0.0,
    e2_complex_double_buf2,
  )
  assert_eq(info_zhfrk, 0)
}

///|
test "test_lapacke2_hp" {
  let info_chpcon = lapacke_chpcon(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    e2_int_buf,
    0.0,
    e2_float_buf,
  )
  assert_eq(info_chpcon, 0)

  let info_zhpcon = lapacke_zhpcon(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    e2_int_buf,
    0.0,
    e2_double_buf,
  )
  assert_eq(info_zhpcon, 0)

  let info_chpev = lapacke_chpev(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    e2_float_buf,
    e2_complex_float_buf2,
    1,
  )
  assert_eq(info_chpev, 0)

  let info_zhpev = lapacke_zhpev(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    e2_double_buf,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zhpev, 0)

  let info_chpevd = lapacke_chpevd(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    e2_float_buf,
    e2_complex_float_buf2,
    1,
  )
  assert_eq(info_chpevd, 0)

  let info_zhpevd = lapacke_zhpevd(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    e2_double_buf,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zhpevd, 0)

  let info_chpevx = lapacke_chpevx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    0.0,
    0.0,
    0,
    0,
    0.0,
    e2_int_buf,
    e2_float_buf,
    e2_complex_float_buf2,
    1,
    e2_int_buf2,
  )
  assert_eq(info_chpevx, 0)

  let info_zhpevx = lapacke_zhpevx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    0.0,
    0.0,
    0,
    0,
    0.0,
    e2_int_buf,
    e2_double_buf,
    e2_complex_double_buf2,
    1,
    e2_int_buf2,
  )
  assert_eq(info_zhpevx, 0)

  let info_chpgst = lapacke_chpgst(
    e2_lapack_col_major,
    1,
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    e2_complex_float_buf2,
  )
  assert_eq(info_chpgst, 0)

  let info_zhpgst = lapacke_zhpgst(
    e2_lapack_col_major,
    1,
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    e2_complex_double_buf2,
  )
  assert_eq(info_zhpgst, 0)

  let info_chpgv = lapacke_chpgv(
    e2_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_float_buf,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_chpgv, 0)

  let info_zhpgv = lapacke_zhpgv(
    e2_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_double_buf,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zhpgv, 0)

  let info_chpgvd = lapacke_chpgvd(
    e2_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_float_buf,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_chpgvd, 0)

  let info_zhpgvd = lapacke_zhpgvd(
    e2_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_double_buf,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zhpgvd, 0)

  let info_chpgvx = lapacke_chpgvx(
    e2_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    0.0,
    0.0,
    0,
    0,
    0.0,
    e2_int_buf,
    e2_float_buf,
    e2_complex_float_buf2,
    1,
    e2_int_buf2,
  )
  assert_eq(info_chpgvx, 0)

  let info_zhpgvx = lapacke_zhpgvx(
    e2_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    0.0,
    0.0,
    0,
    0,
    0.0,
    e2_int_buf,
    e2_double_buf,
    e2_complex_double_buf2,
    1,
    e2_int_buf2,
  )
  assert_eq(info_zhpgvx, 0)

  let info_chprfs = lapacke_chprfs(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_int_buf,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_float_buf,
    e2_float_buf2,
  )
  assert_eq(info_chprfs, 0)

  let info_zhprfs = lapacke_zhprfs(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_int_buf,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_double_buf,
    e2_double_buf2,
  )
  assert_eq(info_zhprfs, 0)

  let info_chpsv = lapacke_chpsv(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_float_buf,
    e2_int_buf,
    e2_complex_float_buf2,
    1,
  )
  assert_eq(info_chpsv, 0)

  let info_zhpsv = lapacke_zhpsv(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_double_buf,
    e2_int_buf,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zhpsv, 0)

  let info_chpsvx = lapacke_chpsvx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_int_buf,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
  )
  assert_eq(info_chpsvx, 0)

  let info_zhpsvx = lapacke_zhpsvx(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_int_buf,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
  )
  assert_eq(info_zhpsvx, 0)

  let info_chptrd = lapacke_chptrd(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    e2_float_buf,
    e2_float_buf2,
    e2_complex_float_buf2,
  )
  assert_eq(info_chptrd, 0)

  let info_zhptrd = lapacke_zhptrd(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    e2_double_buf,
    e2_double_buf2,
    e2_complex_double_buf2,
  )
  assert_eq(info_zhptrd, 0)

  let info_chptrf = lapacke_chptrf(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    e2_int_buf,
  )
  assert_eq(info_chptrf, 0)

  let info_zhptrf = lapacke_zhptrf(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    e2_int_buf,
  )
  assert_eq(info_zhptrf, 0)

  let info_chptri = lapacke_chptri(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    e2_complex_float_buf,
    e2_int_buf,
  )
  assert_eq(info_chptri, 0)

  let info_zhptri = lapacke_zhptri(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    e2_complex_double_buf,
    e2_int_buf,
  )
  assert_eq(info_zhptri, 0)

  let info_chptrs = lapacke_chptrs(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_float_buf,
    e2_int_buf,
    e2_complex_float_buf2,
    1,
  )
  assert_eq(info_chptrs, 0)

  let info_zhptrs = lapacke_zhptrs(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    0,
    0,
    e2_complex_double_buf,
    e2_int_buf,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zhptrs, 0)
}

///|
test "test_lapacke2_hq_and_la" {
  let info_shgeqz = lapacke_shgeqz(
    e2_lapack_col_major,
    'E'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    e2_float_buf2,
    1,
    e2_float_buf,
    1,
  )
  assert_eq(info_shgeqz, 0)

  let info_dhgeqz = lapacke_dhgeqz(
    e2_lapack_col_major,
    'E'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_double_buf,
    1,
    e2_double_buf2,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    e2_double_buf2,
    1,
    e2_double_buf,
    1,
  )
  assert_eq(info_dhgeqz, 0)

  let info_chgeqz = lapacke_chgeqz(
    e2_lapack_col_major,
    'E'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    0,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
  )
  assert_eq(info_chgeqz, -8)

  let info_zhgeqz = lapacke_zhgeqz(
    e2_lapack_col_major,
    'E'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    0,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zhgeqz, -8)

  let info_shseqr = lapacke_shseqr(
    e2_lapack_col_major,
    'E'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
    e2_float_buf,
    e2_float_buf2,
    1,
  )
  assert_eq(info_shseqr, 0)

  let info_dhseqr = lapacke_dhseqr(
    e2_lapack_col_major,
    'E'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_double_buf,
    1,
    e2_double_buf2,
    e2_double_buf,
    e2_double_buf2,
    1,
  )
  assert_eq(info_dhseqr, 0)

  let info_chseqr = lapacke_chseqr(
    e2_lapack_col_major,
    'E'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_chseqr, -7)

  let info_zhseqr = lapacke_zhseqr(
    e2_lapack_col_major,
    'E'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zhseqr, -7)

  let info_shsein = lapacke_shsein(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'Q'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    e2_float_buf,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
    0,
    e2_int_buf,
    e2_int_buf2,
    e2_int_buf2,
  )
  assert_eq(info_shsein, -2)

  let info_dhsein = lapacke_dhsein(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'Q'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    e2_double_buf,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    1,
    e2_double_buf2,
    1,
    0,
    e2_int_buf,
    e2_int_buf2,
    e2_int_buf2,
  )
  assert_eq(info_dhsein, -2)

  let info_chsein = lapacke_chsein(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'Q'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    0,
    e2_int_buf,
    e2_int_buf2,
    e2_int_buf2,
  )
  assert_eq(info_chsein, -2)

  let info_zhsein = lapacke_zhsein(
    e2_lapack_col_major,
    'N'.to_int().to_byte(),
    'Q'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    0,
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    0,
    e2_int_buf,
    e2_int_buf2,
    e2_int_buf2,
  )
  assert_eq(info_zhsein, -2)

  let info_clacgv = lapacke_clacgv(0, e2_complex_float_buf, 1)
  assert_eq(info_clacgv, 0)

  let info_zlacgv = lapacke_zlacgv(0, e2_complex_double_buf, 1)
  assert_eq(info_zlacgv, 0)

  let info_slacn2 = lapacke_slacn2(
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_int_buf,
    e2_float_buf,
    e2_int_buf2,
    e2_int_buf,
  )
  assert_eq(info_slacn2, 0)

  let info_dlacn2 = lapacke_dlacn2(
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_int_buf,
    e2_double_buf,
    e2_int_buf2,
    e2_int_buf,
  )
  assert_eq(info_dlacn2, 0)

  let info_clacn2 = lapacke_clacn2(
    0,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    e2_float_buf,
    e2_int_buf,
    e2_int_buf2,
  )
  assert_eq(info_clacn2, 0)

  let info_zlacn2 = lapacke_zlacn2(
    0,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    e2_double_buf,
    e2_int_buf,
    e2_int_buf2,
  )
  assert_eq(info_zlacn2, 0)

  let info_slacpy = lapacke_slacpy(
    e2_lapack_col_major,
    'A'.to_int().to_byte(),
    1,
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
  )
  assert_eq(info_slacpy, 0)

  let info_dlacpy = lapacke_dlacpy(
    e2_lapack_col_major,
    'A'.to_int().to_byte(),
    1,
    1,
    e2_double_buf,
    1,
    e2_double_buf2,
    1,
  )
  assert_eq(info_dlacpy, 0)

  let info_clacpy = lapacke_clacpy(
    e2_lapack_col_major,
    'A'.to_int().to_byte(),
    0,
    0,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
  )
  assert_eq(info_clacpy, 0)

  let info_zlacpy = lapacke_zlacpy(
    e2_lapack_col_major,
    'A'.to_int().to_byte(),
    0,
    0,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zlacpy, 0)

  let info_clacp2 = lapacke_clacp2(
    e2_lapack_col_major,
    'A'.to_int().to_byte(),
    1,
    1,
    e2_float_buf,
    1,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_clacp2, 0)

  let info_zlacp2 = lapacke_zlacp2(
    e2_lapack_col_major,
    'A'.to_int().to_byte(),
    1,
    1,
    e2_double_buf,
    1,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zlacp2, 0)

  let info_zlag2c = lapacke_zlag2c(
    e2_lapack_col_major,
    1,
    1,
    e2_complex_double_buf,
    1,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_zlag2c, 0)

  let info_slag2d = lapacke_slag2d(
    e2_lapack_col_major,
    1,
    1,
    e2_float_buf,
    1,
    e2_double_buf,
    1,
  )
  assert_eq(info_slag2d, 0)

  let info_dlag2s = lapacke_dlag2s(
    e2_lapack_col_major,
    1,
    1,
    e2_double_buf,
    1,
    e2_float_buf,
    1,
  )
  assert_eq(info_dlag2s, 0)

  let info_clag2z = lapacke_clag2z(
    e2_lapack_col_major,
    1,
    1,
    e2_complex_float_buf,
    1,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_clag2z, 0)

  let info_slagge = lapacke_slagge(
    e2_lapack_col_major,
    1,
    1,
    0,
    0,
    e2_float_buf,
    e2_float_buf2,
    1,
    e2_iseed_buf,
  )
  assert_eq(info_slagge, 0)

  let info_dlagge = lapacke_dlagge(
    e2_lapack_col_major,
    1,
    1,
    0,
    0,
    e2_double_buf,
    e2_double_buf2,
    1,
    e2_iseed_buf,
  )
  assert_eq(info_dlagge, 0)

  let info_clagge = lapacke_clagge(
    e2_lapack_col_major,
    1,
    1,
    0,
    0,
    e2_float_buf,
    e2_complex_float_buf,
    1,
    e2_iseed_buf,
  )
  assert_eq(info_clagge, 0)

  let info_zlagge = lapacke_zlagge(
    e2_lapack_col_major,
    1,
    1,
    0,
    0,
    e2_double_buf,
    e2_complex_double_buf,
    1,
    e2_iseed_buf,
  )
  assert_eq(info_zlagge, 0)

  let slamch = lapacke_slamch('E'.to_int().to_byte())
  assert_eq(slamch > 0.0, true)

  let dlamch = lapacke_dlamch('E'.to_int().to_byte())
  assert_eq(dlamch > 0.0, true)

  let slangb = lapacke_slangb(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    1,
    0,
    0,
    e2_float_buf,
    1,
  )
  assert_eq(slangb, 0.0)

  let dlangb = lapacke_dlangb(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    1,
    0,
    0,
    e2_double_buf,
    1,
  )
  assert_eq(dlangb, 0.0)

  let clangb = lapacke_clangb(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    1,
    0,
    0,
    e2_complex_float_buf,
    1,
  )
  assert_eq(clangb, 0.0)

  let zlangb = lapacke_zlangb(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    1,
    0,
    0,
    e2_complex_double_buf,
    1,
  )
  assert_eq(zlangb, 0.0)

  let slange = lapacke_slange(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    1,
    1,
    e2_float_buf,
    1,
  )
  assert_eq(slange, 0.0)

  let dlange = lapacke_dlange(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    1,
    1,
    e2_double_buf,
    1,
  )
  assert_eq(dlange, 0.0)

  let clange = lapacke_clange(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    1,
    1,
    e2_complex_float_buf,
    1,
  )
  assert_eq(clange, 0.0)

  let zlange = lapacke_zlange(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    1,
    1,
    e2_complex_double_buf,
    1,
  )
  assert_eq(zlange, 0.0)

  let clanhe = lapacke_clanhe(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e2_complex_float_buf,
    1,
  )
  assert_eq(clanhe, 0.0)

  let zlanhe = lapacke_zlanhe(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e2_complex_double_buf,
    1,
  )
  assert_eq(zlanhe, 0.0)

  let info_clacrm = lapacke_clacrm(
    e2_lapack_col_major,
    1,
    1,
    e2_complex_float_buf,
    1,
    e2_float_buf,
    1,
    e2_complex_float_buf2,
    1,
  )
  assert_eq(info_clacrm, 0)

  let info_zlacrm = lapacke_zlacrm(
    e2_lapack_col_major,
    1,
    1,
    e2_complex_double_buf,
    1,
    e2_double_buf,
    1,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zlacrm, 0)

  let info_clarcm = lapacke_clarcm(
    e2_lapack_col_major,
    1,
    1,
    e2_float_buf,
    1,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
  )
  assert_eq(info_clarcm, 0)

  let info_zlarcm = lapacke_zlarcm(
    e2_lapack_col_major,
    1,
    1,
    e2_double_buf,
    1,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
  )
  assert_eq(info_zlarcm, 0)

  let slansy = lapacke_slansy(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e2_float_buf,
    1,
  )
  assert_eq(slansy, 0.0)

  let dlansy = lapacke_dlansy(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e2_double_buf,
    1,
  )
  assert_eq(dlansy, 0.0)

  let clansy = lapacke_clansy(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e2_complex_float_buf,
    1,
  )
  assert_eq(clansy, 0.0)

  let zlansy = lapacke_zlansy(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e2_complex_double_buf,
    1,
  )
  assert_eq(zlansy, 0.0)

  let slantr = lapacke_slantr(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    e2_float_buf,
    1,
  )
  assert_eq(slantr, 0.0)

  let dlantr = lapacke_dlantr(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    e2_double_buf,
    1,
  )
  assert_eq(dlantr, 0.0)

  let clantr = lapacke_clantr(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    e2_complex_float_buf,
    1,
  )
  assert_eq(clantr, 0.0)

  let zlantr = lapacke_zlantr(
    e2_lapack_col_major,
    '1'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    e2_complex_double_buf,
    1,
  )
  assert_eq(zlantr, 0.0)

  let info_slarfb = lapacke_slarfb(
    e2_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'F'.to_int().to_byte(),
    'C'.to_int().to_byte(),
    1,
    1,
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
    e2_float_buf,
    1,
  )
  assert_eq(info_slarfb, 0)

  let info_dlarfb = lapacke_dlarfb(
    e2_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'F'.to_int().to_byte(),
    'C'.to_int().to_byte(),
    1,
    1,
    1,
    e2_double_buf,
    1,
    e2_double_buf2,
    1,
    e2_double_buf,
    1,
  )
  assert_eq(info_dlarfb, 0)

  let info_clarfb = lapacke_clarfb(
    e2_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'F'.to_int().to_byte(),
    'C'.to_int().to_byte(),
    1,
    1,
    1,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    1,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_clarfb, 0)

  let info_zlarfb = lapacke_zlarfb(
    e2_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    'F'.to_int().to_byte(),
    'C'.to_int().to_byte(),
    1,
    1,
    1,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    1,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zlarfb, 0)

  let info_slarfg = lapacke_slarfg(1, e2_float_buf, e2_float_buf2, 1, e2_float_buf)
  assert_eq(info_slarfg, 0)

  let info_dlarfg = lapacke_dlarfg(1, e2_double_buf, e2_double_buf2, 1, e2_double_buf)
  assert_eq(info_dlarfg, 0)

  let info_clarfg = lapacke_clarfg(
    1,
    e2_complex_float_buf,
    e2_complex_float_buf2,
    1,
    e2_complex_float_buf,
  )
  assert_eq(info_clarfg, 0)

  let info_zlarfg = lapacke_zlarfg(
    1,
    e2_complex_double_buf,
    e2_complex_double_buf2,
    1,
    e2_complex_double_buf,
  )
  assert_eq(info_zlarfg, 0)

  let info_slarft = lapacke_slarft(
    e2_lapack_col_major,
    'F'.to_int().to_byte(),
    'C'.to_int().to_byte(),
    1,
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
    e2_float_buf,
    1,
  )
  assert_eq(info_slarft, 0)

  let info_dlarft = lapacke_dlarft(
    e2_lapack_col_major,
    'F'.to_int().to_byte(),
    'C'.to_int().to_byte(),
    1,
    1,
    e2_double_buf,
    1,
    e2_double_buf2,
    e2_double_buf,
    1,
  )
  assert_eq(info_dlarft, 0)

  let info_clarft = lapacke_clarft(
    e2_lapack_col_major,
    'F'.to_int().to_byte(),
    'C'.to_int().to_byte(),
    1,
    1,
    e2_complex_float_buf,
    1,
    e2_complex_float_buf2,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_clarft, 0)

  let info_zlarft = lapacke_zlarft(
    e2_lapack_col_major,
    'F'.to_int().to_byte(),
    'C'.to_int().to_byte(),
    1,
    1,
    e2_complex_double_buf,
    1,
    e2_complex_double_buf2,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zlarft, 0)

  let info_slarfx = lapacke_slarfx(
    e2_lapack_col_major,
    'L'.to_int().to_byte(),
    1,
    1,
    e2_float_buf,
    0.0,
    e2_float_buf2,
    1,
    e2_float_buf,
  )
  assert_eq(info_slarfx, 0)

  let info_dlarfx = lapacke_dlarfx(
    e2_lapack_col_major,
    'L'.to_int().to_byte(),
    1,
    1,
    e2_double_buf,
    0.0,
    e2_double_buf2,
    1,
    e2_double_buf,
  )
  assert_eq(info_dlarfx, 0)

  let info_clarfx = lapacke_clarfx(
    e2_lapack_col_major,
    'L'.to_int().to_byte(),
    1,
    1,
    e2_complex_float_buf,
    e2_complex_float_zero,
    e2_complex_float_buf2,
    1,
    e2_complex_float_buf,
  )
  assert_eq(info_clarfx, 0)

  let info_zlarfx = lapacke_zlarfx(
    e2_lapack_col_major,
    'L'.to_int().to_byte(),
    1,
    1,
    e2_complex_double_buf,
    e2_complex_double_zero,
    e2_complex_double_buf2,
    1,
    e2_complex_double_buf,
  )
  assert_eq(info_zlarfx, 0)

  let info_slarnv = lapacke_slarnv(1, e2_iseed_buf, 1, e2_float_buf)
  assert_eq(info_slarnv, 0)

  let info_dlarnv = lapacke_dlarnv(1, e2_iseed_buf, 1, e2_double_buf)
  assert_eq(info_dlarnv, 0)

  let info_clarnv = lapacke_clarnv(1, e2_iseed_buf, 1, e2_complex_float_buf)
  assert_eq(info_clarnv, 0)

  let info_zlarnv = lapacke_zlarnv(1, e2_iseed_buf, 1, e2_complex_double_buf)
  assert_eq(info_zlarnv, 0)

  let info_slascl = lapacke_slascl(
    e2_lapack_col_major,
    'G'.to_int().to_byte(),
    0,
    0,
    1.0,
    1.0,
    1,
    1,
    e2_float_buf,
    1,
  )
  assert_eq(info_slascl, 0)

  let info_dlascl = lapacke_dlascl(
    e2_lapack_col_major,
    'G'.to_int().to_byte(),
    0,
    0,
    1.0,
    1.0,
    1,
    1,
    e2_double_buf,
    1,
  )
  assert_eq(info_dlascl, 0)

  let info_clascl = lapacke_clascl(
    e2_lapack_col_major,
    'G'.to_int().to_byte(),
    0,
    0,
    1.0,
    1.0,
    1,
    1,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_clascl, 0)

  let info_zlascl = lapacke_zlascl(
    e2_lapack_col_major,
    'G'.to_int().to_byte(),
    0,
    0,
    1.0,
    1.0,
    1,
    1,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zlascl, 0)

  let info_slaset = lapacke_slaset(
    e2_lapack_col_major,
    'A'.to_int().to_byte(),
    1,
    1,
    0.0,
    0.0,
    e2_float_buf,
    1,
  )
  assert_eq(info_slaset, 0)

  let info_dlaset = lapacke_dlaset(
    e2_lapack_col_major,
    'A'.to_int().to_byte(),
    1,
    1,
    0.0,
    0.0,
    e2_double_buf,
    1,
  )
  assert_eq(info_dlaset, 0)

  let info_claset = lapacke_claset(
    e2_lapack_col_major,
    'A'.to_int().to_byte(),
    1,
    1,
    e2_complex_float_zero,
    e2_complex_float_zero,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_claset, 0)

  let info_zlaset = lapacke_zlaset(
    e2_lapack_col_major,
    'A'.to_int().to_byte(),
    1,
    1,
    e2_complex_double_zero,
    e2_complex_double_zero,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zlaset, 0)

  let info_slasrt = lapacke_slasrt('I'.to_int().to_byte(), 1, e2_float_buf)
  assert_eq(info_slasrt, 0)

  let info_dlasrt = lapacke_dlasrt('I'.to_int().to_byte(), 1, e2_double_buf)
  assert_eq(info_dlasrt, 0)

  let info_slassq = lapacke_slassq(1, e2_float_buf, 1, e2_float_buf, e2_float_buf2)
  assert_eq(info_slassq, 0)

  let info_dlassq = lapacke_dlassq(1, e2_double_buf, 1, e2_double_buf, e2_double_buf2)
  assert_eq(info_dlassq, 0)

  let info_classq = lapacke_classq(
    1,
    e2_complex_float_buf,
    1,
    e2_float_buf,
    e2_float_buf2,
  )
  assert_eq(info_classq, 0)

  let info_zlassq = lapacke_zlassq(
    1,
    e2_complex_double_buf,
    1,
    e2_double_buf,
    e2_double_buf2,
  )
  assert_eq(info_zlassq, 0)

  let info_slaswp = lapacke_slaswp(
    e2_lapack_col_major,
    1,
    e2_float_buf,
    1,
    1,
    1,
    e2_int_buf,
    1,
  )
  assert_eq(info_slaswp, 0)

  let info_dlaswp = lapacke_dlaswp(
    e2_lapack_col_major,
    1,
    e2_double_buf,
    1,
    1,
    1,
    e2_int_buf,
    1,
  )
  assert_eq(info_dlaswp, 0)

  let info_claswp = lapacke_claswp(
    e2_lapack_col_major,
    1,
    e2_complex_float_buf,
    1,
    1,
    1,
    e2_int_buf,
    1,
  )
  assert_eq(info_claswp, 0)

  let info_zlaswp = lapacke_zlaswp(
    e2_lapack_col_major,
    1,
    e2_complex_double_buf,
    1,
    1,
    1,
    e2_int_buf,
    1,
  )
  assert_eq(info_zlaswp, 0)

  let info_slatms = lapacke_slatms(
    e2_lapack_col_major,
    1,
    1,
    'U'.to_int().to_byte(),
    e2_iseed_buf,
    'N'.to_int().to_byte(),
    e2_float_buf,
    0,
    1.0,
    1.0,
    0,
    0,
    'N'.to_int().to_byte(),
    e2_float_buf2,
    1,
  )
  assert_eq(info_slatms, 0)

  let info_dlatms = lapacke_dlatms(
    e2_lapack_col_major,
    1,
    1,
    'U'.to_int().to_byte(),
    e2_iseed_buf,
    'N'.to_int().to_byte(),
    e2_double_buf,
    0,
    1.0,
    1.0,
    0,
    0,
    'N'.to_int().to_byte(),
    e2_double_buf2,
    1,
  )
  assert_eq(info_dlatms, 0)

  let info_clatms = lapacke_clatms(
    e2_lapack_col_major,
    1,
    1,
    'U'.to_int().to_byte(),
    e2_iseed_buf,
    'N'.to_int().to_byte(),
    e2_float_buf,
    0,
    1.0,
    1.0,
    0,
    0,
    'N'.to_int().to_byte(),
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_clatms, 0)

  let info_zlatms = lapacke_zlatms(
    e2_lapack_col_major,
    1,
    1,
    'U'.to_int().to_byte(),
    e2_iseed_buf,
    'N'.to_int().to_byte(),
    e2_double_buf,
    0,
    1.0,
    1.0,
    0,
    0,
    'N'.to_int().to_byte(),
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zlatms, 0)

  let info_slauum = lapacke_slauum(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_float_buf,
    1,
  )
  assert_eq(info_slauum, 0)

  let info_dlauum = lapacke_dlauum(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_double_buf,
    1,
  )
  assert_eq(info_dlauum, 0)

  let info_clauum = lapacke_clauum(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_complex_float_buf,
    1,
  )
  assert_eq(info_clauum, 0)

  let info_zlauum = lapacke_zlauum(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_complex_double_buf,
    1,
  )
  assert_eq(info_zlauum, 0)

  let info_sopgtr = lapacke_sopgtr(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    1,
  )
  assert_eq(info_sopgtr, 0)

  let info_dopgtr = lapacke_dopgtr(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    1,
  )
  assert_eq(info_dopgtr, 0)

  let info_sopmtr = lapacke_sopmtr(
    e2_lapack_col_major,
    'L'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    e2_float_buf,
    e2_float_buf2,
    e2_float_buf,
    1,
  )
  assert_eq(info_sopmtr, 0)

  let info_dopmtr = lapacke_dopmtr(
    e2_lapack_col_major,
    'L'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    e2_double_buf,
    e2_double_buf2,
    e2_double_buf,
    1,
  )
  assert_eq(info_dopmtr, 0)

  let info_sorgbr = lapacke_sorgbr(
    e2_lapack_col_major,
    'Q'.to_int().to_byte(),
    1,
    1,
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
  )
  assert_eq(info_sorgbr, 0)

  let info_dorgbr = lapacke_dorgbr(
    e2_lapack_col_major,
    'Q'.to_int().to_byte(),
    1,
    1,
    1,
    e2_double_buf,
    1,
    e2_double_buf2,
  )
  assert_eq(info_dorgbr, 0)

  let info_sorghr = lapacke_sorghr(
    e2_lapack_col_major,
    1,
    1,
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
  )
  assert_eq(info_sorghr, 0)

  let info_dorghr = lapacke_dorghr(
    e2_lapack_col_major,
    1,
    1,
    1,
    e2_double_buf,
    1,
    e2_double_buf2,
  )
  assert_eq(info_dorghr, 0)

  let info_sorglq = lapacke_sorglq(
    e2_lapack_col_major,
    1,
    1,
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
  )
  assert_eq(info_sorglq, 0)

  let info_dorglq = lapacke_dorglq(
    e2_lapack_col_major,
    1,
    1,
    1,
    e2_double_buf,
    1,
    e2_double_buf2,
  )
  assert_eq(info_dorglq, 0)

  let info_sorgql = lapacke_sorgql(
    e2_lapack_col_major,
    1,
    1,
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
  )
  assert_eq(info_sorgql, 0)

  let info_dorgql = lapacke_dorgql(
    e2_lapack_col_major,
    1,
    1,
    1,
    e2_double_buf,
    1,
    e2_double_buf2,
  )
  assert_eq(info_dorgql, 0)

  let info_sorgqr = lapacke_sorgqr(
    e2_lapack_col_major,
    1,
    1,
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
  )
  assert_eq(info_sorgqr, 0)

  let info_dorgqr = lapacke_dorgqr(
    e2_lapack_col_major,
    1,
    1,
    1,
    e2_double_buf,
    1,
    e2_double_buf2,
  )
  assert_eq(info_dorgqr, 0)

  let info_sorgrq = lapacke_sorgrq(
    e2_lapack_col_major,
    1,
    1,
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
  )
  assert_eq(info_sorgrq, 0)

  let info_dorgrq = lapacke_dorgrq(
    e2_lapack_col_major,
    1,
    1,
    1,
    e2_double_buf,
    1,
    e2_double_buf2,
  )
  assert_eq(info_dorgrq, 0)

  let info_sorgtr = lapacke_sorgtr(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
  )
  assert_eq(info_sorgtr, 0)

  let info_dorgtr = lapacke_dorgtr(
    e2_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e2_double_buf,
    1,
    e2_double_buf2,
  )
  assert_eq(info_dorgtr, 0)

  let info_sorgtsqr_row = lapacke_sorgtsqr_row(
    e2_lapack_col_major,
    1,
    1,
    1,
    1,
    e2_float_buf,
    1,
    e2_float_buf2,
    1,
  )
  assert_eq(info_sorgtsqr_row, -4)
}
