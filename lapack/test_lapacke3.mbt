///|
let e3_lapack_col_major = 102

///|
let e3_float_buf : FixedArray[Float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

///|
let e3_float_buf2 : FixedArray[Float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

///|
let e3_double_buf : FixedArray[Double] = [
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

///|
let e3_double_buf2 : FixedArray[Double] = [
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

///|
let e3_complex_float_buf : FixedArray[Float] = [
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

///|
let e3_complex_float_buf2 : FixedArray[Float] = [
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

///|
let e3_complex_double_buf : FixedArray[Double] = [
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

///|
let e3_complex_double_buf2 : FixedArray[Double] = [
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

///|
let e3_int_buf : FixedArray[Int] = [0, 0, 0, 0, 0, 0, 0, 0]

///|
let e3_int_buf2 : FixedArray[Int] = [0, 0, 0, 0, 0, 0, 0, 0]

///|
let e3_byte_buf : FixedArray[Byte] = [0, 0, 0, 0, 0, 0, 0, 0]

///|
test "test_lapacke3_part1" {
  let info_lapacke_dorgtsqr_row = lapacke_dorgtsqr_row(
    101, 1, 1, 1, 1, e3_double_buf, 1, e3_double_buf2, 1,
  )
  inspect(info_lapacke_dorgtsqr_row, content="-4")
  let info_lapacke_sormbr = lapacke_sormbr(
    e3_lapack_col_major,
    'Q'.to_int().to_byte(),
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_sormbr, content="0")
  let info_lapacke_dormbr = lapacke_dormbr(
    e3_lapack_col_major,
    'Q'.to_int().to_byte(),
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dormbr, content="0")
  let info_lapacke_sormhr = lapacke_sormhr(
    e3_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_sormhr, content="0")
  let info_lapacke_dormhr = lapacke_dormhr(
    e3_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dormhr, content="0")
  let info_lapacke_sormlq = lapacke_sormlq(
    e3_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_sormlq, content="0")
  let info_lapacke_dormlq = lapacke_dormlq(
    e3_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dormlq, content="0")
  let info_lapacke_sormql = lapacke_sormql(
    e3_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_sormql, content="0")
  let info_lapacke_dormql = lapacke_dormql(
    e3_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dormql, content="0")
  let info_lapacke_sormqr = lapacke_sormqr(
    e3_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_sormqr, content="0")
  let info_lapacke_dormqr = lapacke_dormqr(
    e3_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dormqr, content="0")
  let info_lapacke_sormrq = lapacke_sormrq(
    e3_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_sormrq, content="0")
  let info_lapacke_dormrq = lapacke_dormrq(
    e3_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dormrq, content="0")
  let info_lapacke_sormrz = lapacke_sormrz(
    e3_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_sormrz, content="0")
  let info_lapacke_dormrz = lapacke_dormrz(
    e3_lapack_col_major,
    'L'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dormrz, content="0")
  let info_lapacke_sormtr = lapacke_sormtr(
    e3_lapack_col_major,
    'L'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_sormtr, content="0")
  let info_lapacke_dormtr = lapacke_dormtr(
    e3_lapack_col_major,
    'L'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dormtr, content="0")
  let info_lapacke_spbcon = lapacke_spbcon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_float_buf,
    1,
    1,
    e3_float_buf2,
  )
  inspect(info_lapacke_spbcon, content="0")
  let info_lapacke_dpbcon = lapacke_dpbcon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_double_buf,
    1,
    1,
    e3_double_buf2,
  )
  inspect(info_lapacke_dpbcon, content="0")
  let info_lapacke_cpbcon = lapacke_cpbcon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_complex_float_buf,
    1,
    1,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_cpbcon, content="0")
  let info_lapacke_zpbcon = lapacke_zpbcon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_complex_double_buf,
    1,
    1,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zpbcon, content="0")
  let info_lapacke_spbequ = lapacke_spbequ(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_float_buf,
    1,
    e3_float_buf2,
    e3_float_buf,
    e3_float_buf2,
  )
  inspect(info_lapacke_spbequ, content="1")
  let info_lapacke_dpbequ = lapacke_dpbequ(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_double_buf,
    1,
    e3_double_buf2,
    e3_double_buf,
    e3_double_buf2,
  )
  inspect(info_lapacke_dpbequ, content="1")
  let info_lapacke_cpbequ = lapacke_cpbequ(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    e3_complex_float_buf,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_cpbequ, content="1")
  let info_lapacke_zpbequ = lapacke_zpbequ(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    e3_complex_double_buf,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zpbequ, content="1")
  let info_lapacke_spbrfs = lapacke_spbrfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
    e3_float_buf,
    e3_float_buf2,
  )
  inspect(info_lapacke_spbrfs, content="0")
  let info_lapacke_dpbrfs = lapacke_dpbrfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    e3_double_buf,
    e3_double_buf2,
  )
  inspect(info_lapacke_dpbrfs, content="0")
  let info_lapacke_cpbrfs = lapacke_cpbrfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    1,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_cpbrfs, content="0")
  let info_lapacke_zpbrfs = lapacke_zpbrfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zpbrfs, content="0")
  let info_lapacke_spbstf = lapacke_spbstf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_spbstf, content="0")
}

///|
test "test_lapacke3_part2" {
  let info_lapacke_dpbstf = lapacke_dpbstf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dpbstf, content="0")
  let info_lapacke_cpbstf = lapacke_cpbstf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_complex_float_buf,
    1,
  )
  inspect(info_lapacke_cpbstf, content="0")
  let info_lapacke_zpbstf = lapacke_zpbstf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_complex_double_buf,
    1,
  )
  inspect(info_lapacke_zpbstf, content="0")
  let info_lapacke_spbsv = lapacke_spbsv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
  )
  inspect(info_lapacke_spbsv, content="0")
  let info_lapacke_dpbsv = lapacke_dpbsv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
  )
  inspect(info_lapacke_dpbsv, content="0")
  let info_lapacke_cpbsv = lapacke_cpbsv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    1,
  )
  inspect(info_lapacke_cpbsv, content="0")
  let info_lapacke_zpbsv = lapacke_zpbsv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
  )
  inspect(info_lapacke_zpbsv, content="0")
  let info_lapacke_spbsvx = lapacke_spbsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
    e3_byte_buf,
    e3_float_buf,
    e3_float_buf2,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    e3_float_buf,
    e3_float_buf2,
  )
  inspect(info_lapacke_spbsvx, content="2")
  let info_lapacke_dpbsvx = lapacke_dpbsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    e3_byte_buf,
    e3_double_buf,
    e3_double_buf2,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    e3_double_buf,
    e3_double_buf2,
  )
  inspect(info_lapacke_dpbsvx, content="2")
  let info_lapacke_cpbsvx = lapacke_cpbsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    1,
    e3_byte_buf,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    1,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    e3_complex_float_buf,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_cpbsvx, content="2")
  let info_lapacke_zpbsvx = lapacke_zpbsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
    e3_byte_buf,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    1,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    e3_complex_double_buf,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zpbsvx, content="2")
  let info_lapacke_spbtrf = lapacke_spbtrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_spbtrf, content="-5")
  let info_lapacke_dpbtrf = lapacke_dpbtrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dpbtrf, content="-5")
  let info_lapacke_cpbtrf = lapacke_cpbtrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_complex_float_buf,
    1,
  )
  inspect(info_lapacke_cpbtrf, content="-5")
  let info_lapacke_zpbtrf = lapacke_zpbtrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    e3_complex_double_buf,
    1,
  )
  inspect(info_lapacke_zpbtrf, content="-5")
  let info_lapacke_spbtrs = lapacke_spbtrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
  )
  inspect(info_lapacke_spbtrs, content="-6")
  let info_lapacke_dpbtrs = lapacke_dpbtrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
  )
  inspect(info_lapacke_dpbtrs, content="-6")
  let info_lapacke_cpbtrs = lapacke_cpbtrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    1,
  )
  inspect(info_lapacke_cpbtrs, content="-6")
  let info_lapacke_zpbtrs = lapacke_zpbtrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    0,
    1,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
  )
  inspect(info_lapacke_zpbtrs, content="-6")
  let info_lapacke_spftrf = lapacke_spftrf(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
  )
  inspect(info_lapacke_spftrf, content="-5")
  let info_lapacke_dpftrf = lapacke_dpftrf(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
  )
  inspect(info_lapacke_dpftrf, content="-5")
  let info_lapacke_cpftrf = lapacke_cpftrf(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
  )
  inspect(info_lapacke_cpftrf, content="-5")
  let info_lapacke_zpftrf = lapacke_zpftrf(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
  )
  inspect(info_lapacke_zpftrf, content="-5")
  let info_lapacke_spftri = lapacke_spftri(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
  )
  inspect(info_lapacke_spftri, content="-5")
  let info_lapacke_dpftri = lapacke_dpftri(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
  )
  inspect(info_lapacke_dpftri, content="-5")
  let info_lapacke_cpftri = lapacke_cpftri(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
  )
  inspect(info_lapacke_cpftri, content="-5")
  let info_lapacke_zpftri = lapacke_zpftri(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
  )
  inspect(info_lapacke_zpftri, content="-5")
  let info_lapacke_spftrs = lapacke_spftrs(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_float_buf,
    e3_float_buf2,
    1,
  )
  inspect(info_lapacke_spftrs, content="-6")
  let info_lapacke_dpftrs = lapacke_dpftrs(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    e3_double_buf2,
    1,
  )
  inspect(info_lapacke_dpftrs, content="-6")
  let info_lapacke_cpftrs = lapacke_cpftrs(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    1,
  )
  inspect(info_lapacke_cpftrs, content="-6")
}

///|
test "test_lapacke3_part3" {
  let info_lapacke_zpftrs = lapacke_zpftrs(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    1,
  )
  inspect(info_lapacke_zpftrs, content="-6")
  let info_lapacke_spocon = lapacke_spocon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    1,
    1,
    e3_float_buf2,
  )
  inspect(info_lapacke_spocon, content="-4")
  let info_lapacke_dpocon = lapacke_dpocon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    1,
    1,
    e3_double_buf2,
  )
  inspect(info_lapacke_dpocon, content="-4")
  let info_lapacke_cpocon = lapacke_cpocon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
    1,
    1,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_cpocon, content="-4")
  let info_lapacke_zpocon = lapacke_zpocon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
    1,
    1,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zpocon, content="-4")
  let info_lapacke_spoequ = lapacke_spoequ(
    e3_lapack_col_major, 1, e3_float_buf, 1, e3_float_buf2, e3_float_buf, e3_float_buf2,
  )
  inspect(info_lapacke_spoequ, content="-3")
  let info_lapacke_dpoequ = lapacke_dpoequ(
    e3_lapack_col_major, 1, e3_double_buf, 1, e3_double_buf2, e3_double_buf, e3_double_buf2,
  )
  inspect(info_lapacke_dpoequ, content="-3")
  let info_lapacke_cpoequ = lapacke_cpoequ(
    e3_lapack_col_major, 1, e3_complex_float_buf, 1, e3_complex_float_buf2, e3_complex_float_buf,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_cpoequ, content="-3")
  let info_lapacke_zpoequ = lapacke_zpoequ(
    e3_lapack_col_major, 1, e3_complex_double_buf, 1, e3_complex_double_buf2, e3_complex_double_buf,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zpoequ, content="-3")
  let info_lapacke_spoequb = lapacke_spoequb(
    e3_lapack_col_major, 1, e3_float_buf, 1, e3_float_buf2, e3_float_buf, e3_float_buf2,
  )
  inspect(info_lapacke_spoequb, content="-3")
  let info_lapacke_dpoequb = lapacke_dpoequb(
    e3_lapack_col_major, 1, e3_double_buf, 1, e3_double_buf2, e3_double_buf, e3_double_buf2,
  )
  inspect(info_lapacke_dpoequb, content="-3")
  let info_lapacke_cpoequb = lapacke_cpoequb(
    e3_lapack_col_major, 1, e3_complex_float_buf, 1, e3_complex_float_buf2, e3_complex_float_buf,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_cpoequb, content="-3")
  let info_lapacke_zpoequb = lapacke_zpoequb(
    e3_lapack_col_major, 1, e3_complex_double_buf, 1, e3_complex_double_buf2, e3_complex_double_buf,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zpoequb, content="-3")
  let info_lapacke_sporfs = lapacke_sporfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
    e3_float_buf,
    e3_float_buf2,
  )
  inspect(info_lapacke_sporfs, content="-5")
  let info_lapacke_dporfs = lapacke_dporfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    e3_double_buf,
    e3_double_buf2,
  )
  inspect(info_lapacke_dporfs, content="-5")
  let info_lapacke_cporfs = lapacke_cporfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    1,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_cporfs, content="-5")
  let info_lapacke_zporfs = lapacke_zporfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zporfs, content="-5")
  let info_lapacke_sposv = lapacke_sposv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
  )
  inspect(info_lapacke_sposv, content="-5")
  let info_lapacke_dposv = lapacke_dposv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
  )
  inspect(info_lapacke_dposv, content="-5")
  let info_lapacke_cposv = lapacke_cposv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    1,
  )
  inspect(info_lapacke_cposv, content="-5")
  let info_lapacke_zposv = lapacke_zposv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
  )
  inspect(info_lapacke_zposv, content="-5")
  let info_lapacke_dsposv = lapacke_dsposv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    e3_double_buf,
    1,
    e3_int_buf,
  )
  inspect(info_lapacke_dsposv, content="-5")
  let info_lapacke_zcposv = lapacke_zcposv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
    e3_complex_double_buf,
    1,
    e3_int_buf,
  )
  inspect(info_lapacke_zcposv, content="-5")
  let info_lapacke_sposvx = lapacke_sposvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
    e3_byte_buf,
    e3_float_buf,
    e3_float_buf2,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    e3_float_buf,
    e3_float_buf2,
  )
  inspect(info_lapacke_sposvx, content="-6")
  let info_lapacke_dposvx = lapacke_dposvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    e3_byte_buf,
    e3_double_buf,
    e3_double_buf2,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    e3_double_buf,
    e3_double_buf2,
  )
  inspect(info_lapacke_dposvx, content="-6")
  let info_lapacke_cposvx = lapacke_cposvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    1,
    e3_byte_buf,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    1,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    e3_complex_float_buf,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_cposvx, content="-6")
  let info_lapacke_zposvx = lapacke_zposvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
    e3_byte_buf,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    1,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    e3_complex_double_buf,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zposvx, content="-6")
  let info_lapacke_spotrf2 = lapacke_spotrf2(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_spotrf2, content="-4")
  let info_lapacke_dpotrf2 = lapacke_dpotrf2(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dpotrf2, content="-4")
  let info_lapacke_cpotrf2 = lapacke_cpotrf2(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
    1,
  )
  inspect(info_lapacke_cpotrf2, content="-4")
}

///|
test "test_lapacke3_part4" {
  let info_lapacke_zpotrf2 = lapacke_zpotrf2(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
    1,
  )
  inspect(info_lapacke_zpotrf2, content="-4")
  let info_lapacke_spotrf = lapacke_spotrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_spotrf, content="-4")
  let info_lapacke_dpotrf = lapacke_dpotrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dpotrf, content="-4")
  let info_lapacke_cpotrf = lapacke_cpotrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
    1,
  )
  inspect(info_lapacke_cpotrf, content="-4")
  let info_lapacke_zpotrf = lapacke_zpotrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
    1,
  )
  inspect(info_lapacke_zpotrf, content="-4")
  let info_lapacke_spotri = lapacke_spotri(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_spotri, content="-4")
  let info_lapacke_dpotri = lapacke_dpotri(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dpotri, content="-4")
  let info_lapacke_cpotri = lapacke_cpotri(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
    1,
  )
  inspect(info_lapacke_cpotri, content="-4")
  let info_lapacke_zpotri = lapacke_zpotri(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
    1,
  )
  inspect(info_lapacke_zpotri, content="-4")
  let info_lapacke_spotrs = lapacke_spotrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
  )
  inspect(info_lapacke_spotrs, content="-5")
  let info_lapacke_dpotrs = lapacke_dpotrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
  )
  inspect(info_lapacke_dpotrs, content="-5")
  let info_lapacke_cpotrs = lapacke_cpotrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    1,
  )
  inspect(info_lapacke_cpotrs, content="-5")
  let info_lapacke_zpotrs = lapacke_zpotrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
  )
  inspect(info_lapacke_zpotrs, content="-5")
  let info_lapacke_sppcon = lapacke_sppcon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
  )
  inspect(info_lapacke_sppcon, content="-4")
  let info_lapacke_dppcon = lapacke_dppcon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
  )
  inspect(info_lapacke_dppcon, content="-4")
  let info_lapacke_cppcon = lapacke_cppcon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_cppcon, content="-4")
  let info_lapacke_zppcon = lapacke_zppcon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zppcon, content="-4")
  let info_lapacke_sppequ = lapacke_sppequ(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_float_buf,
    e3_float_buf2,
  )
  inspect(info_lapacke_sppequ, content="-4")
  let info_lapacke_dppequ = lapacke_dppequ(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_double_buf,
    e3_double_buf2,
  )
  inspect(info_lapacke_dppequ, content="-4")
  let info_lapacke_cppequ = lapacke_cppequ(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    e3_complex_float_buf,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_cppequ, content="-4")
  let info_lapacke_zppequ = lapacke_zppequ(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    e3_complex_double_buf,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zppequ, content="-4")
  let info_lapacke_spprfs = lapacke_spprfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
    e3_float_buf,
    e3_float_buf2,
  )
  inspect(info_lapacke_spprfs, content="-5")
  let info_lapacke_dpprfs = lapacke_dpprfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    e3_double_buf,
    e3_double_buf2,
  )
  inspect(info_lapacke_dpprfs, content="-5")
  let info_lapacke_cpprfs = lapacke_cpprfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_cpprfs, content="-5")
  let info_lapacke_zpprfs = lapacke_zpprfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zpprfs, content="-5")
  let info_lapacke_sppsv = lapacke_sppsv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_float_buf,
    e3_float_buf2,
    1,
  )
  inspect(info_lapacke_sppsv, content="-5")
  let info_lapacke_dppsv = lapacke_dppsv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    e3_double_buf2,
    1,
  )
  inspect(info_lapacke_dppsv, content="-5")
  let info_lapacke_cppsv = lapacke_cppsv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    1,
  )
  inspect(info_lapacke_cppsv, content="-5")
  let info_lapacke_zppsv = lapacke_zppsv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    1,
  )
  inspect(info_lapacke_zppsv, content="-5")
  let info_lapacke_sppsvx = lapacke_sppsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_byte_buf,
    e3_float_buf,
    e3_float_buf2,
    1,
    e3_float_buf,
    1,
    e3_float_buf2,
    e3_float_buf,
    e3_float_buf2,
  )
  inspect(info_lapacke_sppsvx, content="-6")
}

///|
test "test_lapacke3_part5" {
  let info_lapacke_dppsvx = lapacke_dppsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_byte_buf,
    e3_double_buf,
    e3_double_buf2,
    1,
    e3_double_buf,
    1,
    e3_double_buf2,
    e3_double_buf,
    e3_double_buf2,
  )
  inspect(info_lapacke_dppsvx, content="-6")
  let info_lapacke_cppsvx = lapacke_cppsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    e3_byte_buf,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    1,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    e3_complex_float_buf,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_cppsvx, content="-6")
  let info_lapacke_zppsvx = lapacke_zppsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    e3_byte_buf,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    1,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    e3_complex_double_buf,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zppsvx, content="-6")
  let info_lapacke_spptrf = lapacke_spptrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
  )
  inspect(info_lapacke_spptrf, content="-4")
  let info_lapacke_dpptrf = lapacke_dpptrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
  )
  inspect(info_lapacke_dpptrf, content="-4")
  let info_lapacke_cpptrf = lapacke_cpptrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
  )
  inspect(info_lapacke_cpptrf, content="-4")
  let info_lapacke_zpptrf = lapacke_zpptrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
  )
  inspect(info_lapacke_zpptrf, content="-4")
  let info_lapacke_spptri = lapacke_spptri(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
  )
  inspect(info_lapacke_spptri, content="-4")
  let info_lapacke_dpptri = lapacke_dpptri(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
  )
  inspect(info_lapacke_dpptri, content="-4")
  let info_lapacke_cpptri = lapacke_cpptri(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
  )
  inspect(info_lapacke_cpptri, content="-4")
  let info_lapacke_zpptri = lapacke_zpptri(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
  )
  inspect(info_lapacke_zpptri, content="-4")
  let info_lapacke_spptrs = lapacke_spptrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_float_buf,
    e3_float_buf2,
    1,
  )
  inspect(info_lapacke_spptrs, content="-5")
  let info_lapacke_dpptrs = lapacke_dpptrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    e3_double_buf2,
    1,
  )
  inspect(info_lapacke_dpptrs, content="-5")
  let info_lapacke_cpptrs = lapacke_cpptrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    1,
  )
  inspect(info_lapacke_cpptrs, content="-5")
  let info_lapacke_zpptrs = lapacke_zpptrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    1,
  )
  inspect(info_lapacke_zpptrs, content="-5")
  let info_lapacke_spstrf = lapacke_spstrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    1,
    e3_int_buf,
    e3_int_buf2,
    1,
  )
  inspect(info_lapacke_spstrf, content="-4")
  let info_lapacke_dpstrf = lapacke_dpstrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    1,
    e3_int_buf,
    e3_int_buf2,
    1,
  )
  inspect(info_lapacke_dpstrf, content="-4")
  let info_lapacke_cpstrf = lapacke_cpstrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
    1,
    e3_int_buf,
    e3_int_buf2,
    1,
  )
  inspect(info_lapacke_cpstrf, content="-4")
  let info_lapacke_zpstrf = lapacke_zpstrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
    1,
    e3_int_buf,
    e3_int_buf2,
    1,
  )
  inspect(info_lapacke_zpstrf, content="-4")
  let info_lapacke_sptcon = lapacke_sptcon(
    1, e3_float_buf, e3_float_buf2, 1, e3_float_buf,
  )
  inspect(info_lapacke_sptcon, content="-2")
  let info_lapacke_dptcon = lapacke_dptcon(
    1, e3_double_buf, e3_double_buf2, 1, e3_double_buf,
  )
  inspect(info_lapacke_dptcon, content="-2")
  let info_lapacke_cptcon = lapacke_cptcon(
    1, e3_complex_float_buf, e3_complex_float_buf2, 1, e3_complex_float_buf,
  )
  inspect(info_lapacke_cptcon, content="-2")
  let info_lapacke_zptcon = lapacke_zptcon(
    1, e3_complex_double_buf, e3_complex_double_buf2, 1, e3_complex_double_buf,
  )
  inspect(info_lapacke_zptcon, content="-2")
  let info_lapacke_spteqr = lapacke_spteqr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_spteqr, content="-4")
  let info_lapacke_dpteqr = lapacke_dpteqr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dpteqr, content="-4")
  let info_lapacke_cpteqr = lapacke_cpteqr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    e3_complex_float_buf,
    1,
  )
  inspect(info_lapacke_cpteqr, content="-4")
  let info_lapacke_zpteqr = lapacke_zpteqr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    e3_complex_double_buf,
    1,
  )
  inspect(info_lapacke_zpteqr, content="-4")
  let info_lapacke_sptrfs = lapacke_sptrfs(
    e3_lapack_col_major, 1, 1, e3_float_buf, e3_float_buf2, e3_float_buf, e3_float_buf2,
    e3_float_buf, 1, e3_float_buf2, 1, e3_float_buf, e3_float_buf2,
  )
  inspect(info_lapacke_sptrfs, content="-8")
  let info_lapacke_dptrfs = lapacke_dptrfs(
    e3_lapack_col_major, 1, 1, e3_double_buf, e3_double_buf2, e3_double_buf, e3_double_buf2,
    e3_double_buf, 1, e3_double_buf2, 1, e3_double_buf, e3_double_buf2,
  )
  inspect(info_lapacke_dptrfs, content="-8")
  let info_lapacke_cptrfs = lapacke_cptrfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_cptrfs, content="-9")
}

///|
test "test_lapacke3_part6" {
  let info_lapacke_zptrfs = lapacke_zptrfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zptrfs, content="-9")
  let info_lapacke_sptsv = lapacke_sptsv(
    e3_lapack_col_major, 1, 1, e3_float_buf, e3_float_buf2, e3_float_buf, 1,
  )
  inspect(info_lapacke_sptsv, content="-6")
  let info_lapacke_dptsv = lapacke_dptsv(
    e3_lapack_col_major, 1, 1, e3_double_buf, e3_double_buf2, e3_double_buf, 1,
  )
  inspect(info_lapacke_dptsv, content="-6")
  let info_lapacke_cptsv = lapacke_cptsv(
    e3_lapack_col_major, 1, 1, e3_complex_float_buf, e3_complex_float_buf2, e3_complex_float_buf,
    1,
  )
  inspect(info_lapacke_cptsv, content="-6")
  let info_lapacke_zptsv = lapacke_zptsv(
    e3_lapack_col_major, 1, 1, e3_complex_double_buf, e3_complex_double_buf2, e3_complex_double_buf,
    1,
  )
  inspect(info_lapacke_zptsv, content="-6")
  let info_lapacke_sptsvx = lapacke_sptsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_float_buf,
    e3_float_buf2,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_float_buf,
  )
  inspect(info_lapacke_sptsvx, content="-9")
  let info_lapacke_dptsvx = lapacke_dptsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_double_buf,
    e3_double_buf2,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_double_buf,
  )
  inspect(info_lapacke_dptsvx, content="-9")
  let info_lapacke_cptsvx = lapacke_cptsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    e3_complex_float_buf,
  )
  inspect(info_lapacke_cptsvx, content="-9")
  let info_lapacke_zptsvx = lapacke_zptsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    e3_complex_double_buf,
  )
  inspect(info_lapacke_zptsvx, content="-9")
  let info_lapacke_spttrf = lapacke_spttrf(1, e3_float_buf, e3_float_buf2)
  inspect(info_lapacke_spttrf, content="-2")
  let info_lapacke_dpttrf = lapacke_dpttrf(1, e3_double_buf, e3_double_buf2)
  inspect(info_lapacke_dpttrf, content="-2")
  let info_lapacke_cpttrf = lapacke_cpttrf(
    1, e3_complex_float_buf, e3_complex_float_buf2,
  )
  inspect(info_lapacke_cpttrf, content="-2")
  let info_lapacke_zpttrf = lapacke_zpttrf(
    1, e3_complex_double_buf, e3_complex_double_buf2,
  )
  inspect(info_lapacke_zpttrf, content="-2")
  let info_lapacke_spttrs = lapacke_spttrs(
    e3_lapack_col_major, 1, 1, e3_float_buf, e3_float_buf2, e3_float_buf, 1,
  )
  inspect(info_lapacke_spttrs, content="-6")
  let info_lapacke_dpttrs = lapacke_dpttrs(
    e3_lapack_col_major, 1, 1, e3_double_buf, e3_double_buf2, e3_double_buf, 1,
  )
  inspect(info_lapacke_dpttrs, content="-6")
  let info_lapacke_cpttrs = lapacke_cpttrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    e3_complex_float_buf,
    1,
  )
  inspect(info_lapacke_cpttrs, content="-7")
  let info_lapacke_zpttrs = lapacke_zpttrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    e3_complex_double_buf,
    1,
  )
  inspect(info_lapacke_zpttrs, content="-7")
  let info_lapacke_ssbev = lapacke_ssbev(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e3_float_buf,
    1,
    e3_float_buf2,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_ssbev, content="-6")
  let info_lapacke_dsbev = lapacke_dsbev(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e3_double_buf,
    1,
    e3_double_buf2,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dsbev, content="-6")
  let info_lapacke_ssbevd = lapacke_ssbevd(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e3_float_buf,
    1,
    e3_float_buf2,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_ssbevd, content="-6")
  let info_lapacke_dsbevd = lapacke_dsbevd(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e3_double_buf,
    1,
    e3_double_buf2,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dsbevd, content="-6")
  let info_lapacke_ssbevx = lapacke_ssbevx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
    1,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_float_buf,
    e3_float_buf2,
    1,
    e3_int_buf2,
  )
  inspect(info_lapacke_ssbevx, content="-7")
  let info_lapacke_dsbevx = lapacke_dsbevx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    1,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_double_buf,
    e3_double_buf2,
    1,
    e3_int_buf2,
  )
  inspect(info_lapacke_dsbevx, content="-7")
  let info_lapacke_ssbgst = lapacke_ssbgst(
    e3_lapack_col_major,
    'Q'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    0,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_ssbgst, content="-7")
  let info_lapacke_dsbgst = lapacke_dsbgst(
    e3_lapack_col_major,
    'Q'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    0,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dsbgst, content="-7")
  let info_lapacke_ssbgv = lapacke_ssbgv(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    0,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
    e3_float_buf,
    e3_float_buf2,
    1,
  )
  inspect(info_lapacke_ssbgv, content="-7")
  let info_lapacke_dsbgv = lapacke_dsbgv(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    0,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    e3_double_buf,
    e3_double_buf2,
    1,
  )
  inspect(info_lapacke_dsbgv, content="-7")
  let info_lapacke_ssbgvd = lapacke_ssbgvd(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    0,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
    e3_float_buf,
    e3_float_buf2,
    1,
  )
  inspect(info_lapacke_ssbgvd, content="-7")
  let info_lapacke_dsbgvd = lapacke_dsbgvd(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    0,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    e3_double_buf,
    e3_double_buf2,
    1,
  )
  inspect(info_lapacke_dsbgvd, content="-7")
  let info_lapacke_ssbgvx = lapacke_ssbgvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    0,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
    e3_float_buf,
    1,
    1,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_float_buf2,
    e3_float_buf,
    1,
    e3_int_buf2,
  )
  inspect(info_lapacke_ssbgvx, content="-8")
}

///|
test "test_lapacke3_part7" {
  let info_lapacke_dsbgvx = lapacke_dsbgvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    0,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    e3_double_buf,
    1,
    1,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_double_buf2,
    e3_double_buf,
    1,
    e3_int_buf2,
  )
  inspect(info_lapacke_dsbgvx, content="-8")
  let info_lapacke_ssbtrd = lapacke_ssbtrd(
    e3_lapack_col_major,
    'Q'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e3_float_buf,
    1,
    e3_float_buf2,
    e3_float_buf,
    e3_float_buf2,
    1,
  )
  inspect(info_lapacke_ssbtrd, content="-6")
  let info_lapacke_dsbtrd = lapacke_dsbtrd(
    e3_lapack_col_major,
    'Q'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    0,
    e3_double_buf,
    1,
    e3_double_buf2,
    e3_double_buf,
    e3_double_buf2,
    1,
  )
  inspect(info_lapacke_dsbtrd, content="-6")
  let info_lapacke_ssfrk = lapacke_ssfrk(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e3_float_buf,
    1,
    1,
    e3_float_buf2,
  )
  inspect(info_lapacke_ssfrk, content="-8")
  let info_lapacke_dsfrk = lapacke_dsfrk(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    'N'.to_int().to_byte(),
    1,
    1,
    1,
    e3_double_buf,
    1,
    1,
    e3_double_buf2,
  )
  inspect(info_lapacke_dsfrk, content="-8")
  let info_lapacke_sspcon = lapacke_sspcon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_int_buf,
    1,
    e3_float_buf2,
  )
  inspect(info_lapacke_sspcon, content="-4")
  let info_lapacke_dspcon = lapacke_dspcon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_int_buf,
    1,
    e3_double_buf2,
  )
  inspect(info_lapacke_dspcon, content="-4")
  let info_lapacke_cspcon = lapacke_cspcon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
    e3_int_buf,
    1,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_cspcon, content="-4")
  let info_lapacke_zspcon = lapacke_zspcon(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
    e3_int_buf,
    1,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zspcon, content="-4")
  let info_lapacke_sspev = lapacke_sspev(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_sspev, content="-5")
  let info_lapacke_dspev = lapacke_dspev(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dspev, content="-5")
  let info_lapacke_sspevd = lapacke_sspevd(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_sspevd, content="-5")
  let info_lapacke_dspevd = lapacke_dspevd(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dspevd, content="-5")
  let info_lapacke_sspevx = lapacke_sspevx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    1,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_float_buf2,
    e3_float_buf,
    1,
    e3_int_buf2,
  )
  inspect(info_lapacke_sspevx, content="-6")
  let info_lapacke_dspevx = lapacke_dspevx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    1,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_double_buf2,
    e3_double_buf,
    1,
    e3_int_buf2,
  )
  inspect(info_lapacke_dspevx, content="-6")
  let info_lapacke_sspgst = lapacke_sspgst(
    e3_lapack_col_major,
    1,
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_float_buf2,
  )
  inspect(info_lapacke_sspgst, content="-5")
  let info_lapacke_dspgst = lapacke_dspgst(
    e3_lapack_col_major,
    1,
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_double_buf2,
  )
  inspect(info_lapacke_dspgst, content="-5")
  let info_lapacke_sspgv = lapacke_sspgv(
    e3_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_float_buf,
    e3_float_buf2,
    1,
  )
  inspect(info_lapacke_sspgv, content="-6")
  let info_lapacke_dspgv = lapacke_dspgv(
    e3_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_double_buf,
    e3_double_buf2,
    1,
  )
  inspect(info_lapacke_dspgv, content="-6")
  let info_lapacke_sspgvd = lapacke_sspgvd(
    e3_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_float_buf,
    e3_float_buf2,
    1,
  )
  inspect(info_lapacke_sspgvd, content="-6")
  let info_lapacke_dspgvd = lapacke_dspgvd(
    e3_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_double_buf,
    e3_double_buf2,
    1,
  )
  inspect(info_lapacke_dspgvd, content="-6")
  let info_lapacke_sspgvx = lapacke_sspgvx(
    e3_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_float_buf2,
    1,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_float_buf,
    e3_float_buf2,
    1,
    e3_int_buf2,
  )
  inspect(info_lapacke_sspgvx, content="-7")
  let info_lapacke_dspgvx = lapacke_dspgvx(
    e3_lapack_col_major,
    1,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_double_buf2,
    1,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_double_buf,
    e3_double_buf2,
    1,
    e3_int_buf2,
  )
  inspect(info_lapacke_dspgvx, content="-7")
  let info_lapacke_ssprfs = lapacke_ssprfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_int_buf,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
    e3_float_buf,
    e3_float_buf2,
  )
  inspect(info_lapacke_ssprfs, content="-5")
  let info_lapacke_dsprfs = lapacke_dsprfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_int_buf,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    e3_double_buf,
    e3_double_buf2,
  )
  inspect(info_lapacke_dsprfs, content="-5")
  let info_lapacke_csprfs = lapacke_csprfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    e3_int_buf,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
  )
  inspect(info_lapacke_csprfs, content="-5")
  let info_lapacke_zsprfs = lapacke_zsprfs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    e3_int_buf,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
  )
  inspect(info_lapacke_zsprfs, content="-5")
  let info_lapacke_sspsv = lapacke_sspsv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_float_buf,
    e3_int_buf,
    e3_float_buf2,
    1,
  )
  inspect(info_lapacke_sspsv, content="-5")
  let info_lapacke_dspsv = lapacke_dspsv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    e3_int_buf,
    e3_double_buf2,
    1,
  )
  inspect(info_lapacke_dspsv, content="-5")
  let info_lapacke_cspsv = lapacke_cspsv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    e3_int_buf,
    e3_complex_float_buf2,
    1,
  )
  inspect(info_lapacke_cspsv, content="-5")
}

///|
test "test_lapacke3_part8" {
  let info_lapacke_zspsv = lapacke_zspsv(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    e3_int_buf,
    e3_complex_double_buf2,
    1,
  )
  inspect(info_lapacke_zspsv, content="-5")
  let info_lapacke_sspsvx = lapacke_sspsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_int_buf,
    e3_float_buf,
    1,
    e3_float_buf2,
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_float_buf,
  )
  inspect(info_lapacke_sspsvx, content="-6")
  let info_lapacke_dspsvx = lapacke_dspsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_int_buf,
    e3_double_buf,
    1,
    e3_double_buf2,
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_double_buf,
  )
  inspect(info_lapacke_dspsvx, content="-6")
  let info_lapacke_cspsvx = lapacke_cspsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    e3_int_buf,
    e3_complex_float_buf,
    1,
    e3_complex_float_buf2,
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    e3_complex_float_buf,
  )
  inspect(info_lapacke_cspsvx, content="-6")
  let info_lapacke_zspsvx = lapacke_zspsvx(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    e3_int_buf,
    e3_complex_double_buf,
    1,
    e3_complex_double_buf2,
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    e3_complex_double_buf,
  )
  inspect(info_lapacke_zspsvx, content="-6")
  let info_lapacke_ssptrd = lapacke_ssptrd(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_float_buf,
    e3_float_buf2,
  )
  inspect(info_lapacke_ssptrd, content="-4")
  let info_lapacke_dsptrd = lapacke_dsptrd(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_double_buf,
    e3_double_buf2,
  )
  inspect(info_lapacke_dsptrd, content="-4")
  let info_lapacke_ssptrf = lapacke_ssptrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_int_buf,
  )
  inspect(info_lapacke_ssptrf, content="-4")
  let info_lapacke_dsptrf = lapacke_dsptrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_int_buf,
  )
  inspect(info_lapacke_dsptrf, content="-4")
  let info_lapacke_csptrf = lapacke_csptrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
    e3_int_buf,
  )
  inspect(info_lapacke_csptrf, content="-4")
  let info_lapacke_zsptrf = lapacke_zsptrf(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
    e3_int_buf,
  )
  inspect(info_lapacke_zsptrf, content="-4")
  let info_lapacke_ssptri = lapacke_ssptri(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_int_buf,
  )
  inspect(info_lapacke_ssptri, content="-4")
  let info_lapacke_dsptri = lapacke_dsptri(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_int_buf,
  )
  inspect(info_lapacke_dsptri, content="-4")
  let info_lapacke_csptri = lapacke_csptri(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
    e3_int_buf,
  )
  inspect(info_lapacke_csptri, content="-4")
  let info_lapacke_zsptri = lapacke_zsptri(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
    e3_int_buf,
  )
  inspect(info_lapacke_zsptri, content="-4")
  let info_lapacke_ssptrs = lapacke_ssptrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_float_buf,
    e3_int_buf,
    e3_float_buf2,
    1,
  )
  inspect(info_lapacke_ssptrs, content="-5")
  let info_lapacke_dsptrs = lapacke_dsptrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_double_buf,
    e3_int_buf,
    e3_double_buf2,
    1,
  )
  inspect(info_lapacke_dsptrs, content="-5")
  let info_lapacke_csptrs = lapacke_csptrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_float_buf,
    e3_int_buf,
    e3_complex_float_buf2,
    1,
  )
  inspect(info_lapacke_csptrs, content="-5")
  let info_lapacke_zsptrs = lapacke_zsptrs(
    e3_lapack_col_major,
    'U'.to_int().to_byte(),
    1,
    1,
    e3_complex_double_buf,
    e3_int_buf,
    e3_complex_double_buf2,
    1,
  )
  inspect(info_lapacke_zsptrs, content="-5")
  let info_lapacke_sstebz = lapacke_sstebz(
    'A'.to_int().to_byte(),
    'R'.to_int().to_byte(),
    1,
    1,
    1,
    1,
    1,
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_int_buf,
    e3_int_buf2,
    e3_float_buf,
    e3_int_buf,
    e3_int_buf2,
  )
  inspect(info_lapacke_sstebz, content="-9")
  let info_lapacke_dstebz = lapacke_dstebz(
    'A'.to_int().to_byte(),
    'R'.to_int().to_byte(),
    1,
    1,
    1,
    1,
    1,
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_int_buf,
    e3_int_buf2,
    e3_double_buf,
    e3_int_buf,
    e3_int_buf2,
  )
  inspect(info_lapacke_dstebz, content="-9")
  let info_lapacke_sstedc = lapacke_sstedc(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_sstedc, content="-4")
  let info_lapacke_dstedc = lapacke_dstedc(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dstedc, content="-4")
  let info_lapacke_cstedc = lapacke_cstedc(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    e3_complex_float_buf,
    1,
  )
  inspect(info_lapacke_cstedc, content="-4")
  let info_lapacke_zstedc = lapacke_zstedc(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    e3_complex_double_buf,
    1,
  )
  inspect(info_lapacke_zstedc, content="-4")
  let info_lapacke_sstegr = lapacke_sstegr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_float_buf2,
    1,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_float_buf,
    e3_float_buf2,
    1,
    e3_int_buf2,
  )
  inspect(info_lapacke_sstegr, content="-5")
  let info_lapacke_dstegr = lapacke_dstegr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_double_buf2,
    1,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_double_buf,
    e3_double_buf2,
    1,
    e3_int_buf2,
  )
  inspect(info_lapacke_dstegr, content="-5")
  let info_lapacke_cstegr = lapacke_cstegr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    1,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    1,
    e3_int_buf2,
  )
  inspect(info_lapacke_cstegr, content="-5")
  let info_lapacke_zstegr = lapacke_zstegr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    1,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    1,
    e3_int_buf2,
  )
  inspect(info_lapacke_zstegr, content="-5")
  let info_lapacke_sstein = lapacke_sstein(
    e3_lapack_col_major, 1, e3_float_buf, e3_float_buf2, 1, e3_float_buf, e3_int_buf,
    e3_int_buf2, e3_float_buf2, 1, e3_int_buf,
  )
  inspect(info_lapacke_sstein, content="-3")
}

///|
test "test_lapacke3_part9" {
  let info_lapacke_dstein = lapacke_dstein(
    e3_lapack_col_major, 1, e3_double_buf, e3_double_buf2, 1, e3_double_buf, e3_int_buf,
    e3_int_buf2, e3_double_buf2, 1, e3_int_buf,
  )
  inspect(info_lapacke_dstein, content="-3")
  let info_lapacke_cstein = lapacke_cstein(
    e3_lapack_col_major, 1, e3_complex_float_buf, e3_complex_float_buf2, 1, e3_complex_float_buf,
    e3_int_buf, e3_int_buf2, e3_complex_float_buf2, 1, e3_int_buf,
  )
  inspect(info_lapacke_cstein, content="-3")
  let info_lapacke_zstein = lapacke_zstein(
    e3_lapack_col_major, 1, e3_complex_double_buf, e3_complex_double_buf2, 1, e3_complex_double_buf,
    e3_int_buf, e3_int_buf2, e3_complex_double_buf2, 1, e3_int_buf,
  )
  inspect(info_lapacke_zstein, content="-3")
  let info_lapacke_sstemr = lapacke_sstemr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_float_buf2,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_float_buf,
    e3_float_buf2,
    1,
    1,
    e3_int_buf2,
    0,
  )
  inspect(info_lapacke_sstemr, content="-5")
  let info_lapacke_dstemr = lapacke_dstemr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_double_buf2,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_double_buf,
    e3_double_buf2,
    1,
    1,
    e3_int_buf2,
    0,
  )
  inspect(info_lapacke_dstemr, content="-5")
  let info_lapacke_cstemr = lapacke_cstemr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    1,
    1,
    e3_int_buf2,
    0,
  )
  inspect(info_lapacke_cstemr, content="-5")
  let info_lapacke_zstemr = lapacke_zstemr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    'A'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    1,
    1,
    1,
    1,
    e3_int_buf,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    1,
    1,
    e3_int_buf2,
    0,
  )
  inspect(info_lapacke_zstemr, content="-5")
  let info_lapacke_ssteqr = lapacke_ssteqr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    e3_float_buf,
    e3_float_buf2,
    e3_float_buf,
    1,
  )
  inspect(info_lapacke_ssteqr, content="-4")
  let info_lapacke_dsteqr = lapacke_dsteqr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    e3_double_buf,
    e3_double_buf2,
    e3_double_buf,
    1,
  )
  inspect(info_lapacke_dsteqr, content="-4")
  let info_lapacke_csteqr = lapacke_csteqr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    e3_complex_float_buf,
    e3_complex_float_buf2,
    e3_complex_float_buf,
    1,
  )
  inspect(info_lapacke_csteqr, content="-4")
  let info_lapacke_zsteqr = lapacke_zsteqr(
    e3_lapack_col_major,
    'N'.to_int().to_byte(),
    1,
    e3_complex_double_buf,
    e3_complex_double_buf2,
    e3_complex_double_buf,
    1,
  )
  inspect(info_lapacke_zsteqr, content="-4")
  let info_lapacke_ssterf = lapacke_ssterf(1, e3_float_buf, e3_float_buf2)
  inspect(info_lapacke_ssterf, content="-2")
}
