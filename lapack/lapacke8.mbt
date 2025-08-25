///|
///
/// ```c
/// lapack_int LAPACKE_sstev_work(
///     int matrix_layout, char jobz, lapack_int n, float* d, 
///     float* e, float* z, lapack_int ldz, float* work
/// );
/// ```
#borrow(d, e, z, work)
pub extern "C" fn lapacke_sstev_work(
  matrix_layout : Int,
  jobz : Byte,
  n : Lapack_int,
  d : FixedArray[Float],
  e : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sstev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dstev_work(
///     int matrix_layout, char jobz, lapack_int n, double* d, 
///     double* e, double* z, lapack_int ldz, double* work
/// );
/// ```
#borrow(d, e, z, work)
pub extern "C" fn lapacke_dstev_work(
  matrix_layout : Int,
  jobz : Byte,
  n : Lapack_int,
  d : FixedArray[Double],
  e : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dstev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sstevd_work(
///     int matrix_layout, char jobz, lapack_int n, float* d, 
///     float* e, float* z, lapack_int ldz, float* work, 
///     lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(d, e, z, work, iwork)
pub extern "C" fn lapacke_sstevd_work(
  matrix_layout : Int,
  jobz : Byte,
  n : Lapack_int,
  d : FixedArray[Float],
  e : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sstevd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dstevd_work(
///     int matrix_layout, char jobz, lapack_int n, double* d, 
///     double* e, double* z, lapack_int ldz, double* work, 
///     lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(d, e, z, work, iwork)
pub extern "C" fn lapacke_dstevd_work(
  matrix_layout : Int,
  jobz : Byte,
  n : Lapack_int,
  d : FixedArray[Double],
  e : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dstevd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sstevr_work(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     float* d, float* e, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, float* z, lapack_int ldz, lapack_int* isuppz, 
///     float* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(d, e, m, w, z, isuppz, work, iwork)
pub extern "C" fn lapacke_sstevr_work(
  matrix_layout : Int,
  jobz : Byte,
  range : Byte,
  n : Lapack_int,
  d : FixedArray[Float],
  e : FixedArray[Float],
  vl : Float,
  vu : Float,
  il : Lapack_int,
  iu : Lapack_int,
  abstol : Float,
  m : FixedArray[Int],
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  isuppz : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sstevr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dstevr_work(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     double* d, double* e, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, double* z, lapack_int ldz, lapack_int* isuppz, 
///     double* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(d, e, m, w, z, isuppz, work, iwork)
pub extern "C" fn lapacke_dstevr_work(
  matrix_layout : Int,
  jobz : Byte,
  range : Byte,
  n : Lapack_int,
  d : FixedArray[Double],
  e : FixedArray[Double],
  vl : Double,
  vu : Double,
  il : Lapack_int,
  iu : Lapack_int,
  abstol : Double,
  m : FixedArray[Int],
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  isuppz : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dstevr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sstevx_work(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     float* d, float* e, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, float* z, lapack_int ldz, float* work, 
///     lapack_int* iwork, lapack_int* ifail
/// );
/// ```
#borrow(d, e, m, w, z, work, iwork, ifail)
pub extern "C" fn lapacke_sstevx_work(
  matrix_layout : Int,
  jobz : Byte,
  range : Byte,
  n : Lapack_int,
  d : FixedArray[Float],
  e : FixedArray[Float],
  vl : Float,
  vu : Float,
  il : Lapack_int,
  iu : Lapack_int,
  abstol : Float,
  m : FixedArray[Int],
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sstevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dstevx_work(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     double* d, double* e, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, double* z, lapack_int ldz, double* work, 
///     lapack_int* iwork, lapack_int* ifail
/// );
/// ```
#borrow(d, e, m, w, z, work, iwork, ifail)
pub extern "C" fn lapacke_dstevx_work(
  matrix_layout : Int,
  jobz : Byte,
  range : Byte,
  n : Lapack_int,
  d : FixedArray[Double],
  e : FixedArray[Double],
  vl : Double,
  vu : Double,
  il : Lapack_int,
  iu : Lapack_int,
  abstol : Double,
  m : FixedArray[Int],
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dstevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssycon_work(
///     int matrix_layout, char uplo, lapack_int n, const float* a, 
///     lapack_int lda, const lapack_int* ipiv, float anorm, float* rcond, 
///     float* work, lapack_int* iwork
/// );
/// ```
#borrow(a, ipiv, rcond, work, iwork)
pub extern "C" fn lapacke_ssycon_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  anorm : Float,
  rcond : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ssycon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsycon_work(
///     int matrix_layout, char uplo, lapack_int n, const double* a, 
///     lapack_int lda, const lapack_int* ipiv, double anorm, double* rcond, 
///     double* work, lapack_int* iwork
/// );
/// ```
#borrow(a, ipiv, rcond, work, iwork)
pub extern "C" fn lapacke_dsycon_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  anorm : Double,
  rcond : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dsycon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csycon_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv, float anorm, float* rcond, lapack_complex_float* work
/// );
/// ```
#borrow(a, ipiv, rcond, work)
pub extern "C" fn lapacke_csycon_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  anorm : Float,
  rcond : FixedArray[Float],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_csycon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsycon_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv, double anorm, double* rcond, lapack_complex_double* work
/// );
/// ```
#borrow(a, ipiv, rcond, work)
pub extern "C" fn lapacke_zsycon_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  anorm : Double,
  rcond : FixedArray[Double],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zsycon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyequb_work(
///     int matrix_layout, char uplo, lapack_int n, const float* a, 
///     lapack_int lda, float* s, float* scond, float* amax, float* work
/// );
/// ```
#borrow(a, s, scond, amax, work)
pub extern "C" fn lapacke_ssyequb_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  s : FixedArray[Float],
  scond : FixedArray[Float],
  amax : FixedArray[Float],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ssyequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyequb_work(
///     int matrix_layout, char uplo, lapack_int n, const double* a, 
///     lapack_int lda, double* s, double* scond, double* amax, double* work
/// );
/// ```
#borrow(a, s, scond, amax, work)
pub extern "C" fn lapacke_dsyequb_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  s : FixedArray[Double],
  scond : FixedArray[Double],
  amax : FixedArray[Double],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dsyequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csyequb_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, float* s, float* scond, float* amax, lapack_complex_float* work
/// );
/// ```
#borrow(a, s, scond, amax, work)
pub extern "C" fn lapacke_csyequb_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  s : FixedArray[Float],
  scond : FixedArray[Float],
  amax : FixedArray[Float],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_csyequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsyequb_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, double* s, double* scond, double* amax, lapack_complex_double* work
/// );
/// ```
#borrow(a, s, scond, amax, work)
pub extern "C" fn lapacke_zsyequb_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  s : FixedArray[Double],
  scond : FixedArray[Double],
  amax : FixedArray[Double],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zsyequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyev_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     float* a, lapack_int lda, float* w, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, w, work)
pub extern "C" fn lapacke_ssyev_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  w : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ssyev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyev_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     double* a, lapack_int lda, double* w, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, w, work)
pub extern "C" fn lapacke_dsyev_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  w : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dsyev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyevd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     float* a, lapack_int lda, float* w, float* work, 
///     lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(a, w, work, iwork)
pub extern "C" fn lapacke_ssyevd_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  w : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ssyevd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyevd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     double* a, lapack_int lda, double* w, double* work, 
///     lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(a, w, work, iwork)
pub extern "C" fn lapacke_dsyevd_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  w : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dsyevd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyevr_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, float* a, lapack_int lda, float vl, 
///     float vu, lapack_int il, lapack_int iu, float abstol, 
///     lapack_int* m, float* w, float* z, lapack_int ldz, 
///     lapack_int* isuppz, float* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(a, m, w, z, isuppz, work, iwork)
pub extern "C" fn lapacke_ssyevr_work(
  matrix_layout : Int,
  jobz : Byte,
  range : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  vl : Float,
  vu : Float,
  il : Lapack_int,
  iu : Lapack_int,
  abstol : Float,
  m : FixedArray[Int],
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  isuppz : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ssyevr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyevr_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, double* a, lapack_int lda, double vl, 
///     double vu, lapack_int il, lapack_int iu, double abstol, 
///     lapack_int* m, double* w, double* z, lapack_int ldz, 
///     lapack_int* isuppz, double* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(a, m, w, z, isuppz, work, iwork)
pub extern "C" fn lapacke_dsyevr_work(
  matrix_layout : Int,
  jobz : Byte,
  range : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  vl : Double,
  vu : Double,
  il : Lapack_int,
  iu : Lapack_int,
  abstol : Double,
  m : FixedArray[Int],
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  isuppz : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dsyevr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyevx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, float* a, lapack_int lda, float vl, 
///     float vu, lapack_int il, lapack_int iu, float abstol, 
///     lapack_int* m, float* w, float* z, lapack_int ldz, 
///     float* work, lapack_int lwork, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
#borrow(a, m, w, z, work, iwork, ifail)
pub extern "C" fn lapacke_ssyevx_work(
  matrix_layout : Int,
  jobz : Byte,
  range : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  vl : Float,
  vu : Float,
  il : Lapack_int,
  iu : Lapack_int,
  abstol : Float,
  m : FixedArray[Int],
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ssyevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyevx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, double* a, lapack_int lda, double vl, 
///     double vu, lapack_int il, lapack_int iu, double abstol, 
///     lapack_int* m, double* w, double* z, lapack_int ldz, 
///     double* work, lapack_int lwork, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
#borrow(a, m, w, z, work, iwork, ifail)
pub extern "C" fn lapacke_dsyevx_work(
  matrix_layout : Int,
  jobz : Byte,
  range : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  vl : Double,
  vu : Double,
  il : Lapack_int,
  iu : Lapack_int,
  abstol : Double,
  m : FixedArray[Int],
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dsyevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssygst_work(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     float* a, lapack_int lda, const float* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_ssygst_work(
  matrix_layout : Int,
  itype : Lapack_int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ssygst_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsygst_work(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     double* a, lapack_int lda, const double* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_dsygst_work(
  matrix_layout : Int,
  itype : Lapack_int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dsygst_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssygv_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, float* a, lapack_int lda, float* b, 
///     lapack_int ldb, float* w, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, w, work)
pub extern "C" fn lapacke_ssygv_work(
  matrix_layout : Int,
  itype : Lapack_int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  w : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ssygv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsygv_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, double* a, lapack_int lda, double* b, 
///     lapack_int ldb, double* w, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, w, work)
pub extern "C" fn lapacke_dsygv_work(
  matrix_layout : Int,
  itype : Lapack_int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  w : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dsygv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssygvd_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, float* a, lapack_int lda, float* b, 
///     lapack_int ldb, float* w, float* work, lapack_int lwork, 
///     lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(a, b, w, work, iwork)
pub extern "C" fn lapacke_ssygvd_work(
  matrix_layout : Int,
  itype : Lapack_int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  w : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ssygvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsygvd_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, double* a, lapack_int lda, double* b, 
///     lapack_int ldb, double* w, double* work, lapack_int lwork, 
///     lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(a, b, w, work, iwork)
pub extern "C" fn lapacke_dsygvd_work(
  matrix_layout : Int,
  itype : Lapack_int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  w : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dsygvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssygvx_work(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, float* a, lapack_int lda, 
///     float* b, lapack_int ldb, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, float* z, lapack_int ldz, float* work, 
///     lapack_int lwork, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
#borrow(a, b, m, w, z, work, iwork, ifail)
pub extern "C" fn lapacke_ssygvx_work(
  matrix_layout : Int,
  itype : Lapack_int,
  jobz : Byte,
  range : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  vl : Float,
  vu : Float,
  il : Lapack_int,
  iu : Lapack_int,
  abstol : Float,
  m : FixedArray[Int],
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ssygvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsygvx_work(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, double* a, lapack_int lda, 
///     double* b, lapack_int ldb, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, double* z, lapack_int ldz, double* work, 
///     lapack_int lwork, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
#borrow(a, b, m, w, z, work, iwork, ifail)
pub extern "C" fn lapacke_dsygvx_work(
  matrix_layout : Int,
  itype : Lapack_int,
  jobz : Byte,
  range : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  vl : Double,
  vu : Double,
  il : Lapack_int,
  iu : Lapack_int,
  abstol : Double,
  m : FixedArray[Int],
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dsygvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyrfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const float* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const float* b, lapack_int ldb, float* x, 
///     lapack_int ldx, float* ferr, float* berr, float* work, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, b, x, ferr, berr, work, iwork)
pub extern "C" fn lapacke_ssyrfs_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  af : FixedArray[Float],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ssyrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyrfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const double* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const double* b, lapack_int ldb, double* x, 
///     lapack_int ldx, double* ferr, double* berr, double* work, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, b, x, ferr, berr, work, iwork)
pub extern "C" fn lapacke_dsyrfs_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  af : FixedArray[Double],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dsyrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csyrfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* ferr, float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(a, af, ipiv, b, x, ferr, berr, work, rwork)
pub extern "C" fn lapacke_csyrfs_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  af : FixedArray[Float],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_csyrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsyrfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* ferr, double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(a, af, ipiv, b, x, ferr, berr, work, rwork)
pub extern "C" fn lapacke_zsyrfs_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  af : FixedArray[Double],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zsyrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyrfsx_work(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const float* a, lapack_int lda, const float* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const float* s, const float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* rcond, 
///     float* berr, lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, 
///     lapack_int nparams, float* params, float* work, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, s, b, x, rcond, berr, err_bnds_norm, err_bnds_comp, params, work, iwork)
pub extern "C" fn lapacke_ssyrfsx_work(
  matrix_layout : Int,
  uplo : Byte,
  equed : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  af : FixedArray[Float],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  s : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  rcond : FixedArray[Float],
  berr : FixedArray[Float],
  n_err_bnds : Lapack_int,
  err_bnds_norm : FixedArray[Float],
  err_bnds_comp : FixedArray[Float],
  nparams : Lapack_int,
  params : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ssyrfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyrfsx_work(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const double* a, lapack_int lda, const double* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const double* s, const double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* rcond, 
///     double* berr, lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, 
///     lapack_int nparams, double* params, double* work, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, s, b, x, rcond, berr, err_bnds_norm, err_bnds_comp, params, work, iwork)
pub extern "C" fn lapacke_dsyrfsx_work(
  matrix_layout : Int,
  uplo : Byte,
  equed : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  af : FixedArray[Double],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  s : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  rcond : FixedArray[Double],
  berr : FixedArray[Double],
  n_err_bnds : Lapack_int,
  err_bnds_norm : FixedArray[Double],
  err_bnds_comp : FixedArray[Double],
  nparams : Lapack_int,
  params : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dsyrfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csyrfsx_work(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const float* s, const lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond, 
///     float* berr, lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, 
///     lapack_int nparams, float* params, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(a, af, ipiv, s, b, x, rcond, berr, err_bnds_norm, err_bnds_comp, params, work, rwork)
pub extern "C" fn lapacke_csyrfsx_work(
  matrix_layout : Int,
  uplo : Byte,
  equed : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  af : FixedArray[Float],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  s : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  rcond : FixedArray[Float],
  berr : FixedArray[Float],
  n_err_bnds : Lapack_int,
  err_bnds_norm : FixedArray[Float],
  err_bnds_comp : FixedArray[Float],
  nparams : Lapack_int,
  params : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_csyrfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsyrfsx_work(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const double* s, const lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond, 
///     double* berr, lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, 
///     lapack_int nparams, double* params, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(a, af, ipiv, s, b, x, rcond, berr, err_bnds_norm, err_bnds_comp, params, work, rwork)
pub extern "C" fn lapacke_zsyrfsx_work(
  matrix_layout : Int,
  uplo : Byte,
  equed : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  af : FixedArray[Double],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  s : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  rcond : FixedArray[Double],
  berr : FixedArray[Double],
  n_err_bnds : Lapack_int,
  err_bnds_norm : FixedArray[Double],
  err_bnds_comp : FixedArray[Double],
  nparams : Lapack_int,
  params : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zsyrfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssysv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, lapack_int* ipiv, float* b, 
///     lapack_int ldb, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_ssysv_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ssysv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsysv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, lapack_int* ipiv, double* b, 
///     lapack_int ldb, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_dsysv_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dsysv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csysv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_int* ipiv, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_csysv_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_csysv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsysv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_int* ipiv, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_zsysv_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zsysv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssysvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const float* a, lapack_int lda, float* af, 
///     lapack_int ldaf, lapack_int* ipiv, const float* b, lapack_int ldb, 
///     float* x, lapack_int ldx, float* rcond, float* ferr, 
///     float* berr, float* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, b, x, rcond, ferr, berr, work, iwork)
pub extern "C" fn lapacke_ssysvx_work(
  matrix_layout : Int,
  fact : Byte,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  af : FixedArray[Float],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  rcond : FixedArray[Float],
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ssysvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsysvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const double* a, lapack_int lda, double* af, 
///     lapack_int ldaf, lapack_int* ipiv, const double* b, lapack_int ldb, 
///     double* x, lapack_int ldx, double* rcond, double* ferr, 
///     double* berr, double* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, b, x, rcond, ferr, berr, work, iwork)
pub extern "C" fn lapacke_dsysvx_work(
  matrix_layout : Int,
  fact : Byte,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  af : FixedArray[Double],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  rcond : FixedArray[Double],
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dsysvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csysvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, lapack_int* ipiv, const lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* x, lapack_int ldx, float* rcond, float* ferr, 
///     float* berr, lapack_complex_float* work, lapack_int lwork, float* rwork
/// );
/// ```
#borrow(a, af, ipiv, b, x, rcond, ferr, berr, work, rwork)
pub extern "C" fn lapacke_csysvx_work(
  matrix_layout : Int,
  fact : Byte,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  af : FixedArray[Float],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  rcond : FixedArray[Float],
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_csysvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsysvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, lapack_int* ipiv, const lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* x, lapack_int ldx, double* rcond, double* ferr, 
///     double* berr, lapack_complex_double* work, lapack_int lwork, double* rwork
/// );
/// ```
#borrow(a, af, ipiv, b, x, rcond, ferr, berr, work, rwork)
pub extern "C" fn lapacke_zsysvx_work(
  matrix_layout : Int,
  fact : Byte,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  af : FixedArray[Double],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  rcond : FixedArray[Double],
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zsysvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssysvxx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, float* a, lapack_int lda, float* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, float* s, 
///     float* b, lapack_int ldb, float* x, lapack_int ldx, 
///     float* rcond, float* rpvgrw, float* berr, lapack_int n_err_bnds, 
///     float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, float* params, 
///     float* work, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, equed, s, b, x, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, params, work, iwork)
pub extern "C" fn lapacke_ssysvxx_work(
  matrix_layout : Int,
  fact : Byte,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  af : FixedArray[Float],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  s : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  rcond : FixedArray[Float],
  rpvgrw : FixedArray[Float],
  berr : FixedArray[Float],
  n_err_bnds : Lapack_int,
  err_bnds_norm : FixedArray[Float],
  err_bnds_comp : FixedArray[Float],
  nparams : Lapack_int,
  params : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ssysvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsysvxx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, double* a, lapack_int lda, double* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, double* s, 
///     double* b, lapack_int ldb, double* x, lapack_int ldx, 
///     double* rcond, double* rpvgrw, double* berr, lapack_int n_err_bnds, 
///     double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, double* params, 
///     double* work, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, equed, s, b, x, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, params, work, iwork)
pub extern "C" fn lapacke_dsysvxx_work(
  matrix_layout : Int,
  fact : Byte,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  af : FixedArray[Double],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  s : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  rcond : FixedArray[Double],
  rpvgrw : FixedArray[Double],
  berr : FixedArray[Double],
  n_err_bnds : Lapack_int,
  err_bnds_norm : FixedArray[Double],
  err_bnds_comp : FixedArray[Double],
  nparams : Lapack_int,
  params : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dsysvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csysvxx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, float* s, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* rcond, float* rpvgrw, float* berr, lapack_int n_err_bnds, 
///     float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, float* params, 
///     lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(a, af, ipiv, equed, s, b, x, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, params, work, rwork)
pub extern "C" fn lapacke_csysvxx_work(
  matrix_layout : Int,
  fact : Byte,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  af : FixedArray[Float],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  s : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  rcond : FixedArray[Float],
  rpvgrw : FixedArray[Float],
  berr : FixedArray[Float],
  n_err_bnds : Lapack_int,
  err_bnds_norm : FixedArray[Float],
  err_bnds_comp : FixedArray[Float],
  nparams : Lapack_int,
  params : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_csysvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsysvxx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, double* s, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* rcond, double* rpvgrw, double* berr, lapack_int n_err_bnds, 
///     double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, double* params, 
///     lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(a, af, ipiv, equed, s, b, x, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, params, work, rwork)
pub extern "C" fn lapacke_zsysvxx_work(
  matrix_layout : Int,
  fact : Byte,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  af : FixedArray[Double],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  s : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  rcond : FixedArray[Double],
  rpvgrw : FixedArray[Double],
  berr : FixedArray[Double],
  n_err_bnds : Lapack_int,
  err_bnds_norm : FixedArray[Double],
  err_bnds_comp : FixedArray[Double],
  nparams : Lapack_int,
  params : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zsysvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrd_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, float* d, float* e, float* tau, 
///     float* work, lapack_int lwork
/// );
/// ```
#borrow(a, d, e, tau, work)
pub extern "C" fn lapacke_ssytrd_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  d : FixedArray[Float],
  e : FixedArray[Float],
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrd_work(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, double* d, double* e, double* tau, 
///     double* work, lapack_int lwork
/// );
/// ```
#borrow(a, d, e, tau, work)
pub extern "C" fn lapacke_dsytrd_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  d : FixedArray[Double],
  e : FixedArray[Double],
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dsytrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrf_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* ipiv, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_ssytrf_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrf_work(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* ipiv, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_dsytrf_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dsytrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_csytrf_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_csytrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_zsytrf_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zsytrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytri_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, const lapack_int* ipiv, float* work
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_ssytri_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ssytri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytri_work(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, const lapack_int* ipiv, double* work
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_dsytri_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dsytri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csytri_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_complex_float* work
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_csytri_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_csytri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytri_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_complex_double* work
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_zsytri_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zsytri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_ssytrs_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_dsytrs_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dsytrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_csytrs_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_csytrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zsytrs_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zsytrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stbcon_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, lapack_int kd, const float* ab, lapack_int ldab, 
///     float* rcond, float* work, lapack_int* iwork
/// );
/// ```
#borrow(ab, rcond, work, iwork)
pub extern "C" fn lapacke_stbcon_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  rcond : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_stbcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtbcon_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, lapack_int kd, const double* ab, lapack_int ldab, 
///     double* rcond, double* work, lapack_int* iwork
/// );
/// ```
#borrow(ab, rcond, work, iwork)
pub extern "C" fn lapacke_dtbcon_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  rcond : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dtbcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctbcon_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, lapack_int kd, const lapack_complex_float* ab, lapack_int ldab, 
///     float* rcond, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(ab, rcond, work, rwork)
pub extern "C" fn lapacke_ctbcon_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  rcond : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctbcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztbcon_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, lapack_int kd, const lapack_complex_double* ab, lapack_int ldab, 
///     double* rcond, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(ab, rcond, work, rwork)
pub extern "C" fn lapacke_ztbcon_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  rcond : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztbcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stbrfs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const float* ab, 
///     lapack_int ldab, const float* b, lapack_int ldb, const float* x, 
///     lapack_int ldx, float* ferr, float* berr, float* work, lapack_int* iwork
/// );
/// ```
#borrow(ab, b, x, ferr, berr, work, iwork)
pub extern "C" fn lapacke_stbrfs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_stbrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtbrfs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const double* ab, 
///     lapack_int ldab, const double* b, lapack_int ldb, const double* x, 
///     lapack_int ldx, double* ferr, double* berr, double* work, lapack_int* iwork
/// );
/// ```
#borrow(ab, b, x, ferr, berr, work, iwork)
pub extern "C" fn lapacke_dtbrfs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dtbrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctbrfs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const lapack_complex_float* ab, 
///     lapack_int ldab, const lapack_complex_float* b, lapack_int ldb, const lapack_complex_float* x, 
///     lapack_int ldx, float* ferr, float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(ab, b, x, ferr, berr, work, rwork)
pub extern "C" fn lapacke_ctbrfs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctbrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztbrfs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const lapack_complex_double* ab, 
///     lapack_int ldab, const lapack_complex_double* b, lapack_int ldb, const lapack_complex_double* x, 
///     lapack_int ldx, double* ferr, double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(ab, b, x, ferr, berr, work, rwork)
pub extern "C" fn lapacke_ztbrfs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztbrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stbtrs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const float* ab, 
///     lapack_int ldab, float* b, lapack_int ldb
/// );
/// ```
#borrow(ab, b)
pub extern "C" fn lapacke_stbtrs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_stbtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtbtrs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const double* ab, 
///     lapack_int ldab, double* b, lapack_int ldb
/// );
/// ```
#borrow(ab, b)
pub extern "C" fn lapacke_dtbtrs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dtbtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctbtrs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const lapack_complex_float* ab, 
///     lapack_int ldab, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(ab, b)
pub extern "C" fn lapacke_ctbtrs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ctbtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztbtrs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const lapack_complex_double* ab, 
///     lapack_int ldab, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(ab, b)
pub extern "C" fn lapacke_ztbtrs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ztbtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stfsm_work(
///     int matrix_layout, char transr, char side, char uplo, 
///     char trans, char diag, lapack_int m, lapack_int n, 
///     float alpha, const float* a, float* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_stfsm_work(
  matrix_layout : Int,
  transr : Byte,
  side : Byte,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  m : Lapack_int,
  n : Lapack_int,
  alpha : Float,
  a : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_stfsm_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtfsm_work(
///     int matrix_layout, char transr, char side, char uplo, 
///     char trans, char diag, lapack_int m, lapack_int n, 
///     double alpha, const double* a, double* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_dtfsm_work(
  matrix_layout : Int,
  transr : Byte,
  side : Byte,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  m : Lapack_int,
  n : Lapack_int,
  alpha : Double,
  a : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dtfsm_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctfsm_work(
///     int matrix_layout, char transr, char side, char uplo, 
///     char trans, char diag, lapack_int m, lapack_int n, 
///     lapack_complex_float alpha, const lapack_complex_float* a, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_ctfsm_work(
  matrix_layout : Int,
  transr : Byte,
  side : Byte,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  m : Lapack_int,
  n : Lapack_int,
  alpha : Lapack_complex_float,
  a : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ctfsm_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztfsm_work(
///     int matrix_layout, char transr, char side, char uplo, 
///     char trans, char diag, lapack_int m, lapack_int n, 
///     lapack_complex_double alpha, const lapack_complex_double* a, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_ztfsm_work(
  matrix_layout : Int,
  transr : Byte,
  side : Byte,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  m : Lapack_int,
  n : Lapack_int,
  alpha : Lapack_complex_double,
  a : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ztfsm_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stftri_work(
///     int matrix_layout, char transr, char uplo, char diag, 
///     lapack_int n, float* a
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_stftri_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
) -> Lapack_int = "LAPACKE_stftri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtftri_work(
///     int matrix_layout, char transr, char uplo, char diag, 
///     lapack_int n, double* a
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_dtftri_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dtftri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctftri_work(
///     int matrix_layout, char transr, char uplo, char diag, 
///     lapack_int n, lapack_complex_float* a
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_ctftri_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctftri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztftri_work(
///     int matrix_layout, char transr, char uplo, char diag, 
///     lapack_int n, lapack_complex_double* a
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_ztftri_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztftri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stfttp_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const float* arf, float* ap
/// );
/// ```
#borrow(arf, ap)
pub extern "C" fn lapacke_stfttp_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  arf : FixedArray[Float],
  ap : FixedArray[Float],
) -> Lapack_int = "LAPACKE_stfttp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtfttp_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const double* arf, double* ap
/// );
/// ```
#borrow(arf, ap)
pub extern "C" fn lapacke_dtfttp_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  arf : FixedArray[Double],
  ap : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dtfttp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctfttp_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_float* arf, lapack_complex_float* ap
/// );
/// ```
#borrow(arf, ap)
pub extern "C" fn lapacke_ctfttp_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  arf : FixedArray[Float],
  ap : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctfttp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztfttp_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_double* arf, lapack_complex_double* ap
/// );
/// ```
#borrow(arf, ap)
pub extern "C" fn lapacke_ztfttp_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  arf : FixedArray[Double],
  ap : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztfttp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stfttr_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const float* arf, float* a, lapack_int lda
/// );
/// ```
#borrow(arf, a)
pub extern "C" fn lapacke_stfttr_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  arf : FixedArray[Float],
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_stfttr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtfttr_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const double* arf, double* a, lapack_int lda
/// );
/// ```
#borrow(arf, a)
pub extern "C" fn lapacke_dtfttr_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  arf : FixedArray[Double],
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_dtfttr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctfttr_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_float* arf, lapack_complex_float* a, lapack_int lda
/// );
/// ```
#borrow(arf, a)
pub extern "C" fn lapacke_ctfttr_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  arf : FixedArray[Float],
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_ctfttr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztfttr_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_double* arf, lapack_complex_double* a, lapack_int lda
/// );
/// ```
#borrow(arf, a)
pub extern "C" fn lapacke_ztfttr_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  arf : FixedArray[Double],
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_ztfttr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stgevc_work(
///     int matrix_layout, char side, char howmny, const lapack_logical* select, 
///     lapack_int n, const float* s, lapack_int lds, const float* p, 
///     lapack_int ldp, float* vl, lapack_int ldvl, float* vr, 
///     lapack_int ldvr, lapack_int mm, lapack_int* m, float* work
/// );
/// ```
#borrow(select, s, p, vl, vr, m, work)
pub extern "C" fn lapacke_stgevc_work(
  matrix_layout : Int,
  side : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  s : FixedArray[Float],
  lds : Lapack_int,
  p : FixedArray[Float],
  ldp : Lapack_int,
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_stgevc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtgevc_work(
///     int matrix_layout, char side, char howmny, const lapack_logical* select, 
///     lapack_int n, const double* s, lapack_int lds, const double* p, 
///     lapack_int ldp, double* vl, lapack_int ldvl, double* vr, 
///     lapack_int ldvr, lapack_int mm, lapack_int* m, double* work
/// );
/// ```
#borrow(select, s, p, vl, vr, m, work)
pub extern "C" fn lapacke_dtgevc_work(
  matrix_layout : Int,
  side : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  s : FixedArray[Double],
  lds : Lapack_int,
  p : FixedArray[Double],
  ldp : Lapack_int,
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dtgevc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctgevc_work(
///     int matrix_layout, char side, char howmny, const lapack_logical* select, 
///     lapack_int n, const lapack_complex_float* s, lapack_int lds, const lapack_complex_float* p, 
///     lapack_int ldp, lapack_complex_float* vl, lapack_int ldvl, lapack_complex_float* vr, 
///     lapack_int ldvr, lapack_int mm, lapack_int* m, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(select, s, p, vl, vr, m, work, rwork)
pub extern "C" fn lapacke_ctgevc_work(
  matrix_layout : Int,
  side : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  s : FixedArray[Float],
  lds : Lapack_int,
  p : FixedArray[Float],
  ldp : Lapack_int,
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctgevc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztgevc_work(
///     int matrix_layout, char side, char howmny, const lapack_logical* select, 
///     lapack_int n, const lapack_complex_double* s, lapack_int lds, const lapack_complex_double* p, 
///     lapack_int ldp, lapack_complex_double* vl, lapack_int ldvl, lapack_complex_double* vr, 
///     lapack_int ldvr, lapack_int mm, lapack_int* m, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(select, s, p, vl, vr, m, work, rwork)
pub extern "C" fn lapacke_ztgevc_work(
  matrix_layout : Int,
  side : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  s : FixedArray[Double],
  lds : Lapack_int,
  p : FixedArray[Double],
  ldp : Lapack_int,
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztgevc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stgexc_work(
///     int matrix_layout, lapack_logical wantq, lapack_logical wantz, lapack_int n, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* q, lapack_int ldq, float* z, lapack_int ldz, 
///     lapack_int* ifst, lapack_int* ilst, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, q, z, ifst, ilst, work)
pub extern "C" fn lapacke_stgexc_work(
  matrix_layout : Int,
  wantq : Lapack_logical,
  wantz : Lapack_logical,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  z : FixedArray[Float],
  ldz : Lapack_int,
  ifst : FixedArray[Int],
  ilst : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_stgexc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtgexc_work(
///     int matrix_layout, lapack_logical wantq, lapack_logical wantz, lapack_int n, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* q, lapack_int ldq, double* z, lapack_int ldz, 
///     lapack_int* ifst, lapack_int* ilst, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, q, z, ifst, ilst, work)
pub extern "C" fn lapacke_dtgexc_work(
  matrix_layout : Int,
  wantq : Lapack_logical,
  wantz : Lapack_logical,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  z : FixedArray[Double],
  ldz : Lapack_int,
  ifst : FixedArray[Int],
  ilst : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dtgexc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctgexc_work(
///     int matrix_layout, lapack_logical wantq, lapack_logical wantz, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* q, lapack_int ldq, lapack_complex_float* z, lapack_int ldz, 
///     lapack_int ifst, lapack_int ilst
/// );
/// ```
#borrow(a, b, q, z)
pub extern "C" fn lapacke_ctgexc_work(
  matrix_layout : Int,
  wantq : Lapack_logical,
  wantz : Lapack_logical,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  z : FixedArray[Float],
  ldz : Lapack_int,
  ifst : Lapack_int,
  ilst : Lapack_int,
) -> Lapack_int = "LAPACKE_ctgexc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztgexc_work(
///     int matrix_layout, lapack_logical wantq, lapack_logical wantz, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* q, lapack_int ldq, lapack_complex_double* z, lapack_int ldz, 
///     lapack_int ifst, lapack_int ilst
/// );
/// ```
#borrow(a, b, q, z)
pub extern "C" fn lapacke_ztgexc_work(
  matrix_layout : Int,
  wantq : Lapack_logical,
  wantz : Lapack_logical,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  z : FixedArray[Double],
  ldz : Lapack_int,
  ifst : Lapack_int,
  ilst : Lapack_int,
) -> Lapack_int = "LAPACKE_ztgexc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stgsen_work(
///     int matrix_layout, lapack_int ijob, lapack_logical wantq, lapack_logical wantz, 
///     const lapack_logical* select, lapack_int n, float* a, lapack_int lda, 
///     float* b, lapack_int ldb, float* alphar, float* alphai, 
///     float* beta, float* q, lapack_int ldq, float* z, 
///     lapack_int ldz, lapack_int* m, float* pl, float* pr, 
///     float* dif, float* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(select, a, b, alphar, alphai, beta, q, z, m, pl, pr, dif, work, iwork)
pub extern "C" fn lapacke_stgsen_work(
  matrix_layout : Int,
  ijob : Lapack_int,
  wantq : Lapack_logical,
  wantz : Lapack_logical,
  select : Lapack_logical,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  alphar : FixedArray[Float],
  alphai : FixedArray[Float],
  beta : FixedArray[Float],
  q : FixedArray[Float],
  ldq : Lapack_int,
  z : FixedArray[Float],
  ldz : Lapack_int,
  m : FixedArray[Int],
  pl : FixedArray[Float],
  pr : FixedArray[Float],
  dif : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_stgsen_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtgsen_work(
///     int matrix_layout, lapack_int ijob, lapack_logical wantq, lapack_logical wantz, 
///     const lapack_logical* select, lapack_int n, double* a, lapack_int lda, 
///     double* b, lapack_int ldb, double* alphar, double* alphai, 
///     double* beta, double* q, lapack_int ldq, double* z, 
///     lapack_int ldz, lapack_int* m, double* pl, double* pr, 
///     double* dif, double* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(select, a, b, alphar, alphai, beta, q, z, m, pl, pr, dif, work, iwork)
pub extern "C" fn lapacke_dtgsen_work(
  matrix_layout : Int,
  ijob : Lapack_int,
  wantq : Lapack_logical,
  wantz : Lapack_logical,
  select : Lapack_logical,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  alphar : FixedArray[Double],
  alphai : FixedArray[Double],
  beta : FixedArray[Double],
  q : FixedArray[Double],
  ldq : Lapack_int,
  z : FixedArray[Double],
  ldz : Lapack_int,
  m : FixedArray[Int],
  pl : FixedArray[Double],
  pr : FixedArray[Double],
  dif : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dtgsen_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctgsen_work(
///     int matrix_layout, lapack_int ijob, lapack_logical wantq, lapack_logical wantz, 
///     const lapack_logical* select, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* alpha, lapack_complex_float* beta, 
///     lapack_complex_float* q, lapack_int ldq, lapack_complex_float* z, lapack_int ldz, 
///     lapack_int* m, float* pl, float* pr, float* dif, 
///     lapack_complex_float* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(select, a, b, alpha, beta, q, z, m, pl, pr, dif, work, iwork)
pub extern "C" fn lapacke_ctgsen_work(
  matrix_layout : Int,
  ijob : Lapack_int,
  wantq : Lapack_logical,
  wantz : Lapack_logical,
  select : Lapack_logical,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  alpha : FixedArray[Float],
  beta : FixedArray[Float],
  q : FixedArray[Float],
  ldq : Lapack_int,
  z : FixedArray[Float],
  ldz : Lapack_int,
  m : FixedArray[Int],
  pl : FixedArray[Float],
  pr : FixedArray[Float],
  dif : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ctgsen_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztgsen_work(
///     int matrix_layout, lapack_int ijob, lapack_logical wantq, lapack_logical wantz, 
///     const lapack_logical* select, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* alpha, lapack_complex_double* beta, 
///     lapack_complex_double* q, lapack_int ldq, lapack_complex_double* z, lapack_int ldz, 
///     lapack_int* m, double* pl, double* pr, double* dif, 
///     lapack_complex_double* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(select, a, b, alpha, beta, q, z, m, pl, pr, dif, work, iwork)
pub extern "C" fn lapacke_ztgsen_work(
  matrix_layout : Int,
  ijob : Lapack_int,
  wantq : Lapack_logical,
  wantz : Lapack_logical,
  select : Lapack_logical,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  alpha : FixedArray[Double],
  beta : FixedArray[Double],
  q : FixedArray[Double],
  ldq : Lapack_int,
  z : FixedArray[Double],
  ldz : Lapack_int,
  m : FixedArray[Int],
  pl : FixedArray[Double],
  pr : FixedArray[Double],
  dif : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ztgsen_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stgsja_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_int k, 
///     lapack_int l, float* a, lapack_int lda, float* b, 
///     lapack_int ldb, float tola, float tolb, float* alpha, 
///     float* beta, float* u, lapack_int ldu, float* v, 
///     lapack_int ldv, float* q, lapack_int ldq, float* work, lapack_int* ncycle
/// );
/// ```
#borrow(a, b, alpha, beta, u, v, q, work, ncycle)
pub extern "C" fn lapacke_stgsja_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  p : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  tola : Float,
  tolb : Float,
  alpha : FixedArray[Float],
  beta : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  work : FixedArray[Float],
  ncycle : FixedArray[Int],
) -> Lapack_int = "LAPACKE_stgsja_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtgsja_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_int k, 
///     lapack_int l, double* a, lapack_int lda, double* b, 
///     lapack_int ldb, double tola, double tolb, double* alpha, 
///     double* beta, double* u, lapack_int ldu, double* v, 
///     lapack_int ldv, double* q, lapack_int ldq, double* work, lapack_int* ncycle
/// );
/// ```
#borrow(a, b, alpha, beta, u, v, q, work, ncycle)
pub extern "C" fn lapacke_dtgsja_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  p : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  tola : Double,
  tolb : Double,
  alpha : FixedArray[Double],
  beta : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  work : FixedArray[Double],
  ncycle : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dtgsja_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctgsja_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_int k, 
///     lapack_int l, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, 
///     lapack_int ldb, float tola, float tolb, float* alpha, 
///     float* beta, lapack_complex_float* u, lapack_int ldu, lapack_complex_float* v, 
///     lapack_int ldv, lapack_complex_float* q, lapack_int ldq, lapack_complex_float* work, lapack_int* ncycle
/// );
/// ```
#borrow(a, b, alpha, beta, u, v, q, work, ncycle)
pub extern "C" fn lapacke_ctgsja_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  p : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  tola : Float,
  tolb : Float,
  alpha : FixedArray[Float],
  beta : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  work : FixedArray[Float],
  ncycle : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ctgsja_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztgsja_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_int k, 
///     lapack_int l, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, 
///     lapack_int ldb, double tola, double tolb, double* alpha, 
///     double* beta, lapack_complex_double* u, lapack_int ldu, lapack_complex_double* v, 
///     lapack_int ldv, lapack_complex_double* q, lapack_int ldq, lapack_complex_double* work, lapack_int* ncycle
/// );
/// ```
#borrow(a, b, alpha, beta, u, v, q, work, ncycle)
pub extern "C" fn lapacke_ztgsja_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  p : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  tola : Double,
  tolb : Double,
  alpha : FixedArray[Double],
  beta : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  work : FixedArray[Double],
  ncycle : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ztgsja_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stgsna_work(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const float* a, lapack_int lda, const float* b, 
///     lapack_int ldb, const float* vl, lapack_int ldvl, const float* vr, 
///     lapack_int ldvr, float* s, float* dif, lapack_int mm, 
///     lapack_int* m, float* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(select, a, b, vl, vr, s, dif, m, work, iwork)
pub extern "C" fn lapacke_stgsna_work(
  matrix_layout : Int,
  job : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  s : FixedArray[Float],
  dif : FixedArray[Float],
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_stgsna_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtgsna_work(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const double* a, lapack_int lda, const double* b, 
///     lapack_int ldb, const double* vl, lapack_int ldvl, const double* vr, 
///     lapack_int ldvr, double* s, double* dif, lapack_int mm, 
///     lapack_int* m, double* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(select, a, b, vl, vr, s, dif, m, work, iwork)
pub extern "C" fn lapacke_dtgsna_work(
  matrix_layout : Int,
  job : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  s : FixedArray[Double],
  dif : FixedArray[Double],
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dtgsna_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctgsna_work(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* b, 
///     lapack_int ldb, const lapack_complex_float* vl, lapack_int ldvl, const lapack_complex_float* vr, 
///     lapack_int ldvr, float* s, float* dif, lapack_int mm, 
///     lapack_int* m, lapack_complex_float* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(select, a, b, vl, vr, s, dif, m, work, iwork)
pub extern "C" fn lapacke_ctgsna_work(
  matrix_layout : Int,
  job : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  s : FixedArray[Float],
  dif : FixedArray[Float],
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ctgsna_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztgsna_work(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* b, 
///     lapack_int ldb, const lapack_complex_double* vl, lapack_int ldvl, const lapack_complex_double* vr, 
///     lapack_int ldvr, double* s, double* dif, lapack_int mm, 
///     lapack_int* m, lapack_complex_double* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(select, a, b, vl, vr, s, dif, m, work, iwork)
pub extern "C" fn lapacke_ztgsna_work(
  matrix_layout : Int,
  job : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  s : FixedArray[Double],
  dif : FixedArray[Double],
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ztgsna_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stgsyl_work(
///     int matrix_layout, char trans, lapack_int ijob, lapack_int m, 
///     lapack_int n, const float* a, lapack_int lda, const float* b, 
///     lapack_int ldb, float* c, lapack_int ldc, const float* d, 
///     lapack_int ldd, const float* e, lapack_int lde, float* f, 
///     lapack_int ldf, float* scale, float* dif, float* work, 
///     lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, b, c, d, e, f, scale, dif, work, iwork)
pub extern "C" fn lapacke_stgsyl_work(
  matrix_layout : Int,
  trans : Byte,
  ijob : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
  d : FixedArray[Float],
  ldd : Lapack_int,
  e : FixedArray[Float],
  lde : Lapack_int,
  f : FixedArray[Float],
  ldf : Lapack_int,
  scale : FixedArray[Float],
  dif : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_stgsyl_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtgsyl_work(
///     int matrix_layout, char trans, lapack_int ijob, lapack_int m, 
///     lapack_int n, const double* a, lapack_int lda, const double* b, 
///     lapack_int ldb, double* c, lapack_int ldc, const double* d, 
///     lapack_int ldd, const double* e, lapack_int lde, double* f, 
///     lapack_int ldf, double* scale, double* dif, double* work, 
///     lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, b, c, d, e, f, scale, dif, work, iwork)
pub extern "C" fn lapacke_dtgsyl_work(
  matrix_layout : Int,
  trans : Byte,
  ijob : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
  d : FixedArray[Double],
  ldd : Lapack_int,
  e : FixedArray[Double],
  lde : Lapack_int,
  f : FixedArray[Double],
  ldf : Lapack_int,
  scale : FixedArray[Double],
  dif : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dtgsyl_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctgsyl_work(
///     int matrix_layout, char trans, lapack_int ijob, lapack_int m, 
///     lapack_int n, const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* c, lapack_int ldc, const lapack_complex_float* d, 
///     lapack_int ldd, const lapack_complex_float* e, lapack_int lde, lapack_complex_float* f, 
///     lapack_int ldf, float* scale, float* dif, lapack_complex_float* work, 
///     lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, b, c, d, e, f, scale, dif, work, iwork)
pub extern "C" fn lapacke_ctgsyl_work(
  matrix_layout : Int,
  trans : Byte,
  ijob : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
  d : FixedArray[Float],
  ldd : Lapack_int,
  e : FixedArray[Float],
  lde : Lapack_int,
  f : FixedArray[Float],
  ldf : Lapack_int,
  scale : FixedArray[Float],
  dif : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ctgsyl_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztgsyl_work(
///     int matrix_layout, char trans, lapack_int ijob, lapack_int m, 
///     lapack_int n, const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* c, lapack_int ldc, const lapack_complex_double* d, 
///     lapack_int ldd, const lapack_complex_double* e, lapack_int lde, lapack_complex_double* f, 
///     lapack_int ldf, double* scale, double* dif, lapack_complex_double* work, 
///     lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, b, c, d, e, f, scale, dif, work, iwork)
pub extern "C" fn lapacke_ztgsyl_work(
  matrix_layout : Int,
  trans : Byte,
  ijob : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
  d : FixedArray[Double],
  ldd : Lapack_int,
  e : FixedArray[Double],
  lde : Lapack_int,
  f : FixedArray[Double],
  ldf : Lapack_int,
  scale : FixedArray[Double],
  dif : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ztgsyl_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stpcon_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const float* ap, float* rcond, float* work, lapack_int* iwork
/// );
/// ```
#borrow(ap, rcond, work, iwork)
pub extern "C" fn lapacke_stpcon_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  rcond : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_stpcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtpcon_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const double* ap, double* rcond, double* work, lapack_int* iwork
/// );
/// ```
#borrow(ap, rcond, work, iwork)
pub extern "C" fn lapacke_dtpcon_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  rcond : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dtpcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctpcon_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const lapack_complex_float* ap, float* rcond, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(ap, rcond, work, rwork)
pub extern "C" fn lapacke_ctpcon_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  rcond : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctpcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztpcon_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const lapack_complex_double* ap, double* rcond, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(ap, rcond, work, rwork)
pub extern "C" fn lapacke_ztpcon_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  rcond : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztpcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stprfs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const float* ap, const float* b, 
///     lapack_int ldb, const float* x, lapack_int ldx, float* ferr, 
///     float* berr, float* work, lapack_int* iwork
/// );
/// ```
#borrow(ap, b, x, ferr, berr, work, iwork)
pub extern "C" fn lapacke_stprfs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_stprfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtprfs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const double* ap, const double* b, 
///     lapack_int ldb, const double* x, lapack_int ldx, double* ferr, 
///     double* berr, double* work, lapack_int* iwork
/// );
/// ```
#borrow(ap, b, x, ferr, berr, work, iwork)
pub extern "C" fn lapacke_dtprfs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dtprfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctprfs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_float* ap, const lapack_complex_float* b, 
///     lapack_int ldb, const lapack_complex_float* x, lapack_int ldx, float* ferr, 
///     float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(ap, b, x, ferr, berr, work, rwork)
pub extern "C" fn lapacke_ctprfs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctprfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztprfs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_double* ap, const lapack_complex_double* b, 
///     lapack_int ldb, const lapack_complex_double* x, lapack_int ldx, double* ferr, 
///     double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(ap, b, x, ferr, berr, work, rwork)
pub extern "C" fn lapacke_ztprfs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztprfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stptri_work(
///     int matrix_layout, char uplo, char diag, lapack_int n, float* ap
/// );
/// ```
#borrow(ap)
pub extern "C" fn lapacke_stptri_work(
  matrix_layout : Int,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
) -> Lapack_int = "LAPACKE_stptri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtptri_work(
///     int matrix_layout, char uplo, char diag, lapack_int n, double* ap
/// );
/// ```
#borrow(ap)
pub extern "C" fn lapacke_dtptri_work(
  matrix_layout : Int,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dtptri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctptri_work(
///     int matrix_layout, char uplo, char diag, lapack_int n, lapack_complex_float* ap
/// );
/// ```
#borrow(ap)
pub extern "C" fn lapacke_ctptri_work(
  matrix_layout : Int,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctptri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztptri_work(
///     int matrix_layout, char uplo, char diag, lapack_int n, lapack_complex_double* ap
/// );
/// ```
#borrow(ap)
pub extern "C" fn lapacke_ztptri_work(
  matrix_layout : Int,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztptri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stptrs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const float* ap, float* b, lapack_int ldb
/// );
/// ```
#borrow(ap, b)
pub extern "C" fn lapacke_stptrs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_stptrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtptrs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const double* ap, double* b, lapack_int ldb
/// );
/// ```
#borrow(ap, b)
pub extern "C" fn lapacke_dtptrs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dtptrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctptrs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_float* ap, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(ap, b)
pub extern "C" fn lapacke_ctptrs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ctptrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztptrs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_double* ap, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(ap, b)
pub extern "C" fn lapacke_ztptrs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ztptrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stpttf_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const float* ap, float* arf
/// );
/// ```
#borrow(ap, arf)
pub extern "C" fn lapacke_stpttf_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  arf : FixedArray[Float],
) -> Lapack_int = "LAPACKE_stpttf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtpttf_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const double* ap, double* arf
/// );
/// ```
#borrow(ap, arf)
pub extern "C" fn lapacke_dtpttf_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  arf : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dtpttf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctpttf_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_float* ap, lapack_complex_float* arf
/// );
/// ```
#borrow(ap, arf)
pub extern "C" fn lapacke_ctpttf_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  arf : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctpttf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztpttf_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_double* ap, lapack_complex_double* arf
/// );
/// ```
#borrow(ap, arf)
pub extern "C" fn lapacke_ztpttf_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  arf : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztpttf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stpttr_work(
///     int matrix_layout, char uplo, lapack_int n, const float* ap, 
///     float* a, lapack_int lda
/// );
/// ```
#borrow(ap, a)
pub extern "C" fn lapacke_stpttr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_stpttr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtpttr_work(
///     int matrix_layout, char uplo, lapack_int n, const double* ap, 
///     double* a, lapack_int lda
/// );
/// ```
#borrow(ap, a)
pub extern "C" fn lapacke_dtpttr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_dtpttr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctpttr_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* ap, 
///     lapack_complex_float* a, lapack_int lda
/// );
/// ```
#borrow(ap, a)
pub extern "C" fn lapacke_ctpttr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_ctpttr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztpttr_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* ap, 
///     lapack_complex_double* a, lapack_int lda
/// );
/// ```
#borrow(ap, a)
pub extern "C" fn lapacke_ztpttr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_ztpttr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_strcon_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const float* a, lapack_int lda, float* rcond, 
///     float* work, lapack_int* iwork
/// );
/// ```
#borrow(a, rcond, work, iwork)
pub extern "C" fn lapacke_strcon_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  rcond : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_strcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrcon_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const double* a, lapack_int lda, double* rcond, 
///     double* work, lapack_int* iwork
/// );
/// ```
#borrow(a, rcond, work, iwork)
pub extern "C" fn lapacke_dtrcon_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  rcond : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dtrcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrcon_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const lapack_complex_float* a, lapack_int lda, float* rcond, 
///     lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(a, rcond, work, rwork)
pub extern "C" fn lapacke_ctrcon_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  rcond : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctrcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrcon_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const lapack_complex_double* a, lapack_int lda, double* rcond, 
///     lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(a, rcond, work, rwork)
pub extern "C" fn lapacke_ztrcon_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  rcond : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztrcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_strevc_work(
///     int matrix_layout, char side, char howmny, lapack_logical* select, 
///     lapack_int n, const float* t, lapack_int ldt, float* vl, 
///     lapack_int ldvl, float* vr, lapack_int ldvr, lapack_int mm, 
///     lapack_int* m, float* work
/// );
/// ```
#borrow(select, t, vl, vr, m, work)
pub extern "C" fn lapacke_strevc_work(
  matrix_layout : Int,
  side : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_strevc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrevc_work(
///     int matrix_layout, char side, char howmny, lapack_logical* select, 
///     lapack_int n, const double* t, lapack_int ldt, double* vl, 
///     lapack_int ldvl, double* vr, lapack_int ldvr, lapack_int mm, 
///     lapack_int* m, double* work
/// );
/// ```
#borrow(select, t, vl, vr, m, work)
pub extern "C" fn lapacke_dtrevc_work(
  matrix_layout : Int,
  side : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dtrevc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrevc_work(
///     int matrix_layout, char side, char howmny, const lapack_logical* select, 
///     lapack_int n, lapack_complex_float* t, lapack_int ldt, lapack_complex_float* vl, 
///     lapack_int ldvl, lapack_complex_float* vr, lapack_int ldvr, lapack_int mm, 
///     lapack_int* m, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(select, t, vl, vr, m, work, rwork)
pub extern "C" fn lapacke_ctrevc_work(
  matrix_layout : Int,
  side : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctrevc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrevc_work(
///     int matrix_layout, char side, char howmny, const lapack_logical* select, 
///     lapack_int n, lapack_complex_double* t, lapack_int ldt, lapack_complex_double* vl, 
///     lapack_int ldvl, lapack_complex_double* vr, lapack_int ldvr, lapack_int mm, 
///     lapack_int* m, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(select, t, vl, vr, m, work, rwork)
pub extern "C" fn lapacke_ztrevc_work(
  matrix_layout : Int,
  side : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztrevc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_strexc_work(
///     int matrix_layout, char compq, lapack_int n, float* t, 
///     lapack_int ldt, float* q, lapack_int ldq, lapack_int* ifst, 
///     lapack_int* ilst, float* work
/// );
/// ```
#borrow(t, q, ifst, ilst, work)
pub extern "C" fn lapacke_strexc_work(
  matrix_layout : Int,
  compq : Byte,
  n : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  ifst : FixedArray[Int],
  ilst : FixedArray[Int],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_strexc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrexc_work(
///     int matrix_layout, char compq, lapack_int n, double* t, 
///     lapack_int ldt, double* q, lapack_int ldq, lapack_int* ifst, 
///     lapack_int* ilst, double* work
/// );
/// ```
#borrow(t, q, ifst, ilst, work)
pub extern "C" fn lapacke_dtrexc_work(
  matrix_layout : Int,
  compq : Byte,
  n : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  ifst : FixedArray[Int],
  ilst : FixedArray[Int],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dtrexc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrexc_work(
///     int matrix_layout, char compq, lapack_int n, lapack_complex_float* t, 
///     lapack_int ldt, lapack_complex_float* q, lapack_int ldq, lapack_int ifst, lapack_int ilst
/// );
/// ```
#borrow(t, q)
pub extern "C" fn lapacke_ctrexc_work(
  matrix_layout : Int,
  compq : Byte,
  n : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  ifst : Lapack_int,
  ilst : Lapack_int,
) -> Lapack_int = "LAPACKE_ctrexc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrexc_work(
///     int matrix_layout, char compq, lapack_int n, lapack_complex_double* t, 
///     lapack_int ldt, lapack_complex_double* q, lapack_int ldq, lapack_int ifst, lapack_int ilst
/// );
/// ```
#borrow(t, q)
pub extern "C" fn lapacke_ztrexc_work(
  matrix_layout : Int,
  compq : Byte,
  n : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  ifst : Lapack_int,
  ilst : Lapack_int,
) -> Lapack_int = "LAPACKE_ztrexc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_strrfs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const float* a, lapack_int lda, 
///     const float* b, lapack_int ldb, const float* x, lapack_int ldx, 
///     float* ferr, float* berr, float* work, lapack_int* iwork
/// );
/// ```
#borrow(a, b, x, ferr, berr, work, iwork)
pub extern "C" fn lapacke_strrfs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_strrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrrfs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const double* a, lapack_int lda, 
///     const double* b, lapack_int ldb, const double* x, lapack_int ldx, 
///     double* ferr, double* berr, double* work, lapack_int* iwork
/// );
/// ```
#borrow(a, b, x, ferr, berr, work, iwork)
pub extern "C" fn lapacke_dtrrfs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dtrrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrrfs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, 
///     const lapack_complex_float* b, lapack_int ldb, const lapack_complex_float* x, lapack_int ldx, 
///     float* ferr, float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(a, b, x, ferr, berr, work, rwork)
pub extern "C" fn lapacke_ctrrfs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctrrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrrfs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* b, lapack_int ldb, const lapack_complex_double* x, lapack_int ldx, 
///     double* ferr, double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(a, b, x, ferr, berr, work, rwork)
pub extern "C" fn lapacke_ztrrfs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztrrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_strsen_work(
///     int matrix_layout, char job, char compq, const lapack_logical* select, 
///     lapack_int n, float* t, lapack_int ldt, float* q, 
///     lapack_int ldq, float* wr, float* wi, lapack_int* m, 
///     float* s, float* sep, float* work, lapack_int lwork, 
///     lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(select, t, q, wr, wi, m, s, sep, work, iwork)
pub extern "C" fn lapacke_strsen_work(
  matrix_layout : Int,
  job : Byte,
  compq : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  wr : FixedArray[Float],
  wi : FixedArray[Float],
  m : FixedArray[Int],
  s : FixedArray[Float],
  sep : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_strsen_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrsen_work(
///     int matrix_layout, char job, char compq, const lapack_logical* select, 
///     lapack_int n, double* t, lapack_int ldt, double* q, 
///     lapack_int ldq, double* wr, double* wi, lapack_int* m, 
///     double* s, double* sep, double* work, lapack_int lwork, 
///     lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(select, t, q, wr, wi, m, s, sep, work, iwork)
pub extern "C" fn lapacke_dtrsen_work(
  matrix_layout : Int,
  job : Byte,
  compq : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  wr : FixedArray[Double],
  wi : FixedArray[Double],
  m : FixedArray[Int],
  s : FixedArray[Double],
  sep : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dtrsen_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrsen_work(
///     int matrix_layout, char job, char compq, const lapack_logical* select, 
///     lapack_int n, lapack_complex_float* t, lapack_int ldt, lapack_complex_float* q, 
///     lapack_int ldq, lapack_complex_float* w, lapack_int* m, float* s, 
///     float* sep, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(select, t, q, w, m, s, sep, work)
pub extern "C" fn lapacke_ctrsen_work(
  matrix_layout : Int,
  job : Byte,
  compq : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  w : FixedArray[Float],
  m : FixedArray[Int],
  s : FixedArray[Float],
  sep : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ctrsen_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrsen_work(
///     int matrix_layout, char job, char compq, const lapack_logical* select, 
///     lapack_int n, lapack_complex_double* t, lapack_int ldt, lapack_complex_double* q, 
///     lapack_int ldq, lapack_complex_double* w, lapack_int* m, double* s, 
///     double* sep, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(select, t, q, w, m, s, sep, work)
pub extern "C" fn lapacke_ztrsen_work(
  matrix_layout : Int,
  job : Byte,
  compq : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  w : FixedArray[Double],
  m : FixedArray[Int],
  s : FixedArray[Double],
  sep : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ztrsen_work"

///|
///
/// ```c
/// lapack_int LAPACKE_strsna_work(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const float* t, lapack_int ldt, const float* vl, 
///     lapack_int ldvl, const float* vr, lapack_int ldvr, float* s, 
///     float* sep, lapack_int mm, lapack_int* m, float* work, 
///     lapack_int ldwork, lapack_int* iwork
/// );
/// ```
#borrow(select, t, vl, vr, s, sep, m, work, iwork)
pub extern "C" fn lapacke_strsna_work(
  matrix_layout : Int,
  job : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  s : FixedArray[Float],
  sep : FixedArray[Float],
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Float],
  ldwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_strsna_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrsna_work(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const double* t, lapack_int ldt, const double* vl, 
///     lapack_int ldvl, const double* vr, lapack_int ldvr, double* s, 
///     double* sep, lapack_int mm, lapack_int* m, double* work, 
///     lapack_int ldwork, lapack_int* iwork
/// );
/// ```
#borrow(select, t, vl, vr, s, sep, m, work, iwork)
pub extern "C" fn lapacke_dtrsna_work(
  matrix_layout : Int,
  job : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  s : FixedArray[Double],
  sep : FixedArray[Double],
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Double],
  ldwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dtrsna_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrsna_work(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const lapack_complex_float* t, lapack_int ldt, const lapack_complex_float* vl, 
///     lapack_int ldvl, const lapack_complex_float* vr, lapack_int ldvr, float* s, 
///     float* sep, lapack_int mm, lapack_int* m, lapack_complex_float* work, 
///     lapack_int ldwork, float* rwork
/// );
/// ```
#borrow(select, t, vl, vr, s, sep, m, work, rwork)
pub extern "C" fn lapacke_ctrsna_work(
  matrix_layout : Int,
  job : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  s : FixedArray[Float],
  sep : FixedArray[Float],
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Float],
  ldwork : Lapack_int,
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctrsna_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrsna_work(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const lapack_complex_double* t, lapack_int ldt, const lapack_complex_double* vl, 
///     lapack_int ldvl, const lapack_complex_double* vr, lapack_int ldvr, double* s, 
///     double* sep, lapack_int mm, lapack_int* m, lapack_complex_double* work, 
///     lapack_int ldwork, double* rwork
/// );
/// ```
#borrow(select, t, vl, vr, s, sep, m, work, rwork)
pub extern "C" fn lapacke_ztrsna_work(
  matrix_layout : Int,
  job : Byte,
  howmny : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  s : FixedArray[Double],
  sep : FixedArray[Double],
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Double],
  ldwork : Lapack_int,
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztrsna_work"

///|
///
/// ```c
/// lapack_int LAPACKE_strsyl_work(
///     int matrix_layout, char trana, char tranb, lapack_int isgn, 
///     lapack_int m, lapack_int n, const float* a, lapack_int lda, 
///     const float* b, lapack_int ldb, float* c, lapack_int ldc, float* scale
/// );
/// ```
#borrow(a, b, c, scale)
pub extern "C" fn lapacke_strsyl_work(
  matrix_layout : Int,
  trana : Byte,
  tranb : Byte,
  isgn : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
  scale : FixedArray[Float],
) -> Lapack_int = "LAPACKE_strsyl_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrsyl_work(
///     int matrix_layout, char trana, char tranb, lapack_int isgn, 
///     lapack_int m, lapack_int n, const double* a, lapack_int lda, 
///     const double* b, lapack_int ldb, double* c, lapack_int ldc, double* scale
/// );
/// ```
#borrow(a, b, c, scale)
pub extern "C" fn lapacke_dtrsyl_work(
  matrix_layout : Int,
  trana : Byte,
  tranb : Byte,
  isgn : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
  scale : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dtrsyl_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrsyl_work(
///     int matrix_layout, char trana, char tranb, lapack_int isgn, 
///     lapack_int m, lapack_int n, const lapack_complex_float* a, lapack_int lda, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* c, lapack_int ldc, float* scale
/// );
/// ```
#borrow(a, b, c, scale)
pub extern "C" fn lapacke_ctrsyl_work(
  matrix_layout : Int,
  trana : Byte,
  tranb : Byte,
  isgn : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
  scale : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctrsyl_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrsyl_work(
///     int matrix_layout, char trana, char tranb, lapack_int isgn, 
///     lapack_int m, lapack_int n, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* c, lapack_int ldc, double* scale
/// );
/// ```
#borrow(a, b, c, scale)
pub extern "C" fn lapacke_ztrsyl_work(
  matrix_layout : Int,
  trana : Byte,
  tranb : Byte,
  isgn : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
  scale : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztrsyl_work"

///|
///
/// ```c
/// lapack_int LAPACKE_strsyl3_work(
///     int matrix_layout, char trana, char tranb, lapack_int isgn, 
///     lapack_int m, lapack_int n, const float* a, lapack_int lda, 
///     const float* b, lapack_int ldb, float* c, lapack_int ldc, 
///     float* scale, lapack_int* iwork, lapack_int liwork, float* swork, lapack_int ldswork
/// );
/// ```
#borrow(a, b, c, scale, iwork, swork)
pub extern "C" fn lapacke_strsyl3_work(
  matrix_layout : Int,
  trana : Byte,
  tranb : Byte,
  isgn : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
  scale : FixedArray[Float],
  iwork : FixedArray[Int],
  liwork : Lapack_int,
  swork : FixedArray[Float],
  ldswork : Lapack_int,
) -> Lapack_int = "LAPACKE_strsyl3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrsyl3_work(
///     int matrix_layout, char trana, char tranb, lapack_int isgn, 
///     lapack_int m, lapack_int n, const double* a, lapack_int lda, 
///     const double* b, lapack_int ldb, double* c, lapack_int ldc, 
///     double* scale, lapack_int* iwork, lapack_int liwork, double* swork, lapack_int ldswork
/// );
/// ```
#borrow(a, b, c, scale, iwork, swork)
pub extern "C" fn lapacke_dtrsyl3_work(
  matrix_layout : Int,
  trana : Byte,
  tranb : Byte,
  isgn : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
  scale : FixedArray[Double],
  iwork : FixedArray[Int],
  liwork : Lapack_int,
  swork : FixedArray[Double],
  ldswork : Lapack_int,
) -> Lapack_int = "LAPACKE_dtrsyl3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrsyl3_work(
///     int matrix_layout, char trana, char tranb, lapack_int isgn, 
///     lapack_int m, lapack_int n, const lapack_complex_float* a, lapack_int lda, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* c, lapack_int ldc, 
///     float* scale, float* swork, lapack_int ldswork
/// );
/// ```
#borrow(a, b, c, scale, swork)
pub extern "C" fn lapacke_ctrsyl3_work(
  matrix_layout : Int,
  trana : Byte,
  tranb : Byte,
  isgn : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
  scale : FixedArray[Float],
  swork : FixedArray[Float],
  ldswork : Lapack_int,
) -> Lapack_int = "LAPACKE_ctrsyl3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrsyl3_work(
///     int matrix_layout, char trana, char tranb, lapack_int isgn, 
///     lapack_int m, lapack_int n, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* c, lapack_int ldc, 
///     double* scale, double* swork, lapack_int ldswork
/// );
/// ```
#borrow(a, b, c, scale, swork)
pub extern "C" fn lapacke_ztrsyl3_work(
  matrix_layout : Int,
  trana : Byte,
  tranb : Byte,
  isgn : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
  scale : FixedArray[Double],
  swork : FixedArray[Double],
  ldswork : Lapack_int,
) -> Lapack_int = "LAPACKE_ztrsyl3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_strtri_work(
///     int matrix_layout, char uplo, char diag, lapack_int n, 
///     float* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_strtri_work(
  matrix_layout : Int,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_strtri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrtri_work(
///     int matrix_layout, char uplo, char diag, lapack_int n, 
///     double* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_dtrtri_work(
  matrix_layout : Int,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_dtrtri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrtri_work(
///     int matrix_layout, char uplo, char diag, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_ctrtri_work(
  matrix_layout : Int,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_ctrtri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrtri_work(
///     int matrix_layout, char uplo, char diag, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_ztrtri_work(
  matrix_layout : Int,
  uplo : Byte,
  diag : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_ztrtri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_strtrs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const float* a, lapack_int lda, 
///     float* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_strtrs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_strtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrtrs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const double* a, lapack_int lda, 
///     double* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_dtrtrs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dtrtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrtrs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_ctrtrs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ctrtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrtrs_work(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_ztrtrs_work(
  matrix_layout : Int,
  uplo : Byte,
  trans : Byte,
  diag : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ztrtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_strttf_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const float* a, lapack_int lda, float* arf
/// );
/// ```
#borrow(a, arf)
pub extern "C" fn lapacke_strttf_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  arf : FixedArray[Float],
) -> Lapack_int = "LAPACKE_strttf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrttf_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const double* a, lapack_int lda, double* arf
/// );
/// ```
#borrow(a, arf)
pub extern "C" fn lapacke_dtrttf_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  arf : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dtrttf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrttf_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_float* a, lapack_int lda, lapack_complex_float* arf
/// );
/// ```
#borrow(a, arf)
pub extern "C" fn lapacke_ctrttf_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  arf : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctrttf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrttf_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_double* a, lapack_int lda, lapack_complex_double* arf
/// );
/// ```
#borrow(a, arf)
pub extern "C" fn lapacke_ztrttf_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  arf : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztrttf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_strttp_work(
///     int matrix_layout, char uplo, lapack_int n, const float* a, 
///     lapack_int lda, float* ap
/// );
/// ```
#borrow(a, ap)
pub extern "C" fn lapacke_strttp_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ap : FixedArray[Float],
) -> Lapack_int = "LAPACKE_strttp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrttp_work(
///     int matrix_layout, char uplo, lapack_int n, const double* a, 
///     lapack_int lda, double* ap
/// );
/// ```
#borrow(a, ap)
pub extern "C" fn lapacke_dtrttp_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ap : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dtrttp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrttp_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* ap
/// );
/// ```
#borrow(a, ap)
pub extern "C" fn lapacke_ctrttp_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ap : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctrttp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrttp_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* ap
/// );
/// ```
#borrow(a, ap)
pub extern "C" fn lapacke_ztrttp_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ap : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztrttp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stzrzf_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_stzrzf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_stzrzf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtzrzf_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_dtzrzf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dtzrzf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctzrzf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_ctzrzf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ctzrzf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztzrzf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_ztzrzf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ztzrzf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cungbr_work(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int k, lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau, 
///     lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_cungbr_work(
  matrix_layout : Int,
  vect : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cungbr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zungbr_work(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int k, lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau, 
///     lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_zungbr_work(
  matrix_layout : Int,
  vect : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zungbr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cunghr_work(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_cunghr_work(
  matrix_layout : Int,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cunghr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zunghr_work(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_zunghr_work(
  matrix_layout : Int,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zunghr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cunglq_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_cunglq_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cunglq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zunglq_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_zunglq_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zunglq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cungql_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_cungql_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cungql_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zungql_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_zungql_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zungql_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cungqr_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_cungqr_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cungqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zungqr_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_zungqr_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zungqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cungrq_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_cungrq_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cungrq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zungrq_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_zungrq_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zungrq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cungtr_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_complex_float* tau, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_cungtr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cungtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zungtr_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_complex_double* tau, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_zungtr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zungtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cungtsqr_row_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int mb, 
///     lapack_int nb, lapack_complex_float* a, lapack_int lda, const lapack_complex_float* t, 
///     lapack_int ldt, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, t, work)
pub extern "C" fn lapacke_cungtsqr_row_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  mb : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cungtsqr_row_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zungtsqr_row_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int mb, 
///     lapack_int nb, lapack_complex_double* a, lapack_int lda, const lapack_complex_double* t, 
///     lapack_int ldt, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, t, work)
pub extern "C" fn lapacke_zungtsqr_row_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  mb : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zungtsqr_row_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmbr_work(
///     int matrix_layout, char vect, char side, char trans, 
///     lapack_int m, lapack_int n, lapack_int k, const lapack_complex_float* a, 
///     lapack_int lda, const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc, 
///     lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_cunmbr_work(
  matrix_layout : Int,
  vect : Byte,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cunmbr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmbr_work(
///     int matrix_layout, char vect, char side, char trans, 
///     lapack_int m, lapack_int n, lapack_int k, const lapack_complex_double* a, 
///     lapack_int lda, const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc, 
///     lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_zunmbr_work(
  matrix_layout : Int,
  vect : Byte,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zunmbr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmhr_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, const lapack_complex_float* a, 
///     lapack_int lda, const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc, 
///     lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_cunmhr_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cunmhr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmhr_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, const lapack_complex_double* a, 
///     lapack_int lda, const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc, 
///     lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_zunmhr_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zunmhr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmlq_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_float* a, lapack_int lda, 
///     const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_cunmlq_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cunmlq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmlq_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_zunmlq_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zunmlq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmql_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_float* a, lapack_int lda, 
///     const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_cunmql_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cunmql_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmql_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_zunmql_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zunmql_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmqr_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_float* a, lapack_int lda, 
///     const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_cunmqr_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cunmqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmqr_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_zunmqr_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zunmqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmrq_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_float* a, lapack_int lda, 
///     const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_cunmrq_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cunmrq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmrq_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_zunmrq_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zunmrq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmrz_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, const lapack_complex_float* a, 
///     lapack_int lda, const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc, 
///     lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_cunmrz_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cunmrz_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmrz_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, const lapack_complex_double* a, 
///     lapack_int lda, const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc, 
///     lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_zunmrz_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zunmrz_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmtr_work(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const lapack_complex_float* a, lapack_int lda, 
///     const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_cunmtr_work(
  matrix_layout : Int,
  side : Byte,
  uplo : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cunmtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmtr_work(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, c, work)
pub extern "C" fn lapacke_zunmtr_work(
  matrix_layout : Int,
  side : Byte,
  uplo : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zunmtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cupgtr_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* ap, 
///     const lapack_complex_float* tau, lapack_complex_float* q, lapack_int ldq, lapack_complex_float* work
/// );
/// ```
#borrow(ap, tau, q, work)
pub extern "C" fn lapacke_cupgtr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  tau : FixedArray[Float],
  q : FixedArray[Float],
  ldq : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cupgtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zupgtr_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* ap, 
///     const lapack_complex_double* tau, lapack_complex_double* q, lapack_int ldq, lapack_complex_double* work
/// );
/// ```
#borrow(ap, tau, q, work)
pub extern "C" fn lapacke_zupgtr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  tau : FixedArray[Double],
  q : FixedArray[Double],
  ldq : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zupgtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cupmtr_work(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const lapack_complex_float* ap, const lapack_complex_float* tau, 
///     lapack_complex_float* c, lapack_int ldc, lapack_complex_float* work
/// );
/// ```
#borrow(ap, tau, c, work)
pub extern "C" fn lapacke_cupmtr_work(
  matrix_layout : Int,
  side : Byte,
  uplo : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  ap : FixedArray[Float],
  tau : FixedArray[Float],
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cupmtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zupmtr_work(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const lapack_complex_double* ap, const lapack_complex_double* tau, 
///     lapack_complex_double* c, lapack_int ldc, lapack_complex_double* work
/// );
/// ```
#borrow(ap, tau, c, work)
pub extern "C" fn lapacke_zupmtr_work(
  matrix_layout : Int,
  side : Byte,
  uplo : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  ap : FixedArray[Double],
  tau : FixedArray[Double],
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zupmtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_claghe(
///     int matrix_layout, lapack_int n, lapack_int k, const float* d, 
///     lapack_complex_float* a, lapack_int lda, lapack_int* iseed
/// );
/// ```
#borrow(d, a, iseed)
pub extern "C" fn lapacke_claghe(
  matrix_layout : Int,
  n : Lapack_int,
  k : Lapack_int,
  d : FixedArray[Float],
  a : FixedArray[Float],
  lda : Lapack_int,
  iseed : FixedArray[Int],
) -> Lapack_int = "LAPACKE_claghe"

///|
///
/// ```c
/// lapack_int LAPACKE_zlaghe(
///     int matrix_layout, lapack_int n, lapack_int k, const double* d, 
///     lapack_complex_double* a, lapack_int lda, lapack_int* iseed
/// );
/// ```
#borrow(d, a, iseed)
pub extern "C" fn lapacke_zlaghe(
  matrix_layout : Int,
  n : Lapack_int,
  k : Lapack_int,
  d : FixedArray[Double],
  a : FixedArray[Double],
  lda : Lapack_int,
  iseed : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zlaghe"

///|
///
/// ```c
/// lapack_int LAPACKE_slagsy(
///     int matrix_layout, lapack_int n, lapack_int k, const float* d, 
///     float* a, lapack_int lda, lapack_int* iseed
/// );
/// ```
#borrow(d, a, iseed)
pub extern "C" fn lapacke_slagsy(
  matrix_layout : Int,
  n : Lapack_int,
  k : Lapack_int,
  d : FixedArray[Float],
  a : FixedArray[Float],
  lda : Lapack_int,
  iseed : FixedArray[Int],
) -> Lapack_int = "LAPACKE_slagsy"

///|
///
/// ```c
/// lapack_int LAPACKE_dlagsy(
///     int matrix_layout, lapack_int n, lapack_int k, const double* d, 
///     double* a, lapack_int lda, lapack_int* iseed
/// );
/// ```
#borrow(d, a, iseed)
pub extern "C" fn lapacke_dlagsy(
  matrix_layout : Int,
  n : Lapack_int,
  k : Lapack_int,
  d : FixedArray[Double],
  a : FixedArray[Double],
  lda : Lapack_int,
  iseed : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dlagsy"

///|
///
/// ```c
/// lapack_int LAPACKE_clagsy(
///     int matrix_layout, lapack_int n, lapack_int k, const float* d, 
///     lapack_complex_float* a, lapack_int lda, lapack_int* iseed
/// );
/// ```
#borrow(d, a, iseed)
pub extern "C" fn lapacke_clagsy(
  matrix_layout : Int,
  n : Lapack_int,
  k : Lapack_int,
  d : FixedArray[Float],
  a : FixedArray[Float],
  lda : Lapack_int,
  iseed : FixedArray[Int],
) -> Lapack_int = "LAPACKE_clagsy"

///|
///
/// ```c
/// lapack_int LAPACKE_zlagsy(
///     int matrix_layout, lapack_int n, lapack_int k, const double* d, 
///     lapack_complex_double* a, lapack_int lda, lapack_int* iseed
/// );
/// ```
#borrow(d, a, iseed)
pub extern "C" fn lapacke_zlagsy(
  matrix_layout : Int,
  n : Lapack_int,
  k : Lapack_int,
  d : FixedArray[Double],
  a : FixedArray[Double],
  lda : Lapack_int,
  iseed : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zlagsy"

///|
///
/// ```c
/// lapack_int LAPACKE_slapmr(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     float* x, lapack_int ldx, lapack_int* k
/// );
/// ```
#borrow(x, k)
pub extern "C" fn lapacke_slapmr(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_slapmr"

///|
///
/// ```c
/// lapack_int LAPACKE_dlapmr(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     double* x, lapack_int ldx, lapack_int* k
/// );
/// ```
#borrow(x, k)
pub extern "C" fn lapacke_dlapmr(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dlapmr"

///|
///
/// ```c
/// lapack_int LAPACKE_clapmr(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     lapack_complex_float* x, lapack_int ldx, lapack_int* k
/// );
/// ```
#borrow(x, k)
pub extern "C" fn lapacke_clapmr(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_clapmr"

///|
///
/// ```c
/// lapack_int LAPACKE_zlapmr(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     lapack_complex_double* x, lapack_int ldx, lapack_int* k
/// );
/// ```
#borrow(x, k)
pub extern "C" fn lapacke_zlapmr(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zlapmr"

///|
///
/// ```c
/// lapack_int LAPACKE_slapmt(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     float* x, lapack_int ldx, lapack_int* k
/// );
/// ```
#borrow(x, k)
pub extern "C" fn lapacke_slapmt(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_slapmt"

///|
///
/// ```c
/// lapack_int LAPACKE_dlapmt(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     double* x, lapack_int ldx, lapack_int* k
/// );
/// ```
#borrow(x, k)
pub extern "C" fn lapacke_dlapmt(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dlapmt"

///|
///
/// ```c
/// lapack_int LAPACKE_clapmt(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     lapack_complex_float* x, lapack_int ldx, lapack_int* k
/// );
/// ```
#borrow(x, k)
pub extern "C" fn lapacke_clapmt(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_clapmt"

///|
///
/// ```c
/// lapack_int LAPACKE_zlapmt(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     lapack_complex_double* x, lapack_int ldx, lapack_int* k
/// );
/// ```
#borrow(x, k)
pub extern "C" fn lapacke_zlapmt(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zlapmt"

///|
///
/// ```c
/// float LAPACKE_slapy2(float x, float y);
/// ```
pub extern "C" fn lapacke_slapy2(x : Float, y : Float) -> Float = "LAPACKE_slapy2"

///|
///
/// ```c
/// double LAPACKE_dlapy2(double x, double y);
/// ```
pub extern "C" fn lapacke_dlapy2(x : Double, y : Double) -> Double = "LAPACKE_dlapy2"

///|
///
/// ```c
/// float LAPACKE_slapy3(float x, float y, float z);
/// ```
pub extern "C" fn lapacke_slapy3(x : Float, y : Float, z : Float) -> Float = "LAPACKE_slapy3"

///|
///
/// ```c
/// double LAPACKE_dlapy3(double x, double y, double z);
/// ```
pub extern "C" fn lapacke_dlapy3(x : Double, y : Double, z : Double) -> Double = "LAPACKE_dlapy3"

///|
///
/// ```c
/// lapack_int LAPACKE_slartgp(
///     float f, float g, float* cs, float* sn, float* r
/// );
/// ```
#borrow(cs, sn, r)
pub extern "C" fn lapacke_slartgp(
  f : Float,
  g : Float,
  cs : FixedArray[Float],
  sn : FixedArray[Float],
  r : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slartgp"

///|
///
/// ```c
/// lapack_int LAPACKE_dlartgp(
///     double f, double g, double* cs, double* sn, double* r
/// );
/// ```
#borrow(cs, sn, r)
pub extern "C" fn lapacke_dlartgp(
  f : Double,
  g : Double,
  cs : FixedArray[Double],
  sn : FixedArray[Double],
  r : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlartgp"

///|
///
/// ```c
/// lapack_int LAPACKE_slartgs(
///     float x, float y, float sigma, float* cs, float* sn
/// );
/// ```
#borrow(cs, sn)
pub extern "C" fn lapacke_slartgs(
  x : Float,
  y : Float,
  sigma : Float,
  cs : FixedArray[Float],
  sn : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slartgs"

///|
///
/// ```c
/// lapack_int LAPACKE_dlartgs(
///     double x, double y, double sigma, double* cs, double* sn
/// );
/// ```
#borrow(cs, sn)
pub extern "C" fn lapacke_dlartgs(
  x : Double,
  y : Double,
  sigma : Double,
  cs : FixedArray[Double],
  sn : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlartgs"

///|
///
/// ```c
/// lapack_int LAPACKE_cbbcsd(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, lapack_int m, lapack_int p, 
///     lapack_int q, float* theta, float* phi, lapack_complex_float* u1, 
///     lapack_int ldu1, lapack_complex_float* u2, lapack_int ldu2, lapack_complex_float* v1t, 
///     lapack_int ldv1t, lapack_complex_float* v2t, lapack_int ldv2t, float* b11d, 
///     float* b11e, float* b12d, float* b12e, float* b21d, 
///     float* b21e, float* b22d, float* b22e
/// );
/// ```
#borrow(theta, phi, u1, u2, v1t, v2t, b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e)
pub extern "C" fn lapacke_cbbcsd(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  jobv2t : Byte,
  trans : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  theta : FixedArray[Float],
  phi : FixedArray[Float],
  u1 : FixedArray[Float],
  ldu1 : Lapack_int,
  u2 : FixedArray[Float],
  ldu2 : Lapack_int,
  v1t : FixedArray[Float],
  ldv1t : Lapack_int,
  v2t : FixedArray[Float],
  ldv2t : Lapack_int,
  b11d : FixedArray[Float],
  b11e : FixedArray[Float],
  b12d : FixedArray[Float],
  b12e : FixedArray[Float],
  b21d : FixedArray[Float],
  b21e : FixedArray[Float],
  b22d : FixedArray[Float],
  b22e : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cbbcsd"

///|
///
/// ```c
/// lapack_int LAPACKE_cbbcsd_work(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, lapack_int m, lapack_int p, 
///     lapack_int q, float* theta, float* phi, lapack_complex_float* u1, 
///     lapack_int ldu1, lapack_complex_float* u2, lapack_int ldu2, lapack_complex_float* v1t, 
///     lapack_int ldv1t, lapack_complex_float* v2t, lapack_int ldv2t, float* b11d, 
///     float* b11e, float* b12d, float* b12e, float* b21d, 
///     float* b21e, float* b22d, float* b22e, float* rwork, lapack_int lrwork
/// );
/// ```
#borrow(theta, phi, u1, u2, v1t, v2t, b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e, rwork)
pub extern "C" fn lapacke_cbbcsd_work(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  jobv2t : Byte,
  trans : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  theta : FixedArray[Float],
  phi : FixedArray[Float],
  u1 : FixedArray[Float],
  ldu1 : Lapack_int,
  u2 : FixedArray[Float],
  ldu2 : Lapack_int,
  v1t : FixedArray[Float],
  ldv1t : Lapack_int,
  v2t : FixedArray[Float],
  ldv2t : Lapack_int,
  b11d : FixedArray[Float],
  b11e : FixedArray[Float],
  b12d : FixedArray[Float],
  b12e : FixedArray[Float],
  b21d : FixedArray[Float],
  b21e : FixedArray[Float],
  b22d : FixedArray[Float],
  b22e : FixedArray[Float],
  rwork : FixedArray[Float],
  lrwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cbbcsd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cheswapr(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int i1, lapack_int i2
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_cheswapr(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  i1 : Lapack_int,
  i2 : Lapack_int,
) -> Lapack_int = "LAPACKE_cheswapr"

///|
///
/// ```c
/// lapack_int LAPACKE_cheswapr_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int i1, lapack_int i2
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_cheswapr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  i1 : Lapack_int,
  i2 : Lapack_int,
) -> Lapack_int = "LAPACKE_cheswapr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chetri2(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_chetri2(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chetri2"

///|
///
/// ```c
/// lapack_int LAPACKE_chetri2_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_chetri2_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_chetri2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chetri2x(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_int nb
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_chetri2x(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  nb : Lapack_int,
) -> Lapack_int = "LAPACKE_chetri2x"

///|
///
/// ```c
/// lapack_int LAPACKE_chetri2x_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_complex_float* work, lapack_int nb
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_chetri2x_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  nb : Lapack_int,
) -> Lapack_int = "LAPACKE_chetri2x_work"
