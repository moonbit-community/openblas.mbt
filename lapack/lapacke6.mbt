///|
///
/// ```c
/// lapack_int LAPACKE_cggsvp_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* b, lapack_int ldb, float tola, 
///     float tolb, lapack_int* k, lapack_int* l, lapack_complex_float* u, 
///     lapack_int ldu, lapack_complex_float* v, lapack_int ldv, lapack_complex_float* q, 
///     lapack_int ldq, lapack_int* iwork, float* rwork, lapack_complex_float* tau, lapack_complex_float* work
/// );
/// ```
pub extern "C" fn lapacke_cggsvp_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  p : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  tola : Float,
  tolb : Float,
  k : FixedArray[Int],
  l : FixedArray[Int],
  u : FixedArray[Float],
  ldu : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  iwork : FixedArray[Int],
  rwork : FixedArray[Float],
  tau : FixedArray[Float],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cggsvp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zggsvp_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* b, lapack_int ldb, double tola, 
///     double tolb, lapack_int* k, lapack_int* l, lapack_complex_double* u, 
///     lapack_int ldu, lapack_complex_double* v, lapack_int ldv, lapack_complex_double* q, 
///     lapack_int ldq, lapack_int* iwork, double* rwork, lapack_complex_double* tau, lapack_complex_double* work
/// );
/// ```
pub extern "C" fn lapacke_zggsvp_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  p : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  tola : Double,
  tolb : Double,
  k : FixedArray[Int],
  l : FixedArray[Int],
  u : FixedArray[Double],
  ldu : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  iwork : FixedArray[Int],
  rwork : FixedArray[Double],
  tau : FixedArray[Double],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zggsvp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sggsvp3_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, float* a, 
///     lapack_int lda, float* b, lapack_int ldb, float tola, 
///     float tolb, lapack_int* k, lapack_int* l, float* u, 
///     lapack_int ldu, float* v, lapack_int ldv, float* q, 
///     lapack_int ldq, lapack_int* iwork, float* tau, float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sggsvp3_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  p : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  tola : Float,
  tolb : Float,
  k : FixedArray[Int],
  l : FixedArray[Int],
  u : FixedArray[Float],
  ldu : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  iwork : FixedArray[Int],
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sggsvp3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dggsvp3_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, double* a, 
///     lapack_int lda, double* b, lapack_int ldb, double tola, 
///     double tolb, lapack_int* k, lapack_int* l, double* u, 
///     lapack_int ldu, double* v, lapack_int ldv, double* q, 
///     lapack_int ldq, lapack_int* iwork, double* tau, double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dggsvp3_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  p : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  tola : Double,
  tolb : Double,
  k : FixedArray[Int],
  l : FixedArray[Int],
  u : FixedArray[Double],
  ldu : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  iwork : FixedArray[Int],
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dggsvp3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cggsvp3_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* b, lapack_int ldb, float tola, 
///     float tolb, lapack_int* k, lapack_int* l, lapack_complex_float* u, 
///     lapack_int ldu, lapack_complex_float* v, lapack_int ldv, lapack_complex_float* q, 
///     lapack_int ldq, lapack_int* iwork, float* rwork, lapack_complex_float* tau, 
///     lapack_complex_float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_cggsvp3_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  p : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  tola : Float,
  tolb : Float,
  k : FixedArray[Int],
  l : FixedArray[Int],
  u : FixedArray[Float],
  ldu : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  iwork : FixedArray[Int],
  rwork : FixedArray[Float],
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cggsvp3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zggsvp3_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* b, lapack_int ldb, double tola, 
///     double tolb, lapack_int* k, lapack_int* l, lapack_complex_double* u, 
///     lapack_int ldu, lapack_complex_double* v, lapack_int ldv, lapack_complex_double* q, 
///     lapack_int ldq, lapack_int* iwork, double* rwork, lapack_complex_double* tau, 
///     lapack_complex_double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_zggsvp3_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  p : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  tola : Double,
  tolb : Double,
  k : FixedArray[Int],
  l : FixedArray[Int],
  u : FixedArray[Double],
  ldu : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  iwork : FixedArray[Int],
  rwork : FixedArray[Double],
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zggsvp3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgtcon_work(
///     char norm, lapack_int n, const float* dl, const float* d, 
///     const float* du, const float* du2, const lapack_int* ipiv, float anorm, 
///     float* rcond, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sgtcon_work(
  norm : Byte,
  n : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  du2 : FixedArray[Float],
  ipiv : FixedArray[Int],
  anorm : Float,
  rcond : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgtcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgtcon_work(
///     char norm, lapack_int n, const double* dl, const double* d, 
///     const double* du, const double* du2, const lapack_int* ipiv, double anorm, 
///     double* rcond, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dgtcon_work(
  norm : Byte,
  n : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  du2 : FixedArray[Double],
  ipiv : FixedArray[Int],
  anorm : Double,
  rcond : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgtcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgtcon_work(
///     char norm, lapack_int n, const lapack_complex_float* dl, const lapack_complex_float* d, 
///     const lapack_complex_float* du, const lapack_complex_float* du2, const lapack_int* ipiv, float anorm, 
///     float* rcond, lapack_complex_float* work
/// );
/// ```
pub extern "C" fn lapacke_cgtcon_work(
  norm : Byte,
  n : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  du2 : FixedArray[Float],
  ipiv : FixedArray[Int],
  anorm : Float,
  rcond : FixedArray[Float],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgtcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgtcon_work(
///     char norm, lapack_int n, const lapack_complex_double* dl, const lapack_complex_double* d, 
///     const lapack_complex_double* du, const lapack_complex_double* du2, const lapack_int* ipiv, double anorm, 
///     double* rcond, lapack_complex_double* work
/// );
/// ```
pub extern "C" fn lapacke_zgtcon_work(
  norm : Byte,
  n : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  du2 : FixedArray[Double],
  ipiv : FixedArray[Int],
  anorm : Double,
  rcond : FixedArray[Double],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgtcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgtrfs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const float* dl, const float* d, const float* du, const float* dlf, 
///     const float* df, const float* duf, const float* du2, const lapack_int* ipiv, 
///     const float* b, lapack_int ldb, float* x, lapack_int ldx, 
///     float* ferr, float* berr, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sgtrfs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  dlf : FixedArray[Float],
  df : FixedArray[Float],
  duf : FixedArray[Float],
  du2 : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgtrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgtrfs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const double* dl, const double* d, const double* du, const double* dlf, 
///     const double* df, const double* duf, const double* du2, const lapack_int* ipiv, 
///     const double* b, lapack_int ldb, double* x, lapack_int ldx, 
///     double* ferr, double* berr, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dgtrfs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  dlf : FixedArray[Double],
  df : FixedArray[Double],
  duf : FixedArray[Double],
  du2 : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgtrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgtrfs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* dl, const lapack_complex_float* d, const lapack_complex_float* du, const lapack_complex_float* dlf, 
///     const lapack_complex_float* df, const lapack_complex_float* duf, const lapack_complex_float* du2, const lapack_int* ipiv, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* ferr, float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cgtrfs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  dlf : FixedArray[Float],
  df : FixedArray[Float],
  duf : FixedArray[Float],
  du2 : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgtrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgtrfs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* dl, const lapack_complex_double* d, const lapack_complex_double* du, const lapack_complex_double* dlf, 
///     const lapack_complex_double* df, const lapack_complex_double* duf, const lapack_complex_double* du2, const lapack_int* ipiv, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* ferr, double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zgtrfs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  dlf : FixedArray[Double],
  df : FixedArray[Double],
  duf : FixedArray[Double],
  du2 : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgtrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgtsv_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, float* dl, 
///     float* d, float* du, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_sgtsv_work(
  matrix_layout : Int,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_sgtsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgtsv_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, double* dl, 
///     double* d, double* du, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dgtsv_work(
  matrix_layout : Int,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dgtsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgtsv_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, lapack_complex_float* dl, 
///     lapack_complex_float* d, lapack_complex_float* du, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cgtsv_work(
  matrix_layout : Int,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_cgtsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgtsv_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, lapack_complex_double* dl, 
///     lapack_complex_double* d, lapack_complex_double* du, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zgtsv_work(
  matrix_layout : Int,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zgtsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgtsvx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, const float* dl, const float* d, const float* du, 
///     float* dlf, float* df, float* duf, float* du2, 
///     lapack_int* ipiv, const float* b, lapack_int ldb, float* x, 
///     lapack_int ldx, float* rcond, float* ferr, float* berr, 
///     float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sgtsvx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  dlf : FixedArray[Float],
  df : FixedArray[Float],
  duf : FixedArray[Float],
  du2 : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  rcond : FixedArray[Float],
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgtsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgtsvx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, const double* dl, const double* d, const double* du, 
///     double* dlf, double* df, double* duf, double* du2, 
///     lapack_int* ipiv, const double* b, lapack_int ldb, double* x, 
///     lapack_int ldx, double* rcond, double* ferr, double* berr, 
///     double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dgtsvx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  dlf : FixedArray[Double],
  df : FixedArray[Double],
  duf : FixedArray[Double],
  du2 : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  rcond : FixedArray[Double],
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgtsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgtsvx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* dl, const lapack_complex_float* d, const lapack_complex_float* du, 
///     lapack_complex_float* dlf, lapack_complex_float* df, lapack_complex_float* duf, lapack_complex_float* du2, 
///     lapack_int* ipiv, const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* rcond, float* ferr, float* berr, 
///     lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cgtsvx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  dlf : FixedArray[Float],
  df : FixedArray[Float],
  duf : FixedArray[Float],
  du2 : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  rcond : FixedArray[Float],
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgtsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgtsvx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* dl, const lapack_complex_double* d, const lapack_complex_double* du, 
///     lapack_complex_double* dlf, lapack_complex_double* df, lapack_complex_double* duf, lapack_complex_double* du2, 
///     lapack_int* ipiv, const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* rcond, double* ferr, double* berr, 
///     lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zgtsvx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  dlf : FixedArray[Double],
  df : FixedArray[Double],
  duf : FixedArray[Double],
  du2 : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  rcond : FixedArray[Double],
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgtsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgttrf_work(
///     lapack_int n, float* dl, float* d, float* du, 
///     float* du2, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_sgttrf_work(
  n : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  du2 : FixedArray[Float],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgttrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgttrf_work(
///     lapack_int n, double* dl, double* d, double* du, 
///     double* du2, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_dgttrf_work(
  n : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  du2 : FixedArray[Double],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgttrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgttrf_work(
///     lapack_int n, lapack_complex_float* dl, lapack_complex_float* d, lapack_complex_float* du, 
///     lapack_complex_float* du2, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_cgttrf_work(
  n : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  du2 : FixedArray[Float],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cgttrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgttrf_work(
///     lapack_int n, lapack_complex_double* dl, lapack_complex_double* d, lapack_complex_double* du, 
///     lapack_complex_double* du2, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_zgttrf_work(
  n : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  du2 : FixedArray[Double],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zgttrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgttrs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const float* dl, const float* d, const float* du, const float* du2, 
///     const lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_sgttrs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  du2 : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_sgttrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgttrs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const double* dl, const double* d, const double* du, const double* du2, 
///     const lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dgttrs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  du2 : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dgttrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgttrs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* dl, const lapack_complex_float* d, const lapack_complex_float* du, const lapack_complex_float* du2, 
///     const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cgttrs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  du2 : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_cgttrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgttrs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* dl, const lapack_complex_double* d, const lapack_complex_double* du, const lapack_complex_double* du2, 
///     const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zgttrs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  du2 : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zgttrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chbev_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, lapack_complex_float* ab, lapack_int ldab, float* w, 
///     lapack_complex_float* z, lapack_int ldz, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_chbev_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chbev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbev_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, lapack_complex_double* ab, lapack_int ldab, double* w, 
///     lapack_complex_double* z, lapack_int ldz, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zhbev_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhbev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chbevd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, lapack_complex_float* ab, lapack_int ldab, float* w, 
///     lapack_complex_float* z, lapack_int ldz, lapack_complex_float* work, lapack_int lwork, 
///     float* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_chbevd_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_chbevd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbevd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, lapack_complex_double* ab, lapack_int ldab, double* w, 
///     lapack_complex_double* z, lapack_int ldz, lapack_complex_double* work, lapack_int lwork, 
///     double* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_zhbevd_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zhbevd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chbevx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int kd, lapack_complex_float* ab, lapack_int ldab, 
///     lapack_complex_float* q, lapack_int ldq, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, lapack_complex_float* z, lapack_int ldz, lapack_complex_float* work, 
///     float* rwork, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_chbevx_work(
  matrix_layout : Int,
  jobz : Byte,
  range : Byte,
  uplo : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
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
  rwork : FixedArray[Float],
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chbevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbevx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int kd, lapack_complex_double* ab, lapack_int ldab, 
///     lapack_complex_double* q, lapack_int ldq, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, lapack_complex_double* z, lapack_int ldz, lapack_complex_double* work, 
///     double* rwork, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_zhbevx_work(
  matrix_layout : Int,
  jobz : Byte,
  range : Byte,
  uplo : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
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
  rwork : FixedArray[Double],
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhbevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chbgst_work(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, lapack_complex_float* ab, lapack_int ldab, 
///     const lapack_complex_float* bb, lapack_int ldbb, lapack_complex_float* x, lapack_int ldx, 
///     lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_chbgst_work(
  matrix_layout : Int,
  vect : Byte,
  uplo : Byte,
  n : Lapack_int,
  ka : Lapack_int,
  kb : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  bb : FixedArray[Float],
  ldbb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chbgst_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbgst_work(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, lapack_complex_double* ab, lapack_int ldab, 
///     const lapack_complex_double* bb, lapack_int ldbb, lapack_complex_double* x, lapack_int ldx, 
///     lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zhbgst_work(
  matrix_layout : Int,
  vect : Byte,
  uplo : Byte,
  n : Lapack_int,
  ka : Lapack_int,
  kb : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  bb : FixedArray[Double],
  ldbb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhbgst_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chbgv_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, lapack_complex_float* ab, lapack_int ldab, 
///     lapack_complex_float* bb, lapack_int ldbb, float* w, lapack_complex_float* z, 
///     lapack_int ldz, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_chbgv_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ka : Lapack_int,
  kb : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  bb : FixedArray[Float],
  ldbb : Lapack_int,
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chbgv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbgv_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, lapack_complex_double* ab, lapack_int ldab, 
///     lapack_complex_double* bb, lapack_int ldbb, double* w, lapack_complex_double* z, 
///     lapack_int ldz, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zhbgv_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ka : Lapack_int,
  kb : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  bb : FixedArray[Double],
  ldbb : Lapack_int,
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhbgv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chbgvd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, lapack_complex_float* ab, lapack_int ldab, 
///     lapack_complex_float* bb, lapack_int ldbb, float* w, lapack_complex_float* z, 
///     lapack_int ldz, lapack_complex_float* work, lapack_int lwork, float* rwork, 
///     lapack_int lrwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_chbgvd_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ka : Lapack_int,
  kb : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  bb : FixedArray[Float],
  ldbb : Lapack_int,
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_chbgvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbgvd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, lapack_complex_double* ab, lapack_int ldab, 
///     lapack_complex_double* bb, lapack_int ldbb, double* w, lapack_complex_double* z, 
///     lapack_int ldz, lapack_complex_double* work, lapack_int lwork, double* rwork, 
///     lapack_int lrwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_zhbgvd_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ka : Lapack_int,
  kb : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  bb : FixedArray[Double],
  ldbb : Lapack_int,
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zhbgvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chbgvx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int ka, lapack_int kb, lapack_complex_float* ab, 
///     lapack_int ldab, lapack_complex_float* bb, lapack_int ldbb, lapack_complex_float* q, 
///     lapack_int ldq, float vl, float vu, lapack_int il, 
///     lapack_int iu, float abstol, lapack_int* m, float* w, 
///     lapack_complex_float* z, lapack_int ldz, lapack_complex_float* work, float* rwork, 
///     lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_chbgvx_work(
  matrix_layout : Int,
  jobz : Byte,
  range : Byte,
  uplo : Byte,
  n : Lapack_int,
  ka : Lapack_int,
  kb : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  bb : FixedArray[Float],
  ldbb : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
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
  rwork : FixedArray[Float],
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chbgvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbgvx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int ka, lapack_int kb, lapack_complex_double* ab, 
///     lapack_int ldab, lapack_complex_double* bb, lapack_int ldbb, lapack_complex_double* q, 
///     lapack_int ldq, double vl, double vu, lapack_int il, 
///     lapack_int iu, double abstol, lapack_int* m, double* w, 
///     lapack_complex_double* z, lapack_int ldz, lapack_complex_double* work, double* rwork, 
///     lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_zhbgvx_work(
  matrix_layout : Int,
  jobz : Byte,
  range : Byte,
  uplo : Byte,
  n : Lapack_int,
  ka : Lapack_int,
  kb : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  bb : FixedArray[Double],
  ldbb : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
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
  rwork : FixedArray[Double],
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhbgvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chbtrd_work(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int kd, lapack_complex_float* ab, lapack_int ldab, float* d, 
///     float* e, lapack_complex_float* q, lapack_int ldq, lapack_complex_float* work
/// );
/// ```
pub extern "C" fn lapacke_chbtrd_work(
  matrix_layout : Int,
  vect : Byte,
  uplo : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  d : FixedArray[Float],
  e : FixedArray[Float],
  q : FixedArray[Float],
  ldq : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chbtrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbtrd_work(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int kd, lapack_complex_double* ab, lapack_int ldab, double* d, 
///     double* e, lapack_complex_double* q, lapack_int ldq, lapack_complex_double* work
/// );
/// ```
pub extern "C" fn lapacke_zhbtrd_work(
  matrix_layout : Int,
  vect : Byte,
  uplo : Byte,
  n : Lapack_int,
  kd : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  d : FixedArray[Double],
  e : FixedArray[Double],
  q : FixedArray[Double],
  ldq : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhbtrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_checon_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv, float anorm, float* rcond, lapack_complex_float* work
/// );
/// ```
pub extern "C" fn lapacke_checon_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  anorm : Float,
  rcond : FixedArray[Float],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_checon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhecon_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv, double anorm, double* rcond, lapack_complex_double* work
/// );
/// ```
pub extern "C" fn lapacke_zhecon_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  anorm : Double,
  rcond : FixedArray[Double],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhecon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cheequb_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, float* s, float* scond, float* amax, lapack_complex_float* work
/// );
/// ```
pub extern "C" fn lapacke_cheequb_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  s : FixedArray[Float],
  scond : FixedArray[Float],
  amax : FixedArray[Float],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cheequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zheequb_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, double* s, double* scond, double* amax, lapack_complex_double* work
/// );
/// ```
pub extern "C" fn lapacke_zheequb_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  s : FixedArray[Double],
  scond : FixedArray[Double],
  amax : FixedArray[Double],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zheequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cheev_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, float* w, lapack_complex_float* work, 
///     lapack_int lwork, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cheev_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  w : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cheev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zheev_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, double* w, lapack_complex_double* work, 
///     lapack_int lwork, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zheev_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  w : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zheev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cheevd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, float* w, lapack_complex_float* work, 
///     lapack_int lwork, float* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_cheevd_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  w : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cheevd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zheevd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, double* w, lapack_complex_double* work, 
///     lapack_int lwork, double* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_zheevd_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  w : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zheevd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cheevr_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_complex_float* a, lapack_int lda, float vl, 
///     float vu, lapack_int il, lapack_int iu, float abstol, 
///     lapack_int* m, float* w, lapack_complex_float* z, lapack_int ldz, 
///     lapack_int* isuppz, lapack_complex_float* work, lapack_int lwork, float* rwork, 
///     lapack_int lrwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_cheevr_work(
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
  rwork : FixedArray[Float],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cheevr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zheevr_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_complex_double* a, lapack_int lda, double vl, 
///     double vu, lapack_int il, lapack_int iu, double abstol, 
///     lapack_int* m, double* w, lapack_complex_double* z, lapack_int ldz, 
///     lapack_int* isuppz, lapack_complex_double* work, lapack_int lwork, double* rwork, 
///     lapack_int lrwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_zheevr_work(
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
  rwork : FixedArray[Double],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zheevr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cheevx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_complex_float* a, lapack_int lda, float vl, 
///     float vu, lapack_int il, lapack_int iu, float abstol, 
///     lapack_int* m, float* w, lapack_complex_float* z, lapack_int ldz, 
///     lapack_complex_float* work, lapack_int lwork, float* rwork, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_cheevx_work(
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
  rwork : FixedArray[Float],
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cheevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zheevx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_complex_double* a, lapack_int lda, double vl, 
///     double vu, lapack_int il, lapack_int iu, double abstol, 
///     lapack_int* m, double* w, lapack_complex_double* z, lapack_int ldz, 
///     lapack_complex_double* work, lapack_int lwork, double* rwork, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_zheevx_work(
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
  rwork : FixedArray[Double],
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zheevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chegst_work(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, const lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_chegst_work(
  matrix_layout : Int,
  itype : Lapack_int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chegst_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhegst_work(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, const lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zhegst_work(
  matrix_layout : Int,
  itype : Lapack_int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhegst_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chegv_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, 
///     lapack_int ldb, float* w, lapack_complex_float* work, lapack_int lwork, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_chegv_work(
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
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chegv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhegv_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, 
///     lapack_int ldb, double* w, lapack_complex_double* work, lapack_int lwork, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zhegv_work(
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
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhegv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chegvd_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, 
///     lapack_int ldb, float* w, lapack_complex_float* work, lapack_int lwork, 
///     float* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_chegvd_work(
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
  rwork : FixedArray[Float],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_chegvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhegvd_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, 
///     lapack_int ldb, double* w, lapack_complex_double* work, lapack_int lwork, 
///     double* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_zhegvd_work(
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
  rwork : FixedArray[Double],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zhegvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chegvx_work(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, lapack_complex_float* z, lapack_int ldz, lapack_complex_float* work, 
///     lapack_int lwork, float* rwork, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_chegvx_work(
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
  rwork : FixedArray[Float],
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chegvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhegvx_work(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, lapack_complex_double* z, lapack_int ldz, lapack_complex_double* work, 
///     lapack_int lwork, double* rwork, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_zhegvx_work(
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
  rwork : FixedArray[Double],
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhegvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cherfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* ferr, float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cherfs_work(
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
) -> Lapack_int = "LAPACKE_cherfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zherfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* ferr, double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zherfs_work(
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
) -> Lapack_int = "LAPACKE_zherfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cherfsx_work(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const float* s, const lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond, 
///     float* berr, lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, 
///     lapack_int nparams, float* params, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cherfsx_work(
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
) -> Lapack_int = "LAPACKE_cherfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zherfsx_work(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const double* s, const lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond, 
///     double* berr, lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, 
///     lapack_int nparams, double* params, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zherfsx_work(
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
) -> Lapack_int = "LAPACKE_zherfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chesv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_int* ipiv, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_chesv_work(
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
) -> Lapack_int = "LAPACKE_chesv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhesv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_int* ipiv, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_zhesv_work(
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
) -> Lapack_int = "LAPACKE_zhesv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chesvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, lapack_int* ipiv, const lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* x, lapack_int ldx, float* rcond, float* ferr, 
///     float* berr, lapack_complex_float* work, lapack_int lwork, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_chesvx_work(
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
) -> Lapack_int = "LAPACKE_chesvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhesvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, lapack_int* ipiv, const lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* x, lapack_int ldx, double* rcond, double* ferr, 
///     double* berr, lapack_complex_double* work, lapack_int lwork, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zhesvx_work(
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
) -> Lapack_int = "LAPACKE_zhesvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chesvxx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, float* s, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* rcond, float* rpvgrw, float* berr, lapack_int n_err_bnds, 
///     float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, float* params, 
///     lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_chesvxx_work(
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
) -> Lapack_int = "LAPACKE_chesvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhesvxx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, double* s, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* rcond, double* rpvgrw, double* berr, lapack_int n_err_bnds, 
///     double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, double* params, 
///     lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zhesvxx_work(
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
) -> Lapack_int = "LAPACKE_zhesvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrd_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, float* d, float* e, lapack_complex_float* tau, 
///     lapack_complex_float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_chetrd_work(
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
) -> Lapack_int = "LAPACKE_chetrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrd_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, double* d, double* e, lapack_complex_double* tau, 
///     lapack_complex_double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_zhetrd_work(
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
) -> Lapack_int = "LAPACKE_zhetrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_chetrf_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_chetrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_zhetrf_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zhetrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chetri_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_complex_float* work
/// );
/// ```
pub extern "C" fn lapacke_chetri_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chetri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetri_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_complex_double* work
/// );
/// ```
pub extern "C" fn lapacke_zhetri_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhetri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_chetrs_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chetrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zhetrs_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhetrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chfrk_work(
///     int matrix_layout, char transr, char uplo, char trans, 
///     lapack_int n, lapack_int k, float alpha, const lapack_complex_float* a, 
///     lapack_int lda, float beta, lapack_complex_float* c
/// );
/// ```
pub extern "C" fn lapacke_chfrk_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  trans : Byte,
  n : Lapack_int,
  k : Lapack_int,
  alpha : Float,
  a : FixedArray[Float],
  lda : Lapack_int,
  beta : Float,
  c : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chfrk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhfrk_work(
///     int matrix_layout, char transr, char uplo, char trans, 
///     lapack_int n, lapack_int k, double alpha, const lapack_complex_double* a, 
///     lapack_int lda, double beta, lapack_complex_double* c
/// );
/// ```
pub extern "C" fn lapacke_zhfrk_work(
  matrix_layout : Int,
  transr : Byte,
  uplo : Byte,
  trans : Byte,
  n : Lapack_int,
  k : Lapack_int,
  alpha : Double,
  a : FixedArray[Double],
  lda : Lapack_int,
  beta : Double,
  c : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhfrk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_shgeqz_work(
///     int matrix_layout, char job, char compq, char compz, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, float* h, 
///     lapack_int ldh, float* t, lapack_int ldt, float* alphar, 
///     float* alphai, float* beta, float* q, lapack_int ldq, 
///     float* z, lapack_int ldz, float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_shgeqz_work(
  matrix_layout : Int,
  job : Byte,
  compq : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  h : FixedArray[Float],
  ldh : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  alphar : FixedArray[Float],
  alphai : FixedArray[Float],
  beta : FixedArray[Float],
  q : FixedArray[Float],
  ldq : Lapack_int,
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_shgeqz_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dhgeqz_work(
///     int matrix_layout, char job, char compq, char compz, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, double* h, 
///     lapack_int ldh, double* t, lapack_int ldt, double* alphar, 
///     double* alphai, double* beta, double* q, lapack_int ldq, 
///     double* z, lapack_int ldz, double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dhgeqz_work(
  matrix_layout : Int,
  job : Byte,
  compq : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  h : FixedArray[Double],
  ldh : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  alphar : FixedArray[Double],
  alphai : FixedArray[Double],
  beta : FixedArray[Double],
  q : FixedArray[Double],
  ldq : Lapack_int,
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dhgeqz_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chgeqz_work(
///     int matrix_layout, char job, char compq, char compz, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, lapack_complex_float* h, 
///     lapack_int ldh, lapack_complex_float* t, lapack_int ldt, lapack_complex_float* alpha, 
///     lapack_complex_float* beta, lapack_complex_float* q, lapack_int ldq, lapack_complex_float* z, 
///     lapack_int ldz, lapack_complex_float* work, lapack_int lwork, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_chgeqz_work(
  matrix_layout : Int,
  job : Byte,
  compq : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  h : FixedArray[Float],
  ldh : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  alpha : FixedArray[Float],
  beta : FixedArray[Float],
  q : FixedArray[Float],
  ldq : Lapack_int,
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chgeqz_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhgeqz_work(
///     int matrix_layout, char job, char compq, char compz, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, lapack_complex_double* h, 
///     lapack_int ldh, lapack_complex_double* t, lapack_int ldt, lapack_complex_double* alpha, 
///     lapack_complex_double* beta, lapack_complex_double* q, lapack_int ldq, lapack_complex_double* z, 
///     lapack_int ldz, lapack_complex_double* work, lapack_int lwork, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zhgeqz_work(
  matrix_layout : Int,
  job : Byte,
  compq : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  h : FixedArray[Double],
  ldh : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  alpha : FixedArray[Double],
  beta : FixedArray[Double],
  q : FixedArray[Double],
  ldq : Lapack_int,
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhgeqz_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chpcon_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* ap, 
///     const lapack_int* ipiv, float anorm, float* rcond, lapack_complex_float* work
/// );
/// ```
pub extern "C" fn lapacke_chpcon_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  ipiv : FixedArray[Int],
  anorm : Float,
  rcond : FixedArray[Float],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chpcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpcon_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* ap, 
///     const lapack_int* ipiv, double anorm, double* rcond, lapack_complex_double* work
/// );
/// ```
pub extern "C" fn lapacke_zhpcon_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  ipiv : FixedArray[Int],
  anorm : Double,
  rcond : FixedArray[Double],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhpcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chpev_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_float* ap, float* w, lapack_complex_float* z, lapack_int ldz, 
///     lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_chpev_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chpev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpev_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_double* ap, double* w, lapack_complex_double* z, lapack_int ldz, 
///     lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zhpev_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhpev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chpevd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_float* ap, float* w, lapack_complex_float* z, lapack_int ldz, 
///     lapack_complex_float* work, lapack_int lwork, float* rwork, lapack_int lrwork, 
///     lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_chpevd_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_chpevd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpevd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_double* ap, double* w, lapack_complex_double* z, lapack_int ldz, 
///     lapack_complex_double* work, lapack_int lwork, double* rwork, lapack_int lrwork, 
///     lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_zhpevd_work(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zhpevd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chpevx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_complex_float* ap, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, lapack_complex_float* z, lapack_int ldz, lapack_complex_float* work, 
///     float* rwork, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_chpevx_work(
  matrix_layout : Int,
  jobz : Byte,
  range : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
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
  rwork : FixedArray[Float],
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chpevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpevx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_complex_double* ap, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, lapack_complex_double* z, lapack_int ldz, lapack_complex_double* work, 
///     double* rwork, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_zhpevx_work(
  matrix_layout : Int,
  jobz : Byte,
  range : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
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
  rwork : FixedArray[Double],
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhpevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chpgst_work(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     lapack_complex_float* ap, const lapack_complex_float* bp
/// );
/// ```
pub extern "C" fn lapacke_chpgst_work(
  matrix_layout : Int,
  itype : Lapack_int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  bp : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chpgst_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpgst_work(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     lapack_complex_double* ap, const lapack_complex_double* bp
/// );
/// ```
pub extern "C" fn lapacke_zhpgst_work(
  matrix_layout : Int,
  itype : Lapack_int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  bp : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhpgst_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chpgv_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_float* ap, lapack_complex_float* bp, float* w, 
///     lapack_complex_float* z, lapack_int ldz, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_chpgv_work(
  matrix_layout : Int,
  itype : Lapack_int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  bp : FixedArray[Float],
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chpgv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpgv_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_double* ap, lapack_complex_double* bp, double* w, 
///     lapack_complex_double* z, lapack_int ldz, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zhpgv_work(
  matrix_layout : Int,
  itype : Lapack_int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  bp : FixedArray[Double],
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhpgv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chpgvd_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_float* ap, lapack_complex_float* bp, float* w, 
///     lapack_complex_float* z, lapack_int ldz, lapack_complex_float* work, lapack_int lwork, 
///     float* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_chpgvd_work(
  matrix_layout : Int,
  itype : Lapack_int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  bp : FixedArray[Float],
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_chpgvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpgvd_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_double* ap, lapack_complex_double* bp, double* w, 
///     lapack_complex_double* z, lapack_int ldz, lapack_complex_double* work, lapack_int lwork, 
///     double* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_zhpgvd_work(
  matrix_layout : Int,
  itype : Lapack_int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  bp : FixedArray[Double],
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zhpgvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chpgvx_work(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, lapack_complex_float* ap, lapack_complex_float* bp, 
///     float vl, float vu, lapack_int il, lapack_int iu, 
///     float abstol, lapack_int* m, float* w, lapack_complex_float* z, 
///     lapack_int ldz, lapack_complex_float* work, float* rwork, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_chpgvx_work(
  matrix_layout : Int,
  itype : Lapack_int,
  jobz : Byte,
  range : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  bp : FixedArray[Float],
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
  rwork : FixedArray[Float],
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chpgvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpgvx_work(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, lapack_complex_double* ap, lapack_complex_double* bp, 
///     double vl, double vu, lapack_int il, lapack_int iu, 
///     double abstol, lapack_int* m, double* w, lapack_complex_double* z, 
///     lapack_int ldz, lapack_complex_double* work, double* rwork, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_zhpgvx_work(
  matrix_layout : Int,
  itype : Lapack_int,
  jobz : Byte,
  range : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  bp : FixedArray[Double],
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
  rwork : FixedArray[Double],
  iwork : FixedArray[Int],
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhpgvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chprfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* ap, const lapack_complex_float* afp, const lapack_int* ipiv, const lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* ferr, 
///     float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_chprfs_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Float],
  afp : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chprfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhprfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* ap, const lapack_complex_double* afp, const lapack_int* ipiv, const lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* ferr, 
///     double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zhprfs_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Double],
  afp : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhprfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chpsv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* ap, lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_chpsv_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chpsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpsv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* ap, lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zhpsv_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhpsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chpsvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* ap, lapack_complex_float* afp, lapack_int* ipiv, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* rcond, float* ferr, float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_chpsvx_work(
  matrix_layout : Int,
  fact : Byte,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Float],
  afp : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  rcond : FixedArray[Float],
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chpsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpsvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* ap, lapack_complex_double* afp, lapack_int* ipiv, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* rcond, double* ferr, double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zhpsvx_work(
  matrix_layout : Int,
  fact : Byte,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Double],
  afp : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  rcond : FixedArray[Double],
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhpsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chptrd_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* ap, 
///     float* d, float* e, lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_chptrd_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  d : FixedArray[Float],
  e : FixedArray[Float],
  tau : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chptrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhptrd_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* ap, 
///     double* d, double* e, lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zhptrd_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  d : FixedArray[Double],
  e : FixedArray[Double],
  tau : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhptrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chptrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* ap, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_chptrf_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chptrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhptrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* ap, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_zhptrf_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhptrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chptri_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* ap, 
///     const lapack_int* ipiv, lapack_complex_float* work
/// );
/// ```
pub extern "C" fn lapacke_chptri_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chptri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhptri_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* ap, 
///     const lapack_int* ipiv, lapack_complex_double* work
/// );
/// ```
pub extern "C" fn lapacke_zhptri_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhptri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chptrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* ap, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_chptrs_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chptrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhptrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* ap, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zhptrs_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhptrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_shsein_work(
///     int matrix_layout, char job, char eigsrc, char initv, 
///     lapack_logical* select, lapack_int n, const float* h, lapack_int ldh, 
///     float* wr, const float* wi, float* vl, lapack_int ldvl, 
///     float* vr, lapack_int ldvr, lapack_int mm, lapack_int* m, 
///     float* work, lapack_int* ifaill, lapack_int* ifailr
/// );
/// ```
pub extern "C" fn lapacke_shsein_work(
  matrix_layout : Int,
  job : Byte,
  eigsrc : Byte,
  initv : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  h : FixedArray[Float],
  ldh : Lapack_int,
  wr : FixedArray[Float],
  wi : FixedArray[Float],
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Float],
  ifaill : FixedArray[Int],
  ifailr : FixedArray[Int],
) -> Lapack_int = "LAPACKE_shsein_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dhsein_work(
///     int matrix_layout, char job, char eigsrc, char initv, 
///     lapack_logical* select, lapack_int n, const double* h, lapack_int ldh, 
///     double* wr, const double* wi, double* vl, lapack_int ldvl, 
///     double* vr, lapack_int ldvr, lapack_int mm, lapack_int* m, 
///     double* work, lapack_int* ifaill, lapack_int* ifailr
/// );
/// ```
pub extern "C" fn lapacke_dhsein_work(
  matrix_layout : Int,
  job : Byte,
  eigsrc : Byte,
  initv : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  h : FixedArray[Double],
  ldh : Lapack_int,
  wr : FixedArray[Double],
  wi : FixedArray[Double],
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Double],
  ifaill : FixedArray[Int],
  ifailr : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dhsein_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chsein_work(
///     int matrix_layout, char job, char eigsrc, char initv, 
///     const lapack_logical* select, lapack_int n, const lapack_complex_float* h, lapack_int ldh, 
///     lapack_complex_float* w, lapack_complex_float* vl, lapack_int ldvl, lapack_complex_float* vr, 
///     lapack_int ldvr, lapack_int mm, lapack_int* m, lapack_complex_float* work, 
///     float* rwork, lapack_int* ifaill, lapack_int* ifailr
/// );
/// ```
pub extern "C" fn lapacke_chsein_work(
  matrix_layout : Int,
  job : Byte,
  eigsrc : Byte,
  initv : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  h : FixedArray[Float],
  ldh : Lapack_int,
  w : FixedArray[Float],
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
  ifaill : FixedArray[Int],
  ifailr : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chsein_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhsein_work(
///     int matrix_layout, char job, char eigsrc, char initv, 
///     const lapack_logical* select, lapack_int n, const lapack_complex_double* h, lapack_int ldh, 
///     lapack_complex_double* w, lapack_complex_double* vl, lapack_int ldvl, lapack_complex_double* vr, 
///     lapack_int ldvr, lapack_int mm, lapack_int* m, lapack_complex_double* work, 
///     double* rwork, lapack_int* ifaill, lapack_int* ifailr
/// );
/// ```
pub extern "C" fn lapacke_zhsein_work(
  matrix_layout : Int,
  job : Byte,
  eigsrc : Byte,
  initv : Byte,
  select : Lapack_logical,
  n : Lapack_int,
  h : FixedArray[Double],
  ldh : Lapack_int,
  w : FixedArray[Double],
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  mm : Lapack_int,
  m : FixedArray[Int],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
  ifaill : FixedArray[Int],
  ifailr : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhsein_work"

///|
///
/// ```c
/// lapack_int LAPACKE_shseqr_work(
///     int matrix_layout, char job, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, float* h, lapack_int ldh, 
///     float* wr, float* wi, float* z, lapack_int ldz, 
///     float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_shseqr_work(
  matrix_layout : Int,
  job : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  h : FixedArray[Float],
  ldh : Lapack_int,
  wr : FixedArray[Float],
  wi : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_shseqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dhseqr_work(
///     int matrix_layout, char job, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, double* h, lapack_int ldh, 
///     double* wr, double* wi, double* z, lapack_int ldz, 
///     double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dhseqr_work(
  matrix_layout : Int,
  job : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  h : FixedArray[Double],
  ldh : Lapack_int,
  wr : FixedArray[Double],
  wi : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dhseqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chseqr_work(
///     int matrix_layout, char job, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, lapack_complex_float* h, lapack_int ldh, 
///     lapack_complex_float* w, lapack_complex_float* z, lapack_int ldz, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_chseqr_work(
  matrix_layout : Int,
  job : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  h : FixedArray[Float],
  ldh : Lapack_int,
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_chseqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhseqr_work(
///     int matrix_layout, char job, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, lapack_complex_double* h, lapack_int ldh, 
///     lapack_complex_double* w, lapack_complex_double* z, lapack_int ldz, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_zhseqr_work(
  matrix_layout : Int,
  job : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  h : FixedArray[Double],
  ldh : Lapack_int,
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zhseqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clacgv_work(lapack_int n, lapack_complex_float* x, lapack_int incx);
/// ```
pub extern "C" fn lapacke_clacgv_work(
  n : Lapack_int,
  x : FixedArray[Float],
  incx : Lapack_int,
) -> Lapack_int = "LAPACKE_clacgv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlacgv_work(lapack_int n, lapack_complex_double* x, lapack_int incx);
/// ```
pub extern "C" fn lapacke_zlacgv_work(
  n : Lapack_int,
  x : FixedArray[Double],
  incx : Lapack_int,
) -> Lapack_int = "LAPACKE_zlacgv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slacn2_work(
///     lapack_int n, float* v, float* x, lapack_int* isgn, 
///     float* est, lapack_int* kase, lapack_int* isave
/// );
/// ```
pub extern "C" fn lapacke_slacn2_work(
  n : Lapack_int,
  v : FixedArray[Float],
  x : FixedArray[Float],
  isgn : FixedArray[Int],
  est : FixedArray[Float],
  kase : FixedArray[Int],
  isave : FixedArray[Int],
) -> Lapack_int = "LAPACKE_slacn2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlacn2_work(
///     lapack_int n, double* v, double* x, lapack_int* isgn, 
///     double* est, lapack_int* kase, lapack_int* isave
/// );
/// ```
pub extern "C" fn lapacke_dlacn2_work(
  n : Lapack_int,
  v : FixedArray[Double],
  x : FixedArray[Double],
  isgn : FixedArray[Int],
  est : FixedArray[Double],
  kase : FixedArray[Int],
  isave : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dlacn2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clacn2_work(
///     lapack_int n, lapack_complex_float* v, lapack_complex_float* x, float* est, 
///     lapack_int* kase, lapack_int* isave
/// );
/// ```
pub extern "C" fn lapacke_clacn2_work(
  n : Lapack_int,
  v : FixedArray[Float],
  x : FixedArray[Float],
  est : FixedArray[Float],
  kase : FixedArray[Int],
  isave : FixedArray[Int],
) -> Lapack_int = "LAPACKE_clacn2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlacn2_work(
///     lapack_int n, lapack_complex_double* v, lapack_complex_double* x, double* est, 
///     lapack_int* kase, lapack_int* isave
/// );
/// ```
pub extern "C" fn lapacke_zlacn2_work(
  n : Lapack_int,
  v : FixedArray[Double],
  x : FixedArray[Double],
  est : FixedArray[Double],
  kase : FixedArray[Int],
  isave : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zlacn2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slacpy_work(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     const float* a, lapack_int lda, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_slacpy_work(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_slacpy_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlacpy_work(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     const double* a, lapack_int lda, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dlacpy_work(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dlacpy_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clacpy_work(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     const lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_clacpy_work(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_clacpy_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlacpy_work(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     const lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zlacpy_work(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zlacpy_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clacp2_work(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     const float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_clacp2_work(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_clacp2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlacp2_work(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     const double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zlacp2_work(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zlacp2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlag2c_work(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_float* sa, lapack_int ldsa
/// );
/// ```
pub extern "C" fn lapacke_zlag2c_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  sa : FixedArray[Float],
  ldsa : Lapack_int,
) -> Lapack_int = "LAPACKE_zlag2c_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slag2d_work(
///     int matrix_layout, lapack_int m, lapack_int n, const float* sa, 
///     lapack_int ldsa, double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_slag2d_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  sa : FixedArray[Float],
  ldsa : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_slag2d_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlag2s_work(
///     int matrix_layout, lapack_int m, lapack_int n, const double* a, 
///     lapack_int lda, float* sa, lapack_int ldsa
/// );
/// ```
pub extern "C" fn lapacke_dlag2s_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  sa : FixedArray[Float],
  ldsa : Lapack_int,
) -> Lapack_int = "LAPACKE_dlag2s_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clag2z_work(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_float* sa, 
///     lapack_int ldsa, lapack_complex_double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_clag2z_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  sa : FixedArray[Float],
  ldsa : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_clag2z_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slagge_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const float* d, float* a, lapack_int lda, 
///     lapack_int* iseed, float* work
/// );
/// ```
pub extern "C" fn lapacke_slagge_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  d : FixedArray[Float],
  a : FixedArray[Float],
  lda : Lapack_int,
  iseed : FixedArray[Int],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slagge_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlagge_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const double* d, double* a, lapack_int lda, 
///     lapack_int* iseed, double* work
/// );
/// ```
pub extern "C" fn lapacke_dlagge_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  d : FixedArray[Double],
  a : FixedArray[Double],
  lda : Lapack_int,
  iseed : FixedArray[Int],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlagge_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clagge_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const float* d, lapack_complex_float* a, lapack_int lda, 
///     lapack_int* iseed, lapack_complex_float* work
/// );
/// ```
pub extern "C" fn lapacke_clagge_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  d : FixedArray[Float],
  a : FixedArray[Float],
  lda : Lapack_int,
  iseed : FixedArray[Int],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_clagge_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlagge_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const double* d, lapack_complex_double* a, lapack_int lda, 
///     lapack_int* iseed, lapack_complex_double* work
/// );
/// ```
pub extern "C" fn lapacke_zlagge_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  d : FixedArray[Double],
  a : FixedArray[Double],
  lda : Lapack_int,
  iseed : FixedArray[Int],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zlagge_work"

///|
///
/// ```c
/// lapack_int LAPACKE_claghe_work(
///     int matrix_layout, lapack_int n, lapack_int k, const float* d, 
///     lapack_complex_float* a, lapack_int lda, lapack_int* iseed, lapack_complex_float* work
/// );
/// ```
pub extern "C" fn lapacke_claghe_work(
  matrix_layout : Int,
  n : Lapack_int,
  k : Lapack_int,
  d : FixedArray[Float],
  a : FixedArray[Float],
  lda : Lapack_int,
  iseed : FixedArray[Int],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_claghe_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlaghe_work(
///     int matrix_layout, lapack_int n, lapack_int k, const double* d, 
///     lapack_complex_double* a, lapack_int lda, lapack_int* iseed, lapack_complex_double* work
/// );
/// ```
pub extern "C" fn lapacke_zlaghe_work(
  matrix_layout : Int,
  n : Lapack_int,
  k : Lapack_int,
  d : FixedArray[Double],
  a : FixedArray[Double],
  lda : Lapack_int,
  iseed : FixedArray[Int],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zlaghe_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slagsy_work(
///     int matrix_layout, lapack_int n, lapack_int k, const float* d, 
///     float* a, lapack_int lda, lapack_int* iseed, float* work
/// );
/// ```
pub extern "C" fn lapacke_slagsy_work(
  matrix_layout : Int,
  n : Lapack_int,
  k : Lapack_int,
  d : FixedArray[Float],
  a : FixedArray[Float],
  lda : Lapack_int,
  iseed : FixedArray[Int],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slagsy_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlagsy_work(
///     int matrix_layout, lapack_int n, lapack_int k, const double* d, 
///     double* a, lapack_int lda, lapack_int* iseed, double* work
/// );
/// ```
pub extern "C" fn lapacke_dlagsy_work(
  matrix_layout : Int,
  n : Lapack_int,
  k : Lapack_int,
  d : FixedArray[Double],
  a : FixedArray[Double],
  lda : Lapack_int,
  iseed : FixedArray[Int],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlagsy_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clagsy_work(
///     int matrix_layout, lapack_int n, lapack_int k, const float* d, 
///     lapack_complex_float* a, lapack_int lda, lapack_int* iseed, lapack_complex_float* work
/// );
/// ```
pub extern "C" fn lapacke_clagsy_work(
  matrix_layout : Int,
  n : Lapack_int,
  k : Lapack_int,
  d : FixedArray[Float],
  a : FixedArray[Float],
  lda : Lapack_int,
  iseed : FixedArray[Int],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_clagsy_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlagsy_work(
///     int matrix_layout, lapack_int n, lapack_int k, const double* d, 
///     lapack_complex_double* a, lapack_int lda, lapack_int* iseed, lapack_complex_double* work
/// );
/// ```
pub extern "C" fn lapacke_zlagsy_work(
  matrix_layout : Int,
  n : Lapack_int,
  k : Lapack_int,
  d : FixedArray[Double],
  a : FixedArray[Double],
  lda : Lapack_int,
  iseed : FixedArray[Int],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zlagsy_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slapmr_work(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     float* x, lapack_int ldx, lapack_int* k
/// );
/// ```
pub extern "C" fn lapacke_slapmr_work(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_slapmr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlapmr_work(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     double* x, lapack_int ldx, lapack_int* k
/// );
/// ```
pub extern "C" fn lapacke_dlapmr_work(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dlapmr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clapmr_work(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     lapack_complex_float* x, lapack_int ldx, lapack_int* k
/// );
/// ```
pub extern "C" fn lapacke_clapmr_work(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_clapmr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlapmr_work(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     lapack_complex_double* x, lapack_int ldx, lapack_int* k
/// );
/// ```
pub extern "C" fn lapacke_zlapmr_work(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zlapmr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slapmt_work(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     float* x, lapack_int ldx, lapack_int* k
/// );
/// ```
pub extern "C" fn lapacke_slapmt_work(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_slapmt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlapmt_work(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     double* x, lapack_int ldx, lapack_int* k
/// );
/// ```
pub extern "C" fn lapacke_dlapmt_work(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dlapmt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clapmt_work(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     lapack_complex_float* x, lapack_int ldx, lapack_int* k
/// );
/// ```
pub extern "C" fn lapacke_clapmt_work(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_clapmt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlapmt_work(
///     int matrix_layout, lapack_logical forwrd, lapack_int m, lapack_int n, 
///     lapack_complex_double* x, lapack_int ldx, lapack_int* k
/// );
/// ```
pub extern "C" fn lapacke_zlapmt_work(
  matrix_layout : Int,
  forwrd : Lapack_logical,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  k : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zlapmt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slartgp_work(
///     float f, float g, float* cs, float* sn, float* r
/// );
/// ```
pub extern "C" fn lapacke_slartgp_work(
  f : Float,
  g : Float,
  cs : FixedArray[Float],
  sn : FixedArray[Float],
  r : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slartgp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlartgp_work(
///     double f, double g, double* cs, double* sn, double* r
/// );
/// ```
pub extern "C" fn lapacke_dlartgp_work(
  f : Double,
  g : Double,
  cs : FixedArray[Double],
  sn : FixedArray[Double],
  r : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlartgp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slartgs_work(
///     float x, float y, float sigma, float* cs, float* sn
/// );
/// ```
pub extern "C" fn lapacke_slartgs_work(
  x : Float,
  y : Float,
  sigma : Float,
  cs : FixedArray[Float],
  sn : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slartgs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlartgs_work(
///     double x, double y, double sigma, double* cs, double* sn
/// );
/// ```
pub extern "C" fn lapacke_dlartgs_work(
  x : Double,
  y : Double,
  sigma : Double,
  cs : FixedArray[Double],
  sn : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlartgs_work"

///|
///
/// ```c
/// float LAPACKE_slapy2_work(float x, float y);
/// ```
pub extern "C" fn lapacke_slapy2_work(x : Float, y : Float) -> Float = "LAPACKE_slapy2_work"

///|
///
/// ```c
/// double LAPACKE_dlapy2_work(double x, double y);
/// ```
pub extern "C" fn lapacke_dlapy2_work(x : Double, y : Double) -> Double = "LAPACKE_dlapy2_work"

///|
///
/// ```c
/// float LAPACKE_slapy3_work(float x, float y, float z);
/// ```
pub extern "C" fn lapacke_slapy3_work(x : Float, y : Float, z : Float) -> Float = "LAPACKE_slapy3_work"

///|
///
/// ```c
/// double LAPACKE_dlapy3_work(double x, double y, double z);
/// ```
pub extern "C" fn lapacke_dlapy3_work(
  x : Double,
  y : Double,
  z : Double,
) -> Double = "LAPACKE_dlapy3_work"

///|
///
/// ```c
/// float LAPACKE_slamch_work(char cmach);
/// ```
pub extern "C" fn lapacke_slamch_work(cmach : Byte) -> Float = "LAPACKE_slamch_work"

///|
///
/// ```c
/// double LAPACKE_dlamch_work(char cmach);
/// ```
pub extern "C" fn lapacke_dlamch_work(cmach : Byte) -> Double = "LAPACKE_dlamch_work"

///|
///
/// ```c
/// float LAPACKE_slangb_work(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const float* ab, lapack_int ldab, float* work
/// );
/// ```
pub extern "C" fn lapacke_slangb_work(
  matrix_layout : Int,
  norm : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  work : FixedArray[Float],
) -> Float = "LAPACKE_slangb_work"

///|
///
/// ```c
/// double LAPACKE_dlangb_work(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const double* ab, lapack_int ldab, double* work
/// );
/// ```
pub extern "C" fn lapacke_dlangb_work(
  matrix_layout : Int,
  norm : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  work : FixedArray[Double],
) -> Double = "LAPACKE_dlangb_work"

///|
///
/// ```c
/// float LAPACKE_clangb_work(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_float* ab, lapack_int ldab, float* work
/// );
/// ```
pub extern "C" fn lapacke_clangb_work(
  matrix_layout : Int,
  norm : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  work : FixedArray[Float],
) -> Float = "LAPACKE_clangb_work"

///|
///
/// ```c
/// double LAPACKE_zlangb_work(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_double* ab, lapack_int ldab, double* work
/// );
/// ```
pub extern "C" fn lapacke_zlangb_work(
  matrix_layout : Int,
  norm : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  work : FixedArray[Double],
) -> Double = "LAPACKE_zlangb_work"

///|
///
/// ```c
/// float LAPACKE_slange_work(
///     int matrix_layout, char norm, lapack_int m, lapack_int n, 
///     const float* a, lapack_int lda, float* work
/// );
/// ```
pub extern "C" fn lapacke_slange_work(
  matrix_layout : Int,
  norm : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  work : FixedArray[Float],
) -> Float = "LAPACKE_slange_work"

///|
///
/// ```c
/// double LAPACKE_dlange_work(
///     int matrix_layout, char norm, lapack_int m, lapack_int n, 
///     const double* a, lapack_int lda, double* work
/// );
/// ```
pub extern "C" fn lapacke_dlange_work(
  matrix_layout : Int,
  norm : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  work : FixedArray[Double],
) -> Double = "LAPACKE_dlange_work"

///|
///
/// ```c
/// float LAPACKE_clange_work(
///     int matrix_layout, char norm, lapack_int m, lapack_int n, 
///     const lapack_complex_float* a, lapack_int lda, float* work
/// );
/// ```
pub extern "C" fn lapacke_clange_work(
  matrix_layout : Int,
  norm : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  work : FixedArray[Float],
) -> Float = "LAPACKE_clange_work"

///|
///
/// ```c
/// double LAPACKE_zlange_work(
///     int matrix_layout, char norm, lapack_int m, lapack_int n, 
///     const lapack_complex_double* a, lapack_int lda, double* work
/// );
/// ```
pub extern "C" fn lapacke_zlange_work(
  matrix_layout : Int,
  norm : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  work : FixedArray[Double],
) -> Double = "LAPACKE_zlange_work"

///|
///
/// ```c
/// float LAPACKE_clanhe_work(
///     int matrix_layout, char norm, char uplo, lapack_int n, 
///     const lapack_complex_float* a, lapack_int lda, float* work
/// );
/// ```
pub extern "C" fn lapacke_clanhe_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  work : FixedArray[Float],
) -> Float = "LAPACKE_clanhe_work"

///|
///
/// ```c
/// double LAPACKE_zlanhe_work(
///     int matrix_layout, char norm, char uplo, lapack_int n, 
///     const lapack_complex_double* a, lapack_int lda, double* work
/// );
/// ```
pub extern "C" fn lapacke_zlanhe_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  work : FixedArray[Double],
) -> Double = "LAPACKE_zlanhe_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clacrm_work(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, const float* b, lapack_int ldb, lapack_complex_float* c, 
///     lapack_int ldc, float* work
/// );
/// ```
pub extern "C" fn lapacke_clacrm_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_clacrm_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlacrm_work(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, const double* b, lapack_int ldb, lapack_complex_double* c, 
///     lapack_int ldc, double* work
/// );
/// ```
pub extern "C" fn lapacke_zlacrm_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zlacrm_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clarcm_work(
///     int matrix_layout, lapack_int m, lapack_int n, const float* a, 
///     lapack_int lda, const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* c, 
///     lapack_int ldc, float* work
/// );
/// ```
pub extern "C" fn lapacke_clarcm_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_clarcm_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlarcm_work(
///     int matrix_layout, lapack_int m, lapack_int n, const double* a, 
///     lapack_int lda, const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* c, 
///     lapack_int ldc, double* work
/// );
/// ```
pub extern "C" fn lapacke_zlarcm_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zlarcm_work"

///|
///
/// ```c
/// float LAPACKE_slansy_work(
///     int matrix_layout, char norm, char uplo, lapack_int n, 
///     const float* a, lapack_int lda, float* work
/// );
/// ```
pub extern "C" fn lapacke_slansy_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  work : FixedArray[Float],
) -> Float = "LAPACKE_slansy_work"

///|
///
/// ```c
/// double LAPACKE_dlansy_work(
///     int matrix_layout, char norm, char uplo, lapack_int n, 
///     const double* a, lapack_int lda, double* work
/// );
/// ```
pub extern "C" fn lapacke_dlansy_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  work : FixedArray[Double],
) -> Double = "LAPACKE_dlansy_work"

///|
///
/// ```c
/// float LAPACKE_clansy_work(
///     int matrix_layout, char norm, char uplo, lapack_int n, 
///     const lapack_complex_float* a, lapack_int lda, float* work
/// );
/// ```
pub extern "C" fn lapacke_clansy_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  work : FixedArray[Float],
) -> Float = "LAPACKE_clansy_work"

///|
///
/// ```c
/// double LAPACKE_zlansy_work(
///     int matrix_layout, char norm, char uplo, lapack_int n, 
///     const lapack_complex_double* a, lapack_int lda, double* work
/// );
/// ```
pub extern "C" fn lapacke_zlansy_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  work : FixedArray[Double],
) -> Double = "LAPACKE_zlansy_work"

///|
///
/// ```c
/// float LAPACKE_slantr_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int m, lapack_int n, const float* a, lapack_int lda, float* work
/// );
/// ```
pub extern "C" fn lapacke_slantr_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  work : FixedArray[Float],
) -> Float = "LAPACKE_slantr_work"

///|
///
/// ```c
/// double LAPACKE_dlantr_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int m, lapack_int n, const double* a, lapack_int lda, double* work
/// );
/// ```
pub extern "C" fn lapacke_dlantr_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  work : FixedArray[Double],
) -> Double = "LAPACKE_dlantr_work"

///|
///
/// ```c
/// float LAPACKE_clantr_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int m, lapack_int n, const lapack_complex_float* a, lapack_int lda, float* work
/// );
/// ```
pub extern "C" fn lapacke_clantr_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  work : FixedArray[Float],
) -> Float = "LAPACKE_clantr_work"

///|
///
/// ```c
/// double LAPACKE_zlantr_work(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int m, lapack_int n, const lapack_complex_double* a, lapack_int lda, double* work
/// );
/// ```
pub extern "C" fn lapacke_zlantr_work(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  work : FixedArray[Double],
) -> Double = "LAPACKE_zlantr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slarfb_work(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     const float* v, lapack_int ldv, const float* t, lapack_int ldt, 
///     float* c, lapack_int ldc, float* work, lapack_int ldwork
/// );
/// ```
pub extern "C" fn lapacke_slarfb_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  direct : Byte,
  storev : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
  ldwork : Lapack_int,
) -> Lapack_int = "LAPACKE_slarfb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlarfb_work(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     const double* v, lapack_int ldv, const double* t, lapack_int ldt, 
///     double* c, lapack_int ldc, double* work, lapack_int ldwork
/// );
/// ```
pub extern "C" fn lapacke_dlarfb_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  direct : Byte,
  storev : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
  ldwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dlarfb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clarfb_work(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     const lapack_complex_float* v, lapack_int ldv, const lapack_complex_float* t, lapack_int ldt, 
///     lapack_complex_float* c, lapack_int ldc, lapack_complex_float* work, lapack_int ldwork
/// );
/// ```
pub extern "C" fn lapacke_clarfb_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  direct : Byte,
  storev : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
  ldwork : Lapack_int,
) -> Lapack_int = "LAPACKE_clarfb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlarfb_work(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     const lapack_complex_double* v, lapack_int ldv, const lapack_complex_double* t, lapack_int ldt, 
///     lapack_complex_double* c, lapack_int ldc, lapack_complex_double* work, lapack_int ldwork
/// );
/// ```
pub extern "C" fn lapacke_zlarfb_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  direct : Byte,
  storev : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
  ldwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zlarfb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slarfg_work(
///     lapack_int n, float* alpha, float* x, lapack_int incx, float* tau
/// );
/// ```
pub extern "C" fn lapacke_slarfg_work(
  n : Lapack_int,
  alpha : FixedArray[Float],
  x : FixedArray[Float],
  incx : Lapack_int,
  tau : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slarfg_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlarfg_work(
///     lapack_int n, double* alpha, double* x, lapack_int incx, double* tau
/// );
/// ```
pub extern "C" fn lapacke_dlarfg_work(
  n : Lapack_int,
  alpha : FixedArray[Double],
  x : FixedArray[Double],
  incx : Lapack_int,
  tau : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlarfg_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clarfg_work(
///     lapack_int n, lapack_complex_float* alpha, lapack_complex_float* x, lapack_int incx, lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_clarfg_work(
  n : Lapack_int,
  alpha : FixedArray[Float],
  x : FixedArray[Float],
  incx : Lapack_int,
  tau : FixedArray[Float],
) -> Lapack_int = "LAPACKE_clarfg_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlarfg_work(
///     lapack_int n, lapack_complex_double* alpha, lapack_complex_double* x, lapack_int incx, lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zlarfg_work(
  n : Lapack_int,
  alpha : FixedArray[Double],
  x : FixedArray[Double],
  incx : Lapack_int,
  tau : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zlarfg_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slarft_work(
///     int matrix_layout, char direct, char storev, lapack_int n, 
///     lapack_int k, const float* v, lapack_int ldv, const float* tau, 
///     float* t, lapack_int ldt
/// );
/// ```
pub extern "C" fn lapacke_slarft_work(
  matrix_layout : Int,
  direct : Byte,
  storev : Byte,
  n : Lapack_int,
  k : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  tau : FixedArray[Float],
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_slarft_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlarft_work(
///     int matrix_layout, char direct, char storev, lapack_int n, 
///     lapack_int k, const double* v, lapack_int ldv, const double* tau, 
///     double* t, lapack_int ldt
/// );
/// ```
pub extern "C" fn lapacke_dlarft_work(
  matrix_layout : Int,
  direct : Byte,
  storev : Byte,
  n : Lapack_int,
  k : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  tau : FixedArray[Double],
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_dlarft_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clarft_work(
///     int matrix_layout, char direct, char storev, lapack_int n, 
///     lapack_int k, const lapack_complex_float* v, lapack_int ldv, const lapack_complex_float* tau, 
///     lapack_complex_float* t, lapack_int ldt
/// );
/// ```
pub extern "C" fn lapacke_clarft_work(
  matrix_layout : Int,
  direct : Byte,
  storev : Byte,
  n : Lapack_int,
  k : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  tau : FixedArray[Float],
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_clarft_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlarft_work(
///     int matrix_layout, char direct, char storev, lapack_int n, 
///     lapack_int k, const lapack_complex_double* v, lapack_int ldv, const lapack_complex_double* tau, 
///     lapack_complex_double* t, lapack_int ldt
/// );
/// ```
pub extern "C" fn lapacke_zlarft_work(
  matrix_layout : Int,
  direct : Byte,
  storev : Byte,
  n : Lapack_int,
  k : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  tau : FixedArray[Double],
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_zlarft_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slarfx_work(
///     int matrix_layout, char side, lapack_int m, lapack_int n, 
///     const float* v, float tau, float* c, lapack_int ldc, float* work
/// );
/// ```
pub extern "C" fn lapacke_slarfx_work(
  matrix_layout : Int,
  side : Byte,
  m : Lapack_int,
  n : Lapack_int,
  v : FixedArray[Float],
  tau : Float,
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slarfx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlarfx_work(
///     int matrix_layout, char side, lapack_int m, lapack_int n, 
///     const double* v, double tau, double* c, lapack_int ldc, double* work
/// );
/// ```
pub extern "C" fn lapacke_dlarfx_work(
  matrix_layout : Int,
  side : Byte,
  m : Lapack_int,
  n : Lapack_int,
  v : FixedArray[Double],
  tau : Double,
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlarfx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clarfx_work(
///     int matrix_layout, char side, lapack_int m, lapack_int n, 
///     const lapack_complex_float* v, lapack_complex_float tau, lapack_complex_float* c, lapack_int ldc, lapack_complex_float* work
/// );
/// ```
pub extern "C" fn lapacke_clarfx_work(
  matrix_layout : Int,
  side : Byte,
  m : Lapack_int,
  n : Lapack_int,
  v : FixedArray[Float],
  tau : Lapack_complex_float,
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_clarfx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlarfx_work(
///     int matrix_layout, char side, lapack_int m, lapack_int n, 
///     const lapack_complex_double* v, lapack_complex_double tau, lapack_complex_double* c, lapack_int ldc, lapack_complex_double* work
/// );
/// ```
pub extern "C" fn lapacke_zlarfx_work(
  matrix_layout : Int,
  side : Byte,
  m : Lapack_int,
  n : Lapack_int,
  v : FixedArray[Double],
  tau : Lapack_complex_double,
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zlarfx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slarnv_work(
///     lapack_int idist, lapack_int* iseed, lapack_int n, float* x
/// );
/// ```
pub extern "C" fn lapacke_slarnv_work(
  idist : Lapack_int,
  iseed : FixedArray[Int],
  n : Lapack_int,
  x : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slarnv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlarnv_work(
///     lapack_int idist, lapack_int* iseed, lapack_int n, double* x
/// );
/// ```
pub extern "C" fn lapacke_dlarnv_work(
  idist : Lapack_int,
  iseed : FixedArray[Int],
  n : Lapack_int,
  x : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlarnv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clarnv_work(
///     lapack_int idist, lapack_int* iseed, lapack_int n, lapack_complex_float* x
/// );
/// ```
pub extern "C" fn lapacke_clarnv_work(
  idist : Lapack_int,
  iseed : FixedArray[Int],
  n : Lapack_int,
  x : FixedArray[Float],
) -> Lapack_int = "LAPACKE_clarnv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlarnv_work(
///     lapack_int idist, lapack_int* iseed, lapack_int n, lapack_complex_double* x
/// );
/// ```
pub extern "C" fn lapacke_zlarnv_work(
  idist : Lapack_int,
  iseed : FixedArray[Int],
  n : Lapack_int,
  x : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zlarnv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slascl_work(
///     int matrix_layout, char type, lapack_int kl, lapack_int ku, 
///     float cfrom, float cto, lapack_int m, lapack_int n, 
///     float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_slascl_work(
  matrix_layout : Int,
  ty : Byte,
  kl : Lapack_int,
  ku : Lapack_int,
  cfrom : Float,
  cto : Float,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_slascl_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlascl_work(
///     int matrix_layout, char type, lapack_int kl, lapack_int ku, 
///     double cfrom, double cto, lapack_int m, lapack_int n, 
///     double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_dlascl_work(
  matrix_layout : Int,
  ty : Byte,
  kl : Lapack_int,
  ku : Lapack_int,
  cfrom : Double,
  cto : Double,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_dlascl_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clascl_work(
///     int matrix_layout, char type, lapack_int kl, lapack_int ku, 
///     float cfrom, float cto, lapack_int m, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_clascl_work(
  matrix_layout : Int,
  ty : Byte,
  kl : Lapack_int,
  ku : Lapack_int,
  cfrom : Float,
  cto : Float,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_clascl_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlascl_work(
///     int matrix_layout, char type, lapack_int kl, lapack_int ku, 
///     double cfrom, double cto, lapack_int m, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_zlascl_work(
  matrix_layout : Int,
  ty : Byte,
  kl : Lapack_int,
  ku : Lapack_int,
  cfrom : Double,
  cto : Double,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_zlascl_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slaset_work(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     float alpha, float beta, float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_slaset_work(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  alpha : Float,
  beta : Float,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_slaset_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlaset_work(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     double alpha, double beta, double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_dlaset_work(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  alpha : Double,
  beta : Double,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_dlaset_work"

///|
///
/// ```c
/// lapack_int LAPACKE_claset_work(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     lapack_complex_float alpha, lapack_complex_float beta, lapack_complex_float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_claset_work(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  alpha : Lapack_complex_float,
  beta : Lapack_complex_float,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_claset_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlaset_work(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     lapack_complex_double alpha, lapack_complex_double beta, lapack_complex_double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_zlaset_work(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  alpha : Lapack_complex_double,
  beta : Lapack_complex_double,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_zlaset_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slasrt_work(char id, lapack_int n, float* d);
/// ```
pub extern "C" fn lapacke_slasrt_work(
  id : Byte,
  n : Lapack_int,
  d : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slasrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlasrt_work(char id, lapack_int n, double* d);
/// ```
pub extern "C" fn lapacke_dlasrt_work(
  id : Byte,
  n : Lapack_int,
  d : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlasrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slassq_work(
///     lapack_int n, float* x, lapack_int incx, float* scale, float* sumsq
/// );
/// ```
pub extern "C" fn lapacke_slassq_work(
  n : Lapack_int,
  x : FixedArray[Float],
  incx : Lapack_int,
  scale : FixedArray[Float],
  sumsq : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slassq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlassq_work(
///     lapack_int n, double* x, lapack_int incx, double* scale, double* sumsq
/// );
/// ```
pub extern "C" fn lapacke_dlassq_work(
  n : Lapack_int,
  x : FixedArray[Double],
  incx : Lapack_int,
  scale : FixedArray[Double],
  sumsq : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlassq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_classq_work(
///     lapack_int n, lapack_complex_float* x, lapack_int incx, float* scale, float* sumsq
/// );
/// ```
pub extern "C" fn lapacke_classq_work(
  n : Lapack_int,
  x : FixedArray[Float],
  incx : Lapack_int,
  scale : FixedArray[Float],
  sumsq : FixedArray[Float],
) -> Lapack_int = "LAPACKE_classq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlassq_work(
///     lapack_int n, lapack_complex_double* x, lapack_int incx, double* scale, double* sumsq
/// );
/// ```
pub extern "C" fn lapacke_zlassq_work(
  n : Lapack_int,
  x : FixedArray[Double],
  incx : Lapack_int,
  scale : FixedArray[Double],
  sumsq : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zlassq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slaswp_work(
///     int matrix_layout, lapack_int n, float* a, lapack_int lda, 
///     lapack_int k1, lapack_int k2, const lapack_int* ipiv, lapack_int incx
/// );
/// ```
pub extern "C" fn lapacke_slaswp_work(
  matrix_layout : Int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  k1 : Lapack_int,
  k2 : Lapack_int,
  ipiv : FixedArray[Int],
  incx : Lapack_int,
) -> Lapack_int = "LAPACKE_slaswp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlaswp_work(
///     int matrix_layout, lapack_int n, double* a, lapack_int lda, 
///     lapack_int k1, lapack_int k2, const lapack_int* ipiv, lapack_int incx
/// );
/// ```
pub extern "C" fn lapacke_dlaswp_work(
  matrix_layout : Int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  k1 : Lapack_int,
  k2 : Lapack_int,
  ipiv : FixedArray[Int],
  incx : Lapack_int,
) -> Lapack_int = "LAPACKE_dlaswp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_claswp_work(
///     int matrix_layout, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_int k1, lapack_int k2, const lapack_int* ipiv, lapack_int incx
/// );
/// ```
pub extern "C" fn lapacke_claswp_work(
  matrix_layout : Int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  k1 : Lapack_int,
  k2 : Lapack_int,
  ipiv : FixedArray[Int],
  incx : Lapack_int,
) -> Lapack_int = "LAPACKE_claswp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlaswp_work(
///     int matrix_layout, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_int k1, lapack_int k2, const lapack_int* ipiv, lapack_int incx
/// );
/// ```
pub extern "C" fn lapacke_zlaswp_work(
  matrix_layout : Int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  k1 : Lapack_int,
  k2 : Lapack_int,
  ipiv : FixedArray[Int],
  incx : Lapack_int,
) -> Lapack_int = "LAPACKE_zlaswp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slatms_work(
///     int matrix_layout, lapack_int m, lapack_int n, char dist, 
///     lapack_int* iseed, char sym, float* d, lapack_int mode, 
///     float cond, float dmax, lapack_int kl, lapack_int ku, 
///     char pack, float* a, lapack_int lda, float* work
/// );
/// ```
pub extern "C" fn lapacke_slatms_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  dist : Byte,
  iseed : FixedArray[Int],
  sym : Byte,
  d : FixedArray[Float],
  mode : Lapack_int,
  cond : Float,
  dmax : Float,
  kl : Lapack_int,
  ku : Lapack_int,
  pack : Byte,
  a : FixedArray[Float],
  lda : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slatms_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlatms_work(
///     int matrix_layout, lapack_int m, lapack_int n, char dist, 
///     lapack_int* iseed, char sym, double* d, lapack_int mode, 
///     double cond, double dmax, lapack_int kl, lapack_int ku, 
///     char pack, double* a, lapack_int lda, double* work
/// );
/// ```
pub extern "C" fn lapacke_dlatms_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  dist : Byte,
  iseed : FixedArray[Int],
  sym : Byte,
  d : FixedArray[Double],
  mode : Lapack_int,
  cond : Double,
  dmax : Double,
  kl : Lapack_int,
  ku : Lapack_int,
  pack : Byte,
  a : FixedArray[Double],
  lda : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlatms_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clatms_work(
///     int matrix_layout, lapack_int m, lapack_int n, char dist, 
///     lapack_int* iseed, char sym, float* d, lapack_int mode, 
///     float cond, float dmax, lapack_int kl, lapack_int ku, 
///     char pack, lapack_complex_float* a, lapack_int lda, lapack_complex_float* work
/// );
/// ```
pub extern "C" fn lapacke_clatms_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  dist : Byte,
  iseed : FixedArray[Int],
  sym : Byte,
  d : FixedArray[Float],
  mode : Lapack_int,
  cond : Float,
  dmax : Float,
  kl : Lapack_int,
  ku : Lapack_int,
  pack : Byte,
  a : FixedArray[Float],
  lda : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_clatms_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlatms_work(
///     int matrix_layout, lapack_int m, lapack_int n, char dist, 
///     lapack_int* iseed, char sym, double* d, lapack_int mode, 
///     double cond, double dmax, lapack_int kl, lapack_int ku, 
///     char pack, lapack_complex_double* a, lapack_int lda, lapack_complex_double* work
/// );
/// ```
pub extern "C" fn lapacke_zlatms_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  dist : Byte,
  iseed : FixedArray[Int],
  sym : Byte,
  d : FixedArray[Double],
  mode : Lapack_int,
  cond : Double,
  dmax : Double,
  kl : Lapack_int,
  ku : Lapack_int,
  pack : Byte,
  a : FixedArray[Double],
  lda : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zlatms_work"

///|
///
/// ```c
/// lapack_int LAPACKE_slauum_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_slauum_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_slauum_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dlauum_work(
///     int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_dlauum_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_dlauum_work"

///|
///
/// ```c
/// lapack_int LAPACKE_clauum_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_clauum_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_clauum_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zlauum_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_zlauum_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_zlauum_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sopgtr_work(
///     int matrix_layout, char uplo, lapack_int n, const float* ap, 
///     const float* tau, float* q, lapack_int ldq, float* work
/// );
/// ```
pub extern "C" fn lapacke_sopgtr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  tau : FixedArray[Float],
  q : FixedArray[Float],
  ldq : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sopgtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dopgtr_work(
///     int matrix_layout, char uplo, lapack_int n, const double* ap, 
///     const double* tau, double* q, lapack_int ldq, double* work
/// );
/// ```
pub extern "C" fn lapacke_dopgtr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  tau : FixedArray[Double],
  q : FixedArray[Double],
  ldq : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dopgtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sopmtr_work(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const float* ap, const float* tau, 
///     float* c, lapack_int ldc, float* work
/// );
/// ```
pub extern "C" fn lapacke_sopmtr_work(
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
) -> Lapack_int = "LAPACKE_sopmtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dopmtr_work(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const double* ap, const double* tau, 
///     double* c, lapack_int ldc, double* work
/// );
/// ```
pub extern "C" fn lapacke_dopmtr_work(
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
) -> Lapack_int = "LAPACKE_dopmtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sorgbr_work(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int k, float* a, lapack_int lda, const float* tau, 
///     float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sorgbr_work(
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
) -> Lapack_int = "LAPACKE_sorgbr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dorgbr_work(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int k, double* a, lapack_int lda, const double* tau, 
///     double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dorgbr_work(
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
) -> Lapack_int = "LAPACKE_dorgbr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sorghr_work(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     float* a, lapack_int lda, const float* tau, float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sorghr_work(
  matrix_layout : Int,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sorghr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dorghr_work(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     double* a, lapack_int lda, const double* tau, double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dorghr_work(
  matrix_layout : Int,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dorghr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sorglq_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     float* a, lapack_int lda, const float* tau, float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sorglq_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sorglq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dorglq_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     double* a, lapack_int lda, const double* tau, double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dorglq_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dorglq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sorgql_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     float* a, lapack_int lda, const float* tau, float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sorgql_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sorgql_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dorgql_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     double* a, lapack_int lda, const double* tau, double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dorgql_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dorgql_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sorgqr_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     float* a, lapack_int lda, const float* tau, float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sorgqr_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sorgqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dorgqr_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     double* a, lapack_int lda, const double* tau, double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dorgqr_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dorgqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sorgrq_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     float* a, lapack_int lda, const float* tau, float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sorgrq_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sorgrq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dorgrq_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     double* a, lapack_int lda, const double* tau, double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dorgrq_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dorgrq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sorgtr_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, const float* tau, float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sorgtr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sorgtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dorgtr_work(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, const double* tau, double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dorgtr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dorgtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sorgtsqr_row_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int mb, 
///     lapack_int nb, float* a, lapack_int lda, const float* t, 
///     lapack_int ldt, float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sorgtsqr_row_work(
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
) -> Lapack_int = "LAPACKE_sorgtsqr_row_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dorgtsqr_row_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int mb, 
///     lapack_int nb, double* a, lapack_int lda, const double* t, 
///     lapack_int ldt, double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dorgtsqr_row_work(
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
) -> Lapack_int = "LAPACKE_dorgtsqr_row_work"
