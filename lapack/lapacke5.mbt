///|
///
/// ```c
/// lapack_int LAPACKE_sgbrfsx_work(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, const float* ab, 
///     lapack_int ldab, const float* afb, lapack_int ldafb, const lapack_int* ipiv, 
///     const float* r, const float* c, const float* b, lapack_int ldb, 
///     float* x, lapack_int ldx, float* rcond, float* berr, 
///     lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, 
///     float* params, float* work, lapack_int* iwork
/// );
/// ```
#borrow(ab, afb, ipiv, r, c, b, x, rcond, berr, err_bnds_norm, err_bnds_comp, params, work, iwork)
pub extern "C" fn lapacke_sgbrfsx_work(
  matrix_layout : Int,
  trans : Byte,
  equed : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  afb : FixedArray[Float],
  ldafb : Lapack_int,
  ipiv : FixedArray[Int],
  r : FixedArray[Float],
  c : FixedArray[Float],
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
) -> Lapack_int = "LAPACKE_sgbrfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbrfsx_work(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, const double* ab, 
///     lapack_int ldab, const double* afb, lapack_int ldafb, const lapack_int* ipiv, 
///     const double* r, const double* c, const double* b, lapack_int ldb, 
///     double* x, lapack_int ldx, double* rcond, double* berr, 
///     lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, 
///     double* params, double* work, lapack_int* iwork
/// );
/// ```
#borrow(ab, afb, ipiv, r, c, b, x, rcond, berr, err_bnds_norm, err_bnds_comp, params, work, iwork)
pub extern "C" fn lapacke_dgbrfsx_work(
  matrix_layout : Int,
  trans : Byte,
  equed : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  afb : FixedArray[Double],
  ldafb : Lapack_int,
  ipiv : FixedArray[Int],
  r : FixedArray[Double],
  c : FixedArray[Double],
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
) -> Lapack_int = "LAPACKE_dgbrfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbrfsx_work(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, const lapack_complex_float* ab, 
///     lapack_int ldab, const lapack_complex_float* afb, lapack_int ldafb, const lapack_int* ipiv, 
///     const float* r, const float* c, const lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* x, lapack_int ldx, float* rcond, float* berr, 
///     lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, 
///     float* params, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(ab, afb, ipiv, r, c, b, x, rcond, berr, err_bnds_norm, err_bnds_comp, params, work, rwork)
pub extern "C" fn lapacke_cgbrfsx_work(
  matrix_layout : Int,
  trans : Byte,
  equed : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  afb : FixedArray[Float],
  ldafb : Lapack_int,
  ipiv : FixedArray[Int],
  r : FixedArray[Float],
  c : FixedArray[Float],
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
) -> Lapack_int = "LAPACKE_cgbrfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbrfsx_work(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, const lapack_complex_double* ab, 
///     lapack_int ldab, const lapack_complex_double* afb, lapack_int ldafb, const lapack_int* ipiv, 
///     const double* r, const double* c, const lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* x, lapack_int ldx, double* rcond, double* berr, 
///     lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, 
///     double* params, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(ab, afb, ipiv, r, c, b, x, rcond, berr, err_bnds_norm, err_bnds_comp, params, work, rwork)
pub extern "C" fn lapacke_zgbrfsx_work(
  matrix_layout : Int,
  trans : Byte,
  equed : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  afb : FixedArray[Double],
  ldafb : Lapack_int,
  ipiv : FixedArray[Int],
  r : FixedArray[Double],
  c : FixedArray[Double],
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
) -> Lapack_int = "LAPACKE_zgbrfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbsv_work(
///     int matrix_layout, lapack_int n, lapack_int kl, lapack_int ku, 
///     lapack_int nrhs, float* ab, lapack_int ldab, lapack_int* ipiv, 
///     float* b, lapack_int ldb
/// );
/// ```
#borrow(ab, ipiv, b)
pub extern "C" fn lapacke_sgbsv_work(
  matrix_layout : Int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_sgbsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbsv_work(
///     int matrix_layout, lapack_int n, lapack_int kl, lapack_int ku, 
///     lapack_int nrhs, double* ab, lapack_int ldab, lapack_int* ipiv, 
///     double* b, lapack_int ldb
/// );
/// ```
#borrow(ab, ipiv, b)
pub extern "C" fn lapacke_dgbsv_work(
  matrix_layout : Int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dgbsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbsv_work(
///     int matrix_layout, lapack_int n, lapack_int kl, lapack_int ku, 
///     lapack_int nrhs, lapack_complex_float* ab, lapack_int ldab, lapack_int* ipiv, 
///     lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(ab, ipiv, b)
pub extern "C" fn lapacke_cgbsv_work(
  matrix_layout : Int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_cgbsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbsv_work(
///     int matrix_layout, lapack_int n, lapack_int kl, lapack_int ku, 
///     lapack_int nrhs, lapack_complex_double* ab, lapack_int ldab, lapack_int* ipiv, 
///     lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(ab, ipiv, b)
pub extern "C" fn lapacke_zgbsv_work(
  matrix_layout : Int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zgbsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbsvx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, float* ab, 
///     lapack_int ldab, float* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, float* r, float* c, float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* rcond, 
///     float* ferr, float* berr, float* work, lapack_int* iwork
/// );
/// ```
#borrow(ab, afb, ipiv, equed, r, c, b, x, rcond, ferr, berr, work, iwork)
pub extern "C" fn lapacke_sgbsvx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  afb : FixedArray[Float],
  ldafb : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Float],
  c : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  rcond : FixedArray[Float],
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgbsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbsvx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, double* ab, 
///     lapack_int ldab, double* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, double* r, double* c, double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* rcond, 
///     double* ferr, double* berr, double* work, lapack_int* iwork
/// );
/// ```
#borrow(ab, afb, ipiv, equed, r, c, b, x, rcond, ferr, berr, work, iwork)
pub extern "C" fn lapacke_dgbsvx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  afb : FixedArray[Double],
  ldafb : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Double],
  c : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  rcond : FixedArray[Double],
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgbsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbsvx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, lapack_complex_float* ab, 
///     lapack_int ldab, lapack_complex_float* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, float* r, float* c, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond, 
///     float* ferr, float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(ab, afb, ipiv, equed, r, c, b, x, rcond, ferr, berr, work, rwork)
pub extern "C" fn lapacke_cgbsvx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  afb : FixedArray[Float],
  ldafb : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Float],
  c : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  rcond : FixedArray[Float],
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgbsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbsvx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, lapack_complex_double* ab, 
///     lapack_int ldab, lapack_complex_double* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, double* r, double* c, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond, 
///     double* ferr, double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(ab, afb, ipiv, equed, r, c, b, x, rcond, ferr, berr, work, rwork)
pub extern "C" fn lapacke_zgbsvx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  afb : FixedArray[Double],
  ldafb : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Double],
  c : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  rcond : FixedArray[Double],
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgbsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbsvxx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, float* ab, 
///     lapack_int ldab, float* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, float* r, float* c, float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* rcond, 
///     float* rpvgrw, float* berr, lapack_int n_err_bnds, float* err_bnds_norm, 
///     float* err_bnds_comp, lapack_int nparams, float* params, float* work, lapack_int* iwork
/// );
/// ```
#borrow(ab, afb, ipiv, equed, r, c, b, x, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, params, work, iwork)
pub extern "C" fn lapacke_sgbsvxx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  afb : FixedArray[Float],
  ldafb : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Float],
  c : FixedArray[Float],
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
) -> Lapack_int = "LAPACKE_sgbsvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbsvxx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, double* ab, 
///     lapack_int ldab, double* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, double* r, double* c, double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* rcond, 
///     double* rpvgrw, double* berr, lapack_int n_err_bnds, double* err_bnds_norm, 
///     double* err_bnds_comp, lapack_int nparams, double* params, double* work, lapack_int* iwork
/// );
/// ```
#borrow(ab, afb, ipiv, equed, r, c, b, x, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, params, work, iwork)
pub extern "C" fn lapacke_dgbsvxx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  afb : FixedArray[Double],
  ldafb : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Double],
  c : FixedArray[Double],
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
) -> Lapack_int = "LAPACKE_dgbsvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbsvxx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, lapack_complex_float* ab, 
///     lapack_int ldab, lapack_complex_float* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, float* r, float* c, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond, 
///     float* rpvgrw, float* berr, lapack_int n_err_bnds, float* err_bnds_norm, 
///     float* err_bnds_comp, lapack_int nparams, float* params, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(ab, afb, ipiv, equed, r, c, b, x, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, params, work, rwork)
pub extern "C" fn lapacke_cgbsvxx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  afb : FixedArray[Float],
  ldafb : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Float],
  c : FixedArray[Float],
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
) -> Lapack_int = "LAPACKE_cgbsvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbsvxx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, lapack_complex_double* ab, 
///     lapack_int ldab, lapack_complex_double* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, double* r, double* c, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond, 
///     double* rpvgrw, double* berr, lapack_int n_err_bnds, double* err_bnds_norm, 
///     double* err_bnds_comp, lapack_int nparams, double* params, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(ab, afb, ipiv, equed, r, c, b, x, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, params, work, rwork)
pub extern "C" fn lapacke_zgbsvxx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  afb : FixedArray[Double],
  ldafb : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Double],
  c : FixedArray[Double],
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
) -> Lapack_int = "LAPACKE_zgbsvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbtrf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, float* ab, lapack_int ldab, lapack_int* ipiv
/// );
/// ```
#borrow(ab, ipiv)
pub extern "C" fn lapacke_sgbtrf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgbtrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbtrf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, double* ab, lapack_int ldab, lapack_int* ipiv
/// );
/// ```
#borrow(ab, ipiv)
pub extern "C" fn lapacke_dgbtrf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgbtrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbtrf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_complex_float* ab, lapack_int ldab, lapack_int* ipiv
/// );
/// ```
#borrow(ab, ipiv)
pub extern "C" fn lapacke_cgbtrf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cgbtrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbtrf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_complex_double* ab, lapack_int ldab, lapack_int* ipiv
/// );
/// ```
#borrow(ab, ipiv)
pub extern "C" fn lapacke_zgbtrf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zgbtrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbtrs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const float* ab, lapack_int ldab, 
///     const lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
#borrow(ab, ipiv, b)
pub extern "C" fn lapacke_sgbtrs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_sgbtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbtrs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const double* ab, lapack_int ldab, 
///     const lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
#borrow(ab, ipiv, b)
pub extern "C" fn lapacke_dgbtrs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dgbtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbtrs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const lapack_complex_float* ab, lapack_int ldab, 
///     const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(ab, ipiv, b)
pub extern "C" fn lapacke_cgbtrs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_cgbtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbtrs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const lapack_complex_double* ab, lapack_int ldab, 
///     const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(ab, ipiv, b)
pub extern "C" fn lapacke_zgbtrs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  nrhs : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zgbtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgebak_work(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const float* scale, lapack_int m, 
///     float* v, lapack_int ldv
/// );
/// ```
#borrow(scale, v)
pub extern "C" fn lapacke_sgebak_work(
  matrix_layout : Int,
  job : Byte,
  side : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  scale : FixedArray[Float],
  m : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
) -> Lapack_int = "LAPACKE_sgebak_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgebak_work(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const double* scale, lapack_int m, 
///     double* v, lapack_int ldv
/// );
/// ```
#borrow(scale, v)
pub extern "C" fn lapacke_dgebak_work(
  matrix_layout : Int,
  job : Byte,
  side : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  scale : FixedArray[Double],
  m : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
) -> Lapack_int = "LAPACKE_dgebak_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgebak_work(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const float* scale, lapack_int m, 
///     lapack_complex_float* v, lapack_int ldv
/// );
/// ```
#borrow(scale, v)
pub extern "C" fn lapacke_cgebak_work(
  matrix_layout : Int,
  job : Byte,
  side : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  scale : FixedArray[Float],
  m : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
) -> Lapack_int = "LAPACKE_cgebak_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgebak_work(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const double* scale, lapack_int m, 
///     lapack_complex_double* v, lapack_int ldv
/// );
/// ```
#borrow(scale, v)
pub extern "C" fn lapacke_zgebak_work(
  matrix_layout : Int,
  job : Byte,
  side : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  scale : FixedArray[Double],
  m : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
) -> Lapack_int = "LAPACKE_zgebak_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgebal_work(
///     int matrix_layout, char job, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* ilo, lapack_int* ihi, float* scale
/// );
/// ```
#borrow(a, ilo, ihi, scale)
pub extern "C" fn lapacke_sgebal_work(
  matrix_layout : Int,
  job : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  scale : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sgebal_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgebal_work(
///     int matrix_layout, char job, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* ilo, lapack_int* ihi, double* scale
/// );
/// ```
#borrow(a, ilo, ihi, scale)
pub extern "C" fn lapacke_dgebal_work(
  matrix_layout : Int,
  job : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  scale : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dgebal_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgebal_work(
///     int matrix_layout, char job, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ilo, lapack_int* ihi, float* scale
/// );
/// ```
#borrow(a, ilo, ihi, scale)
pub extern "C" fn lapacke_cgebal_work(
  matrix_layout : Int,
  job : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  scale : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgebal_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgebal_work(
///     int matrix_layout, char job, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ilo, lapack_int* ihi, double* scale
/// );
/// ```
#borrow(a, ilo, ihi, scale)
pub extern "C" fn lapacke_zgebal_work(
  matrix_layout : Int,
  job : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  scale : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgebal_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgebrd_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* d, float* e, float* tauq, 
///     float* taup, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, d, e, tauq, taup, work)
pub extern "C" fn lapacke_sgebrd_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  d : FixedArray[Float],
  e : FixedArray[Float],
  tauq : FixedArray[Float],
  taup : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgebrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgebrd_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* d, double* e, double* tauq, 
///     double* taup, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, d, e, tauq, taup, work)
pub extern "C" fn lapacke_dgebrd_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  d : FixedArray[Double],
  e : FixedArray[Double],
  tauq : FixedArray[Double],
  taup : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgebrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgebrd_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, float* d, float* e, lapack_complex_float* tauq, 
///     lapack_complex_float* taup, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, d, e, tauq, taup, work)
pub extern "C" fn lapacke_cgebrd_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  d : FixedArray[Float],
  e : FixedArray[Float],
  tauq : FixedArray[Float],
  taup : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cgebrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgebrd_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, double* d, double* e, lapack_complex_double* tauq, 
///     lapack_complex_double* taup, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, d, e, tauq, taup, work)
pub extern "C" fn lapacke_zgebrd_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  d : FixedArray[Double],
  e : FixedArray[Double],
  tauq : FixedArray[Double],
  taup : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zgebrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgecon_work(
///     int matrix_layout, char norm, lapack_int n, const float* a, 
///     lapack_int lda, float anorm, float* rcond, float* work, lapack_int* iwork
/// );
/// ```
#borrow(a, rcond, work, iwork)
pub extern "C" fn lapacke_sgecon_work(
  matrix_layout : Int,
  norm : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  anorm : Float,
  rcond : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgecon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgecon_work(
///     int matrix_layout, char norm, lapack_int n, const double* a, 
///     lapack_int lda, double anorm, double* rcond, double* work, lapack_int* iwork
/// );
/// ```
#borrow(a, rcond, work, iwork)
pub extern "C" fn lapacke_dgecon_work(
  matrix_layout : Int,
  norm : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  anorm : Double,
  rcond : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgecon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgecon_work(
///     int matrix_layout, char norm, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, float anorm, float* rcond, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(a, rcond, work, rwork)
pub extern "C" fn lapacke_cgecon_work(
  matrix_layout : Int,
  norm : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  anorm : Float,
  rcond : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgecon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgecon_work(
///     int matrix_layout, char norm, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, double anorm, double* rcond, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(a, rcond, work, rwork)
pub extern "C" fn lapacke_zgecon_work(
  matrix_layout : Int,
  norm : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  anorm : Double,
  rcond : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgecon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeequ_work(
///     int matrix_layout, lapack_int m, lapack_int n, const float* a, 
///     lapack_int lda, float* r, float* c, float* rowcnd, 
///     float* colcnd, float* amax
/// );
/// ```
#borrow(a, r, c, rowcnd, colcnd, amax)
pub extern "C" fn lapacke_sgeequ_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  r : FixedArray[Float],
  c : FixedArray[Float],
  rowcnd : FixedArray[Float],
  colcnd : FixedArray[Float],
  amax : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sgeequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeequ_work(
///     int matrix_layout, lapack_int m, lapack_int n, const double* a, 
///     lapack_int lda, double* r, double* c, double* rowcnd, 
///     double* colcnd, double* amax
/// );
/// ```
#borrow(a, r, c, rowcnd, colcnd, amax)
pub extern "C" fn lapacke_dgeequ_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  r : FixedArray[Double],
  c : FixedArray[Double],
  rowcnd : FixedArray[Double],
  colcnd : FixedArray[Double],
  amax : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dgeequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeequ_work(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, float* r, float* c, float* rowcnd, 
///     float* colcnd, float* amax
/// );
/// ```
#borrow(a, r, c, rowcnd, colcnd, amax)
pub extern "C" fn lapacke_cgeequ_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  r : FixedArray[Float],
  c : FixedArray[Float],
  rowcnd : FixedArray[Float],
  colcnd : FixedArray[Float],
  amax : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgeequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeequ_work(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, double* r, double* c, double* rowcnd, 
///     double* colcnd, double* amax
/// );
/// ```
#borrow(a, r, c, rowcnd, colcnd, amax)
pub extern "C" fn lapacke_zgeequ_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  r : FixedArray[Double],
  c : FixedArray[Double],
  rowcnd : FixedArray[Double],
  colcnd : FixedArray[Double],
  amax : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgeequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeequb_work(
///     int matrix_layout, lapack_int m, lapack_int n, const float* a, 
///     lapack_int lda, float* r, float* c, float* rowcnd, 
///     float* colcnd, float* amax
/// );
/// ```
#borrow(a, r, c, rowcnd, colcnd, amax)
pub extern "C" fn lapacke_sgeequb_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  r : FixedArray[Float],
  c : FixedArray[Float],
  rowcnd : FixedArray[Float],
  colcnd : FixedArray[Float],
  amax : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sgeequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeequb_work(
///     int matrix_layout, lapack_int m, lapack_int n, const double* a, 
///     lapack_int lda, double* r, double* c, double* rowcnd, 
///     double* colcnd, double* amax
/// );
/// ```
#borrow(a, r, c, rowcnd, colcnd, amax)
pub extern "C" fn lapacke_dgeequb_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  r : FixedArray[Double],
  c : FixedArray[Double],
  rowcnd : FixedArray[Double],
  colcnd : FixedArray[Double],
  amax : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dgeequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeequb_work(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, float* r, float* c, float* rowcnd, 
///     float* colcnd, float* amax
/// );
/// ```
#borrow(a, r, c, rowcnd, colcnd, amax)
pub extern "C" fn lapacke_cgeequb_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  r : FixedArray[Float],
  c : FixedArray[Float],
  rowcnd : FixedArray[Float],
  colcnd : FixedArray[Float],
  amax : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgeequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeequb_work(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, double* r, double* c, double* rowcnd, 
///     double* colcnd, double* amax
/// );
/// ```
#borrow(a, r, c, rowcnd, colcnd, amax)
pub extern "C" fn lapacke_zgeequb_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  r : FixedArray[Double],
  c : FixedArray[Double],
  rowcnd : FixedArray[Double],
  colcnd : FixedArray[Double],
  amax : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgeequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgees_work(
///     int matrix_layout, char jobvs, char sort, LAPACK_S_SELECT2 select, 
///     lapack_int n, float* a, lapack_int lda, lapack_int* sdim, 
///     float* wr, float* wi, float* vs, lapack_int ldvs, 
///     float* work, lapack_int lwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, sdim, wr, wi, vs, work, bwork)
pub extern "C" fn lapacke_sgees_work(
  matrix_layout : Int,
  jobvs : Byte,
  sort : Byte,
  select : Lapack_s_select2,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  sdim : FixedArray[Int],
  wr : FixedArray[Float],
  wi : FixedArray[Float],
  vs : FixedArray[Float],
  ldvs : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_sgees_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgees_work(
///     int matrix_layout, char jobvs, char sort, LAPACK_D_SELECT2 select, 
///     lapack_int n, double* a, lapack_int lda, lapack_int* sdim, 
///     double* wr, double* wi, double* vs, lapack_int ldvs, 
///     double* work, lapack_int lwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, sdim, wr, wi, vs, work, bwork)
pub extern "C" fn lapacke_dgees_work(
  matrix_layout : Int,
  jobvs : Byte,
  sort : Byte,
  select : Lapack_d_select2,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  sdim : FixedArray[Int],
  wr : FixedArray[Double],
  wi : FixedArray[Double],
  vs : FixedArray[Double],
  ldvs : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_dgees_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgees_work(
///     int matrix_layout, char jobvs, char sort, LAPACK_C_SELECT1 select, 
///     lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_int* sdim, 
///     lapack_complex_float* w, lapack_complex_float* vs, lapack_int ldvs, lapack_complex_float* work, 
///     lapack_int lwork, float* rwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, sdim, w, vs, work, rwork, bwork)
pub extern "C" fn lapacke_cgees_work(
  matrix_layout : Int,
  jobvs : Byte,
  sort : Byte,
  select : Lapack_c_select1,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  sdim : FixedArray[Int],
  w : FixedArray[Float],
  vs : FixedArray[Float],
  ldvs : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_cgees_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgees_work(
///     int matrix_layout, char jobvs, char sort, LAPACK_Z_SELECT1 select, 
///     lapack_int n, lapack_complex_double* a, lapack_int lda, lapack_int* sdim, 
///     lapack_complex_double* w, lapack_complex_double* vs, lapack_int ldvs, lapack_complex_double* work, 
///     lapack_int lwork, double* rwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, sdim, w, vs, work, rwork, bwork)
pub extern "C" fn lapacke_zgees_work(
  matrix_layout : Int,
  jobvs : Byte,
  sort : Byte,
  select : Lapack_z_select1,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  sdim : FixedArray[Int],
  w : FixedArray[Double],
  vs : FixedArray[Double],
  ldvs : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_zgees_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeesx_work(
///     int matrix_layout, char jobvs, char sort, LAPACK_S_SELECT2 select, 
///     char sense, lapack_int n, float* a, lapack_int lda, 
///     lapack_int* sdim, float* wr, float* wi, float* vs, 
///     lapack_int ldvs, float* rconde, float* rcondv, float* work, 
///     lapack_int lwork, lapack_int* iwork, lapack_int liwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, sdim, wr, wi, vs, rconde, rcondv, work, iwork, bwork)
pub extern "C" fn lapacke_sgeesx_work(
  matrix_layout : Int,
  jobvs : Byte,
  sort : Byte,
  select : Lapack_s_select2,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  sdim : FixedArray[Int],
  wr : FixedArray[Float],
  wi : FixedArray[Float],
  vs : FixedArray[Float],
  ldvs : Lapack_int,
  rconde : FixedArray[Float],
  rcondv : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_sgeesx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeesx_work(
///     int matrix_layout, char jobvs, char sort, LAPACK_D_SELECT2 select, 
///     char sense, lapack_int n, double* a, lapack_int lda, 
///     lapack_int* sdim, double* wr, double* wi, double* vs, 
///     lapack_int ldvs, double* rconde, double* rcondv, double* work, 
///     lapack_int lwork, lapack_int* iwork, lapack_int liwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, sdim, wr, wi, vs, rconde, rcondv, work, iwork, bwork)
pub extern "C" fn lapacke_dgeesx_work(
  matrix_layout : Int,
  jobvs : Byte,
  sort : Byte,
  select : Lapack_d_select2,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  sdim : FixedArray[Int],
  wr : FixedArray[Double],
  wi : FixedArray[Double],
  vs : FixedArray[Double],
  ldvs : Lapack_int,
  rconde : FixedArray[Double],
  rcondv : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_dgeesx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeesx_work(
///     int matrix_layout, char jobvs, char sort, LAPACK_C_SELECT1 select, 
///     char sense, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_int* sdim, lapack_complex_float* w, lapack_complex_float* vs, lapack_int ldvs, 
///     float* rconde, float* rcondv, lapack_complex_float* work, lapack_int lwork, 
///     float* rwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, sdim, w, vs, rconde, rcondv, work, rwork, bwork)
pub extern "C" fn lapacke_cgeesx_work(
  matrix_layout : Int,
  jobvs : Byte,
  sort : Byte,
  select : Lapack_c_select1,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  sdim : FixedArray[Int],
  w : FixedArray[Float],
  vs : FixedArray[Float],
  ldvs : Lapack_int,
  rconde : FixedArray[Float],
  rcondv : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_cgeesx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeesx_work(
///     int matrix_layout, char jobvs, char sort, LAPACK_Z_SELECT1 select, 
///     char sense, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_int* sdim, lapack_complex_double* w, lapack_complex_double* vs, lapack_int ldvs, 
///     double* rconde, double* rcondv, lapack_complex_double* work, lapack_int lwork, 
///     double* rwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, sdim, w, vs, rconde, rcondv, work, rwork, bwork)
pub extern "C" fn lapacke_zgeesx_work(
  matrix_layout : Int,
  jobvs : Byte,
  sort : Byte,
  select : Lapack_z_select1,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  sdim : FixedArray[Int],
  w : FixedArray[Double],
  vs : FixedArray[Double],
  ldvs : Lapack_int,
  rconde : FixedArray[Double],
  rcondv : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_zgeesx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeev_work(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     float* a, lapack_int lda, float* wr, float* wi, 
///     float* vl, lapack_int ldvl, float* vr, lapack_int ldvr, 
///     float* work, lapack_int lwork
/// );
/// ```
#borrow(a, wr, wi, vl, vr, work)
pub extern "C" fn lapacke_sgeev_work(
  matrix_layout : Int,
  jobvl : Byte,
  jobvr : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  wr : FixedArray[Float],
  wi : FixedArray[Float],
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgeev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeev_work(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     double* a, lapack_int lda, double* wr, double* wi, 
///     double* vl, lapack_int ldvl, double* vr, lapack_int ldvr, 
///     double* work, lapack_int lwork
/// );
/// ```
#borrow(a, wr, wi, vl, vr, work)
pub extern "C" fn lapacke_dgeev_work(
  matrix_layout : Int,
  jobvl : Byte,
  jobvr : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  wr : FixedArray[Double],
  wi : FixedArray[Double],
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgeev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeev_work(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* w, lapack_complex_float* vl, 
///     lapack_int ldvl, lapack_complex_float* vr, lapack_int ldvr, lapack_complex_float* work, 
///     lapack_int lwork, float* rwork
/// );
/// ```
#borrow(a, w, vl, vr, work, rwork)
pub extern "C" fn lapacke_cgeev_work(
  matrix_layout : Int,
  jobvl : Byte,
  jobvr : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  w : FixedArray[Float],
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgeev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeev_work(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* w, lapack_complex_double* vl, 
///     lapack_int ldvl, lapack_complex_double* vr, lapack_int ldvr, lapack_complex_double* work, 
///     lapack_int lwork, double* rwork
/// );
/// ```
#borrow(a, w, vl, vr, work, rwork)
pub extern "C" fn lapacke_zgeev_work(
  matrix_layout : Int,
  jobvl : Byte,
  jobvr : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  w : FixedArray[Double],
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgeev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeevx_work(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, float* a, lapack_int lda, 
///     float* wr, float* wi, float* vl, lapack_int ldvl, 
///     float* vr, lapack_int ldvr, lapack_int* ilo, lapack_int* ihi, 
///     float* scale, float* abnrm, float* rconde, float* rcondv, 
///     float* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, wr, wi, vl, vr, ilo, ihi, scale, abnrm, rconde, rcondv, work, iwork)
pub extern "C" fn lapacke_sgeevx_work(
  matrix_layout : Int,
  balanc : Byte,
  jobvl : Byte,
  jobvr : Byte,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  wr : FixedArray[Float],
  wi : FixedArray[Float],
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  scale : FixedArray[Float],
  abnrm : FixedArray[Float],
  rconde : FixedArray[Float],
  rcondv : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgeevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeevx_work(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, double* a, lapack_int lda, 
///     double* wr, double* wi, double* vl, lapack_int ldvl, 
///     double* vr, lapack_int ldvr, lapack_int* ilo, lapack_int* ihi, 
///     double* scale, double* abnrm, double* rconde, double* rcondv, 
///     double* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, wr, wi, vl, vr, ilo, ihi, scale, abnrm, rconde, rcondv, work, iwork)
pub extern "C" fn lapacke_dgeevx_work(
  matrix_layout : Int,
  balanc : Byte,
  jobvl : Byte,
  jobvr : Byte,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  wr : FixedArray[Double],
  wi : FixedArray[Double],
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  scale : FixedArray[Double],
  abnrm : FixedArray[Double],
  rconde : FixedArray[Double],
  rcondv : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgeevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeevx_work(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* w, lapack_complex_float* vl, lapack_int ldvl, lapack_complex_float* vr, 
///     lapack_int ldvr, lapack_int* ilo, lapack_int* ihi, float* scale, 
///     float* abnrm, float* rconde, float* rcondv, lapack_complex_float* work, 
///     lapack_int lwork, float* rwork
/// );
/// ```
#borrow(a, w, vl, vr, ilo, ihi, scale, abnrm, rconde, rcondv, work, rwork)
pub extern "C" fn lapacke_cgeevx_work(
  matrix_layout : Int,
  balanc : Byte,
  jobvl : Byte,
  jobvr : Byte,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  w : FixedArray[Float],
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  scale : FixedArray[Float],
  abnrm : FixedArray[Float],
  rconde : FixedArray[Float],
  rcondv : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgeevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeevx_work(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* w, lapack_complex_double* vl, lapack_int ldvl, lapack_complex_double* vr, 
///     lapack_int ldvr, lapack_int* ilo, lapack_int* ihi, double* scale, 
///     double* abnrm, double* rconde, double* rcondv, lapack_complex_double* work, 
///     lapack_int lwork, double* rwork
/// );
/// ```
#borrow(a, w, vl, vr, ilo, ihi, scale, abnrm, rconde, rcondv, work, rwork)
pub extern "C" fn lapacke_zgeevx_work(
  matrix_layout : Int,
  balanc : Byte,
  jobvl : Byte,
  jobvr : Byte,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  w : FixedArray[Double],
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  scale : FixedArray[Double],
  abnrm : FixedArray[Double],
  rconde : FixedArray[Double],
  rcondv : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgeevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgehrd_work(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     float* a, lapack_int lda, float* tau, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_sgehrd_work(
  matrix_layout : Int,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgehrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgehrd_work(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     double* a, lapack_int lda, double* tau, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_dgehrd_work(
  matrix_layout : Int,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgehrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgehrd_work(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* tau, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_cgehrd_work(
  matrix_layout : Int,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cgehrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgehrd_work(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* tau, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_zgehrd_work(
  matrix_layout : Int,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zgehrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgejsv_work(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     char jobr, char jobt, char jobp, lapack_int m, 
///     lapack_int n, float* a, lapack_int lda, float* sva, 
///     float* u, lapack_int ldu, float* v, lapack_int ldv, 
///     float* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, sva, u, v, work, iwork)
pub extern "C" fn lapacke_sgejsv_work(
  matrix_layout : Int,
  joba : Byte,
  jobu : Byte,
  jobv : Byte,
  jobr : Byte,
  jobt : Byte,
  jobp : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  sva : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgejsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgejsv_work(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     char jobr, char jobt, char jobp, lapack_int m, 
///     lapack_int n, double* a, lapack_int lda, double* sva, 
///     double* u, lapack_int ldu, double* v, lapack_int ldv, 
///     double* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, sva, u, v, work, iwork)
pub extern "C" fn lapacke_dgejsv_work(
  matrix_layout : Int,
  joba : Byte,
  jobu : Byte,
  jobv : Byte,
  jobr : Byte,
  jobt : Byte,
  jobp : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  sva : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgejsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgejsv_work(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     char jobr, char jobt, char jobp, lapack_int m, 
///     lapack_int n, lapack_complex_float* a, lapack_int lda, float* sva, 
///     lapack_complex_float* u, lapack_int ldu, lapack_complex_float* v, lapack_int ldv, 
///     lapack_complex_float* cwork, lapack_int lwork, float* work, lapack_int lrwork, lapack_int* iwork
/// );
/// ```
#borrow(a, sva, u, v, cwork, work, iwork)
pub extern "C" fn lapacke_cgejsv_work(
  matrix_layout : Int,
  joba : Byte,
  jobu : Byte,
  jobv : Byte,
  jobr : Byte,
  jobt : Byte,
  jobp : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  sva : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  cwork : FixedArray[Float],
  lwork : Lapack_int,
  work : FixedArray[Float],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cgejsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgejsv_work(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     char jobr, char jobt, char jobp, lapack_int m, 
///     lapack_int n, lapack_complex_double* a, lapack_int lda, double* sva, 
///     lapack_complex_double* u, lapack_int ldu, lapack_complex_double* v, lapack_int ldv, 
///     lapack_complex_double* cwork, lapack_int lwork, double* work, lapack_int lrwork, lapack_int* iwork
/// );
/// ```
#borrow(a, sva, u, v, cwork, work, iwork)
pub extern "C" fn lapacke_zgejsv_work(
  matrix_layout : Int,
  joba : Byte,
  jobu : Byte,
  jobv : Byte,
  jobr : Byte,
  jobt : Byte,
  jobp : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  sva : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  cwork : FixedArray[Double],
  lwork : Lapack_int,
  work : FixedArray[Double],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zgejsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgelq2_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau, float* work
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_sgelq2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sgelq2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgelq2_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau, double* work
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_dgelq2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dgelq2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgelq2_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau, lapack_complex_float* work
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_cgelq2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgelq2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgelq2_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau, lapack_complex_double* work
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_zgelq2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgelq2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgelqf_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_sgelqf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgelqf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgelqf_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_dgelqf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgelqf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgelqf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_cgelqf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cgelqf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgelqf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_zgelqf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zgelqf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgels_work(
///     int matrix_layout, char trans, lapack_int m, lapack_int n, 
///     lapack_int nrhs, float* a, lapack_int lda, float* b, 
///     lapack_int ldb, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, work)
pub extern "C" fn lapacke_sgels_work(
  matrix_layout : Int,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgels_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgels_work(
///     int matrix_layout, char trans, lapack_int m, lapack_int n, 
///     lapack_int nrhs, double* a, lapack_int lda, double* b, 
///     lapack_int ldb, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, work)
pub extern "C" fn lapacke_dgels_work(
  matrix_layout : Int,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgels_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgels_work(
///     int matrix_layout, char trans, lapack_int m, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, work)
pub extern "C" fn lapacke_cgels_work(
  matrix_layout : Int,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cgels_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgels_work(
///     int matrix_layout, char trans, lapack_int m, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, work)
pub extern "C" fn lapacke_zgels_work(
  matrix_layout : Int,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zgels_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgelsd_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* s, float rcond, lapack_int* rank, float* work, 
///     lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, b, s, rank, work, iwork)
pub extern "C" fn lapacke_sgelsd_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  s : FixedArray[Float],
  rcond : Float,
  rank : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgelsd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgelsd_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* s, double rcond, lapack_int* rank, double* work, 
///     lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, b, s, rank, work, iwork)
pub extern "C" fn lapacke_dgelsd_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  s : FixedArray[Double],
  rcond : Double,
  rank : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgelsd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgelsd_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     float* s, float rcond, lapack_int* rank, lapack_complex_float* work, 
///     lapack_int lwork, float* rwork, lapack_int* iwork
/// );
/// ```
#borrow(a, b, s, rank, work, rwork, iwork)
pub extern "C" fn lapacke_cgelsd_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  s : FixedArray[Float],
  rcond : Float,
  rank : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cgelsd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgelsd_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     double* s, double rcond, lapack_int* rank, lapack_complex_double* work, 
///     lapack_int lwork, double* rwork, lapack_int* iwork
/// );
/// ```
#borrow(a, b, s, rank, work, rwork, iwork)
pub extern "C" fn lapacke_zgelsd_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  s : FixedArray[Double],
  rcond : Double,
  rank : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zgelsd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgelss_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* s, float rcond, lapack_int* rank, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, s, rank, work)
pub extern "C" fn lapacke_sgelss_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  s : FixedArray[Float],
  rcond : Float,
  rank : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgelss_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgelss_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* s, double rcond, lapack_int* rank, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, s, rank, work)
pub extern "C" fn lapacke_dgelss_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  s : FixedArray[Double],
  rcond : Double,
  rank : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgelss_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgelss_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     float* s, float rcond, lapack_int* rank, lapack_complex_float* work, 
///     lapack_int lwork, float* rwork
/// );
/// ```
#borrow(a, b, s, rank, work, rwork)
pub extern "C" fn lapacke_cgelss_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  s : FixedArray[Float],
  rcond : Float,
  rank : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgelss_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgelss_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     double* s, double rcond, lapack_int* rank, lapack_complex_double* work, 
///     lapack_int lwork, double* rwork
/// );
/// ```
#borrow(a, b, s, rank, work, rwork)
pub extern "C" fn lapacke_zgelss_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  s : FixedArray[Double],
  rcond : Double,
  rank : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgelss_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgelsy_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     lapack_int* jpvt, float rcond, lapack_int* rank, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, jpvt, rank, work)
pub extern "C" fn lapacke_sgelsy_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  jpvt : FixedArray[Int],
  rcond : Float,
  rank : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgelsy_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgelsy_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     lapack_int* jpvt, double rcond, lapack_int* rank, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, jpvt, rank, work)
pub extern "C" fn lapacke_dgelsy_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  jpvt : FixedArray[Int],
  rcond : Double,
  rank : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgelsy_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgelsy_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     lapack_int* jpvt, float rcond, lapack_int* rank, lapack_complex_float* work, 
///     lapack_int lwork, float* rwork
/// );
/// ```
#borrow(a, b, jpvt, rank, work, rwork)
pub extern "C" fn lapacke_cgelsy_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  jpvt : FixedArray[Int],
  rcond : Float,
  rank : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgelsy_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgelsy_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_int* jpvt, double rcond, lapack_int* rank, lapack_complex_double* work, 
///     lapack_int lwork, double* rwork
/// );
/// ```
#borrow(a, b, jpvt, rank, work, rwork)
pub extern "C" fn lapacke_zgelsy_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  jpvt : FixedArray[Int],
  rcond : Double,
  rank : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgelsy_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqlf_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_sgeqlf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgeqlf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqlf_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_dgeqlf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgeqlf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqlf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_cgeqlf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cgeqlf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqlf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_zgeqlf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zgeqlf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqp3_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* jpvt, float* tau, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, jpvt, tau, work)
pub extern "C" fn lapacke_sgeqp3_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  jpvt : FixedArray[Int],
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgeqp3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqp3_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* jpvt, double* tau, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, jpvt, tau, work)
pub extern "C" fn lapacke_dgeqp3_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  jpvt : FixedArray[Int],
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgeqp3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqp3_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* jpvt, lapack_complex_float* tau, lapack_complex_float* work, 
///     lapack_int lwork, float* rwork
/// );
/// ```
#borrow(a, jpvt, tau, work, rwork)
pub extern "C" fn lapacke_cgeqp3_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  jpvt : FixedArray[Int],
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgeqp3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqp3_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* jpvt, lapack_complex_double* tau, lapack_complex_double* work, 
///     lapack_int lwork, double* rwork
/// );
/// ```
#borrow(a, jpvt, tau, work, rwork)
pub extern "C" fn lapacke_zgeqp3_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  jpvt : FixedArray[Int],
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgeqp3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqpf_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* jpvt, float* tau, float* work
/// );
/// ```
#borrow(a, jpvt, tau, work)
pub extern "C" fn lapacke_sgeqpf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  jpvt : FixedArray[Int],
  tau : FixedArray[Float],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sgeqpf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqpf_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* jpvt, double* tau, double* work
/// );
/// ```
#borrow(a, jpvt, tau, work)
pub extern "C" fn lapacke_dgeqpf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  jpvt : FixedArray[Int],
  tau : FixedArray[Double],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dgeqpf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqpf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* jpvt, lapack_complex_float* tau, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(a, jpvt, tau, work, rwork)
pub extern "C" fn lapacke_cgeqpf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  jpvt : FixedArray[Int],
  tau : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgeqpf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqpf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* jpvt, lapack_complex_double* tau, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(a, jpvt, tau, work, rwork)
pub extern "C" fn lapacke_zgeqpf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  jpvt : FixedArray[Int],
  tau : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgeqpf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqr2_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau, float* work
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_sgeqr2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sgeqr2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqr2_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau, double* work
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_dgeqr2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dgeqr2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqr2_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau, lapack_complex_float* work
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_cgeqr2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgeqr2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqr2_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau, lapack_complex_double* work
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_zgeqr2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgeqr2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqrf_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_sgeqrf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgeqrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqrf_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_dgeqrf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgeqrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqrf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_cgeqrf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cgeqrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqrf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_zgeqrf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zgeqrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqrfp_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_sgeqrfp_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgeqrfp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqrfp_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_dgeqrfp_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgeqrfp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqrfp_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_cgeqrfp_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cgeqrfp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqrfp_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_zgeqrfp_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zgeqrfp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgerfs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const float* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const float* b, lapack_int ldb, float* x, 
///     lapack_int ldx, float* ferr, float* berr, float* work, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, b, x, ferr, berr, work, iwork)
pub extern "C" fn lapacke_sgerfs_work(
  matrix_layout : Int,
  trans : Byte,
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
) -> Lapack_int = "LAPACKE_sgerfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgerfs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const double* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const double* b, lapack_int ldb, double* x, 
///     lapack_int ldx, double* ferr, double* berr, double* work, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, b, x, ferr, berr, work, iwork)
pub extern "C" fn lapacke_dgerfs_work(
  matrix_layout : Int,
  trans : Byte,
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
) -> Lapack_int = "LAPACKE_dgerfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgerfs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* ferr, float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(a, af, ipiv, b, x, ferr, berr, work, rwork)
pub extern "C" fn lapacke_cgerfs_work(
  matrix_layout : Int,
  trans : Byte,
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
) -> Lapack_int = "LAPACKE_cgerfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgerfs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* ferr, double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(a, af, ipiv, b, x, ferr, berr, work, rwork)
pub extern "C" fn lapacke_zgerfs_work(
  matrix_layout : Int,
  trans : Byte,
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
) -> Lapack_int = "LAPACKE_zgerfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgerfsx_work(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int nrhs, const float* a, lapack_int lda, const float* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const float* r, const float* c, 
///     const float* b, lapack_int ldb, float* x, lapack_int ldx, 
///     float* rcond, float* berr, lapack_int n_err_bnds, float* err_bnds_norm, 
///     float* err_bnds_comp, lapack_int nparams, float* params, float* work, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, r, c, b, x, rcond, berr, err_bnds_norm, err_bnds_comp, params, work, iwork)
pub extern "C" fn lapacke_sgerfsx_work(
  matrix_layout : Int,
  trans : Byte,
  equed : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  af : FixedArray[Float],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  r : FixedArray[Float],
  c : FixedArray[Float],
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
) -> Lapack_int = "LAPACKE_sgerfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgerfsx_work(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int nrhs, const double* a, lapack_int lda, const double* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const double* r, const double* c, 
///     const double* b, lapack_int ldb, double* x, lapack_int ldx, 
///     double* rcond, double* berr, lapack_int n_err_bnds, double* err_bnds_norm, 
///     double* err_bnds_comp, lapack_int nparams, double* params, double* work, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, r, c, b, x, rcond, berr, err_bnds_norm, err_bnds_comp, params, work, iwork)
pub extern "C" fn lapacke_dgerfsx_work(
  matrix_layout : Int,
  trans : Byte,
  equed : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  af : FixedArray[Double],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  r : FixedArray[Double],
  c : FixedArray[Double],
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
) -> Lapack_int = "LAPACKE_dgerfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgerfsx_work(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const float* r, const float* c, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* rcond, float* berr, lapack_int n_err_bnds, float* err_bnds_norm, 
///     float* err_bnds_comp, lapack_int nparams, float* params, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(a, af, ipiv, r, c, b, x, rcond, berr, err_bnds_norm, err_bnds_comp, params, work, rwork)
pub extern "C" fn lapacke_cgerfsx_work(
  matrix_layout : Int,
  trans : Byte,
  equed : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  af : FixedArray[Float],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  r : FixedArray[Float],
  c : FixedArray[Float],
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
) -> Lapack_int = "LAPACKE_cgerfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgerfsx_work(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const double* r, const double* c, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* rcond, double* berr, lapack_int n_err_bnds, double* err_bnds_norm, 
///     double* err_bnds_comp, lapack_int nparams, double* params, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(a, af, ipiv, r, c, b, x, rcond, berr, err_bnds_norm, err_bnds_comp, params, work, rwork)
pub extern "C" fn lapacke_zgerfsx_work(
  matrix_layout : Int,
  trans : Byte,
  equed : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  af : FixedArray[Double],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  r : FixedArray[Double],
  c : FixedArray[Double],
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
) -> Lapack_int = "LAPACKE_zgerfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgerqf_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_sgerqf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgerqf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgerqf_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_dgerqf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgerqf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgerqf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_cgerqf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cgerqf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgerqf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, tau, work)
pub extern "C" fn lapacke_zgerqf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zgerqf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesdd_work(
///     int matrix_layout, char jobz, lapack_int m, lapack_int n, 
///     float* a, lapack_int lda, float* s, float* u, 
///     lapack_int ldu, float* vt, lapack_int ldvt, float* work, 
///     lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, s, u, vt, work, iwork)
pub extern "C" fn lapacke_sgesdd_work(
  matrix_layout : Int,
  jobz : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  s : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  vt : FixedArray[Float],
  ldvt : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgesdd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesdd_work(
///     int matrix_layout, char jobz, lapack_int m, lapack_int n, 
///     double* a, lapack_int lda, double* s, double* u, 
///     lapack_int ldu, double* vt, lapack_int ldvt, double* work, 
///     lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, s, u, vt, work, iwork)
pub extern "C" fn lapacke_dgesdd_work(
  matrix_layout : Int,
  jobz : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  s : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  vt : FixedArray[Double],
  ldvt : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgesdd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesdd_work(
///     int matrix_layout, char jobz, lapack_int m, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, float* s, lapack_complex_float* u, 
///     lapack_int ldu, lapack_complex_float* vt, lapack_int ldvt, lapack_complex_float* work, 
///     lapack_int lwork, float* rwork, lapack_int* iwork
/// );
/// ```
#borrow(a, s, u, vt, work, rwork, iwork)
pub extern "C" fn lapacke_cgesdd_work(
  matrix_layout : Int,
  jobz : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  s : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  vt : FixedArray[Float],
  ldvt : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cgesdd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesdd_work(
///     int matrix_layout, char jobz, lapack_int m, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, double* s, lapack_complex_double* u, 
///     lapack_int ldu, lapack_complex_double* vt, lapack_int ldvt, lapack_complex_double* work, 
///     lapack_int lwork, double* rwork, lapack_int* iwork
/// );
/// ```
#borrow(a, s, u, vt, work, rwork, iwork)
pub extern "C" fn lapacke_zgesdd_work(
  matrix_layout : Int,
  jobz : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  s : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  vt : FixedArray[Double],
  ldvt : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zgesdd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgedmd_work(
///     int matrix_layout, char jobs, char jobz, char jobr, 
///     char jobf, lapack_int whtsvd, lapack_int m, lapack_int n, 
///     float* x, lapack_int ldx, float* y, lapack_int ldy, 
///     lapack_int nrnk, float* tol, lapack_int k, float* reig, 
///     float* imeig, float* z, lapack_int ldz, float* res, 
///     float* b, lapack_int ldb, float* w, lapack_int ldw, 
///     float* s, lapack_int lds, float* work, lapack_int lwork, 
///     lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(x, y, tol, reig, imeig, z, res, b, w, s, work, iwork)
pub extern "C" fn lapacke_sgedmd_work(
  matrix_layout : Int,
  jobs : Byte,
  jobz : Byte,
  jobr : Byte,
  jobf : Byte,
  whtsvd : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  y : FixedArray[Float],
  ldy : Lapack_int,
  nrnk : Lapack_int,
  tol : FixedArray[Float],
  k : Lapack_int,
  reig : FixedArray[Float],
  imeig : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  res : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  w : FixedArray[Float],
  ldw : Lapack_int,
  s : FixedArray[Float],
  lds : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgedmd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgedmd_work(
///     int matrix_layout, char jobs, char jobz, char jobr, 
///     char jobf, lapack_int whtsvd, lapack_int m, lapack_int n, 
///     double* x, lapack_int ldx, double* y, lapack_int ldy, 
///     lapack_int nrnk, double* tol, lapack_int k, double* reig, 
///     double* imeig, double* z, lapack_int ldz, double* res, 
///     double* b, lapack_int ldb, double* w, lapack_int ldw, 
///     double* s, lapack_int lds, double* work, lapack_int lwork, 
///     lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(x, y, tol, reig, imeig, z, res, b, w, s, work, iwork)
pub extern "C" fn lapacke_dgedmd_work(
  matrix_layout : Int,
  jobs : Byte,
  jobz : Byte,
  jobr : Byte,
  jobf : Byte,
  whtsvd : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  y : FixedArray[Double],
  ldy : Lapack_int,
  nrnk : Lapack_int,
  tol : FixedArray[Double],
  k : Lapack_int,
  reig : FixedArray[Double],
  imeig : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  res : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  w : FixedArray[Double],
  ldw : Lapack_int,
  s : FixedArray[Double],
  lds : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgedmd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgedmd_work(
///     int matrix_layout, char jobs, char jobz, char jobr, 
///     char jobf, lapack_int whtsvd, lapack_int m, lapack_int n, 
///     lapack_complex_float* x, lapack_int ldx, lapack_complex_float* y, lapack_int ldy, 
///     lapack_int nrnk, float* tol, lapack_int k, lapack_complex_float* eigs, 
///     lapack_complex_float* z, lapack_int ldz, float* res, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* w, lapack_int ldw, lapack_complex_float* s, 
///     lapack_int lds, lapack_complex_float* zwork, lapack_int lzwork, float* work, 
///     lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(x, y, tol, eigs, z, res, b, w, s, zwork, work, iwork)
pub extern "C" fn lapacke_cgedmd_work(
  matrix_layout : Int,
  jobs : Byte,
  jobz : Byte,
  jobr : Byte,
  jobf : Byte,
  whtsvd : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  y : FixedArray[Float],
  ldy : Lapack_int,
  nrnk : Lapack_int,
  tol : FixedArray[Float],
  k : Lapack_int,
  eigs : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  res : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  w : FixedArray[Float],
  ldw : Lapack_int,
  s : FixedArray[Float],
  lds : Lapack_int,
  zwork : FixedArray[Float],
  lzwork : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cgedmd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgedmd_work(
///     int matrix_layout, char jobs, char jobz, char jobr, 
///     char jobf, lapack_int whtsvd, lapack_int m, lapack_int n, 
///     lapack_complex_double* x, lapack_int ldx, lapack_complex_double* y, lapack_int ldy, 
///     lapack_int nrnk, double* tol, lapack_int k, lapack_complex_double* eigs, 
///     lapack_complex_double* z, lapack_int ldz, double* res, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* w, lapack_int ldw, lapack_complex_double* s, 
///     lapack_int lds, lapack_complex_double* zwork, lapack_int lzwork, double* work, 
///     lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(x, y, tol, eigs, z, res, b, w, s, zwork, work, iwork)
pub extern "C" fn lapacke_zgedmd_work(
  matrix_layout : Int,
  jobs : Byte,
  jobz : Byte,
  jobr : Byte,
  jobf : Byte,
  whtsvd : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  y : FixedArray[Double],
  ldy : Lapack_int,
  nrnk : Lapack_int,
  tol : FixedArray[Double],
  k : Lapack_int,
  eigs : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  res : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  w : FixedArray[Double],
  ldw : Lapack_int,
  s : FixedArray[Double],
  lds : Lapack_int,
  zwork : FixedArray[Double],
  lzwork : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zgedmd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgedmdq_work(
///     int matrix_layout, char jobs, char jobz, char jobr, 
///     char jobq, char jobt, char jobf, lapack_int whtsvd, 
///     lapack_int m, lapack_int n, float* f, lapack_int ldf, 
///     float* x, lapack_int ldx, float* y, lapack_int ldy, 
///     lapack_int nrnk, float* tol, lapack_int k, float* reig, 
///     float* imeig, float* z, lapack_int ldz, float* res, 
///     float* b, lapack_int ldb, float* v, lapack_int ldv, 
///     float* s, lapack_int lds, float* work, lapack_int lwork, 
///     lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(f, x, y, tol, reig, imeig, z, res, b, v, s, work, iwork)
pub extern "C" fn lapacke_sgedmdq_work(
  matrix_layout : Int,
  jobs : Byte,
  jobz : Byte,
  jobr : Byte,
  jobq : Byte,
  jobt : Byte,
  jobf : Byte,
  whtsvd : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  f : FixedArray[Float],
  ldf : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  y : FixedArray[Float],
  ldy : Lapack_int,
  nrnk : Lapack_int,
  tol : FixedArray[Float],
  k : Lapack_int,
  reig : FixedArray[Float],
  imeig : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  res : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  s : FixedArray[Float],
  lds : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgedmdq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgedmdq_work(
///     int matrix_layout, char jobs, char jobz, char jobr, 
///     char jobq, char jobt, char jobf, lapack_int whtsvd, 
///     lapack_int m, lapack_int n, double* f, lapack_int ldf, 
///     double* x, lapack_int ldx, double* y, lapack_int ldy, 
///     lapack_int nrnk, double* tol, lapack_int k, double* reig, 
///     double* imeig, double* z, lapack_int ldz, double* res, 
///     double* b, lapack_int ldb, double* v, lapack_int ldv, 
///     double* s, lapack_int lds, double* work, lapack_int lwork, 
///     lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(f, x, y, tol, reig, imeig, z, res, b, v, s, work, iwork)
pub extern "C" fn lapacke_dgedmdq_work(
  matrix_layout : Int,
  jobs : Byte,
  jobz : Byte,
  jobr : Byte,
  jobq : Byte,
  jobt : Byte,
  jobf : Byte,
  whtsvd : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  f : FixedArray[Double],
  ldf : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  y : FixedArray[Double],
  ldy : Lapack_int,
  nrnk : Lapack_int,
  tol : FixedArray[Double],
  k : Lapack_int,
  reig : FixedArray[Double],
  imeig : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  res : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  s : FixedArray[Double],
  lds : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgedmdq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgedmdq_work(
///     int matrix_layout, char jobs, char jobz, char jobr, 
///     char jobq, char jobt, char jobf, lapack_int whtsvd, 
///     lapack_int m, lapack_int n, lapack_complex_float* f, lapack_int ldf, 
///     lapack_complex_float* x, lapack_int ldx, lapack_complex_float* y, lapack_int ldy, 
///     lapack_int nrnk, float* tol, lapack_int k, lapack_complex_float* eigs, 
///     lapack_complex_float* z, lapack_int ldz, float* res, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* v, lapack_int ldv, lapack_complex_float* s, 
///     lapack_int lds, lapack_complex_float* zwork, lapack_int lzwork, float* work, 
///     lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(f, x, y, tol, eigs, z, res, b, v, s, zwork, work, iwork)
pub extern "C" fn lapacke_cgedmdq_work(
  matrix_layout : Int,
  jobs : Byte,
  jobz : Byte,
  jobr : Byte,
  jobq : Byte,
  jobt : Byte,
  jobf : Byte,
  whtsvd : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  f : FixedArray[Float],
  ldf : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  y : FixedArray[Float],
  ldy : Lapack_int,
  nrnk : Lapack_int,
  tol : FixedArray[Float],
  k : Lapack_int,
  eigs : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
  res : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  s : FixedArray[Float],
  lds : Lapack_int,
  zwork : FixedArray[Float],
  lzwork : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cgedmdq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgedmdq_work(
///     int matrix_layout, char jobs, char jobz, char jobr, 
///     char jobq, char jobt, char jobf, lapack_int whtsvd, 
///     lapack_int m, lapack_int n, lapack_complex_double* f, lapack_int ldf, 
///     lapack_complex_double* x, lapack_int ldx, lapack_complex_double* y, lapack_int ldy, 
///     lapack_int nrnk, double* tol, lapack_int k, lapack_complex_double* eigs, 
///     lapack_complex_double* z, lapack_int ldz, double* res, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* v, lapack_int ldv, lapack_complex_double* s, 
///     lapack_int lds, lapack_complex_double* zwork, lapack_int lzwork, double* work, 
///     lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
#borrow(f, x, y, tol, eigs, z, res, b, v, s, zwork, work, iwork)
pub extern "C" fn lapacke_zgedmdq_work(
  matrix_layout : Int,
  jobs : Byte,
  jobz : Byte,
  jobr : Byte,
  jobq : Byte,
  jobt : Byte,
  jobf : Byte,
  whtsvd : Lapack_int,
  m : Lapack_int,
  n : Lapack_int,
  f : FixedArray[Double],
  ldf : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  y : FixedArray[Double],
  ldy : Lapack_int,
  nrnk : Lapack_int,
  tol : FixedArray[Double],
  k : Lapack_int,
  eigs : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
  res : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  s : FixedArray[Double],
  lds : Lapack_int,
  zwork : FixedArray[Double],
  lzwork : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zgedmdq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesv_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, float* a, 
///     lapack_int lda, lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_sgesv_work(
  matrix_layout : Int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_sgesv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesv_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, double* a, 
///     lapack_int lda, lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_dgesv_work(
  matrix_layout : Int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dgesv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesv_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_cgesv_work(
  matrix_layout : Int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_cgesv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesv_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zgesv_work(
  matrix_layout : Int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zgesv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsgesv_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, double* a, 
///     lapack_int lda, lapack_int* ipiv, double* b, lapack_int ldb, 
///     double* x, lapack_int ldx, double* work, float* swork, lapack_int* iter
/// );
/// ```
#borrow(a, ipiv, b, x, work, swork, iter)
pub extern "C" fn lapacke_dsgesv_work(
  matrix_layout : Int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  work : FixedArray[Double],
  swork : FixedArray[Float],
  iter : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dsgesv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zcgesv_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* x, lapack_int ldx, lapack_complex_double* work, lapack_complex_float* swork, 
///     double* rwork, lapack_int* iter
/// );
/// ```
#borrow(a, ipiv, b, x, work, swork, rwork, iter)
pub extern "C" fn lapacke_zcgesv_work(
  matrix_layout : Int,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  work : FixedArray[Double],
  swork : FixedArray[Float],
  rwork : FixedArray[Double],
  iter : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zcgesv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesvd_work(
///     int matrix_layout, char jobu, char jobvt, lapack_int m, 
///     lapack_int n, float* a, lapack_int lda, float* s, 
///     float* u, lapack_int ldu, float* vt, lapack_int ldvt, 
///     float* work, lapack_int lwork
/// );
/// ```
#borrow(a, s, u, vt, work)
pub extern "C" fn lapacke_sgesvd_work(
  matrix_layout : Int,
  jobu : Byte,
  jobvt : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  s : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  vt : FixedArray[Float],
  ldvt : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgesvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesvd_work(
///     int matrix_layout, char jobu, char jobvt, lapack_int m, 
///     lapack_int n, double* a, lapack_int lda, double* s, 
///     double* u, lapack_int ldu, double* vt, lapack_int ldvt, 
///     double* work, lapack_int lwork
/// );
/// ```
#borrow(a, s, u, vt, work)
pub extern "C" fn lapacke_dgesvd_work(
  matrix_layout : Int,
  jobu : Byte,
  jobvt : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  s : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  vt : FixedArray[Double],
  ldvt : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgesvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesvd_work(
///     int matrix_layout, char jobu, char jobvt, lapack_int m, 
///     lapack_int n, lapack_complex_float* a, lapack_int lda, float* s, 
///     lapack_complex_float* u, lapack_int ldu, lapack_complex_float* vt, lapack_int ldvt, 
///     lapack_complex_float* work, lapack_int lwork, float* rwork
/// );
/// ```
#borrow(a, s, u, vt, work, rwork)
pub extern "C" fn lapacke_cgesvd_work(
  matrix_layout : Int,
  jobu : Byte,
  jobvt : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  s : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  vt : FixedArray[Float],
  ldvt : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgesvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesvd_work(
///     int matrix_layout, char jobu, char jobvt, lapack_int m, 
///     lapack_int n, lapack_complex_double* a, lapack_int lda, double* s, 
///     lapack_complex_double* u, lapack_int ldu, lapack_complex_double* vt, lapack_int ldvt, 
///     lapack_complex_double* work, lapack_int lwork, double* rwork
/// );
/// ```
#borrow(a, s, u, vt, work, rwork)
pub extern "C" fn lapacke_zgesvd_work(
  matrix_layout : Int,
  jobu : Byte,
  jobvt : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  s : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  vt : FixedArray[Double],
  ldvt : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgesvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesvdx_work(
///     int matrix_layout, char jobu, char jobvt, char range, 
///     lapack_int m, lapack_int n, float* a, lapack_int lda, 
///     float vl, float vu, lapack_int il, lapack_int iu, 
///     lapack_int* ns, float* s, float* u, lapack_int ldu, 
///     float* vt, lapack_int ldvt, float* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, ns, s, u, vt, work, iwork)
pub extern "C" fn lapacke_sgesvdx_work(
  matrix_layout : Int,
  jobu : Byte,
  jobvt : Byte,
  range : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  vl : Float,
  vu : Float,
  il : Lapack_int,
  iu : Lapack_int,
  ns : FixedArray[Int],
  s : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  vt : FixedArray[Float],
  ldvt : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgesvdx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesvdx_work(
///     int matrix_layout, char jobu, char jobvt, char range, 
///     lapack_int m, lapack_int n, double* a, lapack_int lda, 
///     double vl, double vu, lapack_int il, lapack_int iu, 
///     lapack_int* ns, double* s, double* u, lapack_int ldu, 
///     double* vt, lapack_int ldvt, double* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(a, ns, s, u, vt, work, iwork)
pub extern "C" fn lapacke_dgesvdx_work(
  matrix_layout : Int,
  jobu : Byte,
  jobvt : Byte,
  range : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  vl : Double,
  vu : Double,
  il : Lapack_int,
  iu : Lapack_int,
  ns : FixedArray[Int],
  s : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  vt : FixedArray[Double],
  ldvt : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgesvdx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesvdx_work(
///     int matrix_layout, char jobu, char jobvt, char range, 
///     lapack_int m, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     float vl, float vu, lapack_int il, lapack_int iu, 
///     lapack_int* ns, float* s, lapack_complex_float* u, lapack_int ldu, 
///     lapack_complex_float* vt, lapack_int ldvt, lapack_complex_float* work, lapack_int lwork, 
///     float* rwork, lapack_int* iwork
/// );
/// ```
#borrow(a, ns, s, u, vt, work, rwork, iwork)
pub extern "C" fn lapacke_cgesvdx_work(
  matrix_layout : Int,
  jobu : Byte,
  jobvt : Byte,
  range : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  vl : Float,
  vu : Float,
  il : Lapack_int,
  iu : Lapack_int,
  ns : FixedArray[Int],
  s : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  vt : FixedArray[Float],
  ldvt : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cgesvdx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesvdx_work(
///     int matrix_layout, char jobu, char jobvt, char range, 
///     lapack_int m, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     double vl, double vu, lapack_int il, lapack_int iu, 
///     lapack_int* ns, double* s, lapack_complex_double* u, lapack_int ldu, 
///     lapack_complex_double* vt, lapack_int ldvt, lapack_complex_double* work, lapack_int lwork, 
///     double* rwork, lapack_int* iwork
/// );
/// ```
#borrow(a, ns, s, u, vt, work, rwork, iwork)
pub extern "C" fn lapacke_zgesvdx_work(
  matrix_layout : Int,
  jobu : Byte,
  jobvt : Byte,
  range : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  vl : Double,
  vu : Double,
  il : Lapack_int,
  iu : Lapack_int,
  ns : FixedArray[Int],
  s : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  vt : FixedArray[Double],
  ldvt : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zgesvdx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesvdq_work(
///     int matrix_layout, char joba, char jobp, char jobr, 
///     char jobu, char jobv, lapack_int m, lapack_int n, 
///     float* a, lapack_int lda, float* s, float* u, 
///     lapack_int ldu, float* v, lapack_int ldv, lapack_int* numrank, 
///     lapack_int* iwork, lapack_int liwork, float* work, lapack_int lwork, 
///     float* rwork, lapack_int lrwork
/// );
/// ```
#borrow(a, s, u, v, numrank, iwork, work, rwork)
pub extern "C" fn lapacke_sgesvdq_work(
  matrix_layout : Int,
  joba : Byte,
  jobp : Byte,
  jobr : Byte,
  jobu : Byte,
  jobv : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  s : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  numrank : FixedArray[Int],
  iwork : FixedArray[Int],
  liwork : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  lrwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgesvdq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesvdq_work(
///     int matrix_layout, char joba, char jobp, char jobr, 
///     char jobu, char jobv, lapack_int m, lapack_int n, 
///     double* a, lapack_int lda, double* s, double* u, 
///     lapack_int ldu, double* v, lapack_int ldv, lapack_int* numrank, 
///     lapack_int* iwork, lapack_int liwork, double* work, lapack_int lwork, 
///     double* rwork, lapack_int lrwork
/// );
/// ```
#borrow(a, s, u, v, numrank, iwork, work, rwork)
pub extern "C" fn lapacke_dgesvdq_work(
  matrix_layout : Int,
  joba : Byte,
  jobp : Byte,
  jobr : Byte,
  jobu : Byte,
  jobv : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  s : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  numrank : FixedArray[Int],
  iwork : FixedArray[Int],
  liwork : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  lrwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgesvdq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesvdq_work(
///     int matrix_layout, char joba, char jobp, char jobr, 
///     char jobu, char jobv, lapack_int m, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, float* s, lapack_complex_float* u, 
///     lapack_int ldu, lapack_complex_float* v, lapack_int ldv, lapack_int* numrank, 
///     lapack_int* iwork, lapack_int liwork, lapack_complex_float* cwork, lapack_int lcwork, 
///     float* rwork, lapack_int lrwork
/// );
/// ```
#borrow(a, s, u, v, numrank, iwork, cwork, rwork)
pub extern "C" fn lapacke_cgesvdq_work(
  matrix_layout : Int,
  joba : Byte,
  jobp : Byte,
  jobr : Byte,
  jobu : Byte,
  jobv : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  s : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  numrank : FixedArray[Int],
  iwork : FixedArray[Int],
  liwork : Lapack_int,
  cwork : FixedArray[Float],
  lcwork : Lapack_int,
  rwork : FixedArray[Float],
  lrwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cgesvdq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesvdq_work(
///     int matrix_layout, char joba, char jobp, char jobr, 
///     char jobu, char jobv, lapack_int m, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, double* s, lapack_complex_double* u, 
///     lapack_int ldu, lapack_complex_double* v, lapack_int ldv, lapack_int* numrank, 
///     lapack_int* iwork, lapack_int liwork, lapack_complex_double* cwork, lapack_int lcwork, 
///     double* rwork, lapack_int lrwork
/// );
/// ```
#borrow(a, s, u, v, numrank, iwork, cwork, rwork)
pub extern "C" fn lapacke_zgesvdq_work(
  matrix_layout : Int,
  joba : Byte,
  jobp : Byte,
  jobr : Byte,
  jobu : Byte,
  jobv : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  s : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  numrank : FixedArray[Int],
  iwork : FixedArray[Int],
  liwork : Lapack_int,
  cwork : FixedArray[Double],
  lcwork : Lapack_int,
  rwork : FixedArray[Double],
  lrwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zgesvdq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesvj_work(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     lapack_int m, lapack_int n, float* a, lapack_int lda, 
///     float* sva, lapack_int mv, float* v, lapack_int ldv, 
///     float* work, lapack_int lwork
/// );
/// ```
#borrow(a, sva, v, work)
pub extern "C" fn lapacke_sgesvj_work(
  matrix_layout : Int,
  joba : Byte,
  jobu : Byte,
  jobv : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  sva : FixedArray[Float],
  mv : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgesvj_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesvj_work(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     lapack_int m, lapack_int n, double* a, lapack_int lda, 
///     double* sva, lapack_int mv, double* v, lapack_int ldv, 
///     double* work, lapack_int lwork
/// );
/// ```
#borrow(a, sva, v, work)
pub extern "C" fn lapacke_dgesvj_work(
  matrix_layout : Int,
  joba : Byte,
  jobu : Byte,
  jobv : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  sva : FixedArray[Double],
  mv : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgesvj_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesvj_work(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     lapack_int m, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     float* sva, lapack_int mv, lapack_complex_float* v, lapack_int ldv, 
///     lapack_complex_float* cwork, lapack_int lwork, float* rwork, lapack_int lrwork
/// );
/// ```
#borrow(a, sva, v, cwork, rwork)
pub extern "C" fn lapacke_cgesvj_work(
  matrix_layout : Int,
  joba : Byte,
  jobu : Byte,
  jobv : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  sva : FixedArray[Float],
  mv : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  cwork : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  lrwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cgesvj_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesvj_work(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     lapack_int m, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     double* sva, lapack_int mv, lapack_complex_double* v, lapack_int ldv, 
///     lapack_complex_double* cwork, lapack_int lwork, double* rwork, lapack_int lrwork
/// );
/// ```
#borrow(a, sva, v, cwork, rwork)
pub extern "C" fn lapacke_zgesvj_work(
  matrix_layout : Int,
  joba : Byte,
  jobu : Byte,
  jobv : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  sva : FixedArray[Double],
  mv : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  cwork : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  lrwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zgesvj_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesvx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, float* a, lapack_int lda, float* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, float* r, 
///     float* c, float* b, lapack_int ldb, float* x, 
///     lapack_int ldx, float* rcond, float* ferr, float* berr, 
///     float* work, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, equed, r, c, b, x, rcond, ferr, berr, work, iwork)
pub extern "C" fn lapacke_sgesvx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  af : FixedArray[Float],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Float],
  c : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  rcond : FixedArray[Float],
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgesvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesvx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, double* a, lapack_int lda, double* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, double* r, 
///     double* c, double* b, lapack_int ldb, double* x, 
///     lapack_int ldx, double* rcond, double* ferr, double* berr, 
///     double* work, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, equed, r, c, b, x, rcond, ferr, berr, work, iwork)
pub extern "C" fn lapacke_dgesvx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  af : FixedArray[Double],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Double],
  c : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  rcond : FixedArray[Double],
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgesvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesvx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, float* r, 
///     float* c, lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* rcond, float* ferr, float* berr, 
///     lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(a, af, ipiv, equed, r, c, b, x, rcond, ferr, berr, work, rwork)
pub extern "C" fn lapacke_cgesvx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  af : FixedArray[Float],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Float],
  c : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  x : FixedArray[Float],
  ldx : Lapack_int,
  rcond : FixedArray[Float],
  ferr : FixedArray[Float],
  berr : FixedArray[Float],
  work : FixedArray[Float],
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgesvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesvx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, double* r, 
///     double* c, lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* rcond, double* ferr, double* berr, 
///     lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(a, af, ipiv, equed, r, c, b, x, rcond, ferr, berr, work, rwork)
pub extern "C" fn lapacke_zgesvx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  af : FixedArray[Double],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Double],
  c : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  x : FixedArray[Double],
  ldx : Lapack_int,
  rcond : FixedArray[Double],
  ferr : FixedArray[Double],
  berr : FixedArray[Double],
  work : FixedArray[Double],
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgesvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesvxx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, float* a, lapack_int lda, float* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, float* r, 
///     float* c, float* b, lapack_int ldb, float* x, 
///     lapack_int ldx, float* rcond, float* rpvgrw, float* berr, 
///     lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, 
///     float* params, float* work, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, equed, r, c, b, x, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, params, work, iwork)
pub extern "C" fn lapacke_sgesvxx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  af : FixedArray[Float],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Float],
  c : FixedArray[Float],
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
) -> Lapack_int = "LAPACKE_sgesvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesvxx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, double* a, lapack_int lda, double* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, double* r, 
///     double* c, double* b, lapack_int ldb, double* x, 
///     lapack_int ldx, double* rcond, double* rpvgrw, double* berr, 
///     lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, 
///     double* params, double* work, lapack_int* iwork
/// );
/// ```
#borrow(a, af, ipiv, equed, r, c, b, x, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, params, work, iwork)
pub extern "C" fn lapacke_dgesvxx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  af : FixedArray[Double],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Double],
  c : FixedArray[Double],
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
) -> Lapack_int = "LAPACKE_dgesvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesvxx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, float* r, 
///     float* c, lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* rcond, float* rpvgrw, float* berr, 
///     lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, 
///     float* params, lapack_complex_float* work, float* rwork
/// );
/// ```
#borrow(a, af, ipiv, equed, r, c, b, x, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, params, work, rwork)
pub extern "C" fn lapacke_cgesvxx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  af : FixedArray[Float],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Float],
  c : FixedArray[Float],
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
) -> Lapack_int = "LAPACKE_cgesvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesvxx_work(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, double* r, 
///     double* c, lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* rcond, double* rpvgrw, double* berr, 
///     lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, 
///     double* params, lapack_complex_double* work, double* rwork
/// );
/// ```
#borrow(a, af, ipiv, equed, r, c, b, x, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, params, work, rwork)
pub extern "C" fn lapacke_zgesvxx_work(
  matrix_layout : Int,
  fact : Byte,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  af : FixedArray[Double],
  ldaf : Lapack_int,
  ipiv : FixedArray[Int],
  equed : FixedArray[Byte],
  r : FixedArray[Double],
  c : FixedArray[Double],
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
) -> Lapack_int = "LAPACKE_zgesvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgetf2_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_sgetf2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgetf2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgetf2_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_dgetf2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgetf2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgetf2_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_cgetf2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cgetf2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgetf2_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_zgetf2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zgetf2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgetrf_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_sgetrf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgetrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgetrf_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_dgetrf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgetrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgetrf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_cgetrf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cgetrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgetrf_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_zgetrf_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zgetrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgetrf2_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_sgetrf2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgetrf2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgetrf2_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_dgetrf2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgetrf2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgetrf2_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_cgetrf2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cgetrf2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgetrf2_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_zgetrf2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zgetrf2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgetri_work(
///     int matrix_layout, lapack_int n, float* a, lapack_int lda, 
///     const lapack_int* ipiv, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_sgetri_work(
  matrix_layout : Int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgetri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgetri_work(
///     int matrix_layout, lapack_int n, double* a, lapack_int lda, 
///     const lapack_int* ipiv, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_dgetri_work(
  matrix_layout : Int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgetri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgetri_work(
///     int matrix_layout, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     const lapack_int* ipiv, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_cgetri_work(
  matrix_layout : Int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cgetri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgetri_work(
///     int matrix_layout, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     const lapack_int* ipiv, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_zgetri_work(
  matrix_layout : Int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zgetri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgetrs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_sgetrs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_sgetrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgetrs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_dgetrs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dgetrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgetrs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_cgetrs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_cgetrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgetrs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zgetrs_work(
  matrix_layout : Int,
  trans : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zgetrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sggbak_work(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const float* lscale, const float* rscale, 
///     lapack_int m, float* v, lapack_int ldv
/// );
/// ```
#borrow(lscale, rscale, v)
pub extern "C" fn lapacke_sggbak_work(
  matrix_layout : Int,
  job : Byte,
  side : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  lscale : FixedArray[Float],
  rscale : FixedArray[Float],
  m : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
) -> Lapack_int = "LAPACKE_sggbak_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dggbak_work(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const double* lscale, const double* rscale, 
///     lapack_int m, double* v, lapack_int ldv
/// );
/// ```
#borrow(lscale, rscale, v)
pub extern "C" fn lapacke_dggbak_work(
  matrix_layout : Int,
  job : Byte,
  side : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  lscale : FixedArray[Double],
  rscale : FixedArray[Double],
  m : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
) -> Lapack_int = "LAPACKE_dggbak_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cggbak_work(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const float* lscale, const float* rscale, 
///     lapack_int m, lapack_complex_float* v, lapack_int ldv
/// );
/// ```
#borrow(lscale, rscale, v)
pub extern "C" fn lapacke_cggbak_work(
  matrix_layout : Int,
  job : Byte,
  side : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  lscale : FixedArray[Float],
  rscale : FixedArray[Float],
  m : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
) -> Lapack_int = "LAPACKE_cggbak_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zggbak_work(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const double* lscale, const double* rscale, 
///     lapack_int m, lapack_complex_double* v, lapack_int ldv
/// );
/// ```
#borrow(lscale, rscale, v)
pub extern "C" fn lapacke_zggbak_work(
  matrix_layout : Int,
  job : Byte,
  side : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  lscale : FixedArray[Double],
  rscale : FixedArray[Double],
  m : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
) -> Lapack_int = "LAPACKE_zggbak_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sggbal_work(
///     int matrix_layout, char job, lapack_int n, float* a, 
///     lapack_int lda, float* b, lapack_int ldb, lapack_int* ilo, 
///     lapack_int* ihi, float* lscale, float* rscale, float* work
/// );
/// ```
#borrow(a, b, ilo, ihi, lscale, rscale, work)
pub extern "C" fn lapacke_sggbal_work(
  matrix_layout : Int,
  job : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  lscale : FixedArray[Float],
  rscale : FixedArray[Float],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sggbal_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dggbal_work(
///     int matrix_layout, char job, lapack_int n, double* a, 
///     lapack_int lda, double* b, lapack_int ldb, lapack_int* ilo, 
///     lapack_int* ihi, double* lscale, double* rscale, double* work
/// );
/// ```
#borrow(a, b, ilo, ihi, lscale, rscale, work)
pub extern "C" fn lapacke_dggbal_work(
  matrix_layout : Int,
  job : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  lscale : FixedArray[Double],
  rscale : FixedArray[Double],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dggbal_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cggbal_work(
///     int matrix_layout, char job, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* b, lapack_int ldb, lapack_int* ilo, 
///     lapack_int* ihi, float* lscale, float* rscale, float* work
/// );
/// ```
#borrow(a, b, ilo, ihi, lscale, rscale, work)
pub extern "C" fn lapacke_cggbal_work(
  matrix_layout : Int,
  job : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  lscale : FixedArray[Float],
  rscale : FixedArray[Float],
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cggbal_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zggbal_work(
///     int matrix_layout, char job, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* b, lapack_int ldb, lapack_int* ilo, 
///     lapack_int* ihi, double* lscale, double* rscale, double* work
/// );
/// ```
#borrow(a, b, ilo, ihi, lscale, rscale, work)
pub extern "C" fn lapacke_zggbal_work(
  matrix_layout : Int,
  job : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  lscale : FixedArray[Double],
  rscale : FixedArray[Double],
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zggbal_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgges_work(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_S_SELECT3 selctg, lapack_int n, float* a, lapack_int lda, 
///     float* b, lapack_int ldb, lapack_int* sdim, float* alphar, 
///     float* alphai, float* beta, float* vsl, lapack_int ldvsl, 
///     float* vsr, lapack_int ldvsr, float* work, lapack_int lwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, sdim, alphar, alphai, beta, vsl, vsr, work, bwork)
pub extern "C" fn lapacke_sgges_work(
  matrix_layout : Int,
  jobvsl : Byte,
  jobvsr : Byte,
  sort : Byte,
  selctg : Lapack_s_select3,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  sdim : FixedArray[Int],
  alphar : FixedArray[Float],
  alphai : FixedArray[Float],
  beta : FixedArray[Float],
  vsl : FixedArray[Float],
  ldvsl : Lapack_int,
  vsr : FixedArray[Float],
  ldvsr : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_sgges_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgges_work(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_D_SELECT3 selctg, lapack_int n, double* a, lapack_int lda, 
///     double* b, lapack_int ldb, lapack_int* sdim, double* alphar, 
///     double* alphai, double* beta, double* vsl, lapack_int ldvsl, 
///     double* vsr, lapack_int ldvsr, double* work, lapack_int lwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, sdim, alphar, alphai, beta, vsl, vsr, work, bwork)
pub extern "C" fn lapacke_dgges_work(
  matrix_layout : Int,
  jobvsl : Byte,
  jobvsr : Byte,
  sort : Byte,
  selctg : Lapack_d_select3,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  sdim : FixedArray[Int],
  alphar : FixedArray[Double],
  alphai : FixedArray[Double],
  beta : FixedArray[Double],
  vsl : FixedArray[Double],
  ldvsl : Lapack_int,
  vsr : FixedArray[Double],
  ldvsr : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_dgges_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgges_work(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_C_SELECT2 selctg, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb, lapack_int* sdim, lapack_complex_float* alpha, 
///     lapack_complex_float* beta, lapack_complex_float* vsl, lapack_int ldvsl, lapack_complex_float* vsr, 
///     lapack_int ldvsr, lapack_complex_float* work, lapack_int lwork, float* rwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, sdim, alpha, beta, vsl, vsr, work, rwork, bwork)
pub extern "C" fn lapacke_cgges_work(
  matrix_layout : Int,
  jobvsl : Byte,
  jobvsr : Byte,
  sort : Byte,
  selctg : Lapack_c_select2,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  sdim : FixedArray[Int],
  alpha : FixedArray[Float],
  beta : FixedArray[Float],
  vsl : FixedArray[Float],
  ldvsl : Lapack_int,
  vsr : FixedArray[Float],
  ldvsr : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_cgges_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgges_work(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_Z_SELECT2 selctg, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb, lapack_int* sdim, lapack_complex_double* alpha, 
///     lapack_complex_double* beta, lapack_complex_double* vsl, lapack_int ldvsl, lapack_complex_double* vsr, 
///     lapack_int ldvsr, lapack_complex_double* work, lapack_int lwork, double* rwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, sdim, alpha, beta, vsl, vsr, work, rwork, bwork)
pub extern "C" fn lapacke_zgges_work(
  matrix_layout : Int,
  jobvsl : Byte,
  jobvsr : Byte,
  sort : Byte,
  selctg : Lapack_z_select2,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  sdim : FixedArray[Int],
  alpha : FixedArray[Double],
  beta : FixedArray[Double],
  vsl : FixedArray[Double],
  ldvsl : Lapack_int,
  vsr : FixedArray[Double],
  ldvsr : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_zgges_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgges3_work(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_S_SELECT3 selctg, lapack_int n, float* a, lapack_int lda, 
///     float* b, lapack_int ldb, lapack_int* sdim, float* alphar, 
///     float* alphai, float* beta, float* vsl, lapack_int ldvsl, 
///     float* vsr, lapack_int ldvsr, float* work, lapack_int lwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, sdim, alphar, alphai, beta, vsl, vsr, work, bwork)
pub extern "C" fn lapacke_sgges3_work(
  matrix_layout : Int,
  jobvsl : Byte,
  jobvsr : Byte,
  sort : Byte,
  selctg : Lapack_s_select3,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  sdim : FixedArray[Int],
  alphar : FixedArray[Float],
  alphai : FixedArray[Float],
  beta : FixedArray[Float],
  vsl : FixedArray[Float],
  ldvsl : Lapack_int,
  vsr : FixedArray[Float],
  ldvsr : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_sgges3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgges3_work(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_D_SELECT3 selctg, lapack_int n, double* a, lapack_int lda, 
///     double* b, lapack_int ldb, lapack_int* sdim, double* alphar, 
///     double* alphai, double* beta, double* vsl, lapack_int ldvsl, 
///     double* vsr, lapack_int ldvsr, double* work, lapack_int lwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, sdim, alphar, alphai, beta, vsl, vsr, work, bwork)
pub extern "C" fn lapacke_dgges3_work(
  matrix_layout : Int,
  jobvsl : Byte,
  jobvsr : Byte,
  sort : Byte,
  selctg : Lapack_d_select3,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  sdim : FixedArray[Int],
  alphar : FixedArray[Double],
  alphai : FixedArray[Double],
  beta : FixedArray[Double],
  vsl : FixedArray[Double],
  ldvsl : Lapack_int,
  vsr : FixedArray[Double],
  ldvsr : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_dgges3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgges3_work(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_C_SELECT2 selctg, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb, lapack_int* sdim, lapack_complex_float* alpha, 
///     lapack_complex_float* beta, lapack_complex_float* vsl, lapack_int ldvsl, lapack_complex_float* vsr, 
///     lapack_int ldvsr, lapack_complex_float* work, lapack_int lwork, float* rwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, sdim, alpha, beta, vsl, vsr, work, rwork, bwork)
pub extern "C" fn lapacke_cgges3_work(
  matrix_layout : Int,
  jobvsl : Byte,
  jobvsr : Byte,
  sort : Byte,
  selctg : Lapack_c_select2,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  sdim : FixedArray[Int],
  alpha : FixedArray[Float],
  beta : FixedArray[Float],
  vsl : FixedArray[Float],
  ldvsl : Lapack_int,
  vsr : FixedArray[Float],
  ldvsr : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_cgges3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgges3_work(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_Z_SELECT2 selctg, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb, lapack_int* sdim, lapack_complex_double* alpha, 
///     lapack_complex_double* beta, lapack_complex_double* vsl, lapack_int ldvsl, lapack_complex_double* vsr, 
///     lapack_int ldvsr, lapack_complex_double* work, lapack_int lwork, double* rwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, sdim, alpha, beta, vsl, vsr, work, rwork, bwork)
pub extern "C" fn lapacke_zgges3_work(
  matrix_layout : Int,
  jobvsl : Byte,
  jobvsr : Byte,
  sort : Byte,
  selctg : Lapack_z_select2,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  sdim : FixedArray[Int],
  alpha : FixedArray[Double],
  beta : FixedArray[Double],
  vsl : FixedArray[Double],
  ldvsl : Lapack_int,
  vsr : FixedArray[Double],
  ldvsr : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_zgges3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sggesx_work(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_S_SELECT3 selctg, char sense, lapack_int n, float* a, 
///     lapack_int lda, float* b, lapack_int ldb, lapack_int* sdim, 
///     float* alphar, float* alphai, float* beta, float* vsl, 
///     lapack_int ldvsl, float* vsr, lapack_int ldvsr, float* rconde, 
///     float* rcondv, float* work, lapack_int lwork, lapack_int* iwork, 
///     lapack_int liwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, sdim, alphar, alphai, beta, vsl, vsr, rconde, rcondv, work, iwork, bwork)
pub extern "C" fn lapacke_sggesx_work(
  matrix_layout : Int,
  jobvsl : Byte,
  jobvsr : Byte,
  sort : Byte,
  selctg : Lapack_s_select3,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  sdim : FixedArray[Int],
  alphar : FixedArray[Float],
  alphai : FixedArray[Float],
  beta : FixedArray[Float],
  vsl : FixedArray[Float],
  ldvsl : Lapack_int,
  vsr : FixedArray[Float],
  ldvsr : Lapack_int,
  rconde : FixedArray[Float],
  rcondv : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_sggesx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dggesx_work(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_D_SELECT3 selctg, char sense, lapack_int n, double* a, 
///     lapack_int lda, double* b, lapack_int ldb, lapack_int* sdim, 
///     double* alphar, double* alphai, double* beta, double* vsl, 
///     lapack_int ldvsl, double* vsr, lapack_int ldvsr, double* rconde, 
///     double* rcondv, double* work, lapack_int lwork, lapack_int* iwork, 
///     lapack_int liwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, sdim, alphar, alphai, beta, vsl, vsr, rconde, rcondv, work, iwork, bwork)
pub extern "C" fn lapacke_dggesx_work(
  matrix_layout : Int,
  jobvsl : Byte,
  jobvsr : Byte,
  sort : Byte,
  selctg : Lapack_d_select3,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  sdim : FixedArray[Int],
  alphar : FixedArray[Double],
  alphai : FixedArray[Double],
  beta : FixedArray[Double],
  vsl : FixedArray[Double],
  ldvsl : Lapack_int,
  vsr : FixedArray[Double],
  ldvsr : Lapack_int,
  rconde : FixedArray[Double],
  rcondv : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  liwork : Lapack_int,
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_dggesx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cggesx_work(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_C_SELECT2 selctg, char sense, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* b, lapack_int ldb, lapack_int* sdim, 
///     lapack_complex_float* alpha, lapack_complex_float* beta, lapack_complex_float* vsl, lapack_int ldvsl, 
///     lapack_complex_float* vsr, lapack_int ldvsr, float* rconde, float* rcondv, 
///     lapack_complex_float* work, lapack_int lwork, float* rwork, lapack_int* iwork, 
///     lapack_int liwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, sdim, alpha, beta, vsl, vsr, rconde, rcondv, work, rwork, iwork, bwork)
pub extern "C" fn lapacke_cggesx_work(
  matrix_layout : Int,
  jobvsl : Byte,
  jobvsr : Byte,
  sort : Byte,
  selctg : Lapack_c_select2,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  sdim : FixedArray[Int],
  alpha : FixedArray[Float],
  beta : FixedArray[Float],
  vsl : FixedArray[Float],
  ldvsl : Lapack_int,
  vsr : FixedArray[Float],
  ldvsr : Lapack_int,
  rconde : FixedArray[Float],
  rcondv : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  iwork : FixedArray[Int],
  liwork : Lapack_int,
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_cggesx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zggesx_work(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_Z_SELECT2 selctg, char sense, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* b, lapack_int ldb, lapack_int* sdim, 
///     lapack_complex_double* alpha, lapack_complex_double* beta, lapack_complex_double* vsl, lapack_int ldvsl, 
///     lapack_complex_double* vsr, lapack_int ldvsr, double* rconde, double* rcondv, 
///     lapack_complex_double* work, lapack_int lwork, double* rwork, lapack_int* iwork, 
///     lapack_int liwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, sdim, alpha, beta, vsl, vsr, rconde, rcondv, work, rwork, iwork, bwork)
pub extern "C" fn lapacke_zggesx_work(
  matrix_layout : Int,
  jobvsl : Byte,
  jobvsr : Byte,
  sort : Byte,
  selctg : Lapack_z_select2,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  sdim : FixedArray[Int],
  alpha : FixedArray[Double],
  beta : FixedArray[Double],
  vsl : FixedArray[Double],
  ldvsl : Lapack_int,
  vsr : FixedArray[Double],
  ldvsr : Lapack_int,
  rconde : FixedArray[Double],
  rcondv : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  iwork : FixedArray[Int],
  liwork : Lapack_int,
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_zggesx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sggev_work(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* alphar, float* alphai, float* beta, float* vl, 
///     lapack_int ldvl, float* vr, lapack_int ldvr, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, alphar, alphai, beta, vl, vr, work)
pub extern "C" fn lapacke_sggev_work(
  matrix_layout : Int,
  jobvl : Byte,
  jobvr : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  alphar : FixedArray[Float],
  alphai : FixedArray[Float],
  beta : FixedArray[Float],
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sggev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dggev_work(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* alphar, double* alphai, double* beta, double* vl, 
///     lapack_int ldvl, double* vr, lapack_int ldvr, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, alphar, alphai, beta, vl, vr, work)
pub extern "C" fn lapacke_dggev_work(
  matrix_layout : Int,
  jobvl : Byte,
  jobvr : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  alphar : FixedArray[Double],
  alphai : FixedArray[Double],
  beta : FixedArray[Double],
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dggev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cggev_work(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* alpha, lapack_complex_float* beta, lapack_complex_float* vl, lapack_int ldvl, 
///     lapack_complex_float* vr, lapack_int ldvr, lapack_complex_float* work, lapack_int lwork, float* rwork
/// );
/// ```
#borrow(a, b, alpha, beta, vl, vr, work, rwork)
pub extern "C" fn lapacke_cggev_work(
  matrix_layout : Int,
  jobvl : Byte,
  jobvr : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  alpha : FixedArray[Float],
  beta : FixedArray[Float],
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cggev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zggev_work(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* alpha, lapack_complex_double* beta, lapack_complex_double* vl, lapack_int ldvl, 
///     lapack_complex_double* vr, lapack_int ldvr, lapack_complex_double* work, lapack_int lwork, double* rwork
/// );
/// ```
#borrow(a, b, alpha, beta, vl, vr, work, rwork)
pub extern "C" fn lapacke_zggev_work(
  matrix_layout : Int,
  jobvl : Byte,
  jobvr : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  alpha : FixedArray[Double],
  beta : FixedArray[Double],
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zggev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sggev3_work(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* alphar, float* alphai, float* beta, float* vl, 
///     lapack_int ldvl, float* vr, lapack_int ldvr, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, alphar, alphai, beta, vl, vr, work)
pub extern "C" fn lapacke_sggev3_work(
  matrix_layout : Int,
  jobvl : Byte,
  jobvr : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  alphar : FixedArray[Float],
  alphai : FixedArray[Float],
  beta : FixedArray[Float],
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sggev3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dggev3_work(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* alphar, double* alphai, double* beta, double* vl, 
///     lapack_int ldvl, double* vr, lapack_int ldvr, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, alphar, alphai, beta, vl, vr, work)
pub extern "C" fn lapacke_dggev3_work(
  matrix_layout : Int,
  jobvl : Byte,
  jobvr : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  alphar : FixedArray[Double],
  alphai : FixedArray[Double],
  beta : FixedArray[Double],
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dggev3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cggev3_work(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* alpha, lapack_complex_float* beta, lapack_complex_float* vl, lapack_int ldvl, 
///     lapack_complex_float* vr, lapack_int ldvr, lapack_complex_float* work, lapack_int lwork, float* rwork
/// );
/// ```
#borrow(a, b, alpha, beta, vl, vr, work, rwork)
pub extern "C" fn lapacke_cggev3_work(
  matrix_layout : Int,
  jobvl : Byte,
  jobvr : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  alpha : FixedArray[Float],
  beta : FixedArray[Float],
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cggev3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zggev3_work(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* alpha, lapack_complex_double* beta, lapack_complex_double* vl, lapack_int ldvl, 
///     lapack_complex_double* vr, lapack_int ldvr, lapack_complex_double* work, lapack_int lwork, double* rwork
/// );
/// ```
#borrow(a, b, alpha, beta, vl, vr, work, rwork)
pub extern "C" fn lapacke_zggev3_work(
  matrix_layout : Int,
  jobvl : Byte,
  jobvr : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  alpha : FixedArray[Double],
  beta : FixedArray[Double],
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zggev3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sggevx_work(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, float* a, lapack_int lda, 
///     float* b, lapack_int ldb, float* alphar, float* alphai, 
///     float* beta, float* vl, lapack_int ldvl, float* vr, 
///     lapack_int ldvr, lapack_int* ilo, lapack_int* ihi, float* lscale, 
///     float* rscale, float* abnrm, float* bbnrm, float* rconde, 
///     float* rcondv, float* work, lapack_int lwork, lapack_int* iwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, alphar, alphai, beta, vl, vr, ilo, ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv, work, iwork, bwork)
pub extern "C" fn lapacke_sggevx_work(
  matrix_layout : Int,
  balanc : Byte,
  jobvl : Byte,
  jobvr : Byte,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  alphar : FixedArray[Float],
  alphai : FixedArray[Float],
  beta : FixedArray[Float],
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  lscale : FixedArray[Float],
  rscale : FixedArray[Float],
  abnrm : FixedArray[Float],
  bbnrm : FixedArray[Float],
  rconde : FixedArray[Float],
  rcondv : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_sggevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dggevx_work(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, double* a, lapack_int lda, 
///     double* b, lapack_int ldb, double* alphar, double* alphai, 
///     double* beta, double* vl, lapack_int ldvl, double* vr, 
///     lapack_int ldvr, lapack_int* ilo, lapack_int* ihi, double* lscale, 
///     double* rscale, double* abnrm, double* bbnrm, double* rconde, 
///     double* rcondv, double* work, lapack_int lwork, lapack_int* iwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, alphar, alphai, beta, vl, vr, ilo, ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv, work, iwork, bwork)
pub extern "C" fn lapacke_dggevx_work(
  matrix_layout : Int,
  balanc : Byte,
  jobvl : Byte,
  jobvr : Byte,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  alphar : FixedArray[Double],
  alphai : FixedArray[Double],
  beta : FixedArray[Double],
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  lscale : FixedArray[Double],
  rscale : FixedArray[Double],
  abnrm : FixedArray[Double],
  bbnrm : FixedArray[Double],
  rconde : FixedArray[Double],
  rcondv : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_dggevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cggevx_work(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* alpha, lapack_complex_float* beta, 
///     lapack_complex_float* vl, lapack_int ldvl, lapack_complex_float* vr, lapack_int ldvr, 
///     lapack_int* ilo, lapack_int* ihi, float* lscale, float* rscale, 
///     float* abnrm, float* bbnrm, float* rconde, float* rcondv, 
///     lapack_complex_float* work, lapack_int lwork, float* rwork, lapack_int* iwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, alpha, beta, vl, vr, ilo, ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv, work, rwork, iwork, bwork)
pub extern "C" fn lapacke_cggevx_work(
  matrix_layout : Int,
  balanc : Byte,
  jobvl : Byte,
  jobvr : Byte,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  alpha : FixedArray[Float],
  beta : FixedArray[Float],
  vl : FixedArray[Float],
  ldvl : Lapack_int,
  vr : FixedArray[Float],
  ldvr : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  lscale : FixedArray[Float],
  rscale : FixedArray[Float],
  abnrm : FixedArray[Float],
  bbnrm : FixedArray[Float],
  rconde : FixedArray[Float],
  rcondv : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  iwork : FixedArray[Int],
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_cggevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zggevx_work(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* alpha, lapack_complex_double* beta, 
///     lapack_complex_double* vl, lapack_int ldvl, lapack_complex_double* vr, lapack_int ldvr, 
///     lapack_int* ilo, lapack_int* ihi, double* lscale, double* rscale, 
///     double* abnrm, double* bbnrm, double* rconde, double* rcondv, 
///     lapack_complex_double* work, lapack_int lwork, double* rwork, lapack_int* iwork, lapack_logical* bwork
/// );
/// ```
#borrow(a, b, alpha, beta, vl, vr, ilo, ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv, work, rwork, iwork, bwork)
pub extern "C" fn lapacke_zggevx_work(
  matrix_layout : Int,
  balanc : Byte,
  jobvl : Byte,
  jobvr : Byte,
  sense : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  alpha : FixedArray[Double],
  beta : FixedArray[Double],
  vl : FixedArray[Double],
  ldvl : Lapack_int,
  vr : FixedArray[Double],
  ldvr : Lapack_int,
  ilo : FixedArray[Int],
  ihi : FixedArray[Int],
  lscale : FixedArray[Double],
  rscale : FixedArray[Double],
  abnrm : FixedArray[Double],
  bbnrm : FixedArray[Double],
  rconde : FixedArray[Double],
  rcondv : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  iwork : FixedArray[Int],
  bwork : Lapack_logical,
) -> Lapack_int = "LAPACKE_zggevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sggglm_work(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* d, float* x, float* y, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, d, x, y, work)
pub extern "C" fn lapacke_sggglm_work(
  matrix_layout : Int,
  n : Lapack_int,
  m : Lapack_int,
  p : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  d : FixedArray[Float],
  x : FixedArray[Float],
  y : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sggglm_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dggglm_work(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* d, double* x, double* y, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, d, x, y, work)
pub extern "C" fn lapacke_dggglm_work(
  matrix_layout : Int,
  n : Lapack_int,
  m : Lapack_int,
  p : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  d : FixedArray[Double],
  x : FixedArray[Double],
  y : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dggglm_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cggglm_work(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* d, lapack_complex_float* x, lapack_complex_float* y, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, d, x, y, work)
pub extern "C" fn lapacke_cggglm_work(
  matrix_layout : Int,
  n : Lapack_int,
  m : Lapack_int,
  p : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  d : FixedArray[Float],
  x : FixedArray[Float],
  y : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cggglm_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zggglm_work(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* d, lapack_complex_double* x, lapack_complex_double* y, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, d, x, y, work)
pub extern "C" fn lapacke_zggglm_work(
  matrix_layout : Int,
  n : Lapack_int,
  m : Lapack_int,
  p : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  d : FixedArray[Double],
  x : FixedArray[Double],
  y : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zggglm_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgghrd_work(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, float* a, lapack_int lda, 
///     float* b, lapack_int ldb, float* q, lapack_int ldq, 
///     float* z, lapack_int ldz
/// );
/// ```
#borrow(a, b, q, z)
pub extern "C" fn lapacke_sgghrd_work(
  matrix_layout : Int,
  compq : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  z : FixedArray[Float],
  ldz : Lapack_int,
) -> Lapack_int = "LAPACKE_sgghrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgghrd_work(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, double* a, lapack_int lda, 
///     double* b, lapack_int ldb, double* q, lapack_int ldq, 
///     double* z, lapack_int ldz
/// );
/// ```
#borrow(a, b, q, z)
pub extern "C" fn lapacke_dgghrd_work(
  matrix_layout : Int,
  compq : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  z : FixedArray[Double],
  ldz : Lapack_int,
) -> Lapack_int = "LAPACKE_dgghrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgghrd_work(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* q, lapack_int ldq, 
///     lapack_complex_float* z, lapack_int ldz
/// );
/// ```
#borrow(a, b, q, z)
pub extern "C" fn lapacke_cgghrd_work(
  matrix_layout : Int,
  compq : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  z : FixedArray[Float],
  ldz : Lapack_int,
) -> Lapack_int = "LAPACKE_cgghrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgghrd_work(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* q, lapack_int ldq, 
///     lapack_complex_double* z, lapack_int ldz
/// );
/// ```
#borrow(a, b, q, z)
pub extern "C" fn lapacke_zgghrd_work(
  matrix_layout : Int,
  compq : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  z : FixedArray[Double],
  ldz : Lapack_int,
) -> Lapack_int = "LAPACKE_zgghrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgghd3_work(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, float* a, lapack_int lda, 
///     float* b, lapack_int ldb, float* q, lapack_int ldq, 
///     float* z, lapack_int ldz, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, q, z, work)
pub extern "C" fn lapacke_sgghd3_work(
  matrix_layout : Int,
  compq : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgghd3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgghd3_work(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, double* a, lapack_int lda, 
///     double* b, lapack_int ldb, double* q, lapack_int ldq, 
///     double* z, lapack_int ldz, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, q, z, work)
pub extern "C" fn lapacke_dgghd3_work(
  matrix_layout : Int,
  compq : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgghd3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgghd3_work(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* q, lapack_int ldq, 
///     lapack_complex_float* z, lapack_int ldz, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, q, z, work)
pub extern "C" fn lapacke_cgghd3_work(
  matrix_layout : Int,
  compq : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  z : FixedArray[Float],
  ldz : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cgghd3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgghd3_work(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* q, lapack_int ldq, 
///     lapack_complex_double* z, lapack_int ldz, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, q, z, work)
pub extern "C" fn lapacke_zgghd3_work(
  matrix_layout : Int,
  compq : Byte,
  compz : Byte,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  z : FixedArray[Double],
  ldz : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zgghd3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgglse_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int p, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* c, float* d, float* x, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, c, d, x, work)
pub extern "C" fn lapacke_sgglse_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  p : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  c : FixedArray[Float],
  d : FixedArray[Float],
  x : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sgglse_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgglse_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int p, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* c, double* d, double* x, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, c, d, x, work)
pub extern "C" fn lapacke_dgglse_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  p : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  c : FixedArray[Double],
  d : FixedArray[Double],
  x : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dgglse_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgglse_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int p, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* c, lapack_complex_float* d, lapack_complex_float* x, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, c, d, x, work)
pub extern "C" fn lapacke_cgglse_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  p : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  c : FixedArray[Float],
  d : FixedArray[Float],
  x : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cgglse_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgglse_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int p, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* c, lapack_complex_double* d, lapack_complex_double* x, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, b, c, d, x, work)
pub extern "C" fn lapacke_zgglse_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  p : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  c : FixedArray[Double],
  d : FixedArray[Double],
  x : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zgglse_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sggqrf_work(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     float* a, lapack_int lda, float* taua, float* b, 
///     lapack_int ldb, float* taub, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, taua, b, taub, work)
pub extern "C" fn lapacke_sggqrf_work(
  matrix_layout : Int,
  n : Lapack_int,
  m : Lapack_int,
  p : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  taua : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  taub : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sggqrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dggqrf_work(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     double* a, lapack_int lda, double* taua, double* b, 
///     lapack_int ldb, double* taub, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, taua, b, taub, work)
pub extern "C" fn lapacke_dggqrf_work(
  matrix_layout : Int,
  n : Lapack_int,
  m : Lapack_int,
  p : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  taua : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  taub : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dggqrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cggqrf_work(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* taua, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* taub, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, taua, b, taub, work)
pub extern "C" fn lapacke_cggqrf_work(
  matrix_layout : Int,
  n : Lapack_int,
  m : Lapack_int,
  p : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  taua : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  taub : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cggqrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zggqrf_work(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* taua, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* taub, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, taua, b, taub, work)
pub extern "C" fn lapacke_zggqrf_work(
  matrix_layout : Int,
  n : Lapack_int,
  m : Lapack_int,
  p : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  taua : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  taub : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zggqrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sggrqf_work(
///     int matrix_layout, lapack_int m, lapack_int p, lapack_int n, 
///     float* a, lapack_int lda, float* taua, float* b, 
///     lapack_int ldb, float* taub, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, taua, b, taub, work)
pub extern "C" fn lapacke_sggrqf_work(
  matrix_layout : Int,
  m : Lapack_int,
  p : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  taua : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  taub : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sggrqf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dggrqf_work(
///     int matrix_layout, lapack_int m, lapack_int p, lapack_int n, 
///     double* a, lapack_int lda, double* taua, double* b, 
///     lapack_int ldb, double* taub, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, taua, b, taub, work)
pub extern "C" fn lapacke_dggrqf_work(
  matrix_layout : Int,
  m : Lapack_int,
  p : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  taua : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  taub : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dggrqf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cggrqf_work(
///     int matrix_layout, lapack_int m, lapack_int p, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* taua, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* taub, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, taua, b, taub, work)
pub extern "C" fn lapacke_cggrqf_work(
  matrix_layout : Int,
  m : Lapack_int,
  p : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  taua : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
  taub : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cggrqf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zggrqf_work(
///     int matrix_layout, lapack_int m, lapack_int p, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* taua, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* taub, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, taua, b, taub, work)
pub extern "C" fn lapacke_zggrqf_work(
  matrix_layout : Int,
  m : Lapack_int,
  p : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  taua : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
  taub : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zggrqf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sggsvd_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, float* a, lapack_int lda, float* b, 
///     lapack_int ldb, float* alpha, float* beta, float* u, 
///     lapack_int ldu, float* v, lapack_int ldv, float* q, 
///     lapack_int ldq, float* work, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, work, iwork)
pub extern "C" fn lapacke_sggsvd_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  n : Lapack_int,
  p : Lapack_int,
  k : FixedArray[Int],
  l : FixedArray[Int],
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  alpha : FixedArray[Float],
  beta : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  work : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sggsvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dggsvd_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, double* a, lapack_int lda, double* b, 
///     lapack_int ldb, double* alpha, double* beta, double* u, 
///     lapack_int ldu, double* v, lapack_int ldv, double* q, 
///     lapack_int ldq, double* work, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, work, iwork)
pub extern "C" fn lapacke_dggsvd_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  n : Lapack_int,
  p : Lapack_int,
  k : FixedArray[Int],
  l : FixedArray[Int],
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  alpha : FixedArray[Double],
  beta : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  work : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dggsvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cggsvd_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, 
///     lapack_int ldb, float* alpha, float* beta, lapack_complex_float* u, 
///     lapack_int ldu, lapack_complex_float* v, lapack_int ldv, lapack_complex_float* q, 
///     lapack_int ldq, lapack_complex_float* work, float* rwork, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, work, rwork, iwork)
pub extern "C" fn lapacke_cggsvd_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  n : Lapack_int,
  p : Lapack_int,
  k : FixedArray[Int],
  l : FixedArray[Int],
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  alpha : FixedArray[Float],
  beta : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  work : FixedArray[Float],
  rwork : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cggsvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zggsvd_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, 
///     lapack_int ldb, double* alpha, double* beta, lapack_complex_double* u, 
///     lapack_int ldu, lapack_complex_double* v, lapack_int ldv, lapack_complex_double* q, 
///     lapack_int ldq, lapack_complex_double* work, double* rwork, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, work, rwork, iwork)
pub extern "C" fn lapacke_zggsvd_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  n : Lapack_int,
  p : Lapack_int,
  k : FixedArray[Int],
  l : FixedArray[Int],
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  alpha : FixedArray[Double],
  beta : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  work : FixedArray[Double],
  rwork : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zggsvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sggsvd3_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, float* a, lapack_int lda, float* b, 
///     lapack_int ldb, float* alpha, float* beta, float* u, 
///     lapack_int ldu, float* v, lapack_int ldv, float* q, 
///     lapack_int ldq, float* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, work, iwork)
pub extern "C" fn lapacke_sggsvd3_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  n : Lapack_int,
  p : Lapack_int,
  k : FixedArray[Int],
  l : FixedArray[Int],
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  alpha : FixedArray[Float],
  beta : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sggsvd3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dggsvd3_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, double* a, lapack_int lda, double* b, 
///     lapack_int ldb, double* alpha, double* beta, double* u, 
///     lapack_int ldu, double* v, lapack_int ldv, double* q, 
///     lapack_int ldq, double* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, work, iwork)
pub extern "C" fn lapacke_dggsvd3_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  n : Lapack_int,
  p : Lapack_int,
  k : FixedArray[Int],
  l : FixedArray[Int],
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  alpha : FixedArray[Double],
  beta : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dggsvd3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cggsvd3_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, 
///     lapack_int ldb, float* alpha, float* beta, lapack_complex_float* u, 
///     lapack_int ldu, lapack_complex_float* v, lapack_int ldv, lapack_complex_float* q, 
///     lapack_int ldq, lapack_complex_float* work, lapack_int lwork, float* rwork, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, work, rwork, iwork)
pub extern "C" fn lapacke_cggsvd3_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  n : Lapack_int,
  p : Lapack_int,
  k : FixedArray[Int],
  l : FixedArray[Int],
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  alpha : FixedArray[Float],
  beta : FixedArray[Float],
  u : FixedArray[Float],
  ldu : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  q : FixedArray[Float],
  ldq : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cggsvd3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zggsvd3_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, 
///     lapack_int ldb, double* alpha, double* beta, lapack_complex_double* u, 
///     lapack_int ldu, lapack_complex_double* v, lapack_int ldv, lapack_complex_double* q, 
///     lapack_int ldq, lapack_complex_double* work, lapack_int lwork, double* rwork, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, work, rwork, iwork)
pub extern "C" fn lapacke_zggsvd3_work(
  matrix_layout : Int,
  jobu : Byte,
  jobv : Byte,
  jobq : Byte,
  m : Lapack_int,
  n : Lapack_int,
  p : Lapack_int,
  k : FixedArray[Int],
  l : FixedArray[Int],
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  alpha : FixedArray[Double],
  beta : FixedArray[Double],
  u : FixedArray[Double],
  ldu : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  q : FixedArray[Double],
  ldq : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zggsvd3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sggsvp_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, float* a, 
///     lapack_int lda, float* b, lapack_int ldb, float tola, 
///     float tolb, lapack_int* k, lapack_int* l, float* u, 
///     lapack_int ldu, float* v, lapack_int ldv, float* q, 
///     lapack_int ldq, lapack_int* iwork, float* tau, float* work
/// );
/// ```
#borrow(a, b, k, l, u, v, q, iwork, tau, work)
pub extern "C" fn lapacke_sggsvp_work(
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
) -> Lapack_int = "LAPACKE_sggsvp_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dggsvp_work(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, double* a, 
///     lapack_int lda, double* b, lapack_int ldb, double tola, 
///     double tolb, lapack_int* k, lapack_int* l, double* u, 
///     lapack_int ldu, double* v, lapack_int ldv, double* q, 
///     lapack_int ldq, lapack_int* iwork, double* tau, double* work
/// );
/// ```
#borrow(a, b, k, l, u, v, q, iwork, tau, work)
pub extern "C" fn lapacke_dggsvp_work(
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
) -> Lapack_int = "LAPACKE_dggsvp_work"
