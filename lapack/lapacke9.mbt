///|
///
/// ```c
/// lapack_int LAPACKE_chetrs2(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_chetrs2(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chetrs2"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrs2_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* work
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_chetrs2_work(
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
) -> Lapack_int = "LAPACKE_chetrs2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csyconv(
///     int matrix_layout, char uplo, char way, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* e
/// );
/// ```
#borrow(a, ipiv, e)
pub extern "C" fn lapacke_csyconv(
  matrix_layout : Int,
  uplo : Byte,
  way : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  e : FixedArray[Float],
) -> Lapack_int = "LAPACKE_csyconv"

///|
///
/// ```c
/// lapack_int LAPACKE_csyconv_work(
///     int matrix_layout, char uplo, char way, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* e
/// );
/// ```
#borrow(a, ipiv, e)
pub extern "C" fn lapacke_csyconv_work(
  matrix_layout : Int,
  uplo : Byte,
  way : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  e : FixedArray[Float],
) -> Lapack_int = "LAPACKE_csyconv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csyswapr(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int i1, lapack_int i2
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_csyswapr(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  i1 : Lapack_int,
  i2 : Lapack_int,
) -> Lapack_int = "LAPACKE_csyswapr"

///|
///
/// ```c
/// lapack_int LAPACKE_csyswapr_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int i1, lapack_int i2
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_csyswapr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  i1 : Lapack_int,
  i2 : Lapack_int,
) -> Lapack_int = "LAPACKE_csyswapr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csytri2(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_csytri2(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_csytri2"

///|
///
/// ```c
/// lapack_int LAPACKE_csytri2_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_csytri2_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_csytri2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csytri2x(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_int nb
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_csytri2x(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  nb : Lapack_int,
) -> Lapack_int = "LAPACKE_csytri2x"

///|
///
/// ```c
/// lapack_int LAPACKE_csytri2x_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_complex_float* work, lapack_int nb
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_csytri2x_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  nb : Lapack_int,
) -> Lapack_int = "LAPACKE_csytri2x_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrs2(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_csytrs2(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_csytrs2"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrs2_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* work
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_csytrs2_work(
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
) -> Lapack_int = "LAPACKE_csytrs2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cunbdb(
///     int matrix_layout, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, lapack_complex_float* x11, lapack_int ldx11, 
///     lapack_complex_float* x12, lapack_int ldx12, lapack_complex_float* x21, lapack_int ldx21, 
///     lapack_complex_float* x22, lapack_int ldx22, float* theta, float* phi, 
///     lapack_complex_float* taup1, lapack_complex_float* taup2, lapack_complex_float* tauq1, lapack_complex_float* tauq2
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, phi, taup1, taup2, tauq1, tauq2)
pub extern "C" fn lapacke_cunbdb(
  matrix_layout : Int,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Float],
  ldx11 : Lapack_int,
  x12 : FixedArray[Float],
  ldx12 : Lapack_int,
  x21 : FixedArray[Float],
  ldx21 : Lapack_int,
  x22 : FixedArray[Float],
  ldx22 : Lapack_int,
  theta : FixedArray[Float],
  phi : FixedArray[Float],
  taup1 : FixedArray[Float],
  taup2 : FixedArray[Float],
  tauq1 : FixedArray[Float],
  tauq2 : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cunbdb"

///|
///
/// ```c
/// lapack_int LAPACKE_cunbdb_work(
///     int matrix_layout, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, lapack_complex_float* x11, lapack_int ldx11, 
///     lapack_complex_float* x12, lapack_int ldx12, lapack_complex_float* x21, lapack_int ldx21, 
///     lapack_complex_float* x22, lapack_int ldx22, float* theta, float* phi, 
///     lapack_complex_float* taup1, lapack_complex_float* taup2, lapack_complex_float* tauq1, lapack_complex_float* tauq2, 
///     lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, phi, taup1, taup2, tauq1, tauq2, work)
pub extern "C" fn lapacke_cunbdb_work(
  matrix_layout : Int,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Float],
  ldx11 : Lapack_int,
  x12 : FixedArray[Float],
  ldx12 : Lapack_int,
  x21 : FixedArray[Float],
  ldx21 : Lapack_int,
  x22 : FixedArray[Float],
  ldx22 : Lapack_int,
  theta : FixedArray[Float],
  phi : FixedArray[Float],
  taup1 : FixedArray[Float],
  taup2 : FixedArray[Float],
  tauq1 : FixedArray[Float],
  tauq2 : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_cunbdb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cuncsd(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, lapack_complex_float* x11, lapack_int ldx11, 
///     lapack_complex_float* x12, lapack_int ldx12, lapack_complex_float* x21, lapack_int ldx21, 
///     lapack_complex_float* x22, lapack_int ldx22, float* theta, lapack_complex_float* u1, 
///     lapack_int ldu1, lapack_complex_float* u2, lapack_int ldu2, lapack_complex_float* v1t, 
///     lapack_int ldv1t, lapack_complex_float* v2t, lapack_int ldv2t
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, u1, u2, v1t, v2t)
pub extern "C" fn lapacke_cuncsd(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  jobv2t : Byte,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Float],
  ldx11 : Lapack_int,
  x12 : FixedArray[Float],
  ldx12 : Lapack_int,
  x21 : FixedArray[Float],
  ldx21 : Lapack_int,
  x22 : FixedArray[Float],
  ldx22 : Lapack_int,
  theta : FixedArray[Float],
  u1 : FixedArray[Float],
  ldu1 : Lapack_int,
  u2 : FixedArray[Float],
  ldu2 : Lapack_int,
  v1t : FixedArray[Float],
  ldv1t : Lapack_int,
  v2t : FixedArray[Float],
  ldv2t : Lapack_int,
) -> Lapack_int = "LAPACKE_cuncsd"

///|
///
/// ```c
/// lapack_int LAPACKE_cuncsd_work(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, lapack_complex_float* x11, lapack_int ldx11, 
///     lapack_complex_float* x12, lapack_int ldx12, lapack_complex_float* x21, lapack_int ldx21, 
///     lapack_complex_float* x22, lapack_int ldx22, float* theta, lapack_complex_float* u1, 
///     lapack_int ldu1, lapack_complex_float* u2, lapack_int ldu2, lapack_complex_float* v1t, 
///     lapack_int ldv1t, lapack_complex_float* v2t, lapack_int ldv2t, lapack_complex_float* work, 
///     lapack_int lwork, float* rwork, lapack_int lrwork, lapack_int* iwork
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, u1, u2, v1t, v2t, work, rwork, iwork)
pub extern "C" fn lapacke_cuncsd_work(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  jobv2t : Byte,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Float],
  ldx11 : Lapack_int,
  x12 : FixedArray[Float],
  ldx12 : Lapack_int,
  x21 : FixedArray[Float],
  ldx21 : Lapack_int,
  x22 : FixedArray[Float],
  ldx22 : Lapack_int,
  theta : FixedArray[Float],
  u1 : FixedArray[Float],
  ldu1 : Lapack_int,
  u2 : FixedArray[Float],
  ldu2 : Lapack_int,
  v1t : FixedArray[Float],
  ldv1t : Lapack_int,
  v2t : FixedArray[Float],
  ldv2t : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cuncsd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cuncsd2by1(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     lapack_int m, lapack_int p, lapack_int q, lapack_complex_float* x11, 
///     lapack_int ldx11, lapack_complex_float* x21, lapack_int ldx21, float* theta, 
///     lapack_complex_float* u1, lapack_int ldu1, lapack_complex_float* u2, lapack_int ldu2, 
///     lapack_complex_float* v1t, lapack_int ldv1t
/// );
/// ```
#borrow(x11, x21, theta, u1, u2, v1t)
pub extern "C" fn lapacke_cuncsd2by1(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Float],
  ldx11 : Lapack_int,
  x21 : FixedArray[Float],
  ldx21 : Lapack_int,
  theta : FixedArray[Float],
  u1 : FixedArray[Float],
  ldu1 : Lapack_int,
  u2 : FixedArray[Float],
  ldu2 : Lapack_int,
  v1t : FixedArray[Float],
  ldv1t : Lapack_int,
) -> Lapack_int = "LAPACKE_cuncsd2by1"

///|
///
/// ```c
/// lapack_int LAPACKE_cuncsd2by1_work(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     lapack_int m, lapack_int p, lapack_int q, lapack_complex_float* x11, 
///     lapack_int ldx11, lapack_complex_float* x21, lapack_int ldx21, float* theta, 
///     lapack_complex_float* u1, lapack_int ldu1, lapack_complex_float* u2, lapack_int ldu2, 
///     lapack_complex_float* v1t, lapack_int ldv1t, lapack_complex_float* work, lapack_int lwork, 
///     float* rwork, lapack_int lrwork, lapack_int* iwork
/// );
/// ```
#borrow(x11, x21, theta, u1, u2, v1t, work, rwork, iwork)
pub extern "C" fn lapacke_cuncsd2by1_work(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Float],
  ldx11 : Lapack_int,
  x21 : FixedArray[Float],
  ldx21 : Lapack_int,
  theta : FixedArray[Float],
  u1 : FixedArray[Float],
  ldu1 : Lapack_int,
  u2 : FixedArray[Float],
  ldu2 : Lapack_int,
  v1t : FixedArray[Float],
  ldv1t : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  rwork : FixedArray[Float],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cuncsd2by1_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dbbcsd(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, lapack_int m, lapack_int p, 
///     lapack_int q, double* theta, double* phi, double* u1, 
///     lapack_int ldu1, double* u2, lapack_int ldu2, double* v1t, 
///     lapack_int ldv1t, double* v2t, lapack_int ldv2t, double* b11d, 
///     double* b11e, double* b12d, double* b12e, double* b21d, 
///     double* b21e, double* b22d, double* b22e
/// );
/// ```
#borrow(theta, phi, u1, u2, v1t, v2t, b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e)
pub extern "C" fn lapacke_dbbcsd(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  jobv2t : Byte,
  trans : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  theta : FixedArray[Double],
  phi : FixedArray[Double],
  u1 : FixedArray[Double],
  ldu1 : Lapack_int,
  u2 : FixedArray[Double],
  ldu2 : Lapack_int,
  v1t : FixedArray[Double],
  ldv1t : Lapack_int,
  v2t : FixedArray[Double],
  ldv2t : Lapack_int,
  b11d : FixedArray[Double],
  b11e : FixedArray[Double],
  b12d : FixedArray[Double],
  b12e : FixedArray[Double],
  b21d : FixedArray[Double],
  b21e : FixedArray[Double],
  b22d : FixedArray[Double],
  b22e : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dbbcsd"

///|
///
/// ```c
/// lapack_int LAPACKE_dbbcsd_work(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, lapack_int m, lapack_int p, 
///     lapack_int q, double* theta, double* phi, double* u1, 
///     lapack_int ldu1, double* u2, lapack_int ldu2, double* v1t, 
///     lapack_int ldv1t, double* v2t, lapack_int ldv2t, double* b11d, 
///     double* b11e, double* b12d, double* b12e, double* b21d, 
///     double* b21e, double* b22d, double* b22e, double* work, lapack_int lwork
/// );
/// ```
#borrow(theta, phi, u1, u2, v1t, v2t, b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e, work)
pub extern "C" fn lapacke_dbbcsd_work(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  jobv2t : Byte,
  trans : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  theta : FixedArray[Double],
  phi : FixedArray[Double],
  u1 : FixedArray[Double],
  ldu1 : Lapack_int,
  u2 : FixedArray[Double],
  ldu2 : Lapack_int,
  v1t : FixedArray[Double],
  ldv1t : Lapack_int,
  v2t : FixedArray[Double],
  ldv2t : Lapack_int,
  b11d : FixedArray[Double],
  b11e : FixedArray[Double],
  b12d : FixedArray[Double],
  b12e : FixedArray[Double],
  b21d : FixedArray[Double],
  b21e : FixedArray[Double],
  b22d : FixedArray[Double],
  b22e : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dbbcsd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dorbdb(
///     int matrix_layout, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, double* x11, lapack_int ldx11, 
///     double* x12, lapack_int ldx12, double* x21, lapack_int ldx21, 
///     double* x22, lapack_int ldx22, double* theta, double* phi, 
///     double* taup1, double* taup2, double* tauq1, double* tauq2
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, phi, taup1, taup2, tauq1, tauq2)
pub extern "C" fn lapacke_dorbdb(
  matrix_layout : Int,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Double],
  ldx11 : Lapack_int,
  x12 : FixedArray[Double],
  ldx12 : Lapack_int,
  x21 : FixedArray[Double],
  ldx21 : Lapack_int,
  x22 : FixedArray[Double],
  ldx22 : Lapack_int,
  theta : FixedArray[Double],
  phi : FixedArray[Double],
  taup1 : FixedArray[Double],
  taup2 : FixedArray[Double],
  tauq1 : FixedArray[Double],
  tauq2 : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dorbdb"

///|
///
/// ```c
/// lapack_int LAPACKE_dorbdb_work(
///     int matrix_layout, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, double* x11, lapack_int ldx11, 
///     double* x12, lapack_int ldx12, double* x21, lapack_int ldx21, 
///     double* x22, lapack_int ldx22, double* theta, double* phi, 
///     double* taup1, double* taup2, double* tauq1, double* tauq2, 
///     double* work, lapack_int lwork
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, phi, taup1, taup2, tauq1, tauq2, work)
pub extern "C" fn lapacke_dorbdb_work(
  matrix_layout : Int,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Double],
  ldx11 : Lapack_int,
  x12 : FixedArray[Double],
  ldx12 : Lapack_int,
  x21 : FixedArray[Double],
  ldx21 : Lapack_int,
  x22 : FixedArray[Double],
  ldx22 : Lapack_int,
  theta : FixedArray[Double],
  phi : FixedArray[Double],
  taup1 : FixedArray[Double],
  taup2 : FixedArray[Double],
  tauq1 : FixedArray[Double],
  tauq2 : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dorbdb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dorcsd(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, double* x11, lapack_int ldx11, 
///     double* x12, lapack_int ldx12, double* x21, lapack_int ldx21, 
///     double* x22, lapack_int ldx22, double* theta, double* u1, 
///     lapack_int ldu1, double* u2, lapack_int ldu2, double* v1t, 
///     lapack_int ldv1t, double* v2t, lapack_int ldv2t
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, u1, u2, v1t, v2t)
pub extern "C" fn lapacke_dorcsd(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  jobv2t : Byte,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Double],
  ldx11 : Lapack_int,
  x12 : FixedArray[Double],
  ldx12 : Lapack_int,
  x21 : FixedArray[Double],
  ldx21 : Lapack_int,
  x22 : FixedArray[Double],
  ldx22 : Lapack_int,
  theta : FixedArray[Double],
  u1 : FixedArray[Double],
  ldu1 : Lapack_int,
  u2 : FixedArray[Double],
  ldu2 : Lapack_int,
  v1t : FixedArray[Double],
  ldv1t : Lapack_int,
  v2t : FixedArray[Double],
  ldv2t : Lapack_int,
) -> Lapack_int = "LAPACKE_dorcsd"

///|
///
/// ```c
/// lapack_int LAPACKE_dorcsd_work(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, double* x11, lapack_int ldx11, 
///     double* x12, lapack_int ldx12, double* x21, lapack_int ldx21, 
///     double* x22, lapack_int ldx22, double* theta, double* u1, 
///     lapack_int ldu1, double* u2, lapack_int ldu2, double* v1t, 
///     lapack_int ldv1t, double* v2t, lapack_int ldv2t, double* work, 
///     lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, u1, u2, v1t, v2t, work, iwork)
pub extern "C" fn lapacke_dorcsd_work(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  jobv2t : Byte,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Double],
  ldx11 : Lapack_int,
  x12 : FixedArray[Double],
  ldx12 : Lapack_int,
  x21 : FixedArray[Double],
  ldx21 : Lapack_int,
  x22 : FixedArray[Double],
  ldx22 : Lapack_int,
  theta : FixedArray[Double],
  u1 : FixedArray[Double],
  ldu1 : Lapack_int,
  u2 : FixedArray[Double],
  ldu2 : Lapack_int,
  v1t : FixedArray[Double],
  ldv1t : Lapack_int,
  v2t : FixedArray[Double],
  ldv2t : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dorcsd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dorcsd2by1(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     lapack_int m, lapack_int p, lapack_int q, double* x11, 
///     lapack_int ldx11, double* x21, lapack_int ldx21, double* theta, 
///     double* u1, lapack_int ldu1, double* u2, lapack_int ldu2, 
///     double* v1t, lapack_int ldv1t
/// );
/// ```
#borrow(x11, x21, theta, u1, u2, v1t)
pub extern "C" fn lapacke_dorcsd2by1(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Double],
  ldx11 : Lapack_int,
  x21 : FixedArray[Double],
  ldx21 : Lapack_int,
  theta : FixedArray[Double],
  u1 : FixedArray[Double],
  ldu1 : Lapack_int,
  u2 : FixedArray[Double],
  ldu2 : Lapack_int,
  v1t : FixedArray[Double],
  ldv1t : Lapack_int,
) -> Lapack_int = "LAPACKE_dorcsd2by1"

///|
///
/// ```c
/// lapack_int LAPACKE_dorcsd2by1_work(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     lapack_int m, lapack_int p, lapack_int q, double* x11, 
///     lapack_int ldx11, double* x21, lapack_int ldx21, double* theta, 
///     double* u1, lapack_int ldu1, double* u2, lapack_int ldu2, 
///     double* v1t, lapack_int ldv1t, double* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(x11, x21, theta, u1, u2, v1t, work, iwork)
pub extern "C" fn lapacke_dorcsd2by1_work(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Double],
  ldx11 : Lapack_int,
  x21 : FixedArray[Double],
  ldx21 : Lapack_int,
  theta : FixedArray[Double],
  u1 : FixedArray[Double],
  ldu1 : Lapack_int,
  u2 : FixedArray[Double],
  ldu2 : Lapack_int,
  v1t : FixedArray[Double],
  ldv1t : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dorcsd2by1_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyconv(
///     int matrix_layout, char uplo, char way, lapack_int n, 
///     double* a, lapack_int lda, const lapack_int* ipiv, double* e
/// );
/// ```
#borrow(a, ipiv, e)
pub extern "C" fn lapacke_dsyconv(
  matrix_layout : Int,
  uplo : Byte,
  way : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  e : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dsyconv"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyconv_work(
///     int matrix_layout, char uplo, char way, lapack_int n, 
///     double* a, lapack_int lda, const lapack_int* ipiv, double* e
/// );
/// ```
#borrow(a, ipiv, e)
pub extern "C" fn lapacke_dsyconv_work(
  matrix_layout : Int,
  uplo : Byte,
  way : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  e : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dsyconv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyswapr(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, lapack_int i1, lapack_int i2
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_dsyswapr(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  i1 : Lapack_int,
  i2 : Lapack_int,
) -> Lapack_int = "LAPACKE_dsyswapr"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyswapr_work(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, lapack_int i1, lapack_int i2
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_dsyswapr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  i1 : Lapack_int,
  i2 : Lapack_int,
) -> Lapack_int = "LAPACKE_dsyswapr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytri2(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, const lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_dsytri2(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dsytri2"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytri2_work(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, const lapack_int* ipiv, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_dsytri2_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dsytri2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytri2x(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_int nb
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_dsytri2x(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  nb : Lapack_int,
) -> Lapack_int = "LAPACKE_dsytri2x"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytri2x_work(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, const lapack_int* ipiv, double* work, lapack_int nb
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_dsytri2x_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  nb : Lapack_int,
) -> Lapack_int = "LAPACKE_dsytri2x_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrs2(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_dsytrs2(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dsytrs2"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrs2_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const lapack_int* ipiv, double* b, 
///     lapack_int ldb, double* work
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_dsytrs2_work(
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
) -> Lapack_int = "LAPACKE_dsytrs2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sbbcsd(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, lapack_int m, lapack_int p, 
///     lapack_int q, float* theta, float* phi, float* u1, 
///     lapack_int ldu1, float* u2, lapack_int ldu2, float* v1t, 
///     lapack_int ldv1t, float* v2t, lapack_int ldv2t, float* b11d, 
///     float* b11e, float* b12d, float* b12e, float* b21d, 
///     float* b21e, float* b22d, float* b22e
/// );
/// ```
#borrow(theta, phi, u1, u2, v1t, v2t, b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e)
pub extern "C" fn lapacke_sbbcsd(
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
) -> Lapack_int = "LAPACKE_sbbcsd"

///|
///
/// ```c
/// lapack_int LAPACKE_sbbcsd_work(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, lapack_int m, lapack_int p, 
///     lapack_int q, float* theta, float* phi, float* u1, 
///     lapack_int ldu1, float* u2, lapack_int ldu2, float* v1t, 
///     lapack_int ldv1t, float* v2t, lapack_int ldv2t, float* b11d, 
///     float* b11e, float* b12d, float* b12e, float* b21d, 
///     float* b21e, float* b22d, float* b22e, float* work, lapack_int lwork
/// );
/// ```
#borrow(theta, phi, u1, u2, v1t, v2t, b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e, work)
pub extern "C" fn lapacke_sbbcsd_work(
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
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sbbcsd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sorbdb(
///     int matrix_layout, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, float* x11, lapack_int ldx11, 
///     float* x12, lapack_int ldx12, float* x21, lapack_int ldx21, 
///     float* x22, lapack_int ldx22, float* theta, float* phi, 
///     float* taup1, float* taup2, float* tauq1, float* tauq2
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, phi, taup1, taup2, tauq1, tauq2)
pub extern "C" fn lapacke_sorbdb(
  matrix_layout : Int,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Float],
  ldx11 : Lapack_int,
  x12 : FixedArray[Float],
  ldx12 : Lapack_int,
  x21 : FixedArray[Float],
  ldx21 : Lapack_int,
  x22 : FixedArray[Float],
  ldx22 : Lapack_int,
  theta : FixedArray[Float],
  phi : FixedArray[Float],
  taup1 : FixedArray[Float],
  taup2 : FixedArray[Float],
  tauq1 : FixedArray[Float],
  tauq2 : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sorbdb"

///|
///
/// ```c
/// lapack_int LAPACKE_sorbdb_work(
///     int matrix_layout, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, float* x11, lapack_int ldx11, 
///     float* x12, lapack_int ldx12, float* x21, lapack_int ldx21, 
///     float* x22, lapack_int ldx22, float* theta, float* phi, 
///     float* taup1, float* taup2, float* tauq1, float* tauq2, 
///     float* work, lapack_int lwork
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, phi, taup1, taup2, tauq1, tauq2, work)
pub extern "C" fn lapacke_sorbdb_work(
  matrix_layout : Int,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Float],
  ldx11 : Lapack_int,
  x12 : FixedArray[Float],
  ldx12 : Lapack_int,
  x21 : FixedArray[Float],
  ldx21 : Lapack_int,
  x22 : FixedArray[Float],
  ldx22 : Lapack_int,
  theta : FixedArray[Float],
  phi : FixedArray[Float],
  taup1 : FixedArray[Float],
  taup2 : FixedArray[Float],
  tauq1 : FixedArray[Float],
  tauq2 : FixedArray[Float],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_sorbdb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sorcsd(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, float* x11, lapack_int ldx11, 
///     float* x12, lapack_int ldx12, float* x21, lapack_int ldx21, 
///     float* x22, lapack_int ldx22, float* theta, float* u1, 
///     lapack_int ldu1, float* u2, lapack_int ldu2, float* v1t, 
///     lapack_int ldv1t, float* v2t, lapack_int ldv2t
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, u1, u2, v1t, v2t)
pub extern "C" fn lapacke_sorcsd(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  jobv2t : Byte,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Float],
  ldx11 : Lapack_int,
  x12 : FixedArray[Float],
  ldx12 : Lapack_int,
  x21 : FixedArray[Float],
  ldx21 : Lapack_int,
  x22 : FixedArray[Float],
  ldx22 : Lapack_int,
  theta : FixedArray[Float],
  u1 : FixedArray[Float],
  ldu1 : Lapack_int,
  u2 : FixedArray[Float],
  ldu2 : Lapack_int,
  v1t : FixedArray[Float],
  ldv1t : Lapack_int,
  v2t : FixedArray[Float],
  ldv2t : Lapack_int,
) -> Lapack_int = "LAPACKE_sorcsd"

///|
///
/// ```c
/// lapack_int LAPACKE_sorcsd_work(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, float* x11, lapack_int ldx11, 
///     float* x12, lapack_int ldx12, float* x21, lapack_int ldx21, 
///     float* x22, lapack_int ldx22, float* theta, float* u1, 
///     lapack_int ldu1, float* u2, lapack_int ldu2, float* v1t, 
///     lapack_int ldv1t, float* v2t, lapack_int ldv2t, float* work, 
///     lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, u1, u2, v1t, v2t, work, iwork)
pub extern "C" fn lapacke_sorcsd_work(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  jobv2t : Byte,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Float],
  ldx11 : Lapack_int,
  x12 : FixedArray[Float],
  ldx12 : Lapack_int,
  x21 : FixedArray[Float],
  ldx21 : Lapack_int,
  x22 : FixedArray[Float],
  ldx22 : Lapack_int,
  theta : FixedArray[Float],
  u1 : FixedArray[Float],
  ldu1 : Lapack_int,
  u2 : FixedArray[Float],
  ldu2 : Lapack_int,
  v1t : FixedArray[Float],
  ldv1t : Lapack_int,
  v2t : FixedArray[Float],
  ldv2t : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sorcsd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sorcsd2by1(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     lapack_int m, lapack_int p, lapack_int q, float* x11, 
///     lapack_int ldx11, float* x21, lapack_int ldx21, float* theta, 
///     float* u1, lapack_int ldu1, float* u2, lapack_int ldu2, 
///     float* v1t, lapack_int ldv1t
/// );
/// ```
#borrow(x11, x21, theta, u1, u2, v1t)
pub extern "C" fn lapacke_sorcsd2by1(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Float],
  ldx11 : Lapack_int,
  x21 : FixedArray[Float],
  ldx21 : Lapack_int,
  theta : FixedArray[Float],
  u1 : FixedArray[Float],
  ldu1 : Lapack_int,
  u2 : FixedArray[Float],
  ldu2 : Lapack_int,
  v1t : FixedArray[Float],
  ldv1t : Lapack_int,
) -> Lapack_int = "LAPACKE_sorcsd2by1"

///|
///
/// ```c
/// lapack_int LAPACKE_sorcsd2by1_work(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     lapack_int m, lapack_int p, lapack_int q, float* x11, 
///     lapack_int ldx11, float* x21, lapack_int ldx21, float* theta, 
///     float* u1, lapack_int ldu1, float* u2, lapack_int ldu2, 
///     float* v1t, lapack_int ldv1t, float* work, lapack_int lwork, lapack_int* iwork
/// );
/// ```
#borrow(x11, x21, theta, u1, u2, v1t, work, iwork)
pub extern "C" fn lapacke_sorcsd2by1_work(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Float],
  ldx11 : Lapack_int,
  x21 : FixedArray[Float],
  ldx21 : Lapack_int,
  theta : FixedArray[Float],
  u1 : FixedArray[Float],
  ldu1 : Lapack_int,
  u2 : FixedArray[Float],
  ldu2 : Lapack_int,
  v1t : FixedArray[Float],
  ldv1t : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sorcsd2by1_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyconv(
///     int matrix_layout, char uplo, char way, lapack_int n, 
///     float* a, lapack_int lda, const lapack_int* ipiv, float* e
/// );
/// ```
#borrow(a, ipiv, e)
pub extern "C" fn lapacke_ssyconv(
  matrix_layout : Int,
  uplo : Byte,
  way : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  e : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ssyconv"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyconv_work(
///     int matrix_layout, char uplo, char way, lapack_int n, 
///     float* a, lapack_int lda, const lapack_int* ipiv, float* e
/// );
/// ```
#borrow(a, ipiv, e)
pub extern "C" fn lapacke_ssyconv_work(
  matrix_layout : Int,
  uplo : Byte,
  way : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  e : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ssyconv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyswapr(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, lapack_int i1, lapack_int i2
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_ssyswapr(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  i1 : Lapack_int,
  i2 : Lapack_int,
) -> Lapack_int = "LAPACKE_ssyswapr"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyswapr_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, lapack_int i1, lapack_int i2
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_ssyswapr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  i1 : Lapack_int,
  i2 : Lapack_int,
) -> Lapack_int = "LAPACKE_ssyswapr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytri2(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, const lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_ssytri2(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ssytri2"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytri2_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, const lapack_int* ipiv, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_ssytri2_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytri2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytri2x(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_int nb
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_ssytri2x(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  nb : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytri2x"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytri2x_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, const lapack_int* ipiv, float* work, lapack_int nb
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_ssytri2x_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  nb : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytri2x_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrs2(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_ssytrs2(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytrs2"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrs2_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const lapack_int* ipiv, float* b, 
///     lapack_int ldb, float* work
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_ssytrs2_work(
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
) -> Lapack_int = "LAPACKE_ssytrs2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zbbcsd(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, lapack_int m, lapack_int p, 
///     lapack_int q, double* theta, double* phi, lapack_complex_double* u1, 
///     lapack_int ldu1, lapack_complex_double* u2, lapack_int ldu2, lapack_complex_double* v1t, 
///     lapack_int ldv1t, lapack_complex_double* v2t, lapack_int ldv2t, double* b11d, 
///     double* b11e, double* b12d, double* b12e, double* b21d, 
///     double* b21e, double* b22d, double* b22e
/// );
/// ```
#borrow(theta, phi, u1, u2, v1t, v2t, b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e)
pub extern "C" fn lapacke_zbbcsd(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  jobv2t : Byte,
  trans : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  theta : FixedArray[Double],
  phi : FixedArray[Double],
  u1 : FixedArray[Double],
  ldu1 : Lapack_int,
  u2 : FixedArray[Double],
  ldu2 : Lapack_int,
  v1t : FixedArray[Double],
  ldv1t : Lapack_int,
  v2t : FixedArray[Double],
  ldv2t : Lapack_int,
  b11d : FixedArray[Double],
  b11e : FixedArray[Double],
  b12d : FixedArray[Double],
  b12e : FixedArray[Double],
  b21d : FixedArray[Double],
  b21e : FixedArray[Double],
  b22d : FixedArray[Double],
  b22e : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zbbcsd"

///|
///
/// ```c
/// lapack_int LAPACKE_zbbcsd_work(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, lapack_int m, lapack_int p, 
///     lapack_int q, double* theta, double* phi, lapack_complex_double* u1, 
///     lapack_int ldu1, lapack_complex_double* u2, lapack_int ldu2, lapack_complex_double* v1t, 
///     lapack_int ldv1t, lapack_complex_double* v2t, lapack_int ldv2t, double* b11d, 
///     double* b11e, double* b12d, double* b12e, double* b21d, 
///     double* b21e, double* b22d, double* b22e, double* rwork, lapack_int lrwork
/// );
/// ```
#borrow(theta, phi, u1, u2, v1t, v2t, b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e, rwork)
pub extern "C" fn lapacke_zbbcsd_work(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  jobv2t : Byte,
  trans : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  theta : FixedArray[Double],
  phi : FixedArray[Double],
  u1 : FixedArray[Double],
  ldu1 : Lapack_int,
  u2 : FixedArray[Double],
  ldu2 : Lapack_int,
  v1t : FixedArray[Double],
  ldv1t : Lapack_int,
  v2t : FixedArray[Double],
  ldv2t : Lapack_int,
  b11d : FixedArray[Double],
  b11e : FixedArray[Double],
  b12d : FixedArray[Double],
  b12e : FixedArray[Double],
  b21d : FixedArray[Double],
  b21e : FixedArray[Double],
  b22d : FixedArray[Double],
  b22e : FixedArray[Double],
  rwork : FixedArray[Double],
  lrwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zbbcsd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zheswapr(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int i1, lapack_int i2
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_zheswapr(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  i1 : Lapack_int,
  i2 : Lapack_int,
) -> Lapack_int = "LAPACKE_zheswapr"

///|
///
/// ```c
/// lapack_int LAPACKE_zheswapr_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int i1, lapack_int i2
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_zheswapr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  i1 : Lapack_int,
  i2 : Lapack_int,
) -> Lapack_int = "LAPACKE_zheswapr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetri2(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_zhetri2(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhetri2"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetri2_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_zhetri2_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zhetri2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetri2x(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_int nb
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_zhetri2x(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  nb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhetri2x"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetri2x_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_complex_double* work, lapack_int nb
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_zhetri2x_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  nb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhetri2x_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrs2(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zhetrs2(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhetrs2"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrs2_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* work
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_zhetrs2_work(
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
) -> Lapack_int = "LAPACKE_zhetrs2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsyconv(
///     int matrix_layout, char uplo, char way, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* e
/// );
/// ```
#borrow(a, ipiv, e)
pub extern "C" fn lapacke_zsyconv(
  matrix_layout : Int,
  uplo : Byte,
  way : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  e : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zsyconv"

///|
///
/// ```c
/// lapack_int LAPACKE_zsyconv_work(
///     int matrix_layout, char uplo, char way, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* e
/// );
/// ```
#borrow(a, ipiv, e)
pub extern "C" fn lapacke_zsyconv_work(
  matrix_layout : Int,
  uplo : Byte,
  way : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  e : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zsyconv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsyswapr(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int i1, lapack_int i2
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_zsyswapr(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  i1 : Lapack_int,
  i2 : Lapack_int,
) -> Lapack_int = "LAPACKE_zsyswapr"

///|
///
/// ```c
/// lapack_int LAPACKE_zsyswapr_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int i1, lapack_int i2
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_zsyswapr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  i1 : Lapack_int,
  i2 : Lapack_int,
) -> Lapack_int = "LAPACKE_zsyswapr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytri2(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_zsytri2(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zsytri2"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytri2_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_zsytri2_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zsytri2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytri2x(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_int nb
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_zsytri2x(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  nb : Lapack_int,
) -> Lapack_int = "LAPACKE_zsytri2x"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytri2x_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv, lapack_complex_double* work, lapack_int nb
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_zsytri2x_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  nb : Lapack_int,
) -> Lapack_int = "LAPACKE_zsytri2x_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrs2(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zsytrs2(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zsytrs2"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrs2_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* work
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_zsytrs2_work(
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
) -> Lapack_int = "LAPACKE_zsytrs2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zunbdb(
///     int matrix_layout, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, lapack_complex_double* x11, lapack_int ldx11, 
///     lapack_complex_double* x12, lapack_int ldx12, lapack_complex_double* x21, lapack_int ldx21, 
///     lapack_complex_double* x22, lapack_int ldx22, double* theta, double* phi, 
///     lapack_complex_double* taup1, lapack_complex_double* taup2, lapack_complex_double* tauq1, lapack_complex_double* tauq2
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, phi, taup1, taup2, tauq1, tauq2)
pub extern "C" fn lapacke_zunbdb(
  matrix_layout : Int,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Double],
  ldx11 : Lapack_int,
  x12 : FixedArray[Double],
  ldx12 : Lapack_int,
  x21 : FixedArray[Double],
  ldx21 : Lapack_int,
  x22 : FixedArray[Double],
  ldx22 : Lapack_int,
  theta : FixedArray[Double],
  phi : FixedArray[Double],
  taup1 : FixedArray[Double],
  taup2 : FixedArray[Double],
  tauq1 : FixedArray[Double],
  tauq2 : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zunbdb"

///|
///
/// ```c
/// lapack_int LAPACKE_zunbdb_work(
///     int matrix_layout, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, lapack_complex_double* x11, lapack_int ldx11, 
///     lapack_complex_double* x12, lapack_int ldx12, lapack_complex_double* x21, lapack_int ldx21, 
///     lapack_complex_double* x22, lapack_int ldx22, double* theta, double* phi, 
///     lapack_complex_double* taup1, lapack_complex_double* taup2, lapack_complex_double* tauq1, lapack_complex_double* tauq2, 
///     lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, phi, taup1, taup2, tauq1, tauq2, work)
pub extern "C" fn lapacke_zunbdb_work(
  matrix_layout : Int,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Double],
  ldx11 : Lapack_int,
  x12 : FixedArray[Double],
  ldx12 : Lapack_int,
  x21 : FixedArray[Double],
  ldx21 : Lapack_int,
  x22 : FixedArray[Double],
  ldx22 : Lapack_int,
  theta : FixedArray[Double],
  phi : FixedArray[Double],
  taup1 : FixedArray[Double],
  taup2 : FixedArray[Double],
  tauq1 : FixedArray[Double],
  tauq2 : FixedArray[Double],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zunbdb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zuncsd(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, lapack_complex_double* x11, lapack_int ldx11, 
///     lapack_complex_double* x12, lapack_int ldx12, lapack_complex_double* x21, lapack_int ldx21, 
///     lapack_complex_double* x22, lapack_int ldx22, double* theta, lapack_complex_double* u1, 
///     lapack_int ldu1, lapack_complex_double* u2, lapack_int ldu2, lapack_complex_double* v1t, 
///     lapack_int ldv1t, lapack_complex_double* v2t, lapack_int ldv2t
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, u1, u2, v1t, v2t)
pub extern "C" fn lapacke_zuncsd(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  jobv2t : Byte,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Double],
  ldx11 : Lapack_int,
  x12 : FixedArray[Double],
  ldx12 : Lapack_int,
  x21 : FixedArray[Double],
  ldx21 : Lapack_int,
  x22 : FixedArray[Double],
  ldx22 : Lapack_int,
  theta : FixedArray[Double],
  u1 : FixedArray[Double],
  ldu1 : Lapack_int,
  u2 : FixedArray[Double],
  ldu2 : Lapack_int,
  v1t : FixedArray[Double],
  ldv1t : Lapack_int,
  v2t : FixedArray[Double],
  ldv2t : Lapack_int,
) -> Lapack_int = "LAPACKE_zuncsd"

///|
///
/// ```c
/// lapack_int LAPACKE_zuncsd_work(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     char jobv2t, char trans, char signs, lapack_int m, 
///     lapack_int p, lapack_int q, lapack_complex_double* x11, lapack_int ldx11, 
///     lapack_complex_double* x12, lapack_int ldx12, lapack_complex_double* x21, lapack_int ldx21, 
///     lapack_complex_double* x22, lapack_int ldx22, double* theta, lapack_complex_double* u1, 
///     lapack_int ldu1, lapack_complex_double* u2, lapack_int ldu2, lapack_complex_double* v1t, 
///     lapack_int ldv1t, lapack_complex_double* v2t, lapack_int ldv2t, lapack_complex_double* work, 
///     lapack_int lwork, double* rwork, lapack_int lrwork, lapack_int* iwork
/// );
/// ```
#borrow(x11, x12, x21, x22, theta, u1, u2, v1t, v2t, work, rwork, iwork)
pub extern "C" fn lapacke_zuncsd_work(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  jobv2t : Byte,
  trans : Byte,
  signs : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Double],
  ldx11 : Lapack_int,
  x12 : FixedArray[Double],
  ldx12 : Lapack_int,
  x21 : FixedArray[Double],
  ldx21 : Lapack_int,
  x22 : FixedArray[Double],
  ldx22 : Lapack_int,
  theta : FixedArray[Double],
  u1 : FixedArray[Double],
  ldu1 : Lapack_int,
  u2 : FixedArray[Double],
  ldu2 : Lapack_int,
  v1t : FixedArray[Double],
  ldv1t : Lapack_int,
  v2t : FixedArray[Double],
  ldv2t : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zuncsd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zuncsd2by1(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     lapack_int m, lapack_int p, lapack_int q, lapack_complex_double* x11, 
///     lapack_int ldx11, lapack_complex_double* x21, lapack_int ldx21, double* theta, 
///     lapack_complex_double* u1, lapack_int ldu1, lapack_complex_double* u2, lapack_int ldu2, 
///     lapack_complex_double* v1t, lapack_int ldv1t
/// );
/// ```
#borrow(x11, x21, theta, u1, u2, v1t)
pub extern "C" fn lapacke_zuncsd2by1(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Double],
  ldx11 : Lapack_int,
  x21 : FixedArray[Double],
  ldx21 : Lapack_int,
  theta : FixedArray[Double],
  u1 : FixedArray[Double],
  ldu1 : Lapack_int,
  u2 : FixedArray[Double],
  ldu2 : Lapack_int,
  v1t : FixedArray[Double],
  ldv1t : Lapack_int,
) -> Lapack_int = "LAPACKE_zuncsd2by1"

///|
///
/// ```c
/// lapack_int LAPACKE_zuncsd2by1_work(
///     int matrix_layout, char jobu1, char jobu2, char jobv1t, 
///     lapack_int m, lapack_int p, lapack_int q, lapack_complex_double* x11, 
///     lapack_int ldx11, lapack_complex_double* x21, lapack_int ldx21, double* theta, 
///     lapack_complex_double* u1, lapack_int ldu1, lapack_complex_double* u2, lapack_int ldu2, 
///     lapack_complex_double* v1t, lapack_int ldv1t, lapack_complex_double* work, lapack_int lwork, 
///     double* rwork, lapack_int lrwork, lapack_int* iwork
/// );
/// ```
#borrow(x11, x21, theta, u1, u2, v1t, work, rwork, iwork)
pub extern "C" fn lapacke_zuncsd2by1_work(
  matrix_layout : Int,
  jobu1 : Byte,
  jobu2 : Byte,
  jobv1t : Byte,
  m : Lapack_int,
  p : Lapack_int,
  q : Lapack_int,
  x11 : FixedArray[Double],
  ldx11 : Lapack_int,
  x21 : FixedArray[Double],
  ldx21 : Lapack_int,
  theta : FixedArray[Double],
  u1 : FixedArray[Double],
  ldu1 : Lapack_int,
  u2 : FixedArray[Double],
  ldu2 : Lapack_int,
  v1t : FixedArray[Double],
  ldv1t : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
  rwork : FixedArray[Double],
  lrwork : Lapack_int,
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zuncsd2by1_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgemqrt(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int nb, const float* v, 
///     lapack_int ldv, const float* t, lapack_int ldt, float* c, lapack_int ldc
/// );
/// ```
#borrow(v, t, c)
pub extern "C" fn lapacke_sgemqrt(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
) -> Lapack_int = "LAPACKE_sgemqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_dgemqrt(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int nb, const double* v, 
///     lapack_int ldv, const double* t, lapack_int ldt, double* c, lapack_int ldc
/// );
/// ```
#borrow(v, t, c)
pub extern "C" fn lapacke_dgemqrt(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
) -> Lapack_int = "LAPACKE_dgemqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_cgemqrt(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int nb, const lapack_complex_float* v, 
///     lapack_int ldv, const lapack_complex_float* t, lapack_int ldt, lapack_complex_float* c, lapack_int ldc
/// );
/// ```
#borrow(v, t, c)
pub extern "C" fn lapacke_cgemqrt(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
) -> Lapack_int = "LAPACKE_cgemqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_zgemqrt(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int nb, const lapack_complex_double* v, 
///     lapack_int ldv, const lapack_complex_double* t, lapack_int ldt, lapack_complex_double* c, lapack_int ldc
/// );
/// ```
#borrow(v, t, c)
pub extern "C" fn lapacke_zgemqrt(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
) -> Lapack_int = "LAPACKE_zgemqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqrt(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nb, 
///     float* a, lapack_int lda, float* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_sgeqrt(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_sgeqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqrt(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nb, 
///     double* a, lapack_int lda, double* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_dgeqrt(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_dgeqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqrt(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nb, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_cgeqrt(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_cgeqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqrt(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nb, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_zgeqrt(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_zgeqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqrt2(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_sgeqrt2(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_sgeqrt2"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqrt2(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_dgeqrt2(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_dgeqrt2"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqrt2(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_cgeqrt2(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_cgeqrt2"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqrt2(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_zgeqrt2(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_zgeqrt2"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqrt3(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_sgeqrt3(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_sgeqrt3"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqrt3(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_dgeqrt3(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_dgeqrt3"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqrt3(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_cgeqrt3(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_cgeqrt3"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqrt3(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_zgeqrt3(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_zgeqrt3"

///|
///
/// ```c
/// lapack_int LAPACKE_stpmqrt(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, lapack_int nb, 
///     const float* v, lapack_int ldv, const float* t, lapack_int ldt, 
///     float* a, lapack_int lda, float* b, lapack_int ldb
/// );
/// ```
#borrow(v, t, a, b)
pub extern "C" fn lapacke_stpmqrt(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_stpmqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_dtpmqrt(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, lapack_int nb, 
///     const double* v, lapack_int ldv, const double* t, lapack_int ldt, 
///     double* a, lapack_int lda, double* b, lapack_int ldb
/// );
/// ```
#borrow(v, t, a, b)
pub extern "C" fn lapacke_dtpmqrt(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dtpmqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_ctpmqrt(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, lapack_int nb, 
///     const lapack_complex_float* v, lapack_int ldv, const lapack_complex_float* t, lapack_int ldt, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(v, t, a, b)
pub extern "C" fn lapacke_ctpmqrt(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ctpmqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_ztpmqrt(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, lapack_int nb, 
///     const lapack_complex_double* v, lapack_int ldv, const lapack_complex_double* t, lapack_int ldt, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(v, t, a, b)
pub extern "C" fn lapacke_ztpmqrt(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ztpmqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_stpqrt(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     lapack_int nb, float* a, lapack_int lda, float* b, 
///     lapack_int ldb, float* t, lapack_int ldt
/// );
/// ```
#borrow(a, b, t)
pub extern "C" fn lapacke_stpqrt(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_stpqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_dtpqrt(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     lapack_int nb, double* a, lapack_int lda, double* b, 
///     lapack_int ldb, double* t, lapack_int ldt
/// );
/// ```
#borrow(a, b, t)
pub extern "C" fn lapacke_dtpqrt(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_dtpqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_ctpqrt(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     lapack_int nb, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* t, lapack_int ldt
/// );
/// ```
#borrow(a, b, t)
pub extern "C" fn lapacke_ctpqrt(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_ctpqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_ztpqrt(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     lapack_int nb, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* t, lapack_int ldt
/// );
/// ```
#borrow(a, b, t)
pub extern "C" fn lapacke_ztpqrt(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_ztpqrt"

///|
///
/// ```c
/// lapack_int LAPACKE_stpqrt2(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* t, lapack_int ldt
/// );
/// ```
#borrow(a, b, t)
pub extern "C" fn lapacke_stpqrt2(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_stpqrt2"

///|
///
/// ```c
/// lapack_int LAPACKE_dtpqrt2(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* t, lapack_int ldt
/// );
/// ```
#borrow(a, b, t)
pub extern "C" fn lapacke_dtpqrt2(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_dtpqrt2"

///|
///
/// ```c
/// lapack_int LAPACKE_ctpqrt2(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* t, lapack_int ldt
/// );
/// ```
#borrow(a, b, t)
pub extern "C" fn lapacke_ctpqrt2(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_ctpqrt2"

///|
///
/// ```c
/// lapack_int LAPACKE_ztpqrt2(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* t, lapack_int ldt
/// );
/// ```
#borrow(a, b, t)
pub extern "C" fn lapacke_ztpqrt2(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_ztpqrt2"

///|
///
/// ```c
/// lapack_int LAPACKE_stprfb(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_int l, const float* v, lapack_int ldv, const float* t, 
///     lapack_int ldt, float* a, lapack_int lda, float* b, lapack_int ldb
/// );
/// ```
#borrow(v, t, a, b)
pub extern "C" fn lapacke_stprfb(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  direct : Byte,
  storev : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_stprfb"

///|
///
/// ```c
/// lapack_int LAPACKE_dtprfb(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_int l, const double* v, lapack_int ldv, const double* t, 
///     lapack_int ldt, double* a, lapack_int lda, double* b, lapack_int ldb
/// );
/// ```
#borrow(v, t, a, b)
pub extern "C" fn lapacke_dtprfb(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  direct : Byte,
  storev : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dtprfb"

///|
///
/// ```c
/// lapack_int LAPACKE_ctprfb(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_int l, const lapack_complex_float* v, lapack_int ldv, const lapack_complex_float* t, 
///     lapack_int ldt, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(v, t, a, b)
pub extern "C" fn lapacke_ctprfb(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  direct : Byte,
  storev : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ctprfb"

///|
///
/// ```c
/// lapack_int LAPACKE_ztprfb(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_int l, const lapack_complex_double* v, lapack_int ldv, const lapack_complex_double* t, 
///     lapack_int ldt, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(v, t, a, b)
pub extern "C" fn lapacke_ztprfb(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  direct : Byte,
  storev : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ztprfb"

///|
///
/// ```c
/// lapack_int LAPACKE_sgemqrt_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int nb, const float* v, 
///     lapack_int ldv, const float* t, lapack_int ldt, float* c, 
///     lapack_int ldc, float* work
/// );
/// ```
#borrow(v, t, c, work)
pub extern "C" fn lapacke_sgemqrt_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sgemqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgemqrt_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int nb, const double* v, 
///     lapack_int ldv, const double* t, lapack_int ldt, double* c, 
///     lapack_int ldc, double* work
/// );
/// ```
#borrow(v, t, c, work)
pub extern "C" fn lapacke_dgemqrt_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dgemqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgemqrt_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int nb, const lapack_complex_float* v, 
///     lapack_int ldv, const lapack_complex_float* t, lapack_int ldt, lapack_complex_float* c, 
///     lapack_int ldc, lapack_complex_float* work
/// );
/// ```
#borrow(v, t, c, work)
pub extern "C" fn lapacke_cgemqrt_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgemqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgemqrt_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int nb, const lapack_complex_double* v, 
///     lapack_int ldv, const lapack_complex_double* t, lapack_int ldt, lapack_complex_double* c, 
///     lapack_int ldc, lapack_complex_double* work
/// );
/// ```
#borrow(v, t, c, work)
pub extern "C" fn lapacke_zgemqrt_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgemqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqrt_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nb, 
///     float* a, lapack_int lda, float* t, lapack_int ldt, float* work
/// );
/// ```
#borrow(a, t, work)
pub extern "C" fn lapacke_sgeqrt_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sgeqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqrt_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nb, 
///     double* a, lapack_int lda, double* t, lapack_int ldt, double* work
/// );
/// ```
#borrow(a, t, work)
pub extern "C" fn lapacke_dgeqrt_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dgeqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqrt_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nb, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* t, lapack_int ldt, lapack_complex_float* work
/// );
/// ```
#borrow(a, t, work)
pub extern "C" fn lapacke_cgeqrt_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgeqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqrt_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nb, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* t, lapack_int ldt, lapack_complex_double* work
/// );
/// ```
#borrow(a, t, work)
pub extern "C" fn lapacke_zgeqrt_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgeqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqrt2_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_sgeqrt2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_sgeqrt2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqrt2_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_dgeqrt2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_dgeqrt2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqrt2_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_cgeqrt2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_cgeqrt2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqrt2_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_zgeqrt2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_zgeqrt2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqrt3_work(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_sgeqrt3_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_sgeqrt3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqrt3_work(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_dgeqrt3_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_dgeqrt3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqrt3_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_cgeqrt3_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_cgeqrt3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqrt3_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_zgeqrt3_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_zgeqrt3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stpmqrt_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, lapack_int nb, 
///     const float* v, lapack_int ldv, const float* t, lapack_int ldt, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, float* work
/// );
/// ```
#borrow(v, t, a, b, work)
pub extern "C" fn lapacke_stpmqrt_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_stpmqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtpmqrt_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, lapack_int nb, 
///     const double* v, lapack_int ldv, const double* t, lapack_int ldt, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, double* work
/// );
/// ```
#borrow(v, t, a, b, work)
pub extern "C" fn lapacke_dtpmqrt_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dtpmqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctpmqrt_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, lapack_int nb, 
///     const lapack_complex_float* v, lapack_int ldv, const lapack_complex_float* t, lapack_int ldt, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, lapack_complex_float* work
/// );
/// ```
#borrow(v, t, a, b, work)
pub extern "C" fn lapacke_ctpmqrt_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctpmqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztpmqrt_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, lapack_int nb, 
///     const lapack_complex_double* v, lapack_int ldv, const lapack_complex_double* t, lapack_int ldt, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, lapack_complex_double* work
/// );
/// ```
#borrow(v, t, a, b, work)
pub extern "C" fn lapacke_ztpmqrt_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztpmqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stpqrt_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     lapack_int nb, float* a, lapack_int lda, float* b, 
///     lapack_int ldb, float* t, lapack_int ldt, float* work
/// );
/// ```
#borrow(a, b, t, work)
pub extern "C" fn lapacke_stpqrt_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_stpqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtpqrt_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     lapack_int nb, double* a, lapack_int lda, double* b, 
///     lapack_int ldb, double* t, lapack_int ldt, double* work
/// );
/// ```
#borrow(a, b, t, work)
pub extern "C" fn lapacke_dtpqrt_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dtpqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctpqrt_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     lapack_int nb, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* t, lapack_int ldt, lapack_complex_float* work
/// );
/// ```
#borrow(a, b, t, work)
pub extern "C" fn lapacke_ctpqrt_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_ctpqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztpqrt_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     lapack_int nb, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* t, lapack_int ldt, lapack_complex_double* work
/// );
/// ```
#borrow(a, b, t, work)
pub extern "C" fn lapacke_ztpqrt_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_ztpqrt_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stpqrt2_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* t, lapack_int ldt
/// );
/// ```
#borrow(a, b, t)
pub extern "C" fn lapacke_stpqrt2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_stpqrt2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtpqrt2_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* t, lapack_int ldt
/// );
/// ```
#borrow(a, b, t)
pub extern "C" fn lapacke_dtpqrt2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_dtpqrt2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctpqrt2_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* t, lapack_int ldt
/// );
/// ```
#borrow(a, b, t)
pub extern "C" fn lapacke_ctpqrt2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_ctpqrt2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztpqrt2_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int l, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* t, lapack_int ldt
/// );
/// ```
#borrow(a, b, t)
pub extern "C" fn lapacke_ztpqrt2_work(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  l : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_ztpqrt2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_stprfb_work(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_int l, const float* v, lapack_int ldv, const float* t, 
///     lapack_int ldt, float* a, lapack_int lda, float* b, 
///     lapack_int ldb, float* work, lapack_int ldwork
/// );
/// ```
#borrow(v, t, a, b, work)
pub extern "C" fn lapacke_stprfb_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  direct : Byte,
  storev : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  work : FixedArray[Float],
  ldwork : Lapack_int,
) -> Lapack_int = "LAPACKE_stprfb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dtprfb_work(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_int l, const double* v, lapack_int ldv, const double* t, 
///     lapack_int ldt, double* a, lapack_int lda, double* b, 
///     lapack_int ldb, double* work, lapack_int ldwork
/// );
/// ```
#borrow(v, t, a, b, work)
pub extern "C" fn lapacke_dtprfb_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  direct : Byte,
  storev : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  work : FixedArray[Double],
  ldwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dtprfb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ctprfb_work(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_int l, const lapack_complex_float* v, lapack_int ldv, const lapack_complex_float* t, 
///     lapack_int ldt, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* work, lapack_int ldwork
/// );
/// ```
#borrow(v, t, a, b, work)
pub extern "C" fn lapacke_ctprfb_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  direct : Byte,
  storev : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  v : FixedArray[Float],
  ldv : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  work : FixedArray[Float],
  ldwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ctprfb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ztprfb_work(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_int l, const lapack_complex_double* v, lapack_int ldv, const lapack_complex_double* t, 
///     lapack_int ldt, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* work, lapack_int ldwork
/// );
/// ```
#borrow(v, t, a, b, work)
pub extern "C" fn lapacke_ztprfb_work(
  matrix_layout : Int,
  side : Byte,
  trans : Byte,
  direct : Byte,
  storev : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  l : Lapack_int,
  v : FixedArray[Double],
  ldv : Lapack_int,
  t : FixedArray[Double],
  ldt : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  work : FixedArray[Double],
  ldwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ztprfb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssysv_rook(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_ssysv_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ssysv_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_dsysv_rook(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_dsysv_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dsysv_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_csysv_rook(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_csysv_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_csysv_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_zsysv_rook(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zsysv_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zsysv_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrf_rook(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_ssytrf_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ssytrf_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrf_rook(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_dsytrf_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dsytrf_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrf_rook(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_csytrf_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_csytrf_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrf_rook(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_zsytrf_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zsytrf_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrs_rook(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_ssytrs_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytrs_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrs_rook(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_dsytrs_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dsytrs_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrs_rook(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_csytrs_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_csytrs_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrs_rook(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zsytrs_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zsytrs_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrf_rook(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_chetrf_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chetrf_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrf_rook(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_zhetrf_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhetrf_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrs_rook(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_chetrs_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chetrs_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrs_rook(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zhetrs_rook(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhetrs_rook"

///|
///
/// ```c
/// lapack_int LAPACKE_csyr(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float alpha, 
///     const lapack_complex_float* x, lapack_int incx, lapack_complex_float* a, lapack_int lda
/// );
/// ```
#borrow(x, a)
pub extern "C" fn lapacke_csyr(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  alpha : Lapack_complex_float,
  x : FixedArray[Float],
  incx : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_csyr"

///|
///
/// ```c
/// lapack_int LAPACKE_zsyr(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double alpha, 
///     const lapack_complex_double* x, lapack_int incx, lapack_complex_double* a, lapack_int lda
/// );
/// ```
#borrow(x, a)
pub extern "C" fn lapacke_zsyr(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  alpha : Lapack_complex_double,
  x : FixedArray[Double],
  incx : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_zsyr"

///|
///
/// ```c
/// lapack_int LAPACKE_ssysv_rook_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, lapack_int* ipiv, float* b, 
///     lapack_int ldb, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_ssysv_rook_work(
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
) -> Lapack_int = "LAPACKE_ssysv_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsysv_rook_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, lapack_int* ipiv, double* b, 
///     lapack_int ldb, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_dsysv_rook_work(
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
) -> Lapack_int = "LAPACKE_dsysv_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csysv_rook_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_int* ipiv, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_csysv_rook_work(
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
) -> Lapack_int = "LAPACKE_csysv_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsysv_rook_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_int* ipiv, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_zsysv_rook_work(
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
) -> Lapack_int = "LAPACKE_zsysv_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrf_rook_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* ipiv, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_ssytrf_rook_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytrf_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrf_rook_work(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* ipiv, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_dsytrf_rook_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dsytrf_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrf_rook_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_csytrf_rook_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_csytrf_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrf_rook_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_zsytrf_rook_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zsytrf_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrs_rook_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_ssytrs_rook_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytrs_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrs_rook_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_dsytrs_rook_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dsytrs_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrs_rook_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_csytrs_rook_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_csytrs_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrs_rook_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zsytrs_rook_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zsytrs_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrf_rook_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_chetrf_rook_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_chetrf_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrf_rook_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_zhetrf_rook_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zhetrf_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrs_rook_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_chetrs_rook_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chetrs_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrs_rook_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zhetrs_rook_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhetrs_rook_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csyr_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float alpha, 
///     const lapack_complex_float* x, lapack_int incx, lapack_complex_float* a, lapack_int lda
/// );
/// ```
#borrow(x, a)
pub extern "C" fn lapacke_csyr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  alpha : Lapack_complex_float,
  x : FixedArray[Float],
  incx : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_csyr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsyr_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double alpha, 
///     const lapack_complex_double* x, lapack_int incx, lapack_complex_double* a, lapack_int lda
/// );
/// ```
#borrow(x, a)
pub extern "C" fn lapacke_zsyr_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  alpha : Lapack_complex_double,
  x : FixedArray[Double],
  incx : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_zsyr_work"

///|
///
/// ```c
/// void LAPACKE_ilaver(lapack_int* vers_major, lapack_int* vers_minor, lapack_int* vers_patch);
/// ```
#borrow(vers_major, vers_minor, vers_patch)
pub extern "C" fn lapacke_ilaver(
  vers_major : FixedArray[Int],
  vers_minor : FixedArray[Int],
  vers_patch : FixedArray[Int],
) -> Unit = "LAPACKE_ilaver"

///|
///
/// ```c
/// lapack_int LAPACKE_ssysv_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_ssysv_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ssysv_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_ssysv_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, lapack_int* ipiv, float* b, 
///     lapack_int ldb, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_ssysv_aa_work(
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
) -> Lapack_int = "LAPACKE_ssysv_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsysv_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_dsysv_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dsysv_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_dsysv_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, lapack_int* ipiv, double* b, 
///     lapack_int ldb, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_dsysv_aa_work(
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
) -> Lapack_int = "LAPACKE_dsysv_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csysv_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_csysv_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_csysv_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_csysv_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_int* ipiv, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_csysv_aa_work(
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
) -> Lapack_int = "LAPACKE_csysv_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsysv_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zsysv_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zsysv_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_zsysv_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_int* ipiv, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_zsysv_aa_work(
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
) -> Lapack_int = "LAPACKE_zsysv_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chesv_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_chesv_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chesv_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_chesv_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_int* ipiv, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_chesv_aa_work(
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
) -> Lapack_int = "LAPACKE_chesv_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhesv_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zhesv_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhesv_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_zhesv_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_int* ipiv, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_zhesv_aa_work(
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
) -> Lapack_int = "LAPACKE_zhesv_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrf_aa(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_ssytrf_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ssytrf_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrf_aa(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_dsytrf_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dsytrf_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrf_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_csytrf_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_csytrf_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrf_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_zsytrf_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zsytrf_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrf_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_chetrf_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chetrf_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrf_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_zhetrf_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhetrf_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrf_aa_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* ipiv, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_ssytrf_aa_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytrf_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrf_aa_work(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* ipiv, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_dsytrf_aa_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dsytrf_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrf_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_csytrf_aa_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_csytrf_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrf_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_zsytrf_aa_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zsytrf_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrf_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_chetrf_aa_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_chetrf_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrf_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, work)
pub extern "C" fn lapacke_zhetrf_aa_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zhetrf_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrs_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_csytrs_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_csytrs_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrs_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_csytrs_aa_work(
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
) -> Lapack_int = "LAPACKE_csytrs_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrs_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_chetrs_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chetrs_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrs_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_chetrs_aa_work(
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
) -> Lapack_int = "LAPACKE_chetrs_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrs_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_dsytrs_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dsytrs_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrs_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const lapack_int* ipiv, double* b, 
///     lapack_int ldb, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_dsytrs_aa_work(
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
) -> Lapack_int = "LAPACKE_dsytrs_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrs_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_ssytrs_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytrs_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrs_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const lapack_int* ipiv, float* b, 
///     lapack_int ldb, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_ssytrs_aa_work(
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
) -> Lapack_int = "LAPACKE_ssytrs_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrs_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zsytrs_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zsytrs_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrs_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_zsytrs_aa_work(
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
) -> Lapack_int = "LAPACKE_zsytrs_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrs_aa(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zhetrs_aa(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhetrs_aa"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrs_aa_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, ipiv, b, work)
pub extern "C" fn lapacke_zhetrs_aa_work(
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
) -> Lapack_int = "LAPACKE_zhetrs_aa_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssysv_rk(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, float* e, lapack_int* ipiv, 
///     float* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_ssysv_rk(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ssysv_rk"

///|
///
/// ```c
/// lapack_int LAPACKE_ssysv_rk_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, float* e, lapack_int* ipiv, 
///     float* b, lapack_int ldb, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, e, ipiv, b, work)
pub extern "C" fn lapacke_ssysv_rk_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ssysv_rk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsysv_rk(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, double* e, lapack_int* ipiv, 
///     double* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_dsysv_rk(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dsysv_rk"

///|
///
/// ```c
/// lapack_int LAPACKE_dsysv_rk_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, double* e, lapack_int* ipiv, 
///     double* b, lapack_int ldb, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, e, ipiv, b, work)
pub extern "C" fn lapacke_dsysv_rk_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dsysv_rk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csysv_rk(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* e, lapack_int* ipiv, 
///     lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_csysv_rk(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_csysv_rk"

///|
///
/// ```c
/// lapack_int LAPACKE_csysv_rk_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* e, lapack_int* ipiv, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, e, ipiv, b, work)
pub extern "C" fn lapacke_csysv_rk_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_csysv_rk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsysv_rk(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* e, lapack_int* ipiv, 
///     lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_zsysv_rk(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zsysv_rk"

///|
///
/// ```c
/// lapack_int LAPACKE_zsysv_rk_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* e, lapack_int* ipiv, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, e, ipiv, b, work)
pub extern "C" fn lapacke_zsysv_rk_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zsysv_rk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chesv_rk(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* e, lapack_int* ipiv, 
///     lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_chesv_rk(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chesv_rk"

///|
///
/// ```c
/// lapack_int LAPACKE_chesv_rk_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* e, lapack_int* ipiv, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, e, ipiv, b, work)
pub extern "C" fn lapacke_chesv_rk_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_chesv_rk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhesv_rk(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* e, lapack_int* ipiv, 
///     lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_zhesv_rk(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhesv_rk"

///|
///
/// ```c
/// lapack_int LAPACKE_zhesv_rk_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* e, lapack_int* ipiv, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, e, ipiv, b, work)
pub extern "C" fn lapacke_zhesv_rk_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zhesv_rk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrf_rk(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, float* e, lapack_int* ipiv
/// );
/// ```
#borrow(a, e, ipiv)
pub extern "C" fn lapacke_ssytrf_rk(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ssytrf_rk"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrf_rk(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, double* e, lapack_int* ipiv
/// );
/// ```
#borrow(a, e, ipiv)
pub extern "C" fn lapacke_dsytrf_rk(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dsytrf_rk"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrf_rk(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* e, lapack_int* ipiv
/// );
/// ```
#borrow(a, e, ipiv)
pub extern "C" fn lapacke_csytrf_rk(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_csytrf_rk"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrf_rk(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* e, lapack_int* ipiv
/// );
/// ```
#borrow(a, e, ipiv)
pub extern "C" fn lapacke_zsytrf_rk(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zsytrf_rk"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrf_rk(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* e, lapack_int* ipiv
/// );
/// ```
#borrow(a, e, ipiv)
pub extern "C" fn lapacke_chetrf_rk(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chetrf_rk"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrf_rk(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* e, lapack_int* ipiv
/// );
/// ```
#borrow(a, e, ipiv)
pub extern "C" fn lapacke_zhetrf_rk(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhetrf_rk"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrf_rk_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, float* e, lapack_int* ipiv, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, e, ipiv, work)
pub extern "C" fn lapacke_ssytrf_rk_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytrf_rk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrf_rk_work(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, double* e, lapack_int* ipiv, double* work, lapack_int lwork
/// );
/// ```
#borrow(a, e, ipiv, work)
pub extern "C" fn lapacke_dsytrf_rk_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_dsytrf_rk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrf_rk_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* e, lapack_int* ipiv, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, e, ipiv, work)
pub extern "C" fn lapacke_csytrf_rk_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_csytrf_rk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrf_rk_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* e, lapack_int* ipiv, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, e, ipiv, work)
pub extern "C" fn lapacke_zsytrf_rk_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zsytrf_rk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrf_rk_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* e, lapack_int* ipiv, lapack_complex_float* work, lapack_int lwork
/// );
/// ```
#borrow(a, e, ipiv, work)
pub extern "C" fn lapacke_chetrf_rk_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_chetrf_rk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrf_rk_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* e, lapack_int* ipiv, lapack_complex_double* work, lapack_int lwork
/// );
/// ```
#borrow(a, e, ipiv, work)
pub extern "C" fn lapacke_zhetrf_rk_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
  work : FixedArray[Double],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_zhetrf_rk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrs_3(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* e, const lapack_int* ipiv, 
///     lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_csytrs_3(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_csytrs_3"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrs_3_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* e, const lapack_int* ipiv, 
///     lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_csytrs_3_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_csytrs_3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrs_3(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* e, const lapack_int* ipiv, 
///     lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_chetrs_3(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chetrs_3"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrs_3_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* e, const lapack_int* ipiv, 
///     lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_chetrs_3_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chetrs_3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrs_3(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const double* e, const lapack_int* ipiv, 
///     double* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_dsytrs_3(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dsytrs_3"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrs_3_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const double* e, const lapack_int* ipiv, 
///     double* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_dsytrs_3_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dsytrs_3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrs_3(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const float* e, const lapack_int* ipiv, 
///     float* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_ssytrs_3(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytrs_3"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrs_3_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const float* e, const lapack_int* ipiv, 
///     float* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_ssytrs_3_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytrs_3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrs_3(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* e, const lapack_int* ipiv, 
///     lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_zsytrs_3(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zsytrs_3"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrs_3_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* e, const lapack_int* ipiv, 
///     lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_zsytrs_3_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zsytrs_3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrs_3(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* e, const lapack_int* ipiv, 
///     lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_zhetrs_3(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhetrs_3"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrs_3_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* e, const lapack_int* ipiv, 
///     lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, e, ipiv, b)
pub extern "C" fn lapacke_zhetrs_3_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhetrs_3_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytri_3(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, const float* e, const lapack_int* ipiv
/// );
/// ```
#borrow(a, e, ipiv)
pub extern "C" fn lapacke_ssytri_3(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_ssytri_3"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytri_3(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, const double* e, const lapack_int* ipiv
/// );
/// ```
#borrow(a, e, ipiv)
pub extern "C" fn lapacke_dsytri_3(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dsytri_3"

///|
///
/// ```c
/// lapack_int LAPACKE_csytri_3(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_complex_float* e, const lapack_int* ipiv
/// );
/// ```
#borrow(a, e, ipiv)
pub extern "C" fn lapacke_csytri_3(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_csytri_3"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytri_3(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_complex_double* e, const lapack_int* ipiv
/// );
/// ```
#borrow(a, e, ipiv)
pub extern "C" fn lapacke_zsytri_3(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zsytri_3"

///|
///
/// ```c
/// lapack_int LAPACKE_chetri_3(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_complex_float* e, const lapack_int* ipiv
/// );
/// ```
#borrow(a, e, ipiv)
pub extern "C" fn lapacke_chetri_3(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chetri_3"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetri_3(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_complex_double* e, const lapack_int* ipiv
/// );
/// ```
#borrow(a, e, ipiv)
pub extern "C" fn lapacke_zhetri_3(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  e : FixedArray[Double],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhetri_3"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytri_3_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, const float* e, const lapack_int* ipiv, float* work, lapack_int lwork
/// );
/// ```
#borrow(a, e, ipiv, work)
pub extern "C" fn lapacke_ssytri_3_work(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  e : FixedArray[Float],
  ipiv : FixedArray[Int],
  work : FixedArray[Float],
  lwork : Lapack_int,
) -> Lapack_int = "LAPACKE_ssytri_3_work"
