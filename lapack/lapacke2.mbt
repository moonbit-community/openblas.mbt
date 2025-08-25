///|
///
/// ```c
/// lapack_int LAPACKE_zgghd3(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* q, lapack_int ldq, 
///     lapack_complex_double* z, lapack_int ldz
/// );
/// ```
#borrow(a, b, q, z)
pub extern "C" fn lapacke_zgghd3(
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
) -> Lapack_int = "LAPACKE_zgghd3"

///|
///
/// ```c
/// lapack_int LAPACKE_sgglse(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int p, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* c, float* d, float* x
/// );
/// ```
#borrow(a, b, c, d, x)
pub extern "C" fn lapacke_sgglse(
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
) -> Lapack_int = "LAPACKE_sgglse"

///|
///
/// ```c
/// lapack_int LAPACKE_dgglse(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int p, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* c, double* d, double* x
/// );
/// ```
#borrow(a, b, c, d, x)
pub extern "C" fn lapacke_dgglse(
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
) -> Lapack_int = "LAPACKE_dgglse"

///|
///
/// ```c
/// lapack_int LAPACKE_cgglse(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int p, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* c, lapack_complex_float* d, lapack_complex_float* x
/// );
/// ```
#borrow(a, b, c, d, x)
pub extern "C" fn lapacke_cgglse(
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
) -> Lapack_int = "LAPACKE_cgglse"

///|
///
/// ```c
/// lapack_int LAPACKE_zgglse(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int p, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* c, lapack_complex_double* d, lapack_complex_double* x
/// );
/// ```
#borrow(a, b, c, d, x)
pub extern "C" fn lapacke_zgglse(
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
) -> Lapack_int = "LAPACKE_zgglse"

///|
///
/// ```c
/// lapack_int LAPACKE_sggqrf(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     float* a, lapack_int lda, float* taua, float* b, 
///     lapack_int ldb, float* taub
/// );
/// ```
#borrow(a, taua, b, taub)
pub extern "C" fn lapacke_sggqrf(
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
) -> Lapack_int = "LAPACKE_sggqrf"

///|
///
/// ```c
/// lapack_int LAPACKE_dggqrf(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     double* a, lapack_int lda, double* taua, double* b, 
///     lapack_int ldb, double* taub
/// );
/// ```
#borrow(a, taua, b, taub)
pub extern "C" fn lapacke_dggqrf(
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
) -> Lapack_int = "LAPACKE_dggqrf"

///|
///
/// ```c
/// lapack_int LAPACKE_cggqrf(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* taua, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* taub
/// );
/// ```
#borrow(a, taua, b, taub)
pub extern "C" fn lapacke_cggqrf(
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
) -> Lapack_int = "LAPACKE_cggqrf"

///|
///
/// ```c
/// lapack_int LAPACKE_zggqrf(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* taua, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* taub
/// );
/// ```
#borrow(a, taua, b, taub)
pub extern "C" fn lapacke_zggqrf(
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
) -> Lapack_int = "LAPACKE_zggqrf"

///|
///
/// ```c
/// lapack_int LAPACKE_sggrqf(
///     int matrix_layout, lapack_int m, lapack_int p, lapack_int n, 
///     float* a, lapack_int lda, float* taua, float* b, 
///     lapack_int ldb, float* taub
/// );
/// ```
#borrow(a, taua, b, taub)
pub extern "C" fn lapacke_sggrqf(
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
) -> Lapack_int = "LAPACKE_sggrqf"

///|
///
/// ```c
/// lapack_int LAPACKE_dggrqf(
///     int matrix_layout, lapack_int m, lapack_int p, lapack_int n, 
///     double* a, lapack_int lda, double* taua, double* b, 
///     lapack_int ldb, double* taub
/// );
/// ```
#borrow(a, taua, b, taub)
pub extern "C" fn lapacke_dggrqf(
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
) -> Lapack_int = "LAPACKE_dggrqf"

///|
///
/// ```c
/// lapack_int LAPACKE_cggrqf(
///     int matrix_layout, lapack_int m, lapack_int p, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* taua, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* taub
/// );
/// ```
#borrow(a, taua, b, taub)
pub extern "C" fn lapacke_cggrqf(
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
) -> Lapack_int = "LAPACKE_cggrqf"

///|
///
/// ```c
/// lapack_int LAPACKE_zggrqf(
///     int matrix_layout, lapack_int m, lapack_int p, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* taua, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* taub
/// );
/// ```
#borrow(a, taua, b, taub)
pub extern "C" fn lapacke_zggrqf(
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
) -> Lapack_int = "LAPACKE_zggrqf"

///|
///
/// ```c
/// lapack_int LAPACKE_sggsvd(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, float* a, lapack_int lda, float* b, 
///     lapack_int ldb, float* alpha, float* beta, float* u, 
///     lapack_int ldu, float* v, lapack_int ldv, float* q, 
///     lapack_int ldq, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, iwork)
pub extern "C" fn lapacke_sggsvd(
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
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sggsvd"

///|
///
/// ```c
/// lapack_int LAPACKE_dggsvd(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, double* a, lapack_int lda, double* b, 
///     lapack_int ldb, double* alpha, double* beta, double* u, 
///     lapack_int ldu, double* v, lapack_int ldv, double* q, 
///     lapack_int ldq, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, iwork)
pub extern "C" fn lapacke_dggsvd(
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
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dggsvd"

///|
///
/// ```c
/// lapack_int LAPACKE_cggsvd(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, 
///     lapack_int ldb, float* alpha, float* beta, lapack_complex_float* u, 
///     lapack_int ldu, lapack_complex_float* v, lapack_int ldv, lapack_complex_float* q, 
///     lapack_int ldq, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, iwork)
pub extern "C" fn lapacke_cggsvd(
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
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cggsvd"

///|
///
/// ```c
/// lapack_int LAPACKE_zggsvd(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, 
///     lapack_int ldb, double* alpha, double* beta, lapack_complex_double* u, 
///     lapack_int ldu, lapack_complex_double* v, lapack_int ldv, lapack_complex_double* q, 
///     lapack_int ldq, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, iwork)
pub extern "C" fn lapacke_zggsvd(
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
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zggsvd"

///|
///
/// ```c
/// lapack_int LAPACKE_sggsvd3(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, float* a, lapack_int lda, float* b, 
///     lapack_int ldb, float* alpha, float* beta, float* u, 
///     lapack_int ldu, float* v, lapack_int ldv, float* q, 
///     lapack_int ldq, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, iwork)
pub extern "C" fn lapacke_sggsvd3(
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
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sggsvd3"

///|
///
/// ```c
/// lapack_int LAPACKE_dggsvd3(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, double* a, lapack_int lda, double* b, 
///     lapack_int ldb, double* alpha, double* beta, double* u, 
///     lapack_int ldu, double* v, lapack_int ldv, double* q, 
///     lapack_int ldq, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, iwork)
pub extern "C" fn lapacke_dggsvd3(
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
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dggsvd3"

///|
///
/// ```c
/// lapack_int LAPACKE_cggsvd3(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, 
///     lapack_int ldb, float* alpha, float* beta, lapack_complex_float* u, 
///     lapack_int ldu, lapack_complex_float* v, lapack_int ldv, lapack_complex_float* q, 
///     lapack_int ldq, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, iwork)
pub extern "C" fn lapacke_cggsvd3(
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
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cggsvd3"

///|
///
/// ```c
/// lapack_int LAPACKE_zggsvd3(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int n, lapack_int p, lapack_int* k, 
///     lapack_int* l, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, 
///     lapack_int ldb, double* alpha, double* beta, lapack_complex_double* u, 
///     lapack_int ldu, lapack_complex_double* v, lapack_int ldv, lapack_complex_double* q, 
///     lapack_int ldq, lapack_int* iwork
/// );
/// ```
#borrow(k, l, a, b, alpha, beta, u, v, q, iwork)
pub extern "C" fn lapacke_zggsvd3(
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
  iwork : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zggsvd3"

///|
///
/// ```c
/// lapack_int LAPACKE_sggsvp(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, float* a, 
///     lapack_int lda, float* b, lapack_int ldb, float tola, 
///     float tolb, lapack_int* k, lapack_int* l, float* u, 
///     lapack_int ldu, float* v, lapack_int ldv, float* q, lapack_int ldq
/// );
/// ```
#borrow(a, b, k, l, u, v, q)
pub extern "C" fn lapacke_sggsvp(
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
) -> Lapack_int = "LAPACKE_sggsvp"

///|
///
/// ```c
/// lapack_int LAPACKE_dggsvp(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, double* a, 
///     lapack_int lda, double* b, lapack_int ldb, double tola, 
///     double tolb, lapack_int* k, lapack_int* l, double* u, 
///     lapack_int ldu, double* v, lapack_int ldv, double* q, lapack_int ldq
/// );
/// ```
#borrow(a, b, k, l, u, v, q)
pub extern "C" fn lapacke_dggsvp(
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
) -> Lapack_int = "LAPACKE_dggsvp"

///|
///
/// ```c
/// lapack_int LAPACKE_cggsvp(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* b, lapack_int ldb, float tola, 
///     float tolb, lapack_int* k, lapack_int* l, lapack_complex_float* u, 
///     lapack_int ldu, lapack_complex_float* v, lapack_int ldv, lapack_complex_float* q, lapack_int ldq
/// );
/// ```
#borrow(a, b, k, l, u, v, q)
pub extern "C" fn lapacke_cggsvp(
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
) -> Lapack_int = "LAPACKE_cggsvp"

///|
///
/// ```c
/// lapack_int LAPACKE_zggsvp(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* b, lapack_int ldb, double tola, 
///     double tolb, lapack_int* k, lapack_int* l, lapack_complex_double* u, 
///     lapack_int ldu, lapack_complex_double* v, lapack_int ldv, lapack_complex_double* q, lapack_int ldq
/// );
/// ```
#borrow(a, b, k, l, u, v, q)
pub extern "C" fn lapacke_zggsvp(
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
) -> Lapack_int = "LAPACKE_zggsvp"

///|
///
/// ```c
/// lapack_int LAPACKE_sggsvp3(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, float* a, 
///     lapack_int lda, float* b, lapack_int ldb, float tola, 
///     float tolb, lapack_int* k, lapack_int* l, float* u, 
///     lapack_int ldu, float* v, lapack_int ldv, float* q, lapack_int ldq
/// );
/// ```
#borrow(a, b, k, l, u, v, q)
pub extern "C" fn lapacke_sggsvp3(
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
) -> Lapack_int = "LAPACKE_sggsvp3"

///|
///
/// ```c
/// lapack_int LAPACKE_dggsvp3(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, double* a, 
///     lapack_int lda, double* b, lapack_int ldb, double tola, 
///     double tolb, lapack_int* k, lapack_int* l, double* u, 
///     lapack_int ldu, double* v, lapack_int ldv, double* q, lapack_int ldq
/// );
/// ```
#borrow(a, b, k, l, u, v, q)
pub extern "C" fn lapacke_dggsvp3(
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
) -> Lapack_int = "LAPACKE_dggsvp3"

///|
///
/// ```c
/// lapack_int LAPACKE_cggsvp3(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* b, lapack_int ldb, float tola, 
///     float tolb, lapack_int* k, lapack_int* l, lapack_complex_float* u, 
///     lapack_int ldu, lapack_complex_float* v, lapack_int ldv, lapack_complex_float* q, lapack_int ldq
/// );
/// ```
#borrow(a, b, k, l, u, v, q)
pub extern "C" fn lapacke_cggsvp3(
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
) -> Lapack_int = "LAPACKE_cggsvp3"

///|
///
/// ```c
/// lapack_int LAPACKE_zggsvp3(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* b, lapack_int ldb, double tola, 
///     double tolb, lapack_int* k, lapack_int* l, lapack_complex_double* u, 
///     lapack_int ldu, lapack_complex_double* v, lapack_int ldv, lapack_complex_double* q, lapack_int ldq
/// );
/// ```
#borrow(a, b, k, l, u, v, q)
pub extern "C" fn lapacke_zggsvp3(
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
) -> Lapack_int = "LAPACKE_zggsvp3"

///|
///
/// ```c
/// lapack_int LAPACKE_sgtcon(
///     char norm, lapack_int n, const float* dl, const float* d, 
///     const float* du, const float* du2, const lapack_int* ipiv, float anorm, float* rcond
/// );
/// ```
#borrow(dl, d, du, du2, ipiv, rcond)
pub extern "C" fn lapacke_sgtcon(
  norm : Byte,
  n : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  du2 : FixedArray[Float],
  ipiv : FixedArray[Int],
  anorm : Float,
  rcond : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sgtcon"

///|
///
/// ```c
/// lapack_int LAPACKE_dgtcon(
///     char norm, lapack_int n, const double* dl, const double* d, 
///     const double* du, const double* du2, const lapack_int* ipiv, double anorm, double* rcond
/// );
/// ```
#borrow(dl, d, du, du2, ipiv, rcond)
pub extern "C" fn lapacke_dgtcon(
  norm : Byte,
  n : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  du2 : FixedArray[Double],
  ipiv : FixedArray[Int],
  anorm : Double,
  rcond : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dgtcon"

///|
///
/// ```c
/// lapack_int LAPACKE_cgtcon(
///     char norm, lapack_int n, const lapack_complex_float* dl, const lapack_complex_float* d, 
///     const lapack_complex_float* du, const lapack_complex_float* du2, const lapack_int* ipiv, float anorm, float* rcond
/// );
/// ```
#borrow(dl, d, du, du2, ipiv, rcond)
pub extern "C" fn lapacke_cgtcon(
  norm : Byte,
  n : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  du2 : FixedArray[Float],
  ipiv : FixedArray[Int],
  anorm : Float,
  rcond : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cgtcon"

///|
///
/// ```c
/// lapack_int LAPACKE_zgtcon(
///     char norm, lapack_int n, const lapack_complex_double* dl, const lapack_complex_double* d, 
///     const lapack_complex_double* du, const lapack_complex_double* du2, const lapack_int* ipiv, double anorm, double* rcond
/// );
/// ```
#borrow(dl, d, du, du2, ipiv, rcond)
pub extern "C" fn lapacke_zgtcon(
  norm : Byte,
  n : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  du2 : FixedArray[Double],
  ipiv : FixedArray[Int],
  anorm : Double,
  rcond : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zgtcon"

///|
///
/// ```c
/// lapack_int LAPACKE_sgtrfs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const float* dl, const float* d, const float* du, const float* dlf, 
///     const float* df, const float* duf, const float* du2, const lapack_int* ipiv, 
///     const float* b, lapack_int ldb, float* x, lapack_int ldx, 
///     float* ferr, float* berr
/// );
/// ```
#borrow(dl, d, du, dlf, df, duf, du2, ipiv, b, x, ferr, berr)
pub extern "C" fn lapacke_sgtrfs(
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
) -> Lapack_int = "LAPACKE_sgtrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_dgtrfs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const double* dl, const double* d, const double* du, const double* dlf, 
///     const double* df, const double* duf, const double* du2, const lapack_int* ipiv, 
///     const double* b, lapack_int ldb, double* x, lapack_int ldx, 
///     double* ferr, double* berr
/// );
/// ```
#borrow(dl, d, du, dlf, df, duf, du2, ipiv, b, x, ferr, berr)
pub extern "C" fn lapacke_dgtrfs(
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
) -> Lapack_int = "LAPACKE_dgtrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_cgtrfs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* dl, const lapack_complex_float* d, const lapack_complex_float* du, const lapack_complex_float* dlf, 
///     const lapack_complex_float* df, const lapack_complex_float* duf, const lapack_complex_float* du2, const lapack_int* ipiv, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* ferr, float* berr
/// );
/// ```
#borrow(dl, d, du, dlf, df, duf, du2, ipiv, b, x, ferr, berr)
pub extern "C" fn lapacke_cgtrfs(
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
) -> Lapack_int = "LAPACKE_cgtrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_zgtrfs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* dl, const lapack_complex_double* d, const lapack_complex_double* du, const lapack_complex_double* dlf, 
///     const lapack_complex_double* df, const lapack_complex_double* duf, const lapack_complex_double* du2, const lapack_int* ipiv, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* ferr, double* berr
/// );
/// ```
#borrow(dl, d, du, dlf, df, duf, du2, ipiv, b, x, ferr, berr)
pub extern "C" fn lapacke_zgtrfs(
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
) -> Lapack_int = "LAPACKE_zgtrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_sgtsv(
///     int matrix_layout, lapack_int n, lapack_int nrhs, float* dl, 
///     float* d, float* du, float* b, lapack_int ldb
/// );
/// ```
#borrow(dl, d, du, b)
pub extern "C" fn lapacke_sgtsv(
  matrix_layout : Int,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_sgtsv"

///|
///
/// ```c
/// lapack_int LAPACKE_dgtsv(
///     int matrix_layout, lapack_int n, lapack_int nrhs, double* dl, 
///     double* d, double* du, double* b, lapack_int ldb
/// );
/// ```
#borrow(dl, d, du, b)
pub extern "C" fn lapacke_dgtsv(
  matrix_layout : Int,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dgtsv"

///|
///
/// ```c
/// lapack_int LAPACKE_cgtsv(
///     int matrix_layout, lapack_int n, lapack_int nrhs, lapack_complex_float* dl, 
///     lapack_complex_float* d, lapack_complex_float* du, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(dl, d, du, b)
pub extern "C" fn lapacke_cgtsv(
  matrix_layout : Int,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_cgtsv"

///|
///
/// ```c
/// lapack_int LAPACKE_zgtsv(
///     int matrix_layout, lapack_int n, lapack_int nrhs, lapack_complex_double* dl, 
///     lapack_complex_double* d, lapack_complex_double* du, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(dl, d, du, b)
pub extern "C" fn lapacke_zgtsv(
  matrix_layout : Int,
  n : Lapack_int,
  nrhs : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zgtsv"

///|
///
/// ```c
/// lapack_int LAPACKE_sgtsvx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, const float* dl, const float* d, const float* du, 
///     float* dlf, float* df, float* duf, float* du2, 
///     lapack_int* ipiv, const float* b, lapack_int ldb, float* x, 
///     lapack_int ldx, float* rcond, float* ferr, float* berr
/// );
/// ```
#borrow(dl, d, du, dlf, df, duf, du2, ipiv, b, x, rcond, ferr, berr)
pub extern "C" fn lapacke_sgtsvx(
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
) -> Lapack_int = "LAPACKE_sgtsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_dgtsvx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, const double* dl, const double* d, const double* du, 
///     double* dlf, double* df, double* duf, double* du2, 
///     lapack_int* ipiv, const double* b, lapack_int ldb, double* x, 
///     lapack_int ldx, double* rcond, double* ferr, double* berr
/// );
/// ```
#borrow(dl, d, du, dlf, df, duf, du2, ipiv, b, x, rcond, ferr, berr)
pub extern "C" fn lapacke_dgtsvx(
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
) -> Lapack_int = "LAPACKE_dgtsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_cgtsvx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* dl, const lapack_complex_float* d, const lapack_complex_float* du, 
///     lapack_complex_float* dlf, lapack_complex_float* df, lapack_complex_float* duf, lapack_complex_float* du2, 
///     lapack_int* ipiv, const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* rcond, float* ferr, float* berr
/// );
/// ```
#borrow(dl, d, du, dlf, df, duf, du2, ipiv, b, x, rcond, ferr, berr)
pub extern "C" fn lapacke_cgtsvx(
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
) -> Lapack_int = "LAPACKE_cgtsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_zgtsvx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* dl, const lapack_complex_double* d, const lapack_complex_double* du, 
///     lapack_complex_double* dlf, lapack_complex_double* df, lapack_complex_double* duf, lapack_complex_double* du2, 
///     lapack_int* ipiv, const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* rcond, double* ferr, double* berr
/// );
/// ```
#borrow(dl, d, du, dlf, df, duf, du2, ipiv, b, x, rcond, ferr, berr)
pub extern "C" fn lapacke_zgtsvx(
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
) -> Lapack_int = "LAPACKE_zgtsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_sgttrf(
///     lapack_int n, float* dl, float* d, float* du, 
///     float* du2, lapack_int* ipiv
/// );
/// ```
#borrow(dl, d, du, du2, ipiv)
pub extern "C" fn lapacke_sgttrf(
  n : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  du2 : FixedArray[Float],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_sgttrf"

///|
///
/// ```c
/// lapack_int LAPACKE_dgttrf(
///     lapack_int n, double* dl, double* d, double* du, 
///     double* du2, lapack_int* ipiv
/// );
/// ```
#borrow(dl, d, du, du2, ipiv)
pub extern "C" fn lapacke_dgttrf(
  n : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  du2 : FixedArray[Double],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dgttrf"

///|
///
/// ```c
/// lapack_int LAPACKE_cgttrf(
///     lapack_int n, lapack_complex_float* dl, lapack_complex_float* d, lapack_complex_float* du, 
///     lapack_complex_float* du2, lapack_int* ipiv
/// );
/// ```
#borrow(dl, d, du, du2, ipiv)
pub extern "C" fn lapacke_cgttrf(
  n : Lapack_int,
  dl : FixedArray[Float],
  d : FixedArray[Float],
  du : FixedArray[Float],
  du2 : FixedArray[Float],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cgttrf"

///|
///
/// ```c
/// lapack_int LAPACKE_zgttrf(
///     lapack_int n, lapack_complex_double* dl, lapack_complex_double* d, lapack_complex_double* du, 
///     lapack_complex_double* du2, lapack_int* ipiv
/// );
/// ```
#borrow(dl, d, du, du2, ipiv)
pub extern "C" fn lapacke_zgttrf(
  n : Lapack_int,
  dl : FixedArray[Double],
  d : FixedArray[Double],
  du : FixedArray[Double],
  du2 : FixedArray[Double],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zgttrf"

///|
///
/// ```c
/// lapack_int LAPACKE_sgttrs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const float* dl, const float* d, const float* du, const float* du2, 
///     const lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
#borrow(dl, d, du, du2, ipiv, b)
pub extern "C" fn lapacke_sgttrs(
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
) -> Lapack_int = "LAPACKE_sgttrs"

///|
///
/// ```c
/// lapack_int LAPACKE_dgttrs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const double* dl, const double* d, const double* du, const double* du2, 
///     const lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
#borrow(dl, d, du, du2, ipiv, b)
pub extern "C" fn lapacke_dgttrs(
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
) -> Lapack_int = "LAPACKE_dgttrs"

///|
///
/// ```c
/// lapack_int LAPACKE_cgttrs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* dl, const lapack_complex_float* d, const lapack_complex_float* du, const lapack_complex_float* du2, 
///     const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(dl, d, du, du2, ipiv, b)
pub extern "C" fn lapacke_cgttrs(
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
) -> Lapack_int = "LAPACKE_cgttrs"

///|
///
/// ```c
/// lapack_int LAPACKE_zgttrs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* dl, const lapack_complex_double* d, const lapack_complex_double* du, const lapack_complex_double* du2, 
///     const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(dl, d, du, du2, ipiv, b)
pub extern "C" fn lapacke_zgttrs(
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
) -> Lapack_int = "LAPACKE_zgttrs"

///|
///
/// ```c
/// lapack_int LAPACKE_chbev(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, lapack_complex_float* ab, lapack_int ldab, float* w, 
///     lapack_complex_float* z, lapack_int ldz
/// );
/// ```
#borrow(ab, w, z)
pub extern "C" fn lapacke_chbev(
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
) -> Lapack_int = "LAPACKE_chbev"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbev(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, lapack_complex_double* ab, lapack_int ldab, double* w, 
///     lapack_complex_double* z, lapack_int ldz
/// );
/// ```
#borrow(ab, w, z)
pub extern "C" fn lapacke_zhbev(
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
) -> Lapack_int = "LAPACKE_zhbev"

///|
///
/// ```c
/// lapack_int LAPACKE_chbevd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, lapack_complex_float* ab, lapack_int ldab, float* w, 
///     lapack_complex_float* z, lapack_int ldz
/// );
/// ```
#borrow(ab, w, z)
pub extern "C" fn lapacke_chbevd(
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
) -> Lapack_int = "LAPACKE_chbevd"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbevd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, lapack_complex_double* ab, lapack_int ldab, double* w, 
///     lapack_complex_double* z, lapack_int ldz
/// );
/// ```
#borrow(ab, w, z)
pub extern "C" fn lapacke_zhbevd(
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
) -> Lapack_int = "LAPACKE_zhbevd"

///|
///
/// ```c
/// lapack_int LAPACKE_chbevx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int kd, lapack_complex_float* ab, lapack_int ldab, 
///     lapack_complex_float* q, lapack_int ldq, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, lapack_complex_float* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
#borrow(ab, q, m, w, z, ifail)
pub extern "C" fn lapacke_chbevx(
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
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chbevx"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbevx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int kd, lapack_complex_double* ab, lapack_int ldab, 
///     lapack_complex_double* q, lapack_int ldq, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, lapack_complex_double* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
#borrow(ab, q, m, w, z, ifail)
pub extern "C" fn lapacke_zhbevx(
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
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhbevx"

///|
///
/// ```c
/// lapack_int LAPACKE_chbgst(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, lapack_complex_float* ab, lapack_int ldab, 
///     const lapack_complex_float* bb, lapack_int ldbb, lapack_complex_float* x, lapack_int ldx
/// );
/// ```
#borrow(ab, bb, x)
pub extern "C" fn lapacke_chbgst(
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
) -> Lapack_int = "LAPACKE_chbgst"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbgst(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, lapack_complex_double* ab, lapack_int ldab, 
///     const lapack_complex_double* bb, lapack_int ldbb, lapack_complex_double* x, lapack_int ldx
/// );
/// ```
#borrow(ab, bb, x)
pub extern "C" fn lapacke_zhbgst(
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
) -> Lapack_int = "LAPACKE_zhbgst"

///|
///
/// ```c
/// lapack_int LAPACKE_chbgv(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, lapack_complex_float* ab, lapack_int ldab, 
///     lapack_complex_float* bb, lapack_int ldbb, float* w, lapack_complex_float* z, lapack_int ldz
/// );
/// ```
#borrow(ab, bb, w, z)
pub extern "C" fn lapacke_chbgv(
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
) -> Lapack_int = "LAPACKE_chbgv"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbgv(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, lapack_complex_double* ab, lapack_int ldab, 
///     lapack_complex_double* bb, lapack_int ldbb, double* w, lapack_complex_double* z, lapack_int ldz
/// );
/// ```
#borrow(ab, bb, w, z)
pub extern "C" fn lapacke_zhbgv(
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
) -> Lapack_int = "LAPACKE_zhbgv"

///|
///
/// ```c
/// lapack_int LAPACKE_chbgvd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, lapack_complex_float* ab, lapack_int ldab, 
///     lapack_complex_float* bb, lapack_int ldbb, float* w, lapack_complex_float* z, lapack_int ldz
/// );
/// ```
#borrow(ab, bb, w, z)
pub extern "C" fn lapacke_chbgvd(
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
) -> Lapack_int = "LAPACKE_chbgvd"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbgvd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, lapack_complex_double* ab, lapack_int ldab, 
///     lapack_complex_double* bb, lapack_int ldbb, double* w, lapack_complex_double* z, lapack_int ldz
/// );
/// ```
#borrow(ab, bb, w, z)
pub extern "C" fn lapacke_zhbgvd(
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
) -> Lapack_int = "LAPACKE_zhbgvd"

///|
///
/// ```c
/// lapack_int LAPACKE_chbgvx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int ka, lapack_int kb, lapack_complex_float* ab, 
///     lapack_int ldab, lapack_complex_float* bb, lapack_int ldbb, lapack_complex_float* q, 
///     lapack_int ldq, float vl, float vu, lapack_int il, 
///     lapack_int iu, float abstol, lapack_int* m, float* w, 
///     lapack_complex_float* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
#borrow(ab, bb, q, m, w, z, ifail)
pub extern "C" fn lapacke_chbgvx(
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
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chbgvx"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbgvx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int ka, lapack_int kb, lapack_complex_double* ab, 
///     lapack_int ldab, lapack_complex_double* bb, lapack_int ldbb, lapack_complex_double* q, 
///     lapack_int ldq, double vl, double vu, lapack_int il, 
///     lapack_int iu, double abstol, lapack_int* m, double* w, 
///     lapack_complex_double* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
#borrow(ab, bb, q, m, w, z, ifail)
pub extern "C" fn lapacke_zhbgvx(
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
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhbgvx"

///|
///
/// ```c
/// lapack_int LAPACKE_chbtrd(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int kd, lapack_complex_float* ab, lapack_int ldab, float* d, 
///     float* e, lapack_complex_float* q, lapack_int ldq
/// );
/// ```
#borrow(ab, d, e, q)
pub extern "C" fn lapacke_chbtrd(
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
) -> Lapack_int = "LAPACKE_chbtrd"

///|
///
/// ```c
/// lapack_int LAPACKE_zhbtrd(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int kd, lapack_complex_double* ab, lapack_int ldab, double* d, 
///     double* e, lapack_complex_double* q, lapack_int ldq
/// );
/// ```
#borrow(ab, d, e, q)
pub extern "C" fn lapacke_zhbtrd(
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
) -> Lapack_int = "LAPACKE_zhbtrd"

///|
///
/// ```c
/// lapack_int LAPACKE_checon(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv, float anorm, float* rcond
/// );
/// ```
#borrow(a, ipiv, rcond)
pub extern "C" fn lapacke_checon(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  anorm : Float,
  rcond : FixedArray[Float],
) -> Lapack_int = "LAPACKE_checon"

///|
///
/// ```c
/// lapack_int LAPACKE_zhecon(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv, double anorm, double* rcond
/// );
/// ```
#borrow(a, ipiv, rcond)
pub extern "C" fn lapacke_zhecon(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  anorm : Double,
  rcond : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhecon"

///|
///
/// ```c
/// lapack_int LAPACKE_cheequb(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, float* s, float* scond, float* amax
/// );
/// ```
#borrow(a, s, scond, amax)
pub extern "C" fn lapacke_cheequb(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  s : FixedArray[Float],
  scond : FixedArray[Float],
  amax : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cheequb"

///|
///
/// ```c
/// lapack_int LAPACKE_zheequb(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, double* s, double* scond, double* amax
/// );
/// ```
#borrow(a, s, scond, amax)
pub extern "C" fn lapacke_zheequb(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  s : FixedArray[Double],
  scond : FixedArray[Double],
  amax : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zheequb"

///|
///
/// ```c
/// lapack_int LAPACKE_cheev(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, float* w
/// );
/// ```
#borrow(a, w)
pub extern "C" fn lapacke_cheev(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  w : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cheev"

///|
///
/// ```c
/// lapack_int LAPACKE_zheev(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, double* w
/// );
/// ```
#borrow(a, w)
pub extern "C" fn lapacke_zheev(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  w : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zheev"

///|
///
/// ```c
/// lapack_int LAPACKE_cheevd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, float* w
/// );
/// ```
#borrow(a, w)
pub extern "C" fn lapacke_cheevd(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  w : FixedArray[Float],
) -> Lapack_int = "LAPACKE_cheevd"

///|
///
/// ```c
/// lapack_int LAPACKE_zheevd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, double* w
/// );
/// ```
#borrow(a, w)
pub extern "C" fn lapacke_zheevd(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  w : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zheevd"

///|
///
/// ```c
/// lapack_int LAPACKE_cheevr(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_complex_float* a, lapack_int lda, float vl, 
///     float vu, lapack_int il, lapack_int iu, float abstol, 
///     lapack_int* m, float* w, lapack_complex_float* z, lapack_int ldz, lapack_int* isuppz
/// );
/// ```
#borrow(a, m, w, z, isuppz)
pub extern "C" fn lapacke_cheevr(
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
) -> Lapack_int = "LAPACKE_cheevr"

///|
///
/// ```c
/// lapack_int LAPACKE_zheevr(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_complex_double* a, lapack_int lda, double vl, 
///     double vu, lapack_int il, lapack_int iu, double abstol, 
///     lapack_int* m, double* w, lapack_complex_double* z, lapack_int ldz, lapack_int* isuppz
/// );
/// ```
#borrow(a, m, w, z, isuppz)
pub extern "C" fn lapacke_zheevr(
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
) -> Lapack_int = "LAPACKE_zheevr"

///|
///
/// ```c
/// lapack_int LAPACKE_cheevx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_complex_float* a, lapack_int lda, float vl, 
///     float vu, lapack_int il, lapack_int iu, float abstol, 
///     lapack_int* m, float* w, lapack_complex_float* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
#borrow(a, m, w, z, ifail)
pub extern "C" fn lapacke_cheevx(
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
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_cheevx"

///|
///
/// ```c
/// lapack_int LAPACKE_zheevx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_complex_double* a, lapack_int lda, double vl, 
///     double vu, lapack_int il, lapack_int iu, double abstol, 
///     lapack_int* m, double* w, lapack_complex_double* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
#borrow(a, m, w, z, ifail)
pub extern "C" fn lapacke_zheevx(
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
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zheevx"

///|
///
/// ```c
/// lapack_int LAPACKE_chegst(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, const lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_chegst(
  matrix_layout : Int,
  itype : Lapack_int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chegst"

///|
///
/// ```c
/// lapack_int LAPACKE_zhegst(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, const lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_zhegst(
  matrix_layout : Int,
  itype : Lapack_int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhegst"

///|
///
/// ```c
/// lapack_int LAPACKE_chegv(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, 
///     lapack_int ldb, float* w
/// );
/// ```
#borrow(a, b, w)
pub extern "C" fn lapacke_chegv(
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
) -> Lapack_int = "LAPACKE_chegv"

///|
///
/// ```c
/// lapack_int LAPACKE_zhegv(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, 
///     lapack_int ldb, double* w
/// );
/// ```
#borrow(a, b, w)
pub extern "C" fn lapacke_zhegv(
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
) -> Lapack_int = "LAPACKE_zhegv"

///|
///
/// ```c
/// lapack_int LAPACKE_chegvd(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, 
///     lapack_int ldb, float* w
/// );
/// ```
#borrow(a, b, w)
pub extern "C" fn lapacke_chegvd(
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
) -> Lapack_int = "LAPACKE_chegvd"

///|
///
/// ```c
/// lapack_int LAPACKE_zhegvd(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, 
///     lapack_int ldb, double* w
/// );
/// ```
#borrow(a, b, w)
pub extern "C" fn lapacke_zhegvd(
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
) -> Lapack_int = "LAPACKE_zhegvd"

///|
///
/// ```c
/// lapack_int LAPACKE_chegvx(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, lapack_complex_float* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
#borrow(a, b, m, w, z, ifail)
pub extern "C" fn lapacke_chegvx(
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
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chegvx"

///|
///
/// ```c
/// lapack_int LAPACKE_zhegvx(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, lapack_complex_double* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
#borrow(a, b, m, w, z, ifail)
pub extern "C" fn lapacke_zhegvx(
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
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhegvx"

///|
///
/// ```c
/// lapack_int LAPACKE_cherfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* ferr, float* berr
/// );
/// ```
#borrow(a, af, ipiv, b, x, ferr, berr)
pub extern "C" fn lapacke_cherfs(
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
) -> Lapack_int = "LAPACKE_cherfs"

///|
///
/// ```c
/// lapack_int LAPACKE_zherfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* ferr, double* berr
/// );
/// ```
#borrow(a, af, ipiv, b, x, ferr, berr)
pub extern "C" fn lapacke_zherfs(
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
) -> Lapack_int = "LAPACKE_zherfs"

///|
///
/// ```c
/// lapack_int LAPACKE_cherfsx(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const float* s, const lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond, 
///     float* berr, lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, 
///     lapack_int nparams, float* params
/// );
/// ```
#borrow(a, af, ipiv, s, b, x, rcond, berr, err_bnds_norm, err_bnds_comp, params)
pub extern "C" fn lapacke_cherfsx(
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
) -> Lapack_int = "LAPACKE_cherfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_zherfsx(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const double* s, const lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond, 
///     double* berr, lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, 
///     lapack_int nparams, double* params
/// );
/// ```
#borrow(a, af, ipiv, s, b, x, rcond, berr, err_bnds_norm, err_bnds_comp, params)
pub extern "C" fn lapacke_zherfsx(
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
) -> Lapack_int = "LAPACKE_zherfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_chesv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_chesv(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chesv"

///|
///
/// ```c
/// lapack_int LAPACKE_zhesv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zhesv(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhesv"

///|
///
/// ```c
/// lapack_int LAPACKE_chesvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, lapack_int* ipiv, const lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* x, lapack_int ldx, float* rcond, float* ferr, float* berr
/// );
/// ```
#borrow(a, af, ipiv, b, x, rcond, ferr, berr)
pub extern "C" fn lapacke_chesvx(
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
) -> Lapack_int = "LAPACKE_chesvx"

///|
///
/// ```c
/// lapack_int LAPACKE_zhesvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, lapack_int* ipiv, const lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* x, lapack_int ldx, double* rcond, double* ferr, double* berr
/// );
/// ```
#borrow(a, af, ipiv, b, x, rcond, ferr, berr)
pub extern "C" fn lapacke_zhesvx(
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
) -> Lapack_int = "LAPACKE_zhesvx"

///|
///
/// ```c
/// lapack_int LAPACKE_chesvxx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, float* s, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* rcond, float* rpvgrw, float* berr, lapack_int n_err_bnds, 
///     float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, float* params
/// );
/// ```
#borrow(a, af, ipiv, equed, s, b, x, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, params)
pub extern "C" fn lapacke_chesvxx(
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
) -> Lapack_int = "LAPACKE_chesvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_zhesvxx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, double* s, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* rcond, double* rpvgrw, double* berr, lapack_int n_err_bnds, 
///     double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, double* params
/// );
/// ```
#borrow(a, af, ipiv, equed, s, b, x, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, params)
pub extern "C" fn lapacke_zhesvxx(
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
) -> Lapack_int = "LAPACKE_zhesvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrd(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, float* d, float* e, lapack_complex_float* tau
/// );
/// ```
#borrow(a, d, e, tau)
pub extern "C" fn lapacke_chetrd(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  d : FixedArray[Float],
  e : FixedArray[Float],
  tau : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chetrd"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrd(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, double* d, double* e, lapack_complex_double* tau
/// );
/// ```
#borrow(a, d, e, tau)
pub extern "C" fn lapacke_zhetrd(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  d : FixedArray[Double],
  e : FixedArray[Double],
  tau : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhetrd"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_chetrf(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chetrf"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_zhetrf(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhetrf"

///|
///
/// ```c
/// lapack_int LAPACKE_chetri(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_chetri(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chetri"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetri(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_zhetri(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhetri"

///|
///
/// ```c
/// lapack_int LAPACKE_chetrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_chetrs(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chetrs"

///|
///
/// ```c
/// lapack_int LAPACKE_zhetrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, ipiv, b)
pub extern "C" fn lapacke_zhetrs(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhetrs"

///|
///
/// ```c
/// lapack_int LAPACKE_chfrk(
///     int matrix_layout, char transr, char uplo, char trans, 
///     lapack_int n, lapack_int k, float alpha, const lapack_complex_float* a, 
///     lapack_int lda, float beta, lapack_complex_float* c
/// );
/// ```
#borrow(a, c)
pub extern "C" fn lapacke_chfrk(
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
) -> Lapack_int = "LAPACKE_chfrk"

///|
///
/// ```c
/// lapack_int LAPACKE_zhfrk(
///     int matrix_layout, char transr, char uplo, char trans, 
///     lapack_int n, lapack_int k, double alpha, const lapack_complex_double* a, 
///     lapack_int lda, double beta, lapack_complex_double* c
/// );
/// ```
#borrow(a, c)
pub extern "C" fn lapacke_zhfrk(
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
) -> Lapack_int = "LAPACKE_zhfrk"

///|
///
/// ```c
/// lapack_int LAPACKE_shgeqz(
///     int matrix_layout, char job, char compq, char compz, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, float* h, 
///     lapack_int ldh, float* t, lapack_int ldt, float* alphar, 
///     float* alphai, float* beta, float* q, lapack_int ldq, 
///     float* z, lapack_int ldz
/// );
/// ```
#borrow(h, t, alphar, alphai, beta, q, z)
pub extern "C" fn lapacke_shgeqz(
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
) -> Lapack_int = "LAPACKE_shgeqz"

///|
///
/// ```c
/// lapack_int LAPACKE_dhgeqz(
///     int matrix_layout, char job, char compq, char compz, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, double* h, 
///     lapack_int ldh, double* t, lapack_int ldt, double* alphar, 
///     double* alphai, double* beta, double* q, lapack_int ldq, 
///     double* z, lapack_int ldz
/// );
/// ```
#borrow(h, t, alphar, alphai, beta, q, z)
pub extern "C" fn lapacke_dhgeqz(
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
) -> Lapack_int = "LAPACKE_dhgeqz"

///|
///
/// ```c
/// lapack_int LAPACKE_chgeqz(
///     int matrix_layout, char job, char compq, char compz, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, lapack_complex_float* h, 
///     lapack_int ldh, lapack_complex_float* t, lapack_int ldt, lapack_complex_float* alpha, 
///     lapack_complex_float* beta, lapack_complex_float* q, lapack_int ldq, lapack_complex_float* z, lapack_int ldz
/// );
/// ```
#borrow(h, t, alpha, beta, q, z)
pub extern "C" fn lapacke_chgeqz(
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
) -> Lapack_int = "LAPACKE_chgeqz"

///|
///
/// ```c
/// lapack_int LAPACKE_zhgeqz(
///     int matrix_layout, char job, char compq, char compz, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, lapack_complex_double* h, 
///     lapack_int ldh, lapack_complex_double* t, lapack_int ldt, lapack_complex_double* alpha, 
///     lapack_complex_double* beta, lapack_complex_double* q, lapack_int ldq, lapack_complex_double* z, lapack_int ldz
/// );
/// ```
#borrow(h, t, alpha, beta, q, z)
pub extern "C" fn lapacke_zhgeqz(
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
) -> Lapack_int = "LAPACKE_zhgeqz"

///|
///
/// ```c
/// lapack_int LAPACKE_chpcon(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* ap, 
///     const lapack_int* ipiv, float anorm, float* rcond
/// );
/// ```
#borrow(ap, ipiv, rcond)
pub extern "C" fn lapacke_chpcon(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  ipiv : FixedArray[Int],
  anorm : Float,
  rcond : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chpcon"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpcon(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* ap, 
///     const lapack_int* ipiv, double anorm, double* rcond
/// );
/// ```
#borrow(ap, ipiv, rcond)
pub extern "C" fn lapacke_zhpcon(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  ipiv : FixedArray[Int],
  anorm : Double,
  rcond : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhpcon"

///|
///
/// ```c
/// lapack_int LAPACKE_chpev(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_float* ap, float* w, lapack_complex_float* z, lapack_int ldz
/// );
/// ```
#borrow(ap, w, z)
pub extern "C" fn lapacke_chpev(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
) -> Lapack_int = "LAPACKE_chpev"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpev(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_double* ap, double* w, lapack_complex_double* z, lapack_int ldz
/// );
/// ```
#borrow(ap, w, z)
pub extern "C" fn lapacke_zhpev(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
) -> Lapack_int = "LAPACKE_zhpev"

///|
///
/// ```c
/// lapack_int LAPACKE_chpevd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_float* ap, float* w, lapack_complex_float* z, lapack_int ldz
/// );
/// ```
#borrow(ap, w, z)
pub extern "C" fn lapacke_chpevd(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  w : FixedArray[Float],
  z : FixedArray[Float],
  ldz : Lapack_int,
) -> Lapack_int = "LAPACKE_chpevd"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpevd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_complex_double* ap, double* w, lapack_complex_double* z, lapack_int ldz
/// );
/// ```
#borrow(ap, w, z)
pub extern "C" fn lapacke_zhpevd(
  matrix_layout : Int,
  jobz : Byte,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  w : FixedArray[Double],
  z : FixedArray[Double],
  ldz : Lapack_int,
) -> Lapack_int = "LAPACKE_zhpevd"

///|
///
/// ```c
/// lapack_int LAPACKE_chpevx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_complex_float* ap, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, lapack_complex_float* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
#borrow(ap, m, w, z, ifail)
pub extern "C" fn lapacke_chpevx(
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
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chpevx"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpevx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_complex_double* ap, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, lapack_complex_double* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
#borrow(ap, m, w, z, ifail)
pub extern "C" fn lapacke_zhpevx(
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
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhpevx"

///|
///
/// ```c
/// lapack_int LAPACKE_chpgst(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     lapack_complex_float* ap, const lapack_complex_float* bp
/// );
/// ```
#borrow(ap, bp)
pub extern "C" fn lapacke_chpgst(
  matrix_layout : Int,
  itype : Lapack_int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  bp : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chpgst"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpgst(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     lapack_complex_double* ap, const lapack_complex_double* bp
/// );
/// ```
#borrow(ap, bp)
pub extern "C" fn lapacke_zhpgst(
  matrix_layout : Int,
  itype : Lapack_int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  bp : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhpgst"

///|
///
/// ```c
/// lapack_int LAPACKE_chpgv(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_float* ap, lapack_complex_float* bp, float* w, 
///     lapack_complex_float* z, lapack_int ldz
/// );
/// ```
#borrow(ap, bp, w, z)
pub extern "C" fn lapacke_chpgv(
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
) -> Lapack_int = "LAPACKE_chpgv"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpgv(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_double* ap, lapack_complex_double* bp, double* w, 
///     lapack_complex_double* z, lapack_int ldz
/// );
/// ```
#borrow(ap, bp, w, z)
pub extern "C" fn lapacke_zhpgv(
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
) -> Lapack_int = "LAPACKE_zhpgv"

///|
///
/// ```c
/// lapack_int LAPACKE_chpgvd(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_float* ap, lapack_complex_float* bp, float* w, 
///     lapack_complex_float* z, lapack_int ldz
/// );
/// ```
#borrow(ap, bp, w, z)
pub extern "C" fn lapacke_chpgvd(
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
) -> Lapack_int = "LAPACKE_chpgvd"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpgvd(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, lapack_complex_double* ap, lapack_complex_double* bp, double* w, 
///     lapack_complex_double* z, lapack_int ldz
/// );
/// ```
#borrow(ap, bp, w, z)
pub extern "C" fn lapacke_zhpgvd(
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
) -> Lapack_int = "LAPACKE_zhpgvd"

///|
///
/// ```c
/// lapack_int LAPACKE_chpgvx(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, lapack_complex_float* ap, lapack_complex_float* bp, 
///     float vl, float vu, lapack_int il, lapack_int iu, 
///     float abstol, lapack_int* m, float* w, lapack_complex_float* z, 
///     lapack_int ldz, lapack_int* ifail
/// );
/// ```
#borrow(ap, bp, m, w, z, ifail)
pub extern "C" fn lapacke_chpgvx(
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
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chpgvx"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpgvx(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, lapack_complex_double* ap, lapack_complex_double* bp, 
///     double vl, double vu, lapack_int il, lapack_int iu, 
///     double abstol, lapack_int* m, double* w, lapack_complex_double* z, 
///     lapack_int ldz, lapack_int* ifail
/// );
/// ```
#borrow(ap, bp, m, w, z, ifail)
pub extern "C" fn lapacke_zhpgvx(
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
  ifail : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhpgvx"

///|
///
/// ```c
/// lapack_int LAPACKE_chprfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* ap, const lapack_complex_float* afp, const lapack_int* ipiv, const lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* ferr, float* berr
/// );
/// ```
#borrow(ap, afp, ipiv, b, x, ferr, berr)
pub extern "C" fn lapacke_chprfs(
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
) -> Lapack_int = "LAPACKE_chprfs"

///|
///
/// ```c
/// lapack_int LAPACKE_zhprfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* ap, const lapack_complex_double* afp, const lapack_int* ipiv, const lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* ferr, double* berr
/// );
/// ```
#borrow(ap, afp, ipiv, b, x, ferr, berr)
pub extern "C" fn lapacke_zhprfs(
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
) -> Lapack_int = "LAPACKE_zhprfs"

///|
///
/// ```c
/// lapack_int LAPACKE_chpsv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* ap, lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(ap, ipiv, b)
pub extern "C" fn lapacke_chpsv(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chpsv"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpsv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* ap, lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(ap, ipiv, b)
pub extern "C" fn lapacke_zhpsv(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhpsv"

///|
///
/// ```c
/// lapack_int LAPACKE_chpsvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* ap, lapack_complex_float* afp, lapack_int* ipiv, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* rcond, float* ferr, float* berr
/// );
/// ```
#borrow(ap, afp, ipiv, b, x, rcond, ferr, berr)
pub extern "C" fn lapacke_chpsvx(
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
) -> Lapack_int = "LAPACKE_chpsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_zhpsvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* ap, lapack_complex_double* afp, lapack_int* ipiv, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* rcond, double* ferr, double* berr
/// );
/// ```
#borrow(ap, afp, ipiv, b, x, rcond, ferr, berr)
pub extern "C" fn lapacke_zhpsvx(
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
) -> Lapack_int = "LAPACKE_zhpsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_chptrd(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* ap, 
///     float* d, float* e, lapack_complex_float* tau
/// );
/// ```
#borrow(ap, d, e, tau)
pub extern "C" fn lapacke_chptrd(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  d : FixedArray[Float],
  e : FixedArray[Float],
  tau : FixedArray[Float],
) -> Lapack_int = "LAPACKE_chptrd"

///|
///
/// ```c
/// lapack_int LAPACKE_zhptrd(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* ap, 
///     double* d, double* e, lapack_complex_double* tau
/// );
/// ```
#borrow(ap, d, e, tau)
pub extern "C" fn lapacke_zhptrd(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  d : FixedArray[Double],
  e : FixedArray[Double],
  tau : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zhptrd"

///|
///
/// ```c
/// lapack_int LAPACKE_chptrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* ap, lapack_int* ipiv
/// );
/// ```
#borrow(ap, ipiv)
pub extern "C" fn lapacke_chptrf(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chptrf"

///|
///
/// ```c
/// lapack_int LAPACKE_zhptrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* ap, lapack_int* ipiv
/// );
/// ```
#borrow(ap, ipiv)
pub extern "C" fn lapacke_zhptrf(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhptrf"

///|
///
/// ```c
/// lapack_int LAPACKE_chptri(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* ap, const lapack_int* ipiv
/// );
/// ```
#borrow(ap, ipiv)
pub extern "C" fn lapacke_chptri(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chptri"

///|
///
/// ```c
/// lapack_int LAPACKE_zhptri(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* ap, const lapack_int* ipiv
/// );
/// ```
#borrow(ap, ipiv)
pub extern "C" fn lapacke_zhptri(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  ipiv : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhptri"

///|
///
/// ```c
/// lapack_int LAPACKE_chptrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* ap, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(ap, ipiv, b)
pub extern "C" fn lapacke_chptrs(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Float],
  ipiv : FixedArray[Int],
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_chptrs"

///|
///
/// ```c
/// lapack_int LAPACKE_zhptrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* ap, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(ap, ipiv, b)
pub extern "C" fn lapacke_zhptrs(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  nrhs : Lapack_int,
  ap : FixedArray[Double],
  ipiv : FixedArray[Int],
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zhptrs"

///|
///
/// ```c
/// lapack_int LAPACKE_shsein(
///     int matrix_layout, char job, char eigsrc, char initv, 
///     lapack_logical* select, lapack_int n, const float* h, lapack_int ldh, 
///     float* wr, const float* wi, float* vl, lapack_int ldvl, 
///     float* vr, lapack_int ldvr, lapack_int mm, lapack_int* m, 
///     lapack_int* ifaill, lapack_int* ifailr
/// );
/// ```
#borrow(select, h, wr, wi, vl, vr, m, ifaill, ifailr)
pub extern "C" fn lapacke_shsein(
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
  ifaill : FixedArray[Int],
  ifailr : FixedArray[Int],
) -> Lapack_int = "LAPACKE_shsein"

///|
///
/// ```c
/// lapack_int LAPACKE_dhsein(
///     int matrix_layout, char job, char eigsrc, char initv, 
///     lapack_logical* select, lapack_int n, const double* h, lapack_int ldh, 
///     double* wr, const double* wi, double* vl, lapack_int ldvl, 
///     double* vr, lapack_int ldvr, lapack_int mm, lapack_int* m, 
///     lapack_int* ifaill, lapack_int* ifailr
/// );
/// ```
#borrow(select, h, wr, wi, vl, vr, m, ifaill, ifailr)
pub extern "C" fn lapacke_dhsein(
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
  ifaill : FixedArray[Int],
  ifailr : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dhsein"

///|
///
/// ```c
/// lapack_int LAPACKE_chsein(
///     int matrix_layout, char job, char eigsrc, char initv, 
///     const lapack_logical* select, lapack_int n, const lapack_complex_float* h, lapack_int ldh, 
///     lapack_complex_float* w, lapack_complex_float* vl, lapack_int ldvl, lapack_complex_float* vr, 
///     lapack_int ldvr, lapack_int mm, lapack_int* m, lapack_int* ifaill, lapack_int* ifailr
/// );
/// ```
#borrow(select, h, w, vl, vr, m, ifaill, ifailr)
pub extern "C" fn lapacke_chsein(
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
  ifaill : FixedArray[Int],
  ifailr : FixedArray[Int],
) -> Lapack_int = "LAPACKE_chsein"

///|
///
/// ```c
/// lapack_int LAPACKE_zhsein(
///     int matrix_layout, char job, char eigsrc, char initv, 
///     const lapack_logical* select, lapack_int n, const lapack_complex_double* h, lapack_int ldh, 
///     lapack_complex_double* w, lapack_complex_double* vl, lapack_int ldvl, lapack_complex_double* vr, 
///     lapack_int ldvr, lapack_int mm, lapack_int* m, lapack_int* ifaill, lapack_int* ifailr
/// );
/// ```
#borrow(select, h, w, vl, vr, m, ifaill, ifailr)
pub extern "C" fn lapacke_zhsein(
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
  ifaill : FixedArray[Int],
  ifailr : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zhsein"

///|
///
/// ```c
/// lapack_int LAPACKE_shseqr(
///     int matrix_layout, char job, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, float* h, lapack_int ldh, 
///     float* wr, float* wi, float* z, lapack_int ldz
/// );
/// ```
#borrow(h, wr, wi, z)
pub extern "C" fn lapacke_shseqr(
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
) -> Lapack_int = "LAPACKE_shseqr"

///|
///
/// ```c
/// lapack_int LAPACKE_dhseqr(
///     int matrix_layout, char job, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, double* h, lapack_int ldh, 
///     double* wr, double* wi, double* z, lapack_int ldz
/// );
/// ```
#borrow(h, wr, wi, z)
pub extern "C" fn lapacke_dhseqr(
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
) -> Lapack_int = "LAPACKE_dhseqr"

///|
///
/// ```c
/// lapack_int LAPACKE_chseqr(
///     int matrix_layout, char job, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, lapack_complex_float* h, lapack_int ldh, 
///     lapack_complex_float* w, lapack_complex_float* z, lapack_int ldz
/// );
/// ```
#borrow(h, w, z)
pub extern "C" fn lapacke_chseqr(
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
) -> Lapack_int = "LAPACKE_chseqr"

///|
///
/// ```c
/// lapack_int LAPACKE_zhseqr(
///     int matrix_layout, char job, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, lapack_complex_double* h, lapack_int ldh, 
///     lapack_complex_double* w, lapack_complex_double* z, lapack_int ldz
/// );
/// ```
#borrow(h, w, z)
pub extern "C" fn lapacke_zhseqr(
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
) -> Lapack_int = "LAPACKE_zhseqr"

///|
///
/// ```c
/// lapack_int LAPACKE_clacgv(lapack_int n, lapack_complex_float* x, lapack_int incx);
/// ```
#borrow(x)
pub extern "C" fn lapacke_clacgv(
  n : Lapack_int,
  x : FixedArray[Float],
  incx : Lapack_int,
) -> Lapack_int = "LAPACKE_clacgv"

///|
///
/// ```c
/// lapack_int LAPACKE_zlacgv(lapack_int n, lapack_complex_double* x, lapack_int incx);
/// ```
#borrow(x)
pub extern "C" fn lapacke_zlacgv(
  n : Lapack_int,
  x : FixedArray[Double],
  incx : Lapack_int,
) -> Lapack_int = "LAPACKE_zlacgv"

///|
///
/// ```c
/// lapack_int LAPACKE_slacn2(
///     lapack_int n, float* v, float* x, lapack_int* isgn, 
///     float* est, lapack_int* kase, lapack_int* isave
/// );
/// ```
#borrow(v, x, isgn, est, kase, isave)
pub extern "C" fn lapacke_slacn2(
  n : Lapack_int,
  v : FixedArray[Float],
  x : FixedArray[Float],
  isgn : FixedArray[Int],
  est : FixedArray[Float],
  kase : FixedArray[Int],
  isave : FixedArray[Int],
) -> Lapack_int = "LAPACKE_slacn2"

///|
///
/// ```c
/// lapack_int LAPACKE_dlacn2(
///     lapack_int n, double* v, double* x, lapack_int* isgn, 
///     double* est, lapack_int* kase, lapack_int* isave
/// );
/// ```
#borrow(v, x, isgn, est, kase, isave)
pub extern "C" fn lapacke_dlacn2(
  n : Lapack_int,
  v : FixedArray[Double],
  x : FixedArray[Double],
  isgn : FixedArray[Int],
  est : FixedArray[Double],
  kase : FixedArray[Int],
  isave : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dlacn2"

///|
///
/// ```c
/// lapack_int LAPACKE_clacn2(
///     lapack_int n, lapack_complex_float* v, lapack_complex_float* x, float* est, 
///     lapack_int* kase, lapack_int* isave
/// );
/// ```
#borrow(v, x, est, kase, isave)
pub extern "C" fn lapacke_clacn2(
  n : Lapack_int,
  v : FixedArray[Float],
  x : FixedArray[Float],
  est : FixedArray[Float],
  kase : FixedArray[Int],
  isave : FixedArray[Int],
) -> Lapack_int = "LAPACKE_clacn2"

///|
///
/// ```c
/// lapack_int LAPACKE_zlacn2(
///     lapack_int n, lapack_complex_double* v, lapack_complex_double* x, double* est, 
///     lapack_int* kase, lapack_int* isave
/// );
/// ```
#borrow(v, x, est, kase, isave)
pub extern "C" fn lapacke_zlacn2(
  n : Lapack_int,
  v : FixedArray[Double],
  x : FixedArray[Double],
  est : FixedArray[Double],
  kase : FixedArray[Int],
  isave : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zlacn2"

///|
///
/// ```c
/// lapack_int LAPACKE_slacpy(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     const float* a, lapack_int lda, float* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_slacpy(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_slacpy"

///|
///
/// ```c
/// lapack_int LAPACKE_dlacpy(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     const double* a, lapack_int lda, double* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_dlacpy(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_dlacpy"

///|
///
/// ```c
/// lapack_int LAPACKE_clacpy(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     const lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_clacpy(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_clacpy"

///|
///
/// ```c
/// lapack_int LAPACKE_zlacpy(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     const lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_zlacpy(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zlacpy"

///|
///
/// ```c
/// lapack_int LAPACKE_clacp2(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     const float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_clacp2(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_clacp2"

///|
///
/// ```c
/// lapack_int LAPACKE_zlacp2(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     const double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
#borrow(a, b)
pub extern "C" fn lapacke_zlacp2(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
) -> Lapack_int = "LAPACKE_zlacp2"

///|
///
/// ```c
/// lapack_int LAPACKE_zlag2c(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_float* sa, lapack_int ldsa
/// );
/// ```
#borrow(a, sa)
pub extern "C" fn lapacke_zlag2c(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  sa : FixedArray[Float],
  ldsa : Lapack_int,
) -> Lapack_int = "LAPACKE_zlag2c"

///|
///
/// ```c
/// lapack_int LAPACKE_slag2d(
///     int matrix_layout, lapack_int m, lapack_int n, const float* sa, 
///     lapack_int ldsa, double* a, lapack_int lda
/// );
/// ```
#borrow(sa, a)
pub extern "C" fn lapacke_slag2d(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  sa : FixedArray[Float],
  ldsa : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_slag2d"

///|
///
/// ```c
/// lapack_int LAPACKE_dlag2s(
///     int matrix_layout, lapack_int m, lapack_int n, const double* a, 
///     lapack_int lda, float* sa, lapack_int ldsa
/// );
/// ```
#borrow(a, sa)
pub extern "C" fn lapacke_dlag2s(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  sa : FixedArray[Float],
  ldsa : Lapack_int,
) -> Lapack_int = "LAPACKE_dlag2s"

///|
///
/// ```c
/// lapack_int LAPACKE_clag2z(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_float* sa, 
///     lapack_int ldsa, lapack_complex_double* a, lapack_int lda
/// );
/// ```
#borrow(sa, a)
pub extern "C" fn lapacke_clag2z(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  sa : FixedArray[Float],
  ldsa : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_clag2z"

///|
///
/// ```c
/// lapack_int LAPACKE_slagge(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const float* d, float* a, lapack_int lda, lapack_int* iseed
/// );
/// ```
#borrow(d, a, iseed)
pub extern "C" fn lapacke_slagge(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  d : FixedArray[Float],
  a : FixedArray[Float],
  lda : Lapack_int,
  iseed : FixedArray[Int],
) -> Lapack_int = "LAPACKE_slagge"

///|
///
/// ```c
/// lapack_int LAPACKE_dlagge(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const double* d, double* a, lapack_int lda, lapack_int* iseed
/// );
/// ```
#borrow(d, a, iseed)
pub extern "C" fn lapacke_dlagge(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  d : FixedArray[Double],
  a : FixedArray[Double],
  lda : Lapack_int,
  iseed : FixedArray[Int],
) -> Lapack_int = "LAPACKE_dlagge"

///|
///
/// ```c
/// lapack_int LAPACKE_clagge(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const float* d, lapack_complex_float* a, lapack_int lda, lapack_int* iseed
/// );
/// ```
#borrow(d, a, iseed)
pub extern "C" fn lapacke_clagge(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  d : FixedArray[Float],
  a : FixedArray[Float],
  lda : Lapack_int,
  iseed : FixedArray[Int],
) -> Lapack_int = "LAPACKE_clagge"

///|
///
/// ```c
/// lapack_int LAPACKE_zlagge(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const double* d, lapack_complex_double* a, lapack_int lda, lapack_int* iseed
/// );
/// ```
#borrow(d, a, iseed)
pub extern "C" fn lapacke_zlagge(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  d : FixedArray[Double],
  a : FixedArray[Double],
  lda : Lapack_int,
  iseed : FixedArray[Int],
) -> Lapack_int = "LAPACKE_zlagge"

///|
///
/// ```c
/// float LAPACKE_slamch(char cmach);
/// ```
pub extern "C" fn lapacke_slamch(cmach : Byte) -> Float = "LAPACKE_slamch"

///|
///
/// ```c
/// double LAPACKE_dlamch(char cmach);
/// ```
pub extern "C" fn lapacke_dlamch(cmach : Byte) -> Double = "LAPACKE_dlamch"

///|
///
/// ```c
/// float LAPACKE_slangb(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const float* ab, lapack_int ldab
/// );
/// ```
#borrow(ab)
pub extern "C" fn lapacke_slangb(
  matrix_layout : Int,
  norm : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
) -> Float = "LAPACKE_slangb"

///|
///
/// ```c
/// double LAPACKE_dlangb(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const double* ab, lapack_int ldab
/// );
/// ```
#borrow(ab)
pub extern "C" fn lapacke_dlangb(
  matrix_layout : Int,
  norm : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
) -> Double = "LAPACKE_dlangb"

///|
///
/// ```c
/// float LAPACKE_clangb(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_float* ab, lapack_int ldab
/// );
/// ```
#borrow(ab)
pub extern "C" fn lapacke_clangb(
  matrix_layout : Int,
  norm : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  ab : FixedArray[Float],
  ldab : Lapack_int,
) -> Float = "LAPACKE_clangb"

///|
///
/// ```c
/// double LAPACKE_zlangb(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_double* ab, lapack_int ldab
/// );
/// ```
#borrow(ab)
pub extern "C" fn lapacke_zlangb(
  matrix_layout : Int,
  norm : Byte,
  n : Lapack_int,
  kl : Lapack_int,
  ku : Lapack_int,
  ab : FixedArray[Double],
  ldab : Lapack_int,
) -> Double = "LAPACKE_zlangb"

///|
///
/// ```c
/// float LAPACKE_slange(
///     int matrix_layout, char norm, lapack_int m, lapack_int n, 
///     const float* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_slange(
  matrix_layout : Int,
  norm : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Float = "LAPACKE_slange"

///|
///
/// ```c
/// double LAPACKE_dlange(
///     int matrix_layout, char norm, lapack_int m, lapack_int n, 
///     const double* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_dlange(
  matrix_layout : Int,
  norm : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Double = "LAPACKE_dlange"

///|
///
/// ```c
/// float LAPACKE_clange(
///     int matrix_layout, char norm, lapack_int m, lapack_int n, 
///     const lapack_complex_float* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_clange(
  matrix_layout : Int,
  norm : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Float = "LAPACKE_clange"

///|
///
/// ```c
/// double LAPACKE_zlange(
///     int matrix_layout, char norm, lapack_int m, lapack_int n, 
///     const lapack_complex_double* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_zlange(
  matrix_layout : Int,
  norm : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Double = "LAPACKE_zlange"

///|
///
/// ```c
/// float LAPACKE_clanhe(
///     int matrix_layout, char norm, char uplo, lapack_int n, 
///     const lapack_complex_float* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_clanhe(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Float = "LAPACKE_clanhe"

///|
///
/// ```c
/// double LAPACKE_zlanhe(
///     int matrix_layout, char norm, char uplo, lapack_int n, 
///     const lapack_complex_double* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_zlanhe(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Double = "LAPACKE_zlanhe"

///|
///
/// ```c
/// lapack_int LAPACKE_clacrm(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, const float* b, lapack_int ldb, lapack_complex_float* c, lapack_int ldc
/// );
/// ```
#borrow(a, b, c)
pub extern "C" fn lapacke_clacrm(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
) -> Lapack_int = "LAPACKE_clacrm"

///|
///
/// ```c
/// lapack_int LAPACKE_zlacrm(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, const double* b, lapack_int ldb, lapack_complex_double* c, lapack_int ldc
/// );
/// ```
#borrow(a, b, c)
pub extern "C" fn lapacke_zlacrm(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
) -> Lapack_int = "LAPACKE_zlacrm"

///|
///
/// ```c
/// lapack_int LAPACKE_clarcm(
///     int matrix_layout, lapack_int m, lapack_int n, const float* a, 
///     lapack_int lda, const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* c, lapack_int ldc
/// );
/// ```
#borrow(a, b, c)
pub extern "C" fn lapacke_clarcm(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  b : FixedArray[Float],
  ldb : Lapack_int,
  c : FixedArray[Float],
  ldc : Lapack_int,
) -> Lapack_int = "LAPACKE_clarcm"

///|
///
/// ```c
/// lapack_int LAPACKE_zlarcm(
///     int matrix_layout, lapack_int m, lapack_int n, const double* a, 
///     lapack_int lda, const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* c, lapack_int ldc
/// );
/// ```
#borrow(a, b, c)
pub extern "C" fn lapacke_zlarcm(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  b : FixedArray[Double],
  ldb : Lapack_int,
  c : FixedArray[Double],
  ldc : Lapack_int,
) -> Lapack_int = "LAPACKE_zlarcm"

///|
///
/// ```c
/// float LAPACKE_slansy(
///     int matrix_layout, char norm, char uplo, lapack_int n, 
///     const float* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_slansy(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Float = "LAPACKE_slansy"

///|
///
/// ```c
/// double LAPACKE_dlansy(
///     int matrix_layout, char norm, char uplo, lapack_int n, 
///     const double* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_dlansy(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Double = "LAPACKE_dlansy"

///|
///
/// ```c
/// float LAPACKE_clansy(
///     int matrix_layout, char norm, char uplo, lapack_int n, 
///     const lapack_complex_float* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_clansy(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Float = "LAPACKE_clansy"

///|
///
/// ```c
/// double LAPACKE_zlansy(
///     int matrix_layout, char norm, char uplo, lapack_int n, 
///     const lapack_complex_double* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_zlansy(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Double = "LAPACKE_zlansy"

///|
///
/// ```c
/// float LAPACKE_slantr(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int m, lapack_int n, const float* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_slantr(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Float = "LAPACKE_slantr"

///|
///
/// ```c
/// double LAPACKE_dlantr(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int m, lapack_int n, const double* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_dlantr(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Double = "LAPACKE_dlantr"

///|
///
/// ```c
/// float LAPACKE_clantr(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int m, lapack_int n, const lapack_complex_float* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_clantr(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Float = "LAPACKE_clantr"

///|
///
/// ```c
/// double LAPACKE_zlantr(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int m, lapack_int n, const lapack_complex_double* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_zlantr(
  matrix_layout : Int,
  norm : Byte,
  uplo : Byte,
  diag : Byte,
  m : Lapack_int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Double = "LAPACKE_zlantr"

///|
///
/// ```c
/// lapack_int LAPACKE_slarfb(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     const float* v, lapack_int ldv, const float* t, lapack_int ldt, 
///     float* c, lapack_int ldc
/// );
/// ```
#borrow(v, t, c)
pub extern "C" fn lapacke_slarfb(
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
) -> Lapack_int = "LAPACKE_slarfb"

///|
///
/// ```c
/// lapack_int LAPACKE_dlarfb(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     const double* v, lapack_int ldv, const double* t, lapack_int ldt, 
///     double* c, lapack_int ldc
/// );
/// ```
#borrow(v, t, c)
pub extern "C" fn lapacke_dlarfb(
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
) -> Lapack_int = "LAPACKE_dlarfb"

///|
///
/// ```c
/// lapack_int LAPACKE_clarfb(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     const lapack_complex_float* v, lapack_int ldv, const lapack_complex_float* t, lapack_int ldt, 
///     lapack_complex_float* c, lapack_int ldc
/// );
/// ```
#borrow(v, t, c)
pub extern "C" fn lapacke_clarfb(
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
) -> Lapack_int = "LAPACKE_clarfb"

///|
///
/// ```c
/// lapack_int LAPACKE_zlarfb(
///     int matrix_layout, char side, char trans, char direct, 
///     char storev, lapack_int m, lapack_int n, lapack_int k, 
///     const lapack_complex_double* v, lapack_int ldv, const lapack_complex_double* t, lapack_int ldt, 
///     lapack_complex_double* c, lapack_int ldc
/// );
/// ```
#borrow(v, t, c)
pub extern "C" fn lapacke_zlarfb(
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
) -> Lapack_int = "LAPACKE_zlarfb"

///|
///
/// ```c
/// lapack_int LAPACKE_slarfg(
///     lapack_int n, float* alpha, float* x, lapack_int incx, float* tau
/// );
/// ```
#borrow(alpha, x, tau)
pub extern "C" fn lapacke_slarfg(
  n : Lapack_int,
  alpha : FixedArray[Float],
  x : FixedArray[Float],
  incx : Lapack_int,
  tau : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slarfg"

///|
///
/// ```c
/// lapack_int LAPACKE_dlarfg(
///     lapack_int n, double* alpha, double* x, lapack_int incx, double* tau
/// );
/// ```
#borrow(alpha, x, tau)
pub extern "C" fn lapacke_dlarfg(
  n : Lapack_int,
  alpha : FixedArray[Double],
  x : FixedArray[Double],
  incx : Lapack_int,
  tau : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlarfg"

///|
///
/// ```c
/// lapack_int LAPACKE_clarfg(
///     lapack_int n, lapack_complex_float* alpha, lapack_complex_float* x, lapack_int incx, lapack_complex_float* tau
/// );
/// ```
#borrow(alpha, x, tau)
pub extern "C" fn lapacke_clarfg(
  n : Lapack_int,
  alpha : FixedArray[Float],
  x : FixedArray[Float],
  incx : Lapack_int,
  tau : FixedArray[Float],
) -> Lapack_int = "LAPACKE_clarfg"

///|
///
/// ```c
/// lapack_int LAPACKE_zlarfg(
///     lapack_int n, lapack_complex_double* alpha, lapack_complex_double* x, lapack_int incx, lapack_complex_double* tau
/// );
/// ```
#borrow(alpha, x, tau)
pub extern "C" fn lapacke_zlarfg(
  n : Lapack_int,
  alpha : FixedArray[Double],
  x : FixedArray[Double],
  incx : Lapack_int,
  tau : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zlarfg"

///|
///
/// ```c
/// lapack_int LAPACKE_slarft(
///     int matrix_layout, char direct, char storev, lapack_int n, 
///     lapack_int k, const float* v, lapack_int ldv, const float* tau, 
///     float* t, lapack_int ldt
/// );
/// ```
#borrow(v, tau, t)
pub extern "C" fn lapacke_slarft(
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
) -> Lapack_int = "LAPACKE_slarft"

///|
///
/// ```c
/// lapack_int LAPACKE_dlarft(
///     int matrix_layout, char direct, char storev, lapack_int n, 
///     lapack_int k, const double* v, lapack_int ldv, const double* tau, 
///     double* t, lapack_int ldt
/// );
/// ```
#borrow(v, tau, t)
pub extern "C" fn lapacke_dlarft(
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
) -> Lapack_int = "LAPACKE_dlarft"

///|
///
/// ```c
/// lapack_int LAPACKE_clarft(
///     int matrix_layout, char direct, char storev, lapack_int n, 
///     lapack_int k, const lapack_complex_float* v, lapack_int ldv, const lapack_complex_float* tau, 
///     lapack_complex_float* t, lapack_int ldt
/// );
/// ```
#borrow(v, tau, t)
pub extern "C" fn lapacke_clarft(
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
) -> Lapack_int = "LAPACKE_clarft"

///|
///
/// ```c
/// lapack_int LAPACKE_zlarft(
///     int matrix_layout, char direct, char storev, lapack_int n, 
///     lapack_int k, const lapack_complex_double* v, lapack_int ldv, const lapack_complex_double* tau, 
///     lapack_complex_double* t, lapack_int ldt
/// );
/// ```
#borrow(v, tau, t)
pub extern "C" fn lapacke_zlarft(
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
) -> Lapack_int = "LAPACKE_zlarft"

///|
///
/// ```c
/// lapack_int LAPACKE_slarfx(
///     int matrix_layout, char side, lapack_int m, lapack_int n, 
///     const float* v, float tau, float* c, lapack_int ldc, float* work
/// );
/// ```
#borrow(v, c, work)
pub extern "C" fn lapacke_slarfx(
  matrix_layout : Int,
  side : Byte,
  m : Lapack_int,
  n : Lapack_int,
  v : FixedArray[Float],
  tau : Float,
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slarfx"

///|
///
/// ```c
/// lapack_int LAPACKE_dlarfx(
///     int matrix_layout, char side, lapack_int m, lapack_int n, 
///     const double* v, double tau, double* c, lapack_int ldc, double* work
/// );
/// ```
#borrow(v, c, work)
pub extern "C" fn lapacke_dlarfx(
  matrix_layout : Int,
  side : Byte,
  m : Lapack_int,
  n : Lapack_int,
  v : FixedArray[Double],
  tau : Double,
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlarfx"

///|
///
/// ```c
/// lapack_int LAPACKE_clarfx(
///     int matrix_layout, char side, lapack_int m, lapack_int n, 
///     const lapack_complex_float* v, lapack_complex_float tau, lapack_complex_float* c, lapack_int ldc, lapack_complex_float* work
/// );
/// ```
#borrow(v, c, work)
pub extern "C" fn lapacke_clarfx(
  matrix_layout : Int,
  side : Byte,
  m : Lapack_int,
  n : Lapack_int,
  v : FixedArray[Float],
  tau : Lapack_complex_float,
  c : FixedArray[Float],
  ldc : Lapack_int,
  work : FixedArray[Float],
) -> Lapack_int = "LAPACKE_clarfx"

///|
///
/// ```c
/// lapack_int LAPACKE_zlarfx(
///     int matrix_layout, char side, lapack_int m, lapack_int n, 
///     const lapack_complex_double* v, lapack_complex_double tau, lapack_complex_double* c, lapack_int ldc, lapack_complex_double* work
/// );
/// ```
#borrow(v, c, work)
pub extern "C" fn lapacke_zlarfx(
  matrix_layout : Int,
  side : Byte,
  m : Lapack_int,
  n : Lapack_int,
  v : FixedArray[Double],
  tau : Lapack_complex_double,
  c : FixedArray[Double],
  ldc : Lapack_int,
  work : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zlarfx"

///|
///
/// ```c
/// lapack_int LAPACKE_slarnv(
///     lapack_int idist, lapack_int* iseed, lapack_int n, float* x
/// );
/// ```
#borrow(iseed, x)
pub extern "C" fn lapacke_slarnv(
  idist : Lapack_int,
  iseed : FixedArray[Int],
  n : Lapack_int,
  x : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slarnv"

///|
///
/// ```c
/// lapack_int LAPACKE_dlarnv(
///     lapack_int idist, lapack_int* iseed, lapack_int n, double* x
/// );
/// ```
#borrow(iseed, x)
pub extern "C" fn lapacke_dlarnv(
  idist : Lapack_int,
  iseed : FixedArray[Int],
  n : Lapack_int,
  x : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlarnv"

///|
///
/// ```c
/// lapack_int LAPACKE_clarnv(
///     lapack_int idist, lapack_int* iseed, lapack_int n, lapack_complex_float* x
/// );
/// ```
#borrow(iseed, x)
pub extern "C" fn lapacke_clarnv(
  idist : Lapack_int,
  iseed : FixedArray[Int],
  n : Lapack_int,
  x : FixedArray[Float],
) -> Lapack_int = "LAPACKE_clarnv"

///|
///
/// ```c
/// lapack_int LAPACKE_zlarnv(
///     lapack_int idist, lapack_int* iseed, lapack_int n, lapack_complex_double* x
/// );
/// ```
#borrow(iseed, x)
pub extern "C" fn lapacke_zlarnv(
  idist : Lapack_int,
  iseed : FixedArray[Int],
  n : Lapack_int,
  x : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zlarnv"

///|
///
/// ```c
/// lapack_int LAPACKE_slascl(
///     int matrix_layout, char type, lapack_int kl, lapack_int ku, 
///     float cfrom, float cto, lapack_int m, lapack_int n, 
///     float* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_slascl(
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
) -> Lapack_int = "LAPACKE_slascl"

///|
///
/// ```c
/// lapack_int LAPACKE_dlascl(
///     int matrix_layout, char type, lapack_int kl, lapack_int ku, 
///     double cfrom, double cto, lapack_int m, lapack_int n, 
///     double* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_dlascl(
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
) -> Lapack_int = "LAPACKE_dlascl"

///|
///
/// ```c
/// lapack_int LAPACKE_clascl(
///     int matrix_layout, char type, lapack_int kl, lapack_int ku, 
///     float cfrom, float cto, lapack_int m, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_clascl(
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
) -> Lapack_int = "LAPACKE_clascl"

///|
///
/// ```c
/// lapack_int LAPACKE_zlascl(
///     int matrix_layout, char type, lapack_int kl, lapack_int ku, 
///     double cfrom, double cto, lapack_int m, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_zlascl(
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
) -> Lapack_int = "LAPACKE_zlascl"

///|
///
/// ```c
/// lapack_int LAPACKE_slaset(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     float alpha, float beta, float* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_slaset(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  alpha : Float,
  beta : Float,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_slaset"

///|
///
/// ```c
/// lapack_int LAPACKE_dlaset(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     double alpha, double beta, double* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_dlaset(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  alpha : Double,
  beta : Double,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_dlaset"

///|
///
/// ```c
/// lapack_int LAPACKE_claset(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     lapack_complex_float alpha, lapack_complex_float beta, lapack_complex_float* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_claset(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  alpha : Lapack_complex_float,
  beta : Lapack_complex_float,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_claset"

///|
///
/// ```c
/// lapack_int LAPACKE_zlaset(
///     int matrix_layout, char uplo, lapack_int m, lapack_int n, 
///     lapack_complex_double alpha, lapack_complex_double beta, lapack_complex_double* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_zlaset(
  matrix_layout : Int,
  uplo : Byte,
  m : Lapack_int,
  n : Lapack_int,
  alpha : Lapack_complex_double,
  beta : Lapack_complex_double,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_zlaset"

///|
///
/// ```c
/// lapack_int LAPACKE_slasrt(char id, lapack_int n, float* d);
/// ```
#borrow(d)
pub extern "C" fn lapacke_slasrt(
  id : Byte,
  n : Lapack_int,
  d : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slasrt"

///|
///
/// ```c
/// lapack_int LAPACKE_dlasrt(char id, lapack_int n, double* d);
/// ```
#borrow(d)
pub extern "C" fn lapacke_dlasrt(
  id : Byte,
  n : Lapack_int,
  d : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlasrt"

///|
///
/// ```c
/// lapack_int LAPACKE_slassq(
///     lapack_int n, float* x, lapack_int incx, float* scale, float* sumsq
/// );
/// ```
#borrow(x, scale, sumsq)
pub extern "C" fn lapacke_slassq(
  n : Lapack_int,
  x : FixedArray[Float],
  incx : Lapack_int,
  scale : FixedArray[Float],
  sumsq : FixedArray[Float],
) -> Lapack_int = "LAPACKE_slassq"

///|
///
/// ```c
/// lapack_int LAPACKE_dlassq(
///     lapack_int n, double* x, lapack_int incx, double* scale, double* sumsq
/// );
/// ```
#borrow(x, scale, sumsq)
pub extern "C" fn lapacke_dlassq(
  n : Lapack_int,
  x : FixedArray[Double],
  incx : Lapack_int,
  scale : FixedArray[Double],
  sumsq : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dlassq"

///|
///
/// ```c
/// lapack_int LAPACKE_classq(
///     lapack_int n, lapack_complex_float* x, lapack_int incx, float* scale, float* sumsq
/// );
/// ```
#borrow(x, scale, sumsq)
pub extern "C" fn lapacke_classq(
  n : Lapack_int,
  x : FixedArray[Float],
  incx : Lapack_int,
  scale : FixedArray[Float],
  sumsq : FixedArray[Float],
) -> Lapack_int = "LAPACKE_classq"

///|
///
/// ```c
/// lapack_int LAPACKE_zlassq(
///     lapack_int n, lapack_complex_double* x, lapack_int incx, double* scale, double* sumsq
/// );
/// ```
#borrow(x, scale, sumsq)
pub extern "C" fn lapacke_zlassq(
  n : Lapack_int,
  x : FixedArray[Double],
  incx : Lapack_int,
  scale : FixedArray[Double],
  sumsq : FixedArray[Double],
) -> Lapack_int = "LAPACKE_zlassq"

///|
///
/// ```c
/// lapack_int LAPACKE_slaswp(
///     int matrix_layout, lapack_int n, float* a, lapack_int lda, 
///     lapack_int k1, lapack_int k2, const lapack_int* ipiv, lapack_int incx
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_slaswp(
  matrix_layout : Int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  k1 : Lapack_int,
  k2 : Lapack_int,
  ipiv : FixedArray[Int],
  incx : Lapack_int,
) -> Lapack_int = "LAPACKE_slaswp"

///|
///
/// ```c
/// lapack_int LAPACKE_dlaswp(
///     int matrix_layout, lapack_int n, double* a, lapack_int lda, 
///     lapack_int k1, lapack_int k2, const lapack_int* ipiv, lapack_int incx
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_dlaswp(
  matrix_layout : Int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  k1 : Lapack_int,
  k2 : Lapack_int,
  ipiv : FixedArray[Int],
  incx : Lapack_int,
) -> Lapack_int = "LAPACKE_dlaswp"

///|
///
/// ```c
/// lapack_int LAPACKE_claswp(
///     int matrix_layout, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_int k1, lapack_int k2, const lapack_int* ipiv, lapack_int incx
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_claswp(
  matrix_layout : Int,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  k1 : Lapack_int,
  k2 : Lapack_int,
  ipiv : FixedArray[Int],
  incx : Lapack_int,
) -> Lapack_int = "LAPACKE_claswp"

///|
///
/// ```c
/// lapack_int LAPACKE_zlaswp(
///     int matrix_layout, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_int k1, lapack_int k2, const lapack_int* ipiv, lapack_int incx
/// );
/// ```
#borrow(a, ipiv)
pub extern "C" fn lapacke_zlaswp(
  matrix_layout : Int,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  k1 : Lapack_int,
  k2 : Lapack_int,
  ipiv : FixedArray[Int],
  incx : Lapack_int,
) -> Lapack_int = "LAPACKE_zlaswp"

///|
///
/// ```c
/// lapack_int LAPACKE_slatms(
///     int matrix_layout, lapack_int m, lapack_int n, char dist, 
///     lapack_int* iseed, char sym, float* d, lapack_int mode, 
///     float cond, float dmax, lapack_int kl, lapack_int ku, 
///     char pack, float* a, lapack_int lda
/// );
/// ```
#borrow(iseed, d, a)
pub extern "C" fn lapacke_slatms(
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
) -> Lapack_int = "LAPACKE_slatms"

///|
///
/// ```c
/// lapack_int LAPACKE_dlatms(
///     int matrix_layout, lapack_int m, lapack_int n, char dist, 
///     lapack_int* iseed, char sym, double* d, lapack_int mode, 
///     double cond, double dmax, lapack_int kl, lapack_int ku, 
///     char pack, double* a, lapack_int lda
/// );
/// ```
#borrow(iseed, d, a)
pub extern "C" fn lapacke_dlatms(
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
) -> Lapack_int = "LAPACKE_dlatms"

///|
///
/// ```c
/// lapack_int LAPACKE_clatms(
///     int matrix_layout, lapack_int m, lapack_int n, char dist, 
///     lapack_int* iseed, char sym, float* d, lapack_int mode, 
///     float cond, float dmax, lapack_int kl, lapack_int ku, 
///     char pack, lapack_complex_float* a, lapack_int lda
/// );
/// ```
#borrow(iseed, d, a)
pub extern "C" fn lapacke_clatms(
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
) -> Lapack_int = "LAPACKE_clatms"

///|
///
/// ```c
/// lapack_int LAPACKE_zlatms(
///     int matrix_layout, lapack_int m, lapack_int n, char dist, 
///     lapack_int* iseed, char sym, double* d, lapack_int mode, 
///     double cond, double dmax, lapack_int kl, lapack_int ku, 
///     char pack, lapack_complex_double* a, lapack_int lda
/// );
/// ```
#borrow(iseed, d, a)
pub extern "C" fn lapacke_zlatms(
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
) -> Lapack_int = "LAPACKE_zlatms"

///|
///
/// ```c
/// lapack_int LAPACKE_slauum(
///     int matrix_layout, char uplo, lapack_int n, float* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_slauum(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_slauum"

///|
///
/// ```c
/// lapack_int LAPACKE_dlauum(
///     int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_dlauum(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_dlauum"

///|
///
/// ```c
/// lapack_int LAPACKE_clauum(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_clauum(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_clauum"

///|
///
/// ```c
/// lapack_int LAPACKE_zlauum(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda
/// );
/// ```
#borrow(a)
pub extern "C" fn lapacke_zlauum(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
) -> Lapack_int = "LAPACKE_zlauum"

///|
///
/// ```c
/// lapack_int LAPACKE_sopgtr(
///     int matrix_layout, char uplo, lapack_int n, const float* ap, 
///     const float* tau, float* q, lapack_int ldq
/// );
/// ```
#borrow(ap, tau, q)
pub extern "C" fn lapacke_sopgtr(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Float],
  tau : FixedArray[Float],
  q : FixedArray[Float],
  ldq : Lapack_int,
) -> Lapack_int = "LAPACKE_sopgtr"

///|
///
/// ```c
/// lapack_int LAPACKE_dopgtr(
///     int matrix_layout, char uplo, lapack_int n, const double* ap, 
///     const double* tau, double* q, lapack_int ldq
/// );
/// ```
#borrow(ap, tau, q)
pub extern "C" fn lapacke_dopgtr(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  ap : FixedArray[Double],
  tau : FixedArray[Double],
  q : FixedArray[Double],
  ldq : Lapack_int,
) -> Lapack_int = "LAPACKE_dopgtr"

///|
///
/// ```c
/// lapack_int LAPACKE_sopmtr(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const float* ap, const float* tau, 
///     float* c, lapack_int ldc
/// );
/// ```
#borrow(ap, tau, c)
pub extern "C" fn lapacke_sopmtr(
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
) -> Lapack_int = "LAPACKE_sopmtr"

///|
///
/// ```c
/// lapack_int LAPACKE_dopmtr(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const double* ap, const double* tau, 
///     double* c, lapack_int ldc
/// );
/// ```
#borrow(ap, tau, c)
pub extern "C" fn lapacke_dopmtr(
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
) -> Lapack_int = "LAPACKE_dopmtr"

///|
///
/// ```c
/// lapack_int LAPACKE_sorgbr(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int k, float* a, lapack_int lda, const float* tau
/// );
/// ```
#borrow(a, tau)
pub extern "C" fn lapacke_sorgbr(
  matrix_layout : Int,
  vect : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sorgbr"

///|
///
/// ```c
/// lapack_int LAPACKE_dorgbr(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int k, double* a, lapack_int lda, const double* tau
/// );
/// ```
#borrow(a, tau)
pub extern "C" fn lapacke_dorgbr(
  matrix_layout : Int,
  vect : Byte,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dorgbr"

///|
///
/// ```c
/// lapack_int LAPACKE_sorghr(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     float* a, lapack_int lda, const float* tau
/// );
/// ```
#borrow(a, tau)
pub extern "C" fn lapacke_sorghr(
  matrix_layout : Int,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sorghr"

///|
///
/// ```c
/// lapack_int LAPACKE_dorghr(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     double* a, lapack_int lda, const double* tau
/// );
/// ```
#borrow(a, tau)
pub extern "C" fn lapacke_dorghr(
  matrix_layout : Int,
  n : Lapack_int,
  ilo : Lapack_int,
  ihi : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dorghr"

///|
///
/// ```c
/// lapack_int LAPACKE_sorglq(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     float* a, lapack_int lda, const float* tau
/// );
/// ```
#borrow(a, tau)
pub extern "C" fn lapacke_sorglq(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sorglq"

///|
///
/// ```c
/// lapack_int LAPACKE_dorglq(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     double* a, lapack_int lda, const double* tau
/// );
/// ```
#borrow(a, tau)
pub extern "C" fn lapacke_dorglq(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dorglq"

///|
///
/// ```c
/// lapack_int LAPACKE_sorgql(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     float* a, lapack_int lda, const float* tau
/// );
/// ```
#borrow(a, tau)
pub extern "C" fn lapacke_sorgql(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sorgql"

///|
///
/// ```c
/// lapack_int LAPACKE_dorgql(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     double* a, lapack_int lda, const double* tau
/// );
/// ```
#borrow(a, tau)
pub extern "C" fn lapacke_dorgql(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dorgql"

///|
///
/// ```c
/// lapack_int LAPACKE_sorgqr(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     float* a, lapack_int lda, const float* tau
/// );
/// ```
#borrow(a, tau)
pub extern "C" fn lapacke_sorgqr(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sorgqr"

///|
///
/// ```c
/// lapack_int LAPACKE_dorgqr(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     double* a, lapack_int lda, const double* tau
/// );
/// ```
#borrow(a, tau)
pub extern "C" fn lapacke_dorgqr(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dorgqr"

///|
///
/// ```c
/// lapack_int LAPACKE_sorgrq(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     float* a, lapack_int lda, const float* tau
/// );
/// ```
#borrow(a, tau)
pub extern "C" fn lapacke_sorgrq(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sorgrq"

///|
///
/// ```c
/// lapack_int LAPACKE_dorgrq(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     double* a, lapack_int lda, const double* tau
/// );
/// ```
#borrow(a, tau)
pub extern "C" fn lapacke_dorgrq(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  k : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dorgrq"

///|
///
/// ```c
/// lapack_int LAPACKE_sorgtr(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, const float* tau
/// );
/// ```
#borrow(a, tau)
pub extern "C" fn lapacke_sorgtr(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  tau : FixedArray[Float],
) -> Lapack_int = "LAPACKE_sorgtr"

///|
///
/// ```c
/// lapack_int LAPACKE_dorgtr(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, const double* tau
/// );
/// ```
#borrow(a, tau)
pub extern "C" fn lapacke_dorgtr(
  matrix_layout : Int,
  uplo : Byte,
  n : Lapack_int,
  a : FixedArray[Double],
  lda : Lapack_int,
  tau : FixedArray[Double],
) -> Lapack_int = "LAPACKE_dorgtr"

///|
///
/// ```c
/// lapack_int LAPACKE_sorgtsqr_row(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int mb, 
///     lapack_int nb, float* a, lapack_int lda, const float* t, lapack_int ldt
/// );
/// ```
#borrow(a, t)
pub extern "C" fn lapacke_sorgtsqr_row(
  matrix_layout : Int,
  m : Lapack_int,
  n : Lapack_int,
  mb : Lapack_int,
  nb : Lapack_int,
  a : FixedArray[Float],
  lda : Lapack_int,
  t : FixedArray[Float],
  ldt : Lapack_int,
) -> Lapack_int = "LAPACKE_sorgtsqr_row"
