///|
///
/// ```c
/// lapack_complex_float lapack_make_complex_float(float re, float im);
/// ```
pub extern "C" fn lapack_make_complex_float(re: Float, im: Float) -> Lapack_complex_float = "lapack_make_complex_float"

///|
///
/// ```c
/// lapack_complex_double lapack_make_complex_double(double re, double im);
/// ```
pub extern "C" fn lapack_make_complex_double(re: Double, im: Double) -> Lapack_complex_double = "lapack_make_complex_double"

///|
///
/// ```c
/// lapack_int LAPACKE_sbdsdc(
///     int matrix_layout, char uplo, char compq, lapack_int n, 
///     float* d, float* e, float* u, lapack_int ldu, 
///     float* vt, lapack_int ldvt, float* q, lapack_int* iq
/// );
/// ```
pub extern "C" fn lapacke_sbdsdc(matrix_layout: Int, uplo: Byte, compq: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], u: FixedArray[Float], ldu: Lapack_int, vt: FixedArray[Float], ldvt: Lapack_int, q: FixedArray[Float], iq: FixedArray[Int]) -> Lapack_int = "LAPACKE_sbdsdc"

///|
///
/// ```c
/// lapack_int LAPACKE_dbdsdc(
///     int matrix_layout, char uplo, char compq, lapack_int n, 
///     double* d, double* e, double* u, lapack_int ldu, 
///     double* vt, lapack_int ldvt, double* q, lapack_int* iq
/// );
/// ```
pub extern "C" fn lapacke_dbdsdc(matrix_layout: Int, uplo: Byte, compq: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], u: FixedArray[Double], ldu: Lapack_int, vt: FixedArray[Double], ldvt: Lapack_int, q: FixedArray[Double], iq: FixedArray[Int]) -> Lapack_int = "LAPACKE_dbdsdc"

///|
///
/// ```c
/// lapack_int LAPACKE_sbdsqr(
///     int matrix_layout, char uplo, lapack_int n, lapack_int ncvt, 
///     lapack_int nru, lapack_int ncc, float* d, float* e, 
///     float* vt, lapack_int ldvt, float* u, lapack_int ldu, 
///     float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_sbdsqr(matrix_layout: Int, uplo: Byte, n: Lapack_int, ncvt: Lapack_int, nru: Lapack_int, ncc: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vt: FixedArray[Float], ldvt: Lapack_int, u: FixedArray[Float], ldu: Lapack_int, c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_sbdsqr"

///|
///
/// ```c
/// lapack_int LAPACKE_dbdsqr(
///     int matrix_layout, char uplo, lapack_int n, lapack_int ncvt, 
///     lapack_int nru, lapack_int ncc, double* d, double* e, 
///     double* vt, lapack_int ldvt, double* u, lapack_int ldu, 
///     double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_dbdsqr(matrix_layout: Int, uplo: Byte, n: Lapack_int, ncvt: Lapack_int, nru: Lapack_int, ncc: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vt: FixedArray[Double], ldvt: Lapack_int, u: FixedArray[Double], ldu: Lapack_int, c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_dbdsqr"

///|
///
/// ```c
/// lapack_int LAPACKE_cbdsqr(
///     int matrix_layout, char uplo, lapack_int n, lapack_int ncvt, 
///     lapack_int nru, lapack_int ncc, float* d, float* e, 
///     lapack_complex_float* vt, lapack_int ldvt, lapack_complex_float* u, lapack_int ldu, 
///     lapack_complex_float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_cbdsqr(matrix_layout: Int, uplo: Byte, n: Lapack_int, ncvt: Lapack_int, nru: Lapack_int, ncc: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vt: FixedArray[Float], ldvt: Lapack_int, u: FixedArray[Float], ldu: Lapack_int, c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_cbdsqr"

///|
///
/// ```c
/// lapack_int LAPACKE_zbdsqr(
///     int matrix_layout, char uplo, lapack_int n, lapack_int ncvt, 
///     lapack_int nru, lapack_int ncc, double* d, double* e, 
///     lapack_complex_double* vt, lapack_int ldvt, lapack_complex_double* u, lapack_int ldu, 
///     lapack_complex_double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_zbdsqr(matrix_layout: Int, uplo: Byte, n: Lapack_int, ncvt: Lapack_int, nru: Lapack_int, ncc: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vt: FixedArray[Double], ldvt: Lapack_int, u: FixedArray[Double], ldu: Lapack_int, c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_zbdsqr"

///|
///
/// ```c
/// lapack_int LAPACKE_sbdsvdx(
///     int matrix_layout, char uplo, char jobz, char range, 
///     lapack_int n, float* d, float* e, float vl, 
///     float vu, lapack_int il, lapack_int iu, lapack_int* ns, 
///     float* s, float* z, lapack_int ldz, lapack_int* superb
/// );
/// ```
pub extern "C" fn lapacke_sbdsvdx(matrix_layout: Int, uplo: Byte, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, ns: FixedArray[Int], s: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, superb: FixedArray[Int]) -> Lapack_int = "LAPACKE_sbdsvdx"

///|
///
/// ```c
/// lapack_int LAPACKE_dbdsvdx(
///     int matrix_layout, char uplo, char jobz, char range, 
///     lapack_int n, double* d, double* e, double vl, 
///     double vu, lapack_int il, lapack_int iu, lapack_int* ns, 
///     double* s, double* z, lapack_int ldz, lapack_int* superb
/// );
/// ```
pub extern "C" fn lapacke_dbdsvdx(matrix_layout: Int, uplo: Byte, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, ns: FixedArray[Int], s: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, superb: FixedArray[Int]) -> Lapack_int = "LAPACKE_dbdsvdx"

///|
///
/// ```c
/// lapack_int LAPACKE_sdisna(
///     char job, lapack_int m, lapack_int n, const float* d, float* sep
/// );
/// ```
pub extern "C" fn lapacke_sdisna(job: Byte, m: Lapack_int, n: Lapack_int, d: FixedArray[Float], sep: FixedArray[Float]) -> Lapack_int = "LAPACKE_sdisna"

///|
///
/// ```c
/// lapack_int LAPACKE_ddisna(
///     char job, lapack_int m, lapack_int n, const double* d, double* sep
/// );
/// ```
pub extern "C" fn lapacke_ddisna(job: Byte, m: Lapack_int, n: Lapack_int, d: FixedArray[Double], sep: FixedArray[Double]) -> Lapack_int = "LAPACKE_ddisna"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbbrd(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int ncc, lapack_int kl, lapack_int ku, float* ab, 
///     lapack_int ldab, float* d, float* e, float* q, 
///     lapack_int ldq, float* pt, lapack_int ldpt, float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_sgbbrd(matrix_layout: Int, vect: Byte, m: Lapack_int, n: Lapack_int, ncc: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], q: FixedArray[Float], ldq: Lapack_int, pt: FixedArray[Float], ldpt: Lapack_int, c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_sgbbrd"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbbrd(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int ncc, lapack_int kl, lapack_int ku, double* ab, 
///     lapack_int ldab, double* d, double* e, double* q, 
///     lapack_int ldq, double* pt, lapack_int ldpt, double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_dgbbrd(matrix_layout: Int, vect: Byte, m: Lapack_int, n: Lapack_int, ncc: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], q: FixedArray[Double], ldq: Lapack_int, pt: FixedArray[Double], ldpt: Lapack_int, c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_dgbbrd"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbbrd(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int ncc, lapack_int kl, lapack_int ku, lapack_complex_float* ab, 
///     lapack_int ldab, float* d, float* e, lapack_complex_float* q, 
///     lapack_int ldq, lapack_complex_float* pt, lapack_int ldpt, lapack_complex_float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_cgbbrd(matrix_layout: Int, vect: Byte, m: Lapack_int, n: Lapack_int, ncc: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], q: FixedArray[Float], ldq: Lapack_int, pt: FixedArray[Float], ldpt: Lapack_int, c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_cgbbrd"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbbrd(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int ncc, lapack_int kl, lapack_int ku, lapack_complex_double* ab, 
///     lapack_int ldab, double* d, double* e, lapack_complex_double* q, 
///     lapack_int ldq, lapack_complex_double* pt, lapack_int ldpt, lapack_complex_double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_zgbbrd(matrix_layout: Int, vect: Byte, m: Lapack_int, n: Lapack_int, ncc: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], q: FixedArray[Double], ldq: Lapack_int, pt: FixedArray[Double], ldpt: Lapack_int, c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_zgbbrd"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbcon(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const float* ab, lapack_int ldab, const lapack_int* ipiv, 
///     float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_sgbcon(matrix_layout: Int, norm: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, ipiv: FixedArray[Int], anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgbcon"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbcon(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const double* ab, lapack_int ldab, const lapack_int* ipiv, 
///     double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_dgbcon(matrix_layout: Int, norm: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, ipiv: FixedArray[Int], anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgbcon"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbcon(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_float* ab, lapack_int ldab, const lapack_int* ipiv, 
///     float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_cgbcon(matrix_layout: Int, norm: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, ipiv: FixedArray[Int], anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgbcon"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbcon(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_double* ab, lapack_int ldab, const lapack_int* ipiv, 
///     double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_zgbcon(matrix_layout: Int, norm: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, ipiv: FixedArray[Int], anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgbcon"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbequ(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const float* ab, lapack_int ldab, float* r, 
///     float* c, float* rowcnd, float* colcnd, float* amax
/// );
/// ```
pub extern "C" fn lapacke_sgbequ(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, r: FixedArray[Float], c: FixedArray[Float], rowcnd: FixedArray[Float], colcnd: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgbequ"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbequ(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const double* ab, lapack_int ldab, double* r, 
///     double* c, double* rowcnd, double* colcnd, double* amax
/// );
/// ```
pub extern "C" fn lapacke_dgbequ(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, r: FixedArray[Double], c: FixedArray[Double], rowcnd: FixedArray[Double], colcnd: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgbequ"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbequ(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_float* ab, lapack_int ldab, float* r, 
///     float* c, float* rowcnd, float* colcnd, float* amax
/// );
/// ```
pub extern "C" fn lapacke_cgbequ(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, r: FixedArray[Float], c: FixedArray[Float], rowcnd: FixedArray[Float], colcnd: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgbequ"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbequ(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_double* ab, lapack_int ldab, double* r, 
///     double* c, double* rowcnd, double* colcnd, double* amax
/// );
/// ```
pub extern "C" fn lapacke_zgbequ(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, r: FixedArray[Double], c: FixedArray[Double], rowcnd: FixedArray[Double], colcnd: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgbequ"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbequb(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const float* ab, lapack_int ldab, float* r, 
///     float* c, float* rowcnd, float* colcnd, float* amax
/// );
/// ```
pub extern "C" fn lapacke_sgbequb(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, r: FixedArray[Float], c: FixedArray[Float], rowcnd: FixedArray[Float], colcnd: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgbequb"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbequb(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const double* ab, lapack_int ldab, double* r, 
///     double* c, double* rowcnd, double* colcnd, double* amax
/// );
/// ```
pub extern "C" fn lapacke_dgbequb(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, r: FixedArray[Double], c: FixedArray[Double], rowcnd: FixedArray[Double], colcnd: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgbequb"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbequb(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_float* ab, lapack_int ldab, float* r, 
///     float* c, float* rowcnd, float* colcnd, float* amax
/// );
/// ```
pub extern "C" fn lapacke_cgbequb(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, r: FixedArray[Float], c: FixedArray[Float], rowcnd: FixedArray[Float], colcnd: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgbequb"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbequb(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_double* ab, lapack_int ldab, double* r, 
///     double* c, double* rowcnd, double* colcnd, double* amax
/// );
/// ```
pub extern "C" fn lapacke_zgbequb(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, r: FixedArray[Double], c: FixedArray[Double], rowcnd: FixedArray[Double], colcnd: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgbequb"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbrfs(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const float* ab, lapack_int ldab, 
///     const float* afb, lapack_int ldafb, const lapack_int* ipiv, const float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_sgbrfs(matrix_layout: Int, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgbrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbrfs(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const double* ab, lapack_int ldab, 
///     const double* afb, lapack_int ldafb, const lapack_int* ipiv, const double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dgbrfs(matrix_layout: Int, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgbrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbrfs(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const lapack_complex_float* ab, lapack_int ldab, 
///     const lapack_complex_float* afb, lapack_int ldafb, const lapack_int* ipiv, const lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_cgbrfs(matrix_layout: Int, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgbrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbrfs(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const lapack_complex_double* ab, lapack_int ldab, 
///     const lapack_complex_double* afb, lapack_int ldafb, const lapack_int* ipiv, const lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_zgbrfs(matrix_layout: Int, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgbrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbrfsx(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, const float* ab, 
///     lapack_int ldab, const float* afb, lapack_int ldafb, const lapack_int* ipiv, 
///     const float* r, const float* c, const float* b, lapack_int ldb, 
///     float* x, lapack_int ldx, float* rcond, float* berr, 
///     lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_sgbrfsx(matrix_layout: Int, trans: Byte, equed: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, ipiv: FixedArray[Int], r: FixedArray[Float], c: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgbrfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbrfsx(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, const double* ab, 
///     lapack_int ldab, const double* afb, lapack_int ldafb, const lapack_int* ipiv, 
///     const double* r, const double* c, const double* b, lapack_int ldb, 
///     double* x, lapack_int ldx, double* rcond, double* berr, 
///     lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_dgbrfsx(matrix_layout: Int, trans: Byte, equed: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, ipiv: FixedArray[Int], r: FixedArray[Double], c: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgbrfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbrfsx(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, const lapack_complex_float* ab, 
///     lapack_int ldab, const lapack_complex_float* afb, lapack_int ldafb, const lapack_int* ipiv, 
///     const float* r, const float* c, const lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* x, lapack_int ldx, float* rcond, float* berr, 
///     lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_cgbrfsx(matrix_layout: Int, trans: Byte, equed: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, ipiv: FixedArray[Int], r: FixedArray[Float], c: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgbrfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbrfsx(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, const lapack_complex_double* ab, 
///     lapack_int ldab, const lapack_complex_double* afb, lapack_int ldafb, const lapack_int* ipiv, 
///     const double* r, const double* c, const lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* x, lapack_int ldx, double* rcond, double* berr, 
///     lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_zgbrfsx(matrix_layout: Int, trans: Byte, equed: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, ipiv: FixedArray[Int], r: FixedArray[Double], c: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgbrfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbsv(
///     int matrix_layout, lapack_int n, lapack_int kl, lapack_int ku, 
///     lapack_int nrhs, float* ab, lapack_int ldab, lapack_int* ipiv, 
///     float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_sgbsv(matrix_layout: Int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_sgbsv"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbsv(
///     int matrix_layout, lapack_int n, lapack_int kl, lapack_int ku, 
///     lapack_int nrhs, double* ab, lapack_int ldab, lapack_int* ipiv, 
///     double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dgbsv(matrix_layout: Int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dgbsv"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbsv(
///     int matrix_layout, lapack_int n, lapack_int kl, lapack_int ku, 
///     lapack_int nrhs, lapack_complex_float* ab, lapack_int ldab, lapack_int* ipiv, 
///     lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cgbsv(matrix_layout: Int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cgbsv"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbsv(
///     int matrix_layout, lapack_int n, lapack_int kl, lapack_int ku, 
///     lapack_int nrhs, lapack_complex_double* ab, lapack_int ldab, lapack_int* ipiv, 
///     lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zgbsv(matrix_layout: Int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zgbsv"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbsvx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, float* ab, 
///     lapack_int ldab, float* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, float* r, float* c, float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* rcond, 
///     float* ferr, float* berr, float* rpivot
/// );
/// ```
pub extern "C" fn lapacke_sgbsvx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Float], c: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float], rpivot: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgbsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbsvx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, double* ab, 
///     lapack_int ldab, double* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, double* r, double* c, double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* rcond, 
///     double* ferr, double* berr, double* rpivot
/// );
/// ```
pub extern "C" fn lapacke_dgbsvx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Double], c: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double], rpivot: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgbsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbsvx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, lapack_complex_float* ab, 
///     lapack_int ldab, lapack_complex_float* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, float* r, float* c, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond, 
///     float* ferr, float* berr, float* rpivot
/// );
/// ```
pub extern "C" fn lapacke_cgbsvx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Float], c: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float], rpivot: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgbsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbsvx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, lapack_complex_double* ab, 
///     lapack_int ldab, lapack_complex_double* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, double* r, double* c, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond, 
///     double* ferr, double* berr, double* rpivot
/// );
/// ```
pub extern "C" fn lapacke_zgbsvx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Double], c: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double], rpivot: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgbsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbsvxx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, float* ab, 
///     lapack_int ldab, float* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, float* r, float* c, float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* rcond, 
///     float* rpvgrw, float* berr, lapack_int n_err_bnds, float* err_bnds_norm, 
///     float* err_bnds_comp, lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_sgbsvxx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Float], c: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], rpvgrw: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgbsvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbsvxx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, double* ab, 
///     lapack_int ldab, double* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, double* r, double* c, double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* rcond, 
///     double* rpvgrw, double* berr, lapack_int n_err_bnds, double* err_bnds_norm, 
///     double* err_bnds_comp, lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_dgbsvxx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Double], c: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], rpvgrw: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgbsvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbsvxx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, lapack_complex_float* ab, 
///     lapack_int ldab, lapack_complex_float* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, float* r, float* c, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond, 
///     float* rpvgrw, float* berr, lapack_int n_err_bnds, float* err_bnds_norm, 
///     float* err_bnds_comp, lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_cgbsvxx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Float], c: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], rpvgrw: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgbsvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbsvxx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int kl, lapack_int ku, lapack_int nrhs, lapack_complex_double* ab, 
///     lapack_int ldab, lapack_complex_double* afb, lapack_int ldafb, lapack_int* ipiv, 
///     char* equed, double* r, double* c, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond, 
///     double* rpvgrw, double* berr, lapack_int n_err_bnds, double* err_bnds_norm, 
///     double* err_bnds_comp, lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_zgbsvxx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Double], c: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], rpvgrw: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgbsvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbtrf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, float* ab, lapack_int ldab, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_sgbtrf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_sgbtrf"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbtrf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, double* ab, lapack_int ldab, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_dgbtrf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_dgbtrf"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbtrf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_complex_float* ab, lapack_int ldab, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_cgbtrf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_cgbtrf"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbtrf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_complex_double* ab, lapack_int ldab, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_zgbtrf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_zgbtrf"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbtrs(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const float* ab, lapack_int ldab, 
///     const lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_sgbtrs(matrix_layout: Int, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_sgbtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbtrs(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const double* ab, lapack_int ldab, 
///     const lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dgbtrs(matrix_layout: Int, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dgbtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbtrs(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const lapack_complex_float* ab, lapack_int ldab, 
///     const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cgbtrs(matrix_layout: Int, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cgbtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbtrs(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const lapack_complex_double* ab, lapack_int ldab, 
///     const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zgbtrs(matrix_layout: Int, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zgbtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_sgebak(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const float* scale, lapack_int m, 
///     float* v, lapack_int ldv
/// );
/// ```
pub extern "C" fn lapacke_sgebak(matrix_layout: Int, job: Byte, side: Byte, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, scale: FixedArray[Float], m: Lapack_int, v: FixedArray[Float], ldv: Lapack_int) -> Lapack_int = "LAPACKE_sgebak"

///|
///
/// ```c
/// lapack_int LAPACKE_dgebak(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const double* scale, lapack_int m, 
///     double* v, lapack_int ldv
/// );
/// ```
pub extern "C" fn lapacke_dgebak(matrix_layout: Int, job: Byte, side: Byte, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, scale: FixedArray[Double], m: Lapack_int, v: FixedArray[Double], ldv: Lapack_int) -> Lapack_int = "LAPACKE_dgebak"

///|
///
/// ```c
/// lapack_int LAPACKE_cgebak(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const float* scale, lapack_int m, 
///     lapack_complex_float* v, lapack_int ldv
/// );
/// ```
pub extern "C" fn lapacke_cgebak(matrix_layout: Int, job: Byte, side: Byte, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, scale: FixedArray[Float], m: Lapack_int, v: FixedArray[Float], ldv: Lapack_int) -> Lapack_int = "LAPACKE_cgebak"

///|
///
/// ```c
/// lapack_int LAPACKE_zgebak(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const double* scale, lapack_int m, 
///     lapack_complex_double* v, lapack_int ldv
/// );
/// ```
pub extern "C" fn lapacke_zgebak(matrix_layout: Int, job: Byte, side: Byte, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, scale: FixedArray[Double], m: Lapack_int, v: FixedArray[Double], ldv: Lapack_int) -> Lapack_int = "LAPACKE_zgebak"

///|
///
/// ```c
/// lapack_int LAPACKE_sgebal(
///     int matrix_layout, char job, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* ilo, lapack_int* ihi, float* scale
/// );
/// ```
pub extern "C" fn lapacke_sgebal(matrix_layout: Int, job: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], scale: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgebal"

///|
///
/// ```c
/// lapack_int LAPACKE_dgebal(
///     int matrix_layout, char job, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* ilo, lapack_int* ihi, double* scale
/// );
/// ```
pub extern "C" fn lapacke_dgebal(matrix_layout: Int, job: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], scale: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgebal"

///|
///
/// ```c
/// lapack_int LAPACKE_cgebal(
///     int matrix_layout, char job, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ilo, lapack_int* ihi, float* scale
/// );
/// ```
pub extern "C" fn lapacke_cgebal(matrix_layout: Int, job: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], scale: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgebal"

///|
///
/// ```c
/// lapack_int LAPACKE_zgebal(
///     int matrix_layout, char job, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ilo, lapack_int* ihi, double* scale
/// );
/// ```
pub extern "C" fn lapacke_zgebal(matrix_layout: Int, job: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], scale: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgebal"

///|
///
/// ```c
/// lapack_int LAPACKE_sgebrd(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* d, float* e, float* tauq, float* taup
/// );
/// ```
pub extern "C" fn lapacke_sgebrd(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], tauq: FixedArray[Float], taup: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgebrd"

///|
///
/// ```c
/// lapack_int LAPACKE_dgebrd(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* d, double* e, double* tauq, double* taup
/// );
/// ```
pub extern "C" fn lapacke_dgebrd(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], tauq: FixedArray[Double], taup: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgebrd"

///|
///
/// ```c
/// lapack_int LAPACKE_cgebrd(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, float* d, float* e, lapack_complex_float* tauq, lapack_complex_float* taup
/// );
/// ```
pub extern "C" fn lapacke_cgebrd(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], tauq: FixedArray[Float], taup: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgebrd"

///|
///
/// ```c
/// lapack_int LAPACKE_zgebrd(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, double* d, double* e, lapack_complex_double* tauq, lapack_complex_double* taup
/// );
/// ```
pub extern "C" fn lapacke_zgebrd(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], tauq: FixedArray[Double], taup: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgebrd"

///|
///
/// ```c
/// lapack_int LAPACKE_sgecon(
///     int matrix_layout, char norm, lapack_int n, const float* a, 
///     lapack_int lda, float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_sgecon(matrix_layout: Int, norm: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgecon"

///|
///
/// ```c
/// lapack_int LAPACKE_dgecon(
///     int matrix_layout, char norm, lapack_int n, const double* a, 
///     lapack_int lda, double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_dgecon(matrix_layout: Int, norm: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgecon"

///|
///
/// ```c
/// lapack_int LAPACKE_cgecon(
///     int matrix_layout, char norm, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_cgecon(matrix_layout: Int, norm: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgecon"

///|
///
/// ```c
/// lapack_int LAPACKE_zgecon(
///     int matrix_layout, char norm, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_zgecon(matrix_layout: Int, norm: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgecon"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeequ(
///     int matrix_layout, lapack_int m, lapack_int n, const float* a, 
///     lapack_int lda, float* r, float* c, float* rowcnd, 
///     float* colcnd, float* amax
/// );
/// ```
pub extern "C" fn lapacke_sgeequ(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, r: FixedArray[Float], c: FixedArray[Float], rowcnd: FixedArray[Float], colcnd: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgeequ"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeequ(
///     int matrix_layout, lapack_int m, lapack_int n, const double* a, 
///     lapack_int lda, double* r, double* c, double* rowcnd, 
///     double* colcnd, double* amax
/// );
/// ```
pub extern "C" fn lapacke_dgeequ(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, r: FixedArray[Double], c: FixedArray[Double], rowcnd: FixedArray[Double], colcnd: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgeequ"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeequ(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, float* r, float* c, float* rowcnd, 
///     float* colcnd, float* amax
/// );
/// ```
pub extern "C" fn lapacke_cgeequ(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, r: FixedArray[Float], c: FixedArray[Float], rowcnd: FixedArray[Float], colcnd: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgeequ"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeequ(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, double* r, double* c, double* rowcnd, 
///     double* colcnd, double* amax
/// );
/// ```
pub extern "C" fn lapacke_zgeequ(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, r: FixedArray[Double], c: FixedArray[Double], rowcnd: FixedArray[Double], colcnd: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgeequ"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeequb(
///     int matrix_layout, lapack_int m, lapack_int n, const float* a, 
///     lapack_int lda, float* r, float* c, float* rowcnd, 
///     float* colcnd, float* amax
/// );
/// ```
pub extern "C" fn lapacke_sgeequb(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, r: FixedArray[Float], c: FixedArray[Float], rowcnd: FixedArray[Float], colcnd: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgeequb"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeequb(
///     int matrix_layout, lapack_int m, lapack_int n, const double* a, 
///     lapack_int lda, double* r, double* c, double* rowcnd, 
///     double* colcnd, double* amax
/// );
/// ```
pub extern "C" fn lapacke_dgeequb(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, r: FixedArray[Double], c: FixedArray[Double], rowcnd: FixedArray[Double], colcnd: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgeequb"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeequb(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, float* r, float* c, float* rowcnd, 
///     float* colcnd, float* amax
/// );
/// ```
pub extern "C" fn lapacke_cgeequb(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, r: FixedArray[Float], c: FixedArray[Float], rowcnd: FixedArray[Float], colcnd: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgeequb"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeequb(
///     int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, double* r, double* c, double* rowcnd, 
///     double* colcnd, double* amax
/// );
/// ```
pub extern "C" fn lapacke_zgeequb(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, r: FixedArray[Double], c: FixedArray[Double], rowcnd: FixedArray[Double], colcnd: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgeequb"

///|
///
/// ```c
/// lapack_int LAPACKE_sgees(
///     int matrix_layout, char jobvs, char sort, LAPACK_S_SELECT2 select, 
///     lapack_int n, float* a, lapack_int lda, lapack_int* sdim, 
///     float* wr, float* wi, float* vs, lapack_int ldvs
/// );
/// ```
pub extern "C" fn lapacke_sgees(matrix_layout: Int, jobvs: Byte, sort: Byte, select: Lapack_s_select2, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, sdim: FixedArray[Int], wr: FixedArray[Float], wi: FixedArray[Float], vs: FixedArray[Float], ldvs: Lapack_int) -> Lapack_int = "LAPACKE_sgees"

///|
///
/// ```c
/// lapack_int LAPACKE_dgees(
///     int matrix_layout, char jobvs, char sort, LAPACK_D_SELECT2 select, 
///     lapack_int n, double* a, lapack_int lda, lapack_int* sdim, 
///     double* wr, double* wi, double* vs, lapack_int ldvs
/// );
/// ```
pub extern "C" fn lapacke_dgees(matrix_layout: Int, jobvs: Byte, sort: Byte, select: Lapack_d_select2, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, sdim: FixedArray[Int], wr: FixedArray[Double], wi: FixedArray[Double], vs: FixedArray[Double], ldvs: Lapack_int) -> Lapack_int = "LAPACKE_dgees"

///|
///
/// ```c
/// lapack_int LAPACKE_cgees(
///     int matrix_layout, char jobvs, char sort, LAPACK_C_SELECT1 select, 
///     lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_int* sdim, 
///     lapack_complex_float* w, lapack_complex_float* vs, lapack_int ldvs
/// );
/// ```
pub extern "C" fn lapacke_cgees(matrix_layout: Int, jobvs: Byte, sort: Byte, select: Lapack_c_select1, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, sdim: FixedArray[Int], w: FixedArray[Float], vs: FixedArray[Float], ldvs: Lapack_int) -> Lapack_int = "LAPACKE_cgees"

///|
///
/// ```c
/// lapack_int LAPACKE_zgees(
///     int matrix_layout, char jobvs, char sort, LAPACK_Z_SELECT1 select, 
///     lapack_int n, lapack_complex_double* a, lapack_int lda, lapack_int* sdim, 
///     lapack_complex_double* w, lapack_complex_double* vs, lapack_int ldvs
/// );
/// ```
pub extern "C" fn lapacke_zgees(matrix_layout: Int, jobvs: Byte, sort: Byte, select: Lapack_z_select1, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, sdim: FixedArray[Int], w: FixedArray[Double], vs: FixedArray[Double], ldvs: Lapack_int) -> Lapack_int = "LAPACKE_zgees"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeesx(
///     int matrix_layout, char jobvs, char sort, LAPACK_S_SELECT2 select, 
///     char sense, lapack_int n, float* a, lapack_int lda, 
///     lapack_int* sdim, float* wr, float* wi, float* vs, 
///     lapack_int ldvs, float* rconde, float* rcondv
/// );
/// ```
pub extern "C" fn lapacke_sgeesx(matrix_layout: Int, jobvs: Byte, sort: Byte, select: Lapack_s_select2, sense: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, sdim: FixedArray[Int], wr: FixedArray[Float], wi: FixedArray[Float], vs: FixedArray[Float], ldvs: Lapack_int, rconde: FixedArray[Float], rcondv: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgeesx"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeesx(
///     int matrix_layout, char jobvs, char sort, LAPACK_D_SELECT2 select, 
///     char sense, lapack_int n, double* a, lapack_int lda, 
///     lapack_int* sdim, double* wr, double* wi, double* vs, 
///     lapack_int ldvs, double* rconde, double* rcondv
/// );
/// ```
pub extern "C" fn lapacke_dgeesx(matrix_layout: Int, jobvs: Byte, sort: Byte, select: Lapack_d_select2, sense: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, sdim: FixedArray[Int], wr: FixedArray[Double], wi: FixedArray[Double], vs: FixedArray[Double], ldvs: Lapack_int, rconde: FixedArray[Double], rcondv: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgeesx"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeesx(
///     int matrix_layout, char jobvs, char sort, LAPACK_C_SELECT1 select, 
///     char sense, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_int* sdim, lapack_complex_float* w, lapack_complex_float* vs, lapack_int ldvs, 
///     float* rconde, float* rcondv
/// );
/// ```
pub extern "C" fn lapacke_cgeesx(matrix_layout: Int, jobvs: Byte, sort: Byte, select: Lapack_c_select1, sense: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, sdim: FixedArray[Int], w: FixedArray[Float], vs: FixedArray[Float], ldvs: Lapack_int, rconde: FixedArray[Float], rcondv: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgeesx"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeesx(
///     int matrix_layout, char jobvs, char sort, LAPACK_Z_SELECT1 select, 
///     char sense, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_int* sdim, lapack_complex_double* w, lapack_complex_double* vs, lapack_int ldvs, 
///     double* rconde, double* rcondv
/// );
/// ```
pub extern "C" fn lapacke_zgeesx(matrix_layout: Int, jobvs: Byte, sort: Byte, select: Lapack_z_select1, sense: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, sdim: FixedArray[Int], w: FixedArray[Double], vs: FixedArray[Double], ldvs: Lapack_int, rconde: FixedArray[Double], rcondv: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgeesx"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeev(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     float* a, lapack_int lda, float* wr, float* wi, 
///     float* vl, lapack_int ldvl, float* vr, lapack_int ldvr
/// );
/// ```
pub extern "C" fn lapacke_sgeev(matrix_layout: Int, jobvl: Byte, jobvr: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, wr: FixedArray[Float], wi: FixedArray[Float], vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int) -> Lapack_int = "LAPACKE_sgeev"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeev(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     double* a, lapack_int lda, double* wr, double* wi, 
///     double* vl, lapack_int ldvl, double* vr, lapack_int ldvr
/// );
/// ```
pub extern "C" fn lapacke_dgeev(matrix_layout: Int, jobvl: Byte, jobvr: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, wr: FixedArray[Double], wi: FixedArray[Double], vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int) -> Lapack_int = "LAPACKE_dgeev"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeev(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* w, lapack_complex_float* vl, 
///     lapack_int ldvl, lapack_complex_float* vr, lapack_int ldvr
/// );
/// ```
pub extern "C" fn lapacke_cgeev(matrix_layout: Int, jobvl: Byte, jobvr: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, w: FixedArray[Float], vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int) -> Lapack_int = "LAPACKE_cgeev"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeev(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* w, lapack_complex_double* vl, 
///     lapack_int ldvl, lapack_complex_double* vr, lapack_int ldvr
/// );
/// ```
pub extern "C" fn lapacke_zgeev(matrix_layout: Int, jobvl: Byte, jobvr: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, w: FixedArray[Double], vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int) -> Lapack_int = "LAPACKE_zgeev"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeevx(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, float* a, lapack_int lda, 
///     float* wr, float* wi, float* vl, lapack_int ldvl, 
///     float* vr, lapack_int ldvr, lapack_int* ilo, lapack_int* ihi, 
///     float* scale, float* abnrm, float* rconde, float* rcondv
/// );
/// ```
pub extern "C" fn lapacke_sgeevx(matrix_layout: Int, balanc: Byte, jobvl: Byte, jobvr: Byte, sense: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, wr: FixedArray[Float], wi: FixedArray[Float], vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], scale: FixedArray[Float], abnrm: FixedArray[Float], rconde: FixedArray[Float], rcondv: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgeevx"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeevx(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, double* a, lapack_int lda, 
///     double* wr, double* wi, double* vl, lapack_int ldvl, 
///     double* vr, lapack_int ldvr, lapack_int* ilo, lapack_int* ihi, 
///     double* scale, double* abnrm, double* rconde, double* rcondv
/// );
/// ```
pub extern "C" fn lapacke_dgeevx(matrix_layout: Int, balanc: Byte, jobvl: Byte, jobvr: Byte, sense: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, wr: FixedArray[Double], wi: FixedArray[Double], vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], scale: FixedArray[Double], abnrm: FixedArray[Double], rconde: FixedArray[Double], rcondv: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgeevx"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeevx(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* w, lapack_complex_float* vl, lapack_int ldvl, lapack_complex_float* vr, 
///     lapack_int ldvr, lapack_int* ilo, lapack_int* ihi, float* scale, 
///     float* abnrm, float* rconde, float* rcondv
/// );
/// ```
pub extern "C" fn lapacke_cgeevx(matrix_layout: Int, balanc: Byte, jobvl: Byte, jobvr: Byte, sense: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, w: FixedArray[Float], vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], scale: FixedArray[Float], abnrm: FixedArray[Float], rconde: FixedArray[Float], rcondv: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgeevx"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeevx(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* w, lapack_complex_double* vl, lapack_int ldvl, lapack_complex_double* vr, 
///     lapack_int ldvr, lapack_int* ilo, lapack_int* ihi, double* scale, 
///     double* abnrm, double* rconde, double* rcondv
/// );
/// ```
pub extern "C" fn lapacke_zgeevx(matrix_layout: Int, balanc: Byte, jobvl: Byte, jobvr: Byte, sense: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, w: FixedArray[Double], vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], scale: FixedArray[Double], abnrm: FixedArray[Double], rconde: FixedArray[Double], rcondv: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgeevx"

///|
///
/// ```c
/// lapack_int LAPACKE_sgehrd(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     float* a, lapack_int lda, float* tau
/// );
/// ```
pub extern "C" fn lapacke_sgehrd(matrix_layout: Int, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgehrd"

///|
///
/// ```c
/// lapack_int LAPACKE_dgehrd(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     double* a, lapack_int lda, double* tau
/// );
/// ```
pub extern "C" fn lapacke_dgehrd(matrix_layout: Int, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgehrd"

///|
///
/// ```c
/// lapack_int LAPACKE_cgehrd(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cgehrd(matrix_layout: Int, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgehrd"

///|
///
/// ```c
/// lapack_int LAPACKE_zgehrd(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zgehrd(matrix_layout: Int, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgehrd"

///|
///
/// ```c
/// lapack_int LAPACKE_sgejsv(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     char jobr, char jobt, char jobp, lapack_int m, 
///     lapack_int n, float* a, lapack_int lda, float* sva, 
///     float* u, lapack_int ldu, float* v, lapack_int ldv, 
///     float* stat, lapack_int* istat
/// );
/// ```
pub extern "C" fn lapacke_sgejsv(matrix_layout: Int, joba: Byte, jobu: Byte, jobv: Byte, jobr: Byte, jobt: Byte, jobp: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, sva: FixedArray[Float], u: FixedArray[Float], ldu: Lapack_int, v: FixedArray[Float], ldv: Lapack_int, stat: FixedArray[Float], istat: FixedArray[Int]) -> Lapack_int = "LAPACKE_sgejsv"

///|
///
/// ```c
/// lapack_int LAPACKE_dgejsv(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     char jobr, char jobt, char jobp, lapack_int m, 
///     lapack_int n, double* a, lapack_int lda, double* sva, 
///     double* u, lapack_int ldu, double* v, lapack_int ldv, 
///     double* stat, lapack_int* istat
/// );
/// ```
pub extern "C" fn lapacke_dgejsv(matrix_layout: Int, joba: Byte, jobu: Byte, jobv: Byte, jobr: Byte, jobt: Byte, jobp: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, sva: FixedArray[Double], u: FixedArray[Double], ldu: Lapack_int, v: FixedArray[Double], ldv: Lapack_int, stat: FixedArray[Double], istat: FixedArray[Int]) -> Lapack_int = "LAPACKE_dgejsv"

///|
///
/// ```c
/// lapack_int LAPACKE_cgejsv(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     char jobr, char jobt, char jobp, lapack_int m, 
///     lapack_int n, lapack_complex_float* a, lapack_int lda, float* sva, 
///     lapack_complex_float* u, lapack_int ldu, lapack_complex_float* v, lapack_int ldv, 
///     float* stat, lapack_int* istat
/// );
/// ```
pub extern "C" fn lapacke_cgejsv(matrix_layout: Int, joba: Byte, jobu: Byte, jobv: Byte, jobr: Byte, jobt: Byte, jobp: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, sva: FixedArray[Float], u: FixedArray[Float], ldu: Lapack_int, v: FixedArray[Float], ldv: Lapack_int, stat: FixedArray[Float], istat: FixedArray[Int]) -> Lapack_int = "LAPACKE_cgejsv"

///|
///
/// ```c
/// lapack_int LAPACKE_zgejsv(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     char jobr, char jobt, char jobp, lapack_int m, 
///     lapack_int n, lapack_complex_double* a, lapack_int lda, double* sva, 
///     lapack_complex_double* u, lapack_int ldu, lapack_complex_double* v, lapack_int ldv, 
///     double* stat, lapack_int* istat
/// );
/// ```
pub extern "C" fn lapacke_zgejsv(matrix_layout: Int, joba: Byte, jobu: Byte, jobv: Byte, jobr: Byte, jobt: Byte, jobp: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, sva: FixedArray[Double], u: FixedArray[Double], ldu: Lapack_int, v: FixedArray[Double], ldv: Lapack_int, stat: FixedArray[Double], istat: FixedArray[Int]) -> Lapack_int = "LAPACKE_zgejsv"

///|
///
/// ```c
/// lapack_int LAPACKE_sgelq2(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau
/// );
/// ```
pub extern "C" fn lapacke_sgelq2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgelq2"

///|
///
/// ```c
/// lapack_int LAPACKE_dgelq2(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau
/// );
/// ```
pub extern "C" fn lapacke_dgelq2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgelq2"

///|
///
/// ```c
/// lapack_int LAPACKE_cgelq2(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cgelq2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgelq2"

///|
///
/// ```c
/// lapack_int LAPACKE_zgelq2(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zgelq2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgelq2"

///|
///
/// ```c
/// lapack_int LAPACKE_sgelqf(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau
/// );
/// ```
pub extern "C" fn lapacke_sgelqf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgelqf"

///|
///
/// ```c
/// lapack_int LAPACKE_dgelqf(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau
/// );
/// ```
pub extern "C" fn lapacke_dgelqf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgelqf"

///|
///
/// ```c
/// lapack_int LAPACKE_cgelqf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cgelqf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgelqf"

///|
///
/// ```c
/// lapack_int LAPACKE_zgelqf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zgelqf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgelqf"

///|
///
/// ```c
/// lapack_int LAPACKE_sgels(
///     int matrix_layout, char trans, lapack_int m, lapack_int n, 
///     lapack_int nrhs, float* a, lapack_int lda, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_sgels(matrix_layout: Int, trans: Byte, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_sgels"

///|
///
/// ```c
/// lapack_int LAPACKE_dgels(
///     int matrix_layout, char trans, lapack_int m, lapack_int n, 
///     lapack_int nrhs, double* a, lapack_int lda, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dgels(matrix_layout: Int, trans: Byte, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dgels"

///|
///
/// ```c
/// lapack_int LAPACKE_cgels(
///     int matrix_layout, char trans, lapack_int m, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cgels(matrix_layout: Int, trans: Byte, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cgels"

///|
///
/// ```c
/// lapack_int LAPACKE_zgels(
///     int matrix_layout, char trans, lapack_int m, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zgels(matrix_layout: Int, trans: Byte, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zgels"

///|
///
/// ```c
/// lapack_int LAPACKE_sgelsd(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* s, float rcond, lapack_int* rank
/// );
/// ```
pub extern "C" fn lapacke_sgelsd(matrix_layout: Int, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, s: FixedArray[Float], rcond: Float, rank: FixedArray[Int]) -> Lapack_int = "LAPACKE_sgelsd"

///|
///
/// ```c
/// lapack_int LAPACKE_dgelsd(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* s, double rcond, lapack_int* rank
/// );
/// ```
pub extern "C" fn lapacke_dgelsd(matrix_layout: Int, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, s: FixedArray[Double], rcond: Double, rank: FixedArray[Int]) -> Lapack_int = "LAPACKE_dgelsd"

///|
///
/// ```c
/// lapack_int LAPACKE_cgelsd(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     float* s, float rcond, lapack_int* rank
/// );
/// ```
pub extern "C" fn lapacke_cgelsd(matrix_layout: Int, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, s: FixedArray[Float], rcond: Float, rank: FixedArray[Int]) -> Lapack_int = "LAPACKE_cgelsd"

///|
///
/// ```c
/// lapack_int LAPACKE_zgelsd(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     double* s, double rcond, lapack_int* rank
/// );
/// ```
pub extern "C" fn lapacke_zgelsd(matrix_layout: Int, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, s: FixedArray[Double], rcond: Double, rank: FixedArray[Int]) -> Lapack_int = "LAPACKE_zgelsd"

///|
///
/// ```c
/// lapack_int LAPACKE_sgelss(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* s, float rcond, lapack_int* rank
/// );
/// ```
pub extern "C" fn lapacke_sgelss(matrix_layout: Int, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, s: FixedArray[Float], rcond: Float, rank: FixedArray[Int]) -> Lapack_int = "LAPACKE_sgelss"

///|
///
/// ```c
/// lapack_int LAPACKE_dgelss(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* s, double rcond, lapack_int* rank
/// );
/// ```
pub extern "C" fn lapacke_dgelss(matrix_layout: Int, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, s: FixedArray[Double], rcond: Double, rank: FixedArray[Int]) -> Lapack_int = "LAPACKE_dgelss"

///|
///
/// ```c
/// lapack_int LAPACKE_cgelss(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     float* s, float rcond, lapack_int* rank
/// );
/// ```
pub extern "C" fn lapacke_cgelss(matrix_layout: Int, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, s: FixedArray[Float], rcond: Float, rank: FixedArray[Int]) -> Lapack_int = "LAPACKE_cgelss"

///|
///
/// ```c
/// lapack_int LAPACKE_zgelss(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     double* s, double rcond, lapack_int* rank
/// );
/// ```
pub extern "C" fn lapacke_zgelss(matrix_layout: Int, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, s: FixedArray[Double], rcond: Double, rank: FixedArray[Int]) -> Lapack_int = "LAPACKE_zgelss"

///|
///
/// ```c
/// lapack_int LAPACKE_sgelsy(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     lapack_int* jpvt, float rcond, lapack_int* rank
/// );
/// ```
pub extern "C" fn lapacke_sgelsy(matrix_layout: Int, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, jpvt: FixedArray[Int], rcond: Float, rank: FixedArray[Int]) -> Lapack_int = "LAPACKE_sgelsy"

///|
///
/// ```c
/// lapack_int LAPACKE_dgelsy(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     lapack_int* jpvt, double rcond, lapack_int* rank
/// );
/// ```
pub extern "C" fn lapacke_dgelsy(matrix_layout: Int, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, jpvt: FixedArray[Int], rcond: Double, rank: FixedArray[Int]) -> Lapack_int = "LAPACKE_dgelsy"

///|
///
/// ```c
/// lapack_int LAPACKE_cgelsy(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     lapack_int* jpvt, float rcond, lapack_int* rank
/// );
/// ```
pub extern "C" fn lapacke_cgelsy(matrix_layout: Int, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, jpvt: FixedArray[Int], rcond: Float, rank: FixedArray[Int]) -> Lapack_int = "LAPACKE_cgelsy"

///|
///
/// ```c
/// lapack_int LAPACKE_zgelsy(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_int* jpvt, double rcond, lapack_int* rank
/// );
/// ```
pub extern "C" fn lapacke_zgelsy(matrix_layout: Int, m: Lapack_int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, jpvt: FixedArray[Int], rcond: Double, rank: FixedArray[Int]) -> Lapack_int = "LAPACKE_zgelsy"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqlf(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau
/// );
/// ```
pub extern "C" fn lapacke_sgeqlf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgeqlf"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqlf(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau
/// );
/// ```
pub extern "C" fn lapacke_dgeqlf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgeqlf"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqlf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cgeqlf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgeqlf"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqlf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zgeqlf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgeqlf"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqp3(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* jpvt, float* tau
/// );
/// ```
pub extern "C" fn lapacke_sgeqp3(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, jpvt: FixedArray[Int], tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgeqp3"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqp3(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* jpvt, double* tau
/// );
/// ```
pub extern "C" fn lapacke_dgeqp3(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, jpvt: FixedArray[Int], tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgeqp3"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqp3(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* jpvt, lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cgeqp3(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, jpvt: FixedArray[Int], tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgeqp3"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqp3(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* jpvt, lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zgeqp3(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, jpvt: FixedArray[Int], tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgeqp3"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqpf(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* jpvt, float* tau
/// );
/// ```
pub extern "C" fn lapacke_sgeqpf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, jpvt: FixedArray[Int], tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgeqpf"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqpf(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* jpvt, double* tau
/// );
/// ```
pub extern "C" fn lapacke_dgeqpf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, jpvt: FixedArray[Int], tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgeqpf"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqpf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* jpvt, lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cgeqpf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, jpvt: FixedArray[Int], tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgeqpf"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqpf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* jpvt, lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zgeqpf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, jpvt: FixedArray[Int], tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgeqpf"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqr2(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau
/// );
/// ```
pub extern "C" fn lapacke_sgeqr2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgeqr2"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqr2(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau
/// );
/// ```
pub extern "C" fn lapacke_dgeqr2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgeqr2"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqr2(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cgeqr2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgeqr2"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqr2(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zgeqr2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgeqr2"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqrf(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau
/// );
/// ```
pub extern "C" fn lapacke_sgeqrf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgeqrf"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqrf(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau
/// );
/// ```
pub extern "C" fn lapacke_dgeqrf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgeqrf"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqrf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cgeqrf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgeqrf"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqrf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zgeqrf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgeqrf"

///|
///
/// ```c
/// lapack_int LAPACKE_sgeqrfp(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau
/// );
/// ```
pub extern "C" fn lapacke_sgeqrfp(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgeqrfp"

///|
///
/// ```c
/// lapack_int LAPACKE_dgeqrfp(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau
/// );
/// ```
pub extern "C" fn lapacke_dgeqrfp(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgeqrfp"

///|
///
/// ```c
/// lapack_int LAPACKE_cgeqrfp(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cgeqrfp(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgeqrfp"

///|
///
/// ```c
/// lapack_int LAPACKE_zgeqrfp(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zgeqrfp(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgeqrfp"

///|
///
/// ```c
/// lapack_int LAPACKE_sgerfs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const float* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const float* b, lapack_int ldb, float* x, 
///     lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_sgerfs(matrix_layout: Int, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgerfs"

///|
///
/// ```c
/// lapack_int LAPACKE_dgerfs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const double* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const double* b, lapack_int ldb, double* x, 
///     lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dgerfs(matrix_layout: Int, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgerfs"

///|
///
/// ```c
/// lapack_int LAPACKE_cgerfs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_cgerfs(matrix_layout: Int, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgerfs"

///|
///
/// ```c
/// lapack_int LAPACKE_zgerfs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_zgerfs(matrix_layout: Int, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgerfs"

///|
///
/// ```c
/// lapack_int LAPACKE_sgerfsx(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int nrhs, const float* a, lapack_int lda, const float* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const float* r, const float* c, 
///     const float* b, lapack_int ldb, float* x, lapack_int ldx, 
///     float* rcond, float* berr, lapack_int n_err_bnds, float* err_bnds_norm, 
///     float* err_bnds_comp, lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_sgerfsx(matrix_layout: Int, trans: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], r: FixedArray[Float], c: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgerfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_dgerfsx(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int nrhs, const double* a, lapack_int lda, const double* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const double* r, const double* c, 
///     const double* b, lapack_int ldb, double* x, lapack_int ldx, 
///     double* rcond, double* berr, lapack_int n_err_bnds, double* err_bnds_norm, 
///     double* err_bnds_comp, lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_dgerfsx(matrix_layout: Int, trans: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], r: FixedArray[Double], c: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgerfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_cgerfsx(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const float* r, const float* c, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* rcond, float* berr, lapack_int n_err_bnds, float* err_bnds_norm, 
///     float* err_bnds_comp, lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_cgerfsx(matrix_layout: Int, trans: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], r: FixedArray[Float], c: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgerfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_zgerfsx(
///     int matrix_layout, char trans, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const double* r, const double* c, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* rcond, double* berr, lapack_int n_err_bnds, double* err_bnds_norm, 
///     double* err_bnds_comp, lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_zgerfsx(matrix_layout: Int, trans: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], r: FixedArray[Double], c: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgerfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_sgerqf(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau
/// );
/// ```
pub extern "C" fn lapacke_sgerqf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgerqf"

///|
///
/// ```c
/// lapack_int LAPACKE_dgerqf(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau
/// );
/// ```
pub extern "C" fn lapacke_dgerqf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgerqf"

///|
///
/// ```c
/// lapack_int LAPACKE_cgerqf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cgerqf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgerqf"

///|
///
/// ```c
/// lapack_int LAPACKE_zgerqf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zgerqf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgerqf"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesdd(
///     int matrix_layout, char jobz, lapack_int m, lapack_int n, 
///     float* a, lapack_int lda, float* s, float* u, 
///     lapack_int ldu, float* vt, lapack_int ldvt
/// );
/// ```
pub extern "C" fn lapacke_sgesdd(matrix_layout: Int, jobz: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], u: FixedArray[Float], ldu: Lapack_int, vt: FixedArray[Float], ldvt: Lapack_int) -> Lapack_int = "LAPACKE_sgesdd"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesdd(
///     int matrix_layout, char jobz, lapack_int m, lapack_int n, 
///     double* a, lapack_int lda, double* s, double* u, 
///     lapack_int ldu, double* vt, lapack_int ldvt
/// );
/// ```
pub extern "C" fn lapacke_dgesdd(matrix_layout: Int, jobz: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], u: FixedArray[Double], ldu: Lapack_int, vt: FixedArray[Double], ldvt: Lapack_int) -> Lapack_int = "LAPACKE_dgesdd"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesdd(
///     int matrix_layout, char jobz, lapack_int m, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, float* s, lapack_complex_float* u, 
///     lapack_int ldu, lapack_complex_float* vt, lapack_int ldvt
/// );
/// ```
pub extern "C" fn lapacke_cgesdd(matrix_layout: Int, jobz: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], u: FixedArray[Float], ldu: Lapack_int, vt: FixedArray[Float], ldvt: Lapack_int) -> Lapack_int = "LAPACKE_cgesdd"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesdd(
///     int matrix_layout, char jobz, lapack_int m, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, double* s, lapack_complex_double* u, 
///     lapack_int ldu, lapack_complex_double* vt, lapack_int ldvt
/// );
/// ```
pub extern "C" fn lapacke_zgesdd(matrix_layout: Int, jobz: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], u: FixedArray[Double], ldu: Lapack_int, vt: FixedArray[Double], ldvt: Lapack_int) -> Lapack_int = "LAPACKE_zgesdd"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesv(
///     int matrix_layout, lapack_int n, lapack_int nrhs, float* a, 
///     lapack_int lda, lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_sgesv(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_sgesv"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesv(
///     int matrix_layout, lapack_int n, lapack_int nrhs, double* a, 
///     lapack_int lda, lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dgesv(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dgesv"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesv(
///     int matrix_layout, lapack_int n, lapack_int nrhs, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cgesv(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cgesv"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesv(
///     int matrix_layout, lapack_int n, lapack_int nrhs, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zgesv(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zgesv"

///|
///
/// ```c
/// lapack_int LAPACKE_dsgesv(
///     int matrix_layout, lapack_int n, lapack_int nrhs, double* a, 
///     lapack_int lda, lapack_int* ipiv, double* b, lapack_int ldb, 
///     double* x, lapack_int ldx, lapack_int* iter
/// );
/// ```
pub extern "C" fn lapacke_dsgesv(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, iter: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsgesv"

///|
///
/// ```c
/// lapack_int LAPACKE_zcgesv(
///     int matrix_layout, lapack_int n, lapack_int nrhs, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* x, lapack_int ldx, lapack_int* iter
/// );
/// ```
pub extern "C" fn lapacke_zcgesv(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, iter: FixedArray[Int]) -> Lapack_int = "LAPACKE_zcgesv"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesvd(
///     int matrix_layout, char jobu, char jobvt, lapack_int m, 
///     lapack_int n, float* a, lapack_int lda, float* s, 
///     float* u, lapack_int ldu, float* vt, lapack_int ldvt, float* superb
/// );
/// ```
pub extern "C" fn lapacke_sgesvd(matrix_layout: Int, jobu: Byte, jobvt: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], u: FixedArray[Float], ldu: Lapack_int, vt: FixedArray[Float], ldvt: Lapack_int, superb: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgesvd"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesvd(
///     int matrix_layout, char jobu, char jobvt, lapack_int m, 
///     lapack_int n, double* a, lapack_int lda, double* s, 
///     double* u, lapack_int ldu, double* vt, lapack_int ldvt, double* superb
/// );
/// ```
pub extern "C" fn lapacke_dgesvd(matrix_layout: Int, jobu: Byte, jobvt: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], u: FixedArray[Double], ldu: Lapack_int, vt: FixedArray[Double], ldvt: Lapack_int, superb: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgesvd"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesvd(
///     int matrix_layout, char jobu, char jobvt, lapack_int m, 
///     lapack_int n, lapack_complex_float* a, lapack_int lda, float* s, 
///     lapack_complex_float* u, lapack_int ldu, lapack_complex_float* vt, lapack_int ldvt, float* superb
/// );
/// ```
pub extern "C" fn lapacke_cgesvd(matrix_layout: Int, jobu: Byte, jobvt: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], u: FixedArray[Float], ldu: Lapack_int, vt: FixedArray[Float], ldvt: Lapack_int, superb: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgesvd"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesvd(
///     int matrix_layout, char jobu, char jobvt, lapack_int m, 
///     lapack_int n, lapack_complex_double* a, lapack_int lda, double* s, 
///     lapack_complex_double* u, lapack_int ldu, lapack_complex_double* vt, lapack_int ldvt, double* superb
/// );
/// ```
pub extern "C" fn lapacke_zgesvd(matrix_layout: Int, jobu: Byte, jobvt: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], u: FixedArray[Double], ldu: Lapack_int, vt: FixedArray[Double], ldvt: Lapack_int, superb: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgesvd"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesvdx(
///     int matrix_layout, char jobu, char jobvt, char range, 
///     lapack_int m, lapack_int n, float* a, lapack_int lda, 
///     float vl, float vu, lapack_int il, lapack_int iu, 
///     lapack_int* ns, float* s, float* u, lapack_int ldu, 
///     float* vt, lapack_int ldvt, lapack_int* superb
/// );
/// ```
pub extern "C" fn lapacke_sgesvdx(matrix_layout: Int, jobu: Byte, jobvt: Byte, range: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, ns: FixedArray[Int], s: FixedArray[Float], u: FixedArray[Float], ldu: Lapack_int, vt: FixedArray[Float], ldvt: Lapack_int, superb: FixedArray[Int]) -> Lapack_int = "LAPACKE_sgesvdx"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesvdx(
///     int matrix_layout, char jobu, char jobvt, char range, 
///     lapack_int m, lapack_int n, double* a, lapack_int lda, 
///     double vl, double vu, lapack_int il, lapack_int iu, 
///     lapack_int* ns, double* s, double* u, lapack_int ldu, 
///     double* vt, lapack_int ldvt, lapack_int* superb
/// );
/// ```
pub extern "C" fn lapacke_dgesvdx(matrix_layout: Int, jobu: Byte, jobvt: Byte, range: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, ns: FixedArray[Int], s: FixedArray[Double], u: FixedArray[Double], ldu: Lapack_int, vt: FixedArray[Double], ldvt: Lapack_int, superb: FixedArray[Int]) -> Lapack_int = "LAPACKE_dgesvdx"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesvdx(
///     int matrix_layout, char jobu, char jobvt, char range, 
///     lapack_int m, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     float vl, float vu, lapack_int il, lapack_int iu, 
///     lapack_int* ns, float* s, lapack_complex_float* u, lapack_int ldu, 
///     lapack_complex_float* vt, lapack_int ldvt, lapack_int* superb
/// );
/// ```
pub extern "C" fn lapacke_cgesvdx(matrix_layout: Int, jobu: Byte, jobvt: Byte, range: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, ns: FixedArray[Int], s: FixedArray[Float], u: FixedArray[Float], ldu: Lapack_int, vt: FixedArray[Float], ldvt: Lapack_int, superb: FixedArray[Int]) -> Lapack_int = "LAPACKE_cgesvdx"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesvdx(
///     int matrix_layout, char jobu, char jobvt, char range, 
///     lapack_int m, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     double vl, double vu, lapack_int il, lapack_int iu, 
///     lapack_int* ns, double* s, lapack_complex_double* u, lapack_int ldu, 
///     lapack_complex_double* vt, lapack_int ldvt, lapack_int* superb
/// );
/// ```
pub extern "C" fn lapacke_zgesvdx(matrix_layout: Int, jobu: Byte, jobvt: Byte, range: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, ns: FixedArray[Int], s: FixedArray[Double], u: FixedArray[Double], ldu: Lapack_int, vt: FixedArray[Double], ldvt: Lapack_int, superb: FixedArray[Int]) -> Lapack_int = "LAPACKE_zgesvdx"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesvdq(
///     int matrix_layout, char joba, char jobp, char jobr, 
///     char jobu, char jobv, lapack_int m, lapack_int n, 
///     float* a, lapack_int lda, float* s, float* u, 
///     lapack_int ldu, float* v, lapack_int ldv, lapack_int* numrank
/// );
/// ```
pub extern "C" fn lapacke_sgesvdq(matrix_layout: Int, joba: Byte, jobp: Byte, jobr: Byte, jobu: Byte, jobv: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], u: FixedArray[Float], ldu: Lapack_int, v: FixedArray[Float], ldv: Lapack_int, numrank: FixedArray[Int]) -> Lapack_int = "LAPACKE_sgesvdq"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesvdq(
///     int matrix_layout, char joba, char jobp, char jobr, 
///     char jobu, char jobv, lapack_int m, lapack_int n, 
///     double* a, lapack_int lda, double* s, double* u, 
///     lapack_int ldu, double* v, lapack_int ldv, lapack_int* numrank
/// );
/// ```
pub extern "C" fn lapacke_dgesvdq(matrix_layout: Int, joba: Byte, jobp: Byte, jobr: Byte, jobu: Byte, jobv: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], u: FixedArray[Double], ldu: Lapack_int, v: FixedArray[Double], ldv: Lapack_int, numrank: FixedArray[Int]) -> Lapack_int = "LAPACKE_dgesvdq"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesvdq(
///     int matrix_layout, char joba, char jobp, char jobr, 
///     char jobu, char jobv, lapack_int m, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, float* s, lapack_complex_float* u, 
///     lapack_int ldu, lapack_complex_float* v, lapack_int ldv, lapack_int* numrank
/// );
/// ```
pub extern "C" fn lapacke_cgesvdq(matrix_layout: Int, joba: Byte, jobp: Byte, jobr: Byte, jobu: Byte, jobv: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], u: FixedArray[Float], ldu: Lapack_int, v: FixedArray[Float], ldv: Lapack_int, numrank: FixedArray[Int]) -> Lapack_int = "LAPACKE_cgesvdq"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesvdq(
///     int matrix_layout, char joba, char jobp, char jobr, 
///     char jobu, char jobv, lapack_int m, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, double* s, lapack_complex_double* u, 
///     lapack_int ldu, lapack_complex_double* v, lapack_int ldv, lapack_int* numrank
/// );
/// ```
pub extern "C" fn lapacke_zgesvdq(matrix_layout: Int, joba: Byte, jobp: Byte, jobr: Byte, jobu: Byte, jobv: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], u: FixedArray[Double], ldu: Lapack_int, v: FixedArray[Double], ldv: Lapack_int, numrank: FixedArray[Int]) -> Lapack_int = "LAPACKE_zgesvdq"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesvj(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     lapack_int m, lapack_int n, float* a, lapack_int lda, 
///     float* sva, lapack_int mv, float* v, lapack_int ldv, float* stat
/// );
/// ```
pub extern "C" fn lapacke_sgesvj(matrix_layout: Int, joba: Byte, jobu: Byte, jobv: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, sva: FixedArray[Float], mv: Lapack_int, v: FixedArray[Float], ldv: Lapack_int, stat: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgesvj"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesvj(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     lapack_int m, lapack_int n, double* a, lapack_int lda, 
///     double* sva, lapack_int mv, double* v, lapack_int ldv, double* stat
/// );
/// ```
pub extern "C" fn lapacke_dgesvj(matrix_layout: Int, joba: Byte, jobu: Byte, jobv: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, sva: FixedArray[Double], mv: Lapack_int, v: FixedArray[Double], ldv: Lapack_int, stat: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgesvj"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesvj(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     lapack_int m, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     float* sva, lapack_int mv, lapack_complex_float* v, lapack_int ldv, float* stat
/// );
/// ```
pub extern "C" fn lapacke_cgesvj(matrix_layout: Int, joba: Byte, jobu: Byte, jobv: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, sva: FixedArray[Float], mv: Lapack_int, v: FixedArray[Float], ldv: Lapack_int, stat: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgesvj"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesvj(
///     int matrix_layout, char joba, char jobu, char jobv, 
///     lapack_int m, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     double* sva, lapack_int mv, lapack_complex_double* v, lapack_int ldv, double* stat
/// );
/// ```
pub extern "C" fn lapacke_zgesvj(matrix_layout: Int, joba: Byte, jobu: Byte, jobv: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, sva: FixedArray[Double], mv: Lapack_int, v: FixedArray[Double], ldv: Lapack_int, stat: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgesvj"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesvx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, float* a, lapack_int lda, float* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, float* r, 
///     float* c, float* b, lapack_int ldb, float* x, 
///     lapack_int ldx, float* rcond, float* ferr, float* berr, float* rpivot
/// );
/// ```
pub extern "C" fn lapacke_sgesvx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Float], c: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float], rpivot: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgesvx"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesvx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, double* a, lapack_int lda, double* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, double* r, 
///     double* c, double* b, lapack_int ldb, double* x, 
///     lapack_int ldx, double* rcond, double* ferr, double* berr, double* rpivot
/// );
/// ```
pub extern "C" fn lapacke_dgesvx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Double], c: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double], rpivot: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgesvx"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesvx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, float* r, 
///     float* c, lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* rcond, float* ferr, float* berr, float* rpivot
/// );
/// ```
pub extern "C" fn lapacke_cgesvx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Float], c: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float], rpivot: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgesvx"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesvx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, double* r, 
///     double* c, lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* rcond, double* ferr, double* berr, double* rpivot
/// );
/// ```
pub extern "C" fn lapacke_zgesvx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Double], c: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double], rpivot: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgesvx"

///|
///
/// ```c
/// lapack_int LAPACKE_sgesvxx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, float* a, lapack_int lda, float* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, float* r, 
///     float* c, float* b, lapack_int ldb, float* x, 
///     lapack_int ldx, float* rcond, float* rpvgrw, float* berr, 
///     lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_sgesvxx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Float], c: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], rpvgrw: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgesvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_dgesvxx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, double* a, lapack_int lda, double* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, double* r, 
///     double* c, double* b, lapack_int ldb, double* x, 
///     lapack_int ldx, double* rcond, double* rpvgrw, double* berr, 
///     lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_dgesvxx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Double], c: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], rpvgrw: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgesvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_cgesvxx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, float* r, 
///     float* c, lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* rcond, float* rpvgrw, float* berr, 
///     lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_cgesvxx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Float], c: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], rpvgrw: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgesvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_zgesvxx(
///     int matrix_layout, char fact, char trans, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, double* r, 
///     double* c, lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* rcond, double* rpvgrw, double* berr, 
///     lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_zgesvxx(matrix_layout: Int, fact: Byte, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], r: FixedArray[Double], c: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], rpvgrw: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgesvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_sgetf2(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_sgetf2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_sgetf2"

///|
///
/// ```c
/// lapack_int LAPACKE_dgetf2(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_dgetf2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_dgetf2"

///|
///
/// ```c
/// lapack_int LAPACKE_cgetf2(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_cgetf2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_cgetf2"

///|
///
/// ```c
/// lapack_int LAPACKE_zgetf2(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_zgetf2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_zgetf2"

///|
///
/// ```c
/// lapack_int LAPACKE_sgetrf(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_sgetrf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_sgetrf"

///|
///
/// ```c
/// lapack_int LAPACKE_dgetrf(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_dgetrf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_dgetrf"

///|
///
/// ```c
/// lapack_int LAPACKE_cgetrf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_cgetrf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_cgetrf"

///|
///
/// ```c
/// lapack_int LAPACKE_zgetrf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_zgetrf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_zgetrf"

///|
///
/// ```c
/// lapack_int LAPACKE_sgetrf2(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_sgetrf2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_sgetrf2"

///|
///
/// ```c
/// lapack_int LAPACKE_dgetrf2(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_dgetrf2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_dgetrf2"

///|
///
/// ```c
/// lapack_int LAPACKE_cgetrf2(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_cgetrf2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_cgetrf2"

///|
///
/// ```c
/// lapack_int LAPACKE_zgetrf2(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_zgetrf2(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_zgetrf2"

///|
///
/// ```c
/// lapack_int LAPACKE_sgetri(
///     int matrix_layout, lapack_int n, float* a, lapack_int lda, const lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_sgetri(matrix_layout: Int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_sgetri"

///|
///
/// ```c
/// lapack_int LAPACKE_dgetri(
///     int matrix_layout, lapack_int n, double* a, lapack_int lda, const lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_dgetri(matrix_layout: Int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_dgetri"

///|
///
/// ```c
/// lapack_int LAPACKE_cgetri(
///     int matrix_layout, lapack_int n, lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_cgetri(matrix_layout: Int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_cgetri"

///|
///
/// ```c
/// lapack_int LAPACKE_zgetri(
///     int matrix_layout, lapack_int n, lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_zgetri(matrix_layout: Int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_zgetri"

///|
///
/// ```c
/// lapack_int LAPACKE_sgetrs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_sgetrs(matrix_layout: Int, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_sgetrs"

///|
///
/// ```c
/// lapack_int LAPACKE_dgetrs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dgetrs(matrix_layout: Int, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dgetrs"

///|
///
/// ```c
/// lapack_int LAPACKE_cgetrs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cgetrs(matrix_layout: Int, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cgetrs"

///|
///
/// ```c
/// lapack_int LAPACKE_zgetrs(
///     int matrix_layout, char trans, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zgetrs(matrix_layout: Int, trans: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zgetrs"

///|
///
/// ```c
/// lapack_int LAPACKE_sggbak(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const float* lscale, const float* rscale, 
///     lapack_int m, float* v, lapack_int ldv
/// );
/// ```
pub extern "C" fn lapacke_sggbak(matrix_layout: Int, job: Byte, side: Byte, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, lscale: FixedArray[Float], rscale: FixedArray[Float], m: Lapack_int, v: FixedArray[Float], ldv: Lapack_int) -> Lapack_int = "LAPACKE_sggbak"

///|
///
/// ```c
/// lapack_int LAPACKE_dggbak(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const double* lscale, const double* rscale, 
///     lapack_int m, double* v, lapack_int ldv
/// );
/// ```
pub extern "C" fn lapacke_dggbak(matrix_layout: Int, job: Byte, side: Byte, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, lscale: FixedArray[Double], rscale: FixedArray[Double], m: Lapack_int, v: FixedArray[Double], ldv: Lapack_int) -> Lapack_int = "LAPACKE_dggbak"

///|
///
/// ```c
/// lapack_int LAPACKE_cggbak(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const float* lscale, const float* rscale, 
///     lapack_int m, lapack_complex_float* v, lapack_int ldv
/// );
/// ```
pub extern "C" fn lapacke_cggbak(matrix_layout: Int, job: Byte, side: Byte, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, lscale: FixedArray[Float], rscale: FixedArray[Float], m: Lapack_int, v: FixedArray[Float], ldv: Lapack_int) -> Lapack_int = "LAPACKE_cggbak"

///|
///
/// ```c
/// lapack_int LAPACKE_zggbak(
///     int matrix_layout, char job, char side, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, const double* lscale, const double* rscale, 
///     lapack_int m, lapack_complex_double* v, lapack_int ldv
/// );
/// ```
pub extern "C" fn lapacke_zggbak(matrix_layout: Int, job: Byte, side: Byte, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, lscale: FixedArray[Double], rscale: FixedArray[Double], m: Lapack_int, v: FixedArray[Double], ldv: Lapack_int) -> Lapack_int = "LAPACKE_zggbak"

///|
///
/// ```c
/// lapack_int LAPACKE_sggbal(
///     int matrix_layout, char job, lapack_int n, float* a, 
///     lapack_int lda, float* b, lapack_int ldb, lapack_int* ilo, 
///     lapack_int* ihi, float* lscale, float* rscale
/// );
/// ```
pub extern "C" fn lapacke_sggbal(matrix_layout: Int, job: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], lscale: FixedArray[Float], rscale: FixedArray[Float]) -> Lapack_int = "LAPACKE_sggbal"

///|
///
/// ```c
/// lapack_int LAPACKE_dggbal(
///     int matrix_layout, char job, lapack_int n, double* a, 
///     lapack_int lda, double* b, lapack_int ldb, lapack_int* ilo, 
///     lapack_int* ihi, double* lscale, double* rscale
/// );
/// ```
pub extern "C" fn lapacke_dggbal(matrix_layout: Int, job: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], lscale: FixedArray[Double], rscale: FixedArray[Double]) -> Lapack_int = "LAPACKE_dggbal"

///|
///
/// ```c
/// lapack_int LAPACKE_cggbal(
///     int matrix_layout, char job, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* b, lapack_int ldb, lapack_int* ilo, 
///     lapack_int* ihi, float* lscale, float* rscale
/// );
/// ```
pub extern "C" fn lapacke_cggbal(matrix_layout: Int, job: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], lscale: FixedArray[Float], rscale: FixedArray[Float]) -> Lapack_int = "LAPACKE_cggbal"

///|
///
/// ```c
/// lapack_int LAPACKE_zggbal(
///     int matrix_layout, char job, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* b, lapack_int ldb, lapack_int* ilo, 
///     lapack_int* ihi, double* lscale, double* rscale
/// );
/// ```
pub extern "C" fn lapacke_zggbal(matrix_layout: Int, job: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], lscale: FixedArray[Double], rscale: FixedArray[Double]) -> Lapack_int = "LAPACKE_zggbal"

///|
///
/// ```c
/// lapack_int LAPACKE_sgges(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_S_SELECT3 selctg, lapack_int n, float* a, lapack_int lda, 
///     float* b, lapack_int ldb, lapack_int* sdim, float* alphar, 
///     float* alphai, float* beta, float* vsl, lapack_int ldvsl, 
///     float* vsr, lapack_int ldvsr
/// );
/// ```
pub extern "C" fn lapacke_sgges(matrix_layout: Int, jobvsl: Byte, jobvsr: Byte, sort: Byte, selctg: Lapack_s_select3, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, sdim: FixedArray[Int], alphar: FixedArray[Float], alphai: FixedArray[Float], beta: FixedArray[Float], vsl: FixedArray[Float], ldvsl: Lapack_int, vsr: FixedArray[Float], ldvsr: Lapack_int) -> Lapack_int = "LAPACKE_sgges"

///|
///
/// ```c
/// lapack_int LAPACKE_dgges(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_D_SELECT3 selctg, lapack_int n, double* a, lapack_int lda, 
///     double* b, lapack_int ldb, lapack_int* sdim, double* alphar, 
///     double* alphai, double* beta, double* vsl, lapack_int ldvsl, 
///     double* vsr, lapack_int ldvsr
/// );
/// ```
pub extern "C" fn lapacke_dgges(matrix_layout: Int, jobvsl: Byte, jobvsr: Byte, sort: Byte, selctg: Lapack_d_select3, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, sdim: FixedArray[Int], alphar: FixedArray[Double], alphai: FixedArray[Double], beta: FixedArray[Double], vsl: FixedArray[Double], ldvsl: Lapack_int, vsr: FixedArray[Double], ldvsr: Lapack_int) -> Lapack_int = "LAPACKE_dgges"

///|
///
/// ```c
/// lapack_int LAPACKE_cgges(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_C_SELECT2 selctg, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb, lapack_int* sdim, lapack_complex_float* alpha, 
///     lapack_complex_float* beta, lapack_complex_float* vsl, lapack_int ldvsl, lapack_complex_float* vsr, lapack_int ldvsr
/// );
/// ```
pub extern "C" fn lapacke_cgges(matrix_layout: Int, jobvsl: Byte, jobvsr: Byte, sort: Byte, selctg: Lapack_c_select2, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, sdim: FixedArray[Int], alpha: FixedArray[Float], beta: FixedArray[Float], vsl: FixedArray[Float], ldvsl: Lapack_int, vsr: FixedArray[Float], ldvsr: Lapack_int) -> Lapack_int = "LAPACKE_cgges"

///|
///
/// ```c
/// lapack_int LAPACKE_zgges(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_Z_SELECT2 selctg, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb, lapack_int* sdim, lapack_complex_double* alpha, 
///     lapack_complex_double* beta, lapack_complex_double* vsl, lapack_int ldvsl, lapack_complex_double* vsr, lapack_int ldvsr
/// );
/// ```
pub extern "C" fn lapacke_zgges(matrix_layout: Int, jobvsl: Byte, jobvsr: Byte, sort: Byte, selctg: Lapack_z_select2, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, sdim: FixedArray[Int], alpha: FixedArray[Double], beta: FixedArray[Double], vsl: FixedArray[Double], ldvsl: Lapack_int, vsr: FixedArray[Double], ldvsr: Lapack_int) -> Lapack_int = "LAPACKE_zgges"

///|
///
/// ```c
/// lapack_int LAPACKE_sgges3(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_S_SELECT3 selctg, lapack_int n, float* a, lapack_int lda, 
///     float* b, lapack_int ldb, lapack_int* sdim, float* alphar, 
///     float* alphai, float* beta, float* vsl, lapack_int ldvsl, 
///     float* vsr, lapack_int ldvsr
/// );
/// ```
pub extern "C" fn lapacke_sgges3(matrix_layout: Int, jobvsl: Byte, jobvsr: Byte, sort: Byte, selctg: Lapack_s_select3, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, sdim: FixedArray[Int], alphar: FixedArray[Float], alphai: FixedArray[Float], beta: FixedArray[Float], vsl: FixedArray[Float], ldvsl: Lapack_int, vsr: FixedArray[Float], ldvsr: Lapack_int) -> Lapack_int = "LAPACKE_sgges3"

///|
///
/// ```c
/// lapack_int LAPACKE_dgges3(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_D_SELECT3 selctg, lapack_int n, double* a, lapack_int lda, 
///     double* b, lapack_int ldb, lapack_int* sdim, double* alphar, 
///     double* alphai, double* beta, double* vsl, lapack_int ldvsl, 
///     double* vsr, lapack_int ldvsr
/// );
/// ```
pub extern "C" fn lapacke_dgges3(matrix_layout: Int, jobvsl: Byte, jobvsr: Byte, sort: Byte, selctg: Lapack_d_select3, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, sdim: FixedArray[Int], alphar: FixedArray[Double], alphai: FixedArray[Double], beta: FixedArray[Double], vsl: FixedArray[Double], ldvsl: Lapack_int, vsr: FixedArray[Double], ldvsr: Lapack_int) -> Lapack_int = "LAPACKE_dgges3"

///|
///
/// ```c
/// lapack_int LAPACKE_cgges3(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_C_SELECT2 selctg, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb, lapack_int* sdim, lapack_complex_float* alpha, 
///     lapack_complex_float* beta, lapack_complex_float* vsl, lapack_int ldvsl, lapack_complex_float* vsr, lapack_int ldvsr
/// );
/// ```
pub extern "C" fn lapacke_cgges3(matrix_layout: Int, jobvsl: Byte, jobvsr: Byte, sort: Byte, selctg: Lapack_c_select2, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, sdim: FixedArray[Int], alpha: FixedArray[Float], beta: FixedArray[Float], vsl: FixedArray[Float], ldvsl: Lapack_int, vsr: FixedArray[Float], ldvsr: Lapack_int) -> Lapack_int = "LAPACKE_cgges3"

///|
///
/// ```c
/// lapack_int LAPACKE_zgges3(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_Z_SELECT2 selctg, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb, lapack_int* sdim, lapack_complex_double* alpha, 
///     lapack_complex_double* beta, lapack_complex_double* vsl, lapack_int ldvsl, lapack_complex_double* vsr, lapack_int ldvsr
/// );
/// ```
pub extern "C" fn lapacke_zgges3(matrix_layout: Int, jobvsl: Byte, jobvsr: Byte, sort: Byte, selctg: Lapack_z_select2, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, sdim: FixedArray[Int], alpha: FixedArray[Double], beta: FixedArray[Double], vsl: FixedArray[Double], ldvsl: Lapack_int, vsr: FixedArray[Double], ldvsr: Lapack_int) -> Lapack_int = "LAPACKE_zgges3"

///|
///
/// ```c
/// lapack_int LAPACKE_sggesx(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_S_SELECT3 selctg, char sense, lapack_int n, float* a, 
///     lapack_int lda, float* b, lapack_int ldb, lapack_int* sdim, 
///     float* alphar, float* alphai, float* beta, float* vsl, 
///     lapack_int ldvsl, float* vsr, lapack_int ldvsr, float* rconde, float* rcondv
/// );
/// ```
pub extern "C" fn lapacke_sggesx(matrix_layout: Int, jobvsl: Byte, jobvsr: Byte, sort: Byte, selctg: Lapack_s_select3, sense: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, sdim: FixedArray[Int], alphar: FixedArray[Float], alphai: FixedArray[Float], beta: FixedArray[Float], vsl: FixedArray[Float], ldvsl: Lapack_int, vsr: FixedArray[Float], ldvsr: Lapack_int, rconde: FixedArray[Float], rcondv: FixedArray[Float]) -> Lapack_int = "LAPACKE_sggesx"

///|
///
/// ```c
/// lapack_int LAPACKE_dggesx(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_D_SELECT3 selctg, char sense, lapack_int n, double* a, 
///     lapack_int lda, double* b, lapack_int ldb, lapack_int* sdim, 
///     double* alphar, double* alphai, double* beta, double* vsl, 
///     lapack_int ldvsl, double* vsr, lapack_int ldvsr, double* rconde, double* rcondv
/// );
/// ```
pub extern "C" fn lapacke_dggesx(matrix_layout: Int, jobvsl: Byte, jobvsr: Byte, sort: Byte, selctg: Lapack_d_select3, sense: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, sdim: FixedArray[Int], alphar: FixedArray[Double], alphai: FixedArray[Double], beta: FixedArray[Double], vsl: FixedArray[Double], ldvsl: Lapack_int, vsr: FixedArray[Double], ldvsr: Lapack_int, rconde: FixedArray[Double], rcondv: FixedArray[Double]) -> Lapack_int = "LAPACKE_dggesx"

///|
///
/// ```c
/// lapack_int LAPACKE_cggesx(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_C_SELECT2 selctg, char sense, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* b, lapack_int ldb, lapack_int* sdim, 
///     lapack_complex_float* alpha, lapack_complex_float* beta, lapack_complex_float* vsl, lapack_int ldvsl, 
///     lapack_complex_float* vsr, lapack_int ldvsr, float* rconde, float* rcondv
/// );
/// ```
pub extern "C" fn lapacke_cggesx(matrix_layout: Int, jobvsl: Byte, jobvsr: Byte, sort: Byte, selctg: Lapack_c_select2, sense: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, sdim: FixedArray[Int], alpha: FixedArray[Float], beta: FixedArray[Float], vsl: FixedArray[Float], ldvsl: Lapack_int, vsr: FixedArray[Float], ldvsr: Lapack_int, rconde: FixedArray[Float], rcondv: FixedArray[Float]) -> Lapack_int = "LAPACKE_cggesx"

///|
///
/// ```c
/// lapack_int LAPACKE_zggesx(
///     int matrix_layout, char jobvsl, char jobvsr, char sort, 
///     LAPACK_Z_SELECT2 selctg, char sense, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* b, lapack_int ldb, lapack_int* sdim, 
///     lapack_complex_double* alpha, lapack_complex_double* beta, lapack_complex_double* vsl, lapack_int ldvsl, 
///     lapack_complex_double* vsr, lapack_int ldvsr, double* rconde, double* rcondv
/// );
/// ```
pub extern "C" fn lapacke_zggesx(matrix_layout: Int, jobvsl: Byte, jobvsr: Byte, sort: Byte, selctg: Lapack_z_select2, sense: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, sdim: FixedArray[Int], alpha: FixedArray[Double], beta: FixedArray[Double], vsl: FixedArray[Double], ldvsl: Lapack_int, vsr: FixedArray[Double], ldvsr: Lapack_int, rconde: FixedArray[Double], rcondv: FixedArray[Double]) -> Lapack_int = "LAPACKE_zggesx"

///|
///
/// ```c
/// lapack_int LAPACKE_sggev(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* alphar, float* alphai, float* beta, float* vl, 
///     lapack_int ldvl, float* vr, lapack_int ldvr
/// );
/// ```
pub extern "C" fn lapacke_sggev(matrix_layout: Int, jobvl: Byte, jobvr: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, alphar: FixedArray[Float], alphai: FixedArray[Float], beta: FixedArray[Float], vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int) -> Lapack_int = "LAPACKE_sggev"

///|
///
/// ```c
/// lapack_int LAPACKE_dggev(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* alphar, double* alphai, double* beta, double* vl, 
///     lapack_int ldvl, double* vr, lapack_int ldvr
/// );
/// ```
pub extern "C" fn lapacke_dggev(matrix_layout: Int, jobvl: Byte, jobvr: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, alphar: FixedArray[Double], alphai: FixedArray[Double], beta: FixedArray[Double], vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int) -> Lapack_int = "LAPACKE_dggev"

///|
///
/// ```c
/// lapack_int LAPACKE_cggev(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* alpha, lapack_complex_float* beta, lapack_complex_float* vl, lapack_int ldvl, 
///     lapack_complex_float* vr, lapack_int ldvr
/// );
/// ```
pub extern "C" fn lapacke_cggev(matrix_layout: Int, jobvl: Byte, jobvr: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, alpha: FixedArray[Float], beta: FixedArray[Float], vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int) -> Lapack_int = "LAPACKE_cggev"

///|
///
/// ```c
/// lapack_int LAPACKE_zggev(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* alpha, lapack_complex_double* beta, lapack_complex_double* vl, lapack_int ldvl, 
///     lapack_complex_double* vr, lapack_int ldvr
/// );
/// ```
pub extern "C" fn lapacke_zggev(matrix_layout: Int, jobvl: Byte, jobvr: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, alpha: FixedArray[Double], beta: FixedArray[Double], vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int) -> Lapack_int = "LAPACKE_zggev"

///|
///
/// ```c
/// lapack_int LAPACKE_sggev3(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* alphar, float* alphai, float* beta, float* vl, 
///     lapack_int ldvl, float* vr, lapack_int ldvr
/// );
/// ```
pub extern "C" fn lapacke_sggev3(matrix_layout: Int, jobvl: Byte, jobvr: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, alphar: FixedArray[Float], alphai: FixedArray[Float], beta: FixedArray[Float], vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int) -> Lapack_int = "LAPACKE_sggev3"

///|
///
/// ```c
/// lapack_int LAPACKE_dggev3(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* alphar, double* alphai, double* beta, double* vl, 
///     lapack_int ldvl, double* vr, lapack_int ldvr
/// );
/// ```
pub extern "C" fn lapacke_dggev3(matrix_layout: Int, jobvl: Byte, jobvr: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, alphar: FixedArray[Double], alphai: FixedArray[Double], beta: FixedArray[Double], vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int) -> Lapack_int = "LAPACKE_dggev3"

///|
///
/// ```c
/// lapack_int LAPACKE_cggev3(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* alpha, lapack_complex_float* beta, lapack_complex_float* vl, lapack_int ldvl, 
///     lapack_complex_float* vr, lapack_int ldvr
/// );
/// ```
pub extern "C" fn lapacke_cggev3(matrix_layout: Int, jobvl: Byte, jobvr: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, alpha: FixedArray[Float], beta: FixedArray[Float], vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int) -> Lapack_int = "LAPACKE_cggev3"

///|
///
/// ```c
/// lapack_int LAPACKE_zggev3(
///     int matrix_layout, char jobvl, char jobvr, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* alpha, lapack_complex_double* beta, lapack_complex_double* vl, lapack_int ldvl, 
///     lapack_complex_double* vr, lapack_int ldvr
/// );
/// ```
pub extern "C" fn lapacke_zggev3(matrix_layout: Int, jobvl: Byte, jobvr: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, alpha: FixedArray[Double], beta: FixedArray[Double], vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int) -> Lapack_int = "LAPACKE_zggev3"

///|
///
/// ```c
/// lapack_int LAPACKE_sggevx(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, float* a, lapack_int lda, 
///     float* b, lapack_int ldb, float* alphar, float* alphai, 
///     float* beta, float* vl, lapack_int ldvl, float* vr, 
///     lapack_int ldvr, lapack_int* ilo, lapack_int* ihi, float* lscale, 
///     float* rscale, float* abnrm, float* bbnrm, float* rconde, float* rcondv
/// );
/// ```
pub extern "C" fn lapacke_sggevx(matrix_layout: Int, balanc: Byte, jobvl: Byte, jobvr: Byte, sense: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, alphar: FixedArray[Float], alphai: FixedArray[Float], beta: FixedArray[Float], vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], lscale: FixedArray[Float], rscale: FixedArray[Float], abnrm: FixedArray[Float], bbnrm: FixedArray[Float], rconde: FixedArray[Float], rcondv: FixedArray[Float]) -> Lapack_int = "LAPACKE_sggevx"

///|
///
/// ```c
/// lapack_int LAPACKE_dggevx(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, double* a, lapack_int lda, 
///     double* b, lapack_int ldb, double* alphar, double* alphai, 
///     double* beta, double* vl, lapack_int ldvl, double* vr, 
///     lapack_int ldvr, lapack_int* ilo, lapack_int* ihi, double* lscale, 
///     double* rscale, double* abnrm, double* bbnrm, double* rconde, double* rcondv
/// );
/// ```
pub extern "C" fn lapacke_dggevx(matrix_layout: Int, balanc: Byte, jobvl: Byte, jobvr: Byte, sense: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, alphar: FixedArray[Double], alphai: FixedArray[Double], beta: FixedArray[Double], vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], lscale: FixedArray[Double], rscale: FixedArray[Double], abnrm: FixedArray[Double], bbnrm: FixedArray[Double], rconde: FixedArray[Double], rcondv: FixedArray[Double]) -> Lapack_int = "LAPACKE_dggevx"

///|
///
/// ```c
/// lapack_int LAPACKE_cggevx(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* alpha, lapack_complex_float* beta, 
///     lapack_complex_float* vl, lapack_int ldvl, lapack_complex_float* vr, lapack_int ldvr, 
///     lapack_int* ilo, lapack_int* ihi, float* lscale, float* rscale, 
///     float* abnrm, float* bbnrm, float* rconde, float* rcondv
/// );
/// ```
pub extern "C" fn lapacke_cggevx(matrix_layout: Int, balanc: Byte, jobvl: Byte, jobvr: Byte, sense: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, alpha: FixedArray[Float], beta: FixedArray[Float], vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], lscale: FixedArray[Float], rscale: FixedArray[Float], abnrm: FixedArray[Float], bbnrm: FixedArray[Float], rconde: FixedArray[Float], rcondv: FixedArray[Float]) -> Lapack_int = "LAPACKE_cggevx"

///|
///
/// ```c
/// lapack_int LAPACKE_zggevx(
///     int matrix_layout, char balanc, char jobvl, char jobvr, 
///     char sense, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* alpha, lapack_complex_double* beta, 
///     lapack_complex_double* vl, lapack_int ldvl, lapack_complex_double* vr, lapack_int ldvr, 
///     lapack_int* ilo, lapack_int* ihi, double* lscale, double* rscale, 
///     double* abnrm, double* bbnrm, double* rconde, double* rcondv
/// );
/// ```
pub extern "C" fn lapacke_zggevx(matrix_layout: Int, balanc: Byte, jobvl: Byte, jobvr: Byte, sense: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, alpha: FixedArray[Double], beta: FixedArray[Double], vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int, ilo: FixedArray[Int], ihi: FixedArray[Int], lscale: FixedArray[Double], rscale: FixedArray[Double], abnrm: FixedArray[Double], bbnrm: FixedArray[Double], rconde: FixedArray[Double], rcondv: FixedArray[Double]) -> Lapack_int = "LAPACKE_zggevx"

///|
///
/// ```c
/// lapack_int LAPACKE_sggglm(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* d, float* x, float* y
/// );
/// ```
pub extern "C" fn lapacke_sggglm(matrix_layout: Int, n: Lapack_int, m: Lapack_int, p: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, d: FixedArray[Float], x: FixedArray[Float], y: FixedArray[Float]) -> Lapack_int = "LAPACKE_sggglm"

///|
///
/// ```c
/// lapack_int LAPACKE_dggglm(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* d, double* x, double* y
/// );
/// ```
pub extern "C" fn lapacke_dggglm(matrix_layout: Int, n: Lapack_int, m: Lapack_int, p: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, d: FixedArray[Double], x: FixedArray[Double], y: FixedArray[Double]) -> Lapack_int = "LAPACKE_dggglm"

///|
///
/// ```c
/// lapack_int LAPACKE_cggglm(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* d, lapack_complex_float* x, lapack_complex_float* y
/// );
/// ```
pub extern "C" fn lapacke_cggglm(matrix_layout: Int, n: Lapack_int, m: Lapack_int, p: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, d: FixedArray[Float], x: FixedArray[Float], y: FixedArray[Float]) -> Lapack_int = "LAPACKE_cggglm"

///|
///
/// ```c
/// lapack_int LAPACKE_zggglm(
///     int matrix_layout, lapack_int n, lapack_int m, lapack_int p, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* d, lapack_complex_double* x, lapack_complex_double* y
/// );
/// ```
pub extern "C" fn lapacke_zggglm(matrix_layout: Int, n: Lapack_int, m: Lapack_int, p: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, d: FixedArray[Double], x: FixedArray[Double], y: FixedArray[Double]) -> Lapack_int = "LAPACKE_zggglm"

///|
///
/// ```c
/// lapack_int LAPACKE_sgghrd(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, float* a, lapack_int lda, 
///     float* b, lapack_int ldb, float* q, lapack_int ldq, 
///     float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_sgghrd(matrix_layout: Int, compq: Byte, compz: Byte, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_sgghrd"

///|
///
/// ```c
/// lapack_int LAPACKE_dgghrd(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, double* a, lapack_int lda, 
///     double* b, lapack_int ldb, double* q, lapack_int ldq, 
///     double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_dgghrd(matrix_layout: Int, compq: Byte, compz: Byte, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, q: FixedArray[Double], ldq: Lapack_int, z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_dgghrd"

///|
///
/// ```c
/// lapack_int LAPACKE_cgghrd(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* q, lapack_int ldq, 
///     lapack_complex_float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_cgghrd(matrix_layout: Int, compq: Byte, compz: Byte, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_cgghrd"

///|
///
/// ```c
/// lapack_int LAPACKE_zgghrd(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* q, lapack_int ldq, 
///     lapack_complex_double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_zgghrd(matrix_layout: Int, compq: Byte, compz: Byte, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, q: FixedArray[Double], ldq: Lapack_int, z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_zgghrd"

///|
///
/// ```c
/// lapack_int LAPACKE_sgghd3(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, float* a, lapack_int lda, 
///     float* b, lapack_int ldb, float* q, lapack_int ldq, 
///     float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_sgghd3(matrix_layout: Int, compq: Byte, compz: Byte, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_sgghd3"

///|
///
/// ```c
/// lapack_int LAPACKE_dgghd3(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, double* a, lapack_int lda, 
///     double* b, lapack_int ldb, double* q, lapack_int ldq, 
///     double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_dgghd3(matrix_layout: Int, compq: Byte, compz: Byte, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, q: FixedArray[Double], ldq: Lapack_int, z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_dgghd3"

///|
///
/// ```c
/// lapack_int LAPACKE_cgghd3(
///     int matrix_layout, char compq, char compz, lapack_int n, 
///     lapack_int ilo, lapack_int ihi, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* q, lapack_int ldq, 
///     lapack_complex_float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_cgghd3(matrix_layout: Int, compq: Byte, compz: Byte, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_cgghd3"

