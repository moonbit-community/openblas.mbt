///|
///
/// ```c
/// lapack_int LAPACKE_dsterf(lapack_int n, double* d, double* e);
/// ```
pub extern "C" fn lapacke_dsterf(n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsterf"

///|
///
/// ```c
/// lapack_int LAPACKE_sstev(
///     int matrix_layout, char jobz, lapack_int n, float* d, 
///     float* e, float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_sstev(matrix_layout: Int, jobz: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_sstev"

///|
///
/// ```c
/// lapack_int LAPACKE_dstev(
///     int matrix_layout, char jobz, lapack_int n, double* d, 
///     double* e, double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_dstev(matrix_layout: Int, jobz: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_dstev"

///|
///
/// ```c
/// lapack_int LAPACKE_sstevd(
///     int matrix_layout, char jobz, lapack_int n, float* d, 
///     float* e, float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_sstevd(matrix_layout: Int, jobz: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_sstevd"

///|
///
/// ```c
/// lapack_int LAPACKE_dstevd(
///     int matrix_layout, char jobz, lapack_int n, double* d, 
///     double* e, double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_dstevd(matrix_layout: Int, jobz: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_dstevd"

///|
///
/// ```c
/// lapack_int LAPACKE_sstevr(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     float* d, float* e, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, float* z, lapack_int ldz, lapack_int* isuppz
/// );
/// ```
pub extern "C" fn lapacke_sstevr(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, isuppz: FixedArray[Int]) -> Lapack_int = "LAPACKE_sstevr"

///|
///
/// ```c
/// lapack_int LAPACKE_dstevr(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     double* d, double* e, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, double* z, lapack_int ldz, lapack_int* isuppz
/// );
/// ```
pub extern "C" fn lapacke_dstevr(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, isuppz: FixedArray[Int]) -> Lapack_int = "LAPACKE_dstevr"

///|
///
/// ```c
/// lapack_int LAPACKE_sstevx(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     float* d, float* e, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, float* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_sstevx(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_sstevx"

///|
///
/// ```c
/// lapack_int LAPACKE_dstevx(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     double* d, double* e, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, double* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_dstevx(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_dstevx"

///|
///
/// ```c
/// lapack_int LAPACKE_ssycon(
///     int matrix_layout, char uplo, lapack_int n, const float* a, 
///     lapack_int lda, const lapack_int* ipiv, float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_ssycon(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int], anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssycon"

///|
///
/// ```c
/// lapack_int LAPACKE_dsycon(
///     int matrix_layout, char uplo, lapack_int n, const double* a, 
///     lapack_int lda, const lapack_int* ipiv, double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_dsycon(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int], anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsycon"

///|
///
/// ```c
/// lapack_int LAPACKE_csycon(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv, float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_csycon(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int], anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_csycon"

///|
///
/// ```c
/// lapack_int LAPACKE_zsycon(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv, double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_zsycon(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int], anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_zsycon"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyequb(
///     int matrix_layout, char uplo, lapack_int n, const float* a, 
///     lapack_int lda, float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_ssyequb(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssyequb"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyequb(
///     int matrix_layout, char uplo, lapack_int n, const double* a, 
///     lapack_int lda, double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_dsyequb(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsyequb"

///|
///
/// ```c
/// lapack_int LAPACKE_csyequb(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_csyequb(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_csyequb"

///|
///
/// ```c
/// lapack_int LAPACKE_zsyequb(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_zsyequb(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_zsyequb"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyev(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     float* a, lapack_int lda, float* w
/// );
/// ```
pub extern "C" fn lapacke_ssyev(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, w: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssyev"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyev(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     double* a, lapack_int lda, double* w
/// );
/// ```
pub extern "C" fn lapacke_dsyev(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, w: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsyev"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyevd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     float* a, lapack_int lda, float* w
/// );
/// ```
pub extern "C" fn lapacke_ssyevd(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, w: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssyevd"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyevd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     double* a, lapack_int lda, double* w
/// );
/// ```
pub extern "C" fn lapacke_dsyevd(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, w: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsyevd"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyevr(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, float* a, lapack_int lda, float vl, 
///     float vu, lapack_int il, lapack_int iu, float abstol, 
///     lapack_int* m, float* w, float* z, lapack_int ldz, lapack_int* isuppz
/// );
/// ```
pub extern "C" fn lapacke_ssyevr(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, isuppz: FixedArray[Int]) -> Lapack_int = "LAPACKE_ssyevr"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyevr(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, double* a, lapack_int lda, double vl, 
///     double vu, lapack_int il, lapack_int iu, double abstol, 
///     lapack_int* m, double* w, double* z, lapack_int ldz, lapack_int* isuppz
/// );
/// ```
pub extern "C" fn lapacke_dsyevr(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, isuppz: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsyevr"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyevx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, float* a, lapack_int lda, float vl, 
///     float vu, lapack_int il, lapack_int iu, float abstol, 
///     lapack_int* m, float* w, float* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_ssyevx(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_ssyevx"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyevx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, double* a, lapack_int lda, double vl, 
///     double vu, lapack_int il, lapack_int iu, double abstol, 
///     lapack_int* m, double* w, double* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_dsyevx(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsyevx"

///|
///
/// ```c
/// lapack_int LAPACKE_ssygst(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     float* a, lapack_int lda, const float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_ssygst(matrix_layout: Int, itype: Lapack_int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_ssygst"

///|
///
/// ```c
/// lapack_int LAPACKE_dsygst(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     double* a, lapack_int lda, const double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dsygst(matrix_layout: Int, itype: Lapack_int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dsygst"

///|
///
/// ```c
/// lapack_int LAPACKE_ssygv(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, float* a, lapack_int lda, float* b, 
///     lapack_int ldb, float* w
/// );
/// ```
pub extern "C" fn lapacke_ssygv(matrix_layout: Int, itype: Lapack_int, jobz: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, w: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssygv"

///|
///
/// ```c
/// lapack_int LAPACKE_dsygv(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, double* a, lapack_int lda, double* b, 
///     lapack_int ldb, double* w
/// );
/// ```
pub extern "C" fn lapacke_dsygv(matrix_layout: Int, itype: Lapack_int, jobz: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, w: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsygv"

///|
///
/// ```c
/// lapack_int LAPACKE_ssygvd(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, float* a, lapack_int lda, float* b, 
///     lapack_int ldb, float* w
/// );
/// ```
pub extern "C" fn lapacke_ssygvd(matrix_layout: Int, itype: Lapack_int, jobz: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, w: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssygvd"

///|
///
/// ```c
/// lapack_int LAPACKE_dsygvd(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, double* a, lapack_int lda, double* b, 
///     lapack_int ldb, double* w
/// );
/// ```
pub extern "C" fn lapacke_dsygvd(matrix_layout: Int, itype: Lapack_int, jobz: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, w: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsygvd"

///|
///
/// ```c
/// lapack_int LAPACKE_ssygvx(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, float* a, lapack_int lda, 
///     float* b, lapack_int ldb, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, float* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_ssygvx(matrix_layout: Int, itype: Lapack_int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_ssygvx"

///|
///
/// ```c
/// lapack_int LAPACKE_dsygvx(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, double* a, lapack_int lda, 
///     double* b, lapack_int ldb, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, double* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_dsygvx(matrix_layout: Int, itype: Lapack_int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsygvx"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyrfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const float* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const float* b, lapack_int ldb, float* x, 
///     lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_ssyrfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssyrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyrfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const double* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const double* b, lapack_int ldb, double* x, 
///     lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dsyrfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsyrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_csyrfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_csyrfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_csyrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_zsyrfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, lapack_int ldaf, 
///     const lapack_int* ipiv, const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_zsyrfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_zsyrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_ssyrfsx(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const float* a, lapack_int lda, const float* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const float* s, const float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* rcond, 
///     float* berr, lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, 
///     lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_ssyrfsx(matrix_layout: Int, uplo: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssyrfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_dsyrfsx(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const double* a, lapack_int lda, const double* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const double* s, const double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* rcond, 
///     double* berr, lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, 
///     lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_dsyrfsx(matrix_layout: Int, uplo: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsyrfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_csyrfsx(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const float* s, const lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond, 
///     float* berr, lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, 
///     lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_csyrfsx(matrix_layout: Int, uplo: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_csyrfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_zsyrfsx(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, 
///     lapack_int ldaf, const lapack_int* ipiv, const double* s, const lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond, 
///     double* berr, lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, 
///     lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_zsyrfsx(matrix_layout: Int, uplo: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_zsyrfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_ssysv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_ssysv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_ssysv"

///|
///
/// ```c
/// lapack_int LAPACKE_dsysv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dsysv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dsysv"

///|
///
/// ```c
/// lapack_int LAPACKE_csysv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_csysv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_csysv"

///|
///
/// ```c
/// lapack_int LAPACKE_zsysv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zsysv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zsysv"

///|
///
/// ```c
/// lapack_int LAPACKE_ssysvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const float* a, lapack_int lda, float* af, 
///     lapack_int ldaf, lapack_int* ipiv, const float* b, lapack_int ldb, 
///     float* x, lapack_int ldx, float* rcond, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_ssysvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssysvx"

///|
///
/// ```c
/// lapack_int LAPACKE_dsysvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const double* a, lapack_int lda, double* af, 
///     lapack_int ldaf, lapack_int* ipiv, const double* b, lapack_int ldb, 
///     double* x, lapack_int ldx, double* rcond, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dsysvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsysvx"

///|
///
/// ```c
/// lapack_int LAPACKE_csysvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, lapack_int* ipiv, const lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* x, lapack_int ldx, float* rcond, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_csysvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_csysvx"

///|
///
/// ```c
/// lapack_int LAPACKE_zsysvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, lapack_int* ipiv, const lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* x, lapack_int ldx, double* rcond, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_zsysvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_zsysvx"

///|
///
/// ```c
/// lapack_int LAPACKE_ssysvxx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, float* a, lapack_int lda, float* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, float* s, 
///     float* b, lapack_int ldb, float* x, lapack_int ldx, 
///     float* rcond, float* rpvgrw, float* berr, lapack_int n_err_bnds, 
///     float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_ssysvxx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], rpvgrw: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssysvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_dsysvxx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, double* a, lapack_int lda, double* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, double* s, 
///     double* b, lapack_int ldb, double* x, lapack_int ldx, 
///     double* rcond, double* rpvgrw, double* berr, lapack_int n_err_bnds, 
///     double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_dsysvxx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], rpvgrw: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsysvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_csysvxx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, float* s, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* rcond, float* rpvgrw, float* berr, lapack_int n_err_bnds, 
///     float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_csysvxx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], rpvgrw: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_csysvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_zsysvxx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, lapack_int* ipiv, char* equed, double* s, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* rcond, double* rpvgrw, double* berr, lapack_int n_err_bnds, 
///     double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_zsysvxx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, ipiv: FixedArray[Int], equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], rpvgrw: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_zsysvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrd(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, float* d, float* e, float* tau
/// );
/// ```
pub extern "C" fn lapacke_ssytrd(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssytrd"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrd(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, double* d, double* e, double* tau
/// );
/// ```
pub extern "C" fn lapacke_dsytrd(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsytrd"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrf(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_ssytrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_ssytrf"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrf(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_dsytrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsytrf"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_csytrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_csytrf"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_zsytrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_zsytrf"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytri(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, const lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_ssytri(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_ssytri"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytri(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, const lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_dsytri(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsytri"

///|
///
/// ```c
/// lapack_int LAPACKE_csytri(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_csytri(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_csytri"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytri(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_zsytri(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_zsytri"

///|
///
/// ```c
/// lapack_int LAPACKE_ssytrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_ssytrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_ssytrs"

///|
///
/// ```c
/// lapack_int LAPACKE_dsytrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dsytrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dsytrs"

///|
///
/// ```c
/// lapack_int LAPACKE_csytrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_csytrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_csytrs"

///|
///
/// ```c
/// lapack_int LAPACKE_zsytrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zsytrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zsytrs"

///|
///
/// ```c
/// lapack_int LAPACKE_stbcon(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, lapack_int kd, const float* ab, lapack_int ldab, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_stbcon(matrix_layout: Int, norm: Byte, uplo: Byte, diag: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_stbcon"

///|
///
/// ```c
/// lapack_int LAPACKE_dtbcon(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, lapack_int kd, const double* ab, lapack_int ldab, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_dtbcon(matrix_layout: Int, norm: Byte, uplo: Byte, diag: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtbcon"

///|
///
/// ```c
/// lapack_int LAPACKE_ctbcon(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, lapack_int kd, const lapack_complex_float* ab, lapack_int ldab, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_ctbcon(matrix_layout: Int, norm: Byte, uplo: Byte, diag: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctbcon"

///|
///
/// ```c
/// lapack_int LAPACKE_ztbcon(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, lapack_int kd, const lapack_complex_double* ab, lapack_int ldab, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_ztbcon(matrix_layout: Int, norm: Byte, uplo: Byte, diag: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztbcon"

///|
///
/// ```c
/// lapack_int LAPACKE_stbrfs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const float* ab, 
///     lapack_int ldab, const float* b, lapack_int ldb, const float* x, 
///     lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_stbrfs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_stbrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_dtbrfs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const double* ab, 
///     lapack_int ldab, const double* b, lapack_int ldb, const double* x, 
///     lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dtbrfs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtbrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_ctbrfs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const lapack_complex_float* ab, 
///     lapack_int ldab, const lapack_complex_float* b, lapack_int ldb, const lapack_complex_float* x, 
///     lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_ctbrfs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctbrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_ztbrfs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const lapack_complex_double* ab, 
///     lapack_int ldab, const lapack_complex_double* b, lapack_int ldb, const lapack_complex_double* x, 
///     lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_ztbrfs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztbrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_stbtrs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const float* ab, 
///     lapack_int ldab, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_stbtrs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_stbtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_dtbtrs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const double* ab, 
///     lapack_int ldab, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dtbtrs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dtbtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_ctbtrs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const lapack_complex_float* ab, 
///     lapack_int ldab, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_ctbtrs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_ctbtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_ztbtrs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int kd, lapack_int nrhs, const lapack_complex_double* ab, 
///     lapack_int ldab, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_ztbtrs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_ztbtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_stfsm(
///     int matrix_layout, char transr, char side, char uplo, 
///     char trans, char diag, lapack_int m, lapack_int n, 
///     float alpha, const float* a, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_stfsm(matrix_layout: Int, transr: Byte, side: Byte, uplo: Byte, trans: Byte, diag: Byte, m: Lapack_int, n: Lapack_int, alpha: Float, a: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_stfsm"

///|
///
/// ```c
/// lapack_int LAPACKE_dtfsm(
///     int matrix_layout, char transr, char side, char uplo, 
///     char trans, char diag, lapack_int m, lapack_int n, 
///     double alpha, const double* a, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dtfsm(matrix_layout: Int, transr: Byte, side: Byte, uplo: Byte, trans: Byte, diag: Byte, m: Lapack_int, n: Lapack_int, alpha: Double, a: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dtfsm"

///|
///
/// ```c
/// lapack_int LAPACKE_ctfsm(
///     int matrix_layout, char transr, char side, char uplo, 
///     char trans, char diag, lapack_int m, lapack_int n, 
///     lapack_complex_float alpha, const lapack_complex_float* a, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_ctfsm(matrix_layout: Int, transr: Byte, side: Byte, uplo: Byte, trans: Byte, diag: Byte, m: Lapack_int, n: Lapack_int, alpha: Lapack_complex_float, a: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_ctfsm"

///|
///
/// ```c
/// lapack_int LAPACKE_ztfsm(
///     int matrix_layout, char transr, char side, char uplo, 
///     char trans, char diag, lapack_int m, lapack_int n, 
///     lapack_complex_double alpha, const lapack_complex_double* a, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_ztfsm(matrix_layout: Int, transr: Byte, side: Byte, uplo: Byte, trans: Byte, diag: Byte, m: Lapack_int, n: Lapack_int, alpha: Lapack_complex_double, a: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_ztfsm"

///|
///
/// ```c
/// lapack_int LAPACKE_stftri(
///     int matrix_layout, char transr, char uplo, char diag, 
///     lapack_int n, float* a
/// );
/// ```
pub extern "C" fn lapacke_stftri(matrix_layout: Int, transr: Byte, uplo: Byte, diag: Byte, n: Lapack_int, a: FixedArray[Float]) -> Lapack_int = "LAPACKE_stftri"

///|
///
/// ```c
/// lapack_int LAPACKE_dtftri(
///     int matrix_layout, char transr, char uplo, char diag, 
///     lapack_int n, double* a
/// );
/// ```
pub extern "C" fn lapacke_dtftri(matrix_layout: Int, transr: Byte, uplo: Byte, diag: Byte, n: Lapack_int, a: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtftri"

///|
///
/// ```c
/// lapack_int LAPACKE_ctftri(
///     int matrix_layout, char transr, char uplo, char diag, 
///     lapack_int n, lapack_complex_float* a
/// );
/// ```
pub extern "C" fn lapacke_ctftri(matrix_layout: Int, transr: Byte, uplo: Byte, diag: Byte, n: Lapack_int, a: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctftri"

///|
///
/// ```c
/// lapack_int LAPACKE_ztftri(
///     int matrix_layout, char transr, char uplo, char diag, 
///     lapack_int n, lapack_complex_double* a
/// );
/// ```
pub extern "C" fn lapacke_ztftri(matrix_layout: Int, transr: Byte, uplo: Byte, diag: Byte, n: Lapack_int, a: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztftri"

///|
///
/// ```c
/// lapack_int LAPACKE_stfttp(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const float* arf, float* ap
/// );
/// ```
pub extern "C" fn lapacke_stfttp(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, arf: FixedArray[Float], ap: FixedArray[Float]) -> Lapack_int = "LAPACKE_stfttp"

///|
///
/// ```c
/// lapack_int LAPACKE_dtfttp(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const double* arf, double* ap
/// );
/// ```
pub extern "C" fn lapacke_dtfttp(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, arf: FixedArray[Double], ap: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtfttp"

///|
///
/// ```c
/// lapack_int LAPACKE_ctfttp(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_float* arf, lapack_complex_float* ap
/// );
/// ```
pub extern "C" fn lapacke_ctfttp(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, arf: FixedArray[Float], ap: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctfttp"

///|
///
/// ```c
/// lapack_int LAPACKE_ztfttp(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_double* arf, lapack_complex_double* ap
/// );
/// ```
pub extern "C" fn lapacke_ztfttp(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, arf: FixedArray[Double], ap: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztfttp"

///|
///
/// ```c
/// lapack_int LAPACKE_stfttr(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const float* arf, float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_stfttr(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, arf: FixedArray[Float], a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_stfttr"

///|
///
/// ```c
/// lapack_int LAPACKE_dtfttr(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const double* arf, double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_dtfttr(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, arf: FixedArray[Double], a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_dtfttr"

///|
///
/// ```c
/// lapack_int LAPACKE_ctfttr(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_float* arf, lapack_complex_float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_ctfttr(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, arf: FixedArray[Float], a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_ctfttr"

///|
///
/// ```c
/// lapack_int LAPACKE_ztfttr(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_double* arf, lapack_complex_double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_ztfttr(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, arf: FixedArray[Double], a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_ztfttr"

///|
///
/// ```c
/// lapack_int LAPACKE_stgevc(
///     int matrix_layout, char side, char howmny, const lapack_logical* select, 
///     lapack_int n, const float* s, lapack_int lds, const float* p, 
///     lapack_int ldp, float* vl, lapack_int ldvl, float* vr, 
///     lapack_int ldvr, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_stgevc(matrix_layout: Int, side: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, s: FixedArray[Float], lds: Lapack_int, p: FixedArray[Float], ldp: Lapack_int, vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int, mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_stgevc"

///|
///
/// ```c
/// lapack_int LAPACKE_dtgevc(
///     int matrix_layout, char side, char howmny, const lapack_logical* select, 
///     lapack_int n, const double* s, lapack_int lds, const double* p, 
///     lapack_int ldp, double* vl, lapack_int ldvl, double* vr, 
///     lapack_int ldvr, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_dtgevc(matrix_layout: Int, side: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, s: FixedArray[Double], lds: Lapack_int, p: FixedArray[Double], ldp: Lapack_int, vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int, mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_dtgevc"

///|
///
/// ```c
/// lapack_int LAPACKE_ctgevc(
///     int matrix_layout, char side, char howmny, const lapack_logical* select, 
///     lapack_int n, const lapack_complex_float* s, lapack_int lds, const lapack_complex_float* p, 
///     lapack_int ldp, lapack_complex_float* vl, lapack_int ldvl, lapack_complex_float* vr, 
///     lapack_int ldvr, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_ctgevc(matrix_layout: Int, side: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, s: FixedArray[Float], lds: Lapack_int, p: FixedArray[Float], ldp: Lapack_int, vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int, mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_ctgevc"

///|
///
/// ```c
/// lapack_int LAPACKE_ztgevc(
///     int matrix_layout, char side, char howmny, const lapack_logical* select, 
///     lapack_int n, const lapack_complex_double* s, lapack_int lds, const lapack_complex_double* p, 
///     lapack_int ldp, lapack_complex_double* vl, lapack_int ldvl, lapack_complex_double* vr, 
///     lapack_int ldvr, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_ztgevc(matrix_layout: Int, side: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, s: FixedArray[Double], lds: Lapack_int, p: FixedArray[Double], ldp: Lapack_int, vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int, mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_ztgevc"

///|
///
/// ```c
/// lapack_int LAPACKE_stgexc(
///     int matrix_layout, lapack_logical wantq, lapack_logical wantz, lapack_int n, 
///     float* a, lapack_int lda, float* b, lapack_int ldb, 
///     float* q, lapack_int ldq, float* z, lapack_int ldz, 
///     lapack_int* ifst, lapack_int* ilst
/// );
/// ```
pub extern "C" fn lapacke_stgexc(matrix_layout: Int, wantq: Lapack_logical, wantz: Lapack_logical, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, z: FixedArray[Float], ldz: Lapack_int, ifst: FixedArray[Int], ilst: FixedArray[Int]) -> Lapack_int = "LAPACKE_stgexc"

///|
///
/// ```c
/// lapack_int LAPACKE_dtgexc(
///     int matrix_layout, lapack_logical wantq, lapack_logical wantz, lapack_int n, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* q, lapack_int ldq, double* z, lapack_int ldz, 
///     lapack_int* ifst, lapack_int* ilst
/// );
/// ```
pub extern "C" fn lapacke_dtgexc(matrix_layout: Int, wantq: Lapack_logical, wantz: Lapack_logical, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, q: FixedArray[Double], ldq: Lapack_int, z: FixedArray[Double], ldz: Lapack_int, ifst: FixedArray[Int], ilst: FixedArray[Int]) -> Lapack_int = "LAPACKE_dtgexc"

///|
///
/// ```c
/// lapack_int LAPACKE_ctgexc(
///     int matrix_layout, lapack_logical wantq, lapack_logical wantz, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* q, lapack_int ldq, lapack_complex_float* z, lapack_int ldz, 
///     lapack_int ifst, lapack_int ilst
/// );
/// ```
pub extern "C" fn lapacke_ctgexc(matrix_layout: Int, wantq: Lapack_logical, wantz: Lapack_logical, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, z: FixedArray[Float], ldz: Lapack_int, ifst: Lapack_int, ilst: Lapack_int) -> Lapack_int = "LAPACKE_ctgexc"

///|
///
/// ```c
/// lapack_int LAPACKE_ztgexc(
///     int matrix_layout, lapack_logical wantq, lapack_logical wantz, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* q, lapack_int ldq, lapack_complex_double* z, lapack_int ldz, 
///     lapack_int ifst, lapack_int ilst
/// );
/// ```
pub extern "C" fn lapacke_ztgexc(matrix_layout: Int, wantq: Lapack_logical, wantz: Lapack_logical, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, q: FixedArray[Double], ldq: Lapack_int, z: FixedArray[Double], ldz: Lapack_int, ifst: Lapack_int, ilst: Lapack_int) -> Lapack_int = "LAPACKE_ztgexc"

///|
///
/// ```c
/// lapack_int LAPACKE_stgsen(
///     int matrix_layout, lapack_int ijob, lapack_logical wantq, lapack_logical wantz, 
///     const lapack_logical* select, lapack_int n, float* a, lapack_int lda, 
///     float* b, lapack_int ldb, float* alphar, float* alphai, 
///     float* beta, float* q, lapack_int ldq, float* z, 
///     lapack_int ldz, lapack_int* m, float* pl, float* pr, float* dif
/// );
/// ```
pub extern "C" fn lapacke_stgsen(matrix_layout: Int, ijob: Lapack_int, wantq: Lapack_logical, wantz: Lapack_logical, select: Lapack_logical, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, alphar: FixedArray[Float], alphai: FixedArray[Float], beta: FixedArray[Float], q: FixedArray[Float], ldq: Lapack_int, z: FixedArray[Float], ldz: Lapack_int, m: FixedArray[Int], pl: FixedArray[Float], pr: FixedArray[Float], dif: FixedArray[Float]) -> Lapack_int = "LAPACKE_stgsen"

///|
///
/// ```c
/// lapack_int LAPACKE_dtgsen(
///     int matrix_layout, lapack_int ijob, lapack_logical wantq, lapack_logical wantz, 
///     const lapack_logical* select, lapack_int n, double* a, lapack_int lda, 
///     double* b, lapack_int ldb, double* alphar, double* alphai, 
///     double* beta, double* q, lapack_int ldq, double* z, 
///     lapack_int ldz, lapack_int* m, double* pl, double* pr, double* dif
/// );
/// ```
pub extern "C" fn lapacke_dtgsen(matrix_layout: Int, ijob: Lapack_int, wantq: Lapack_logical, wantz: Lapack_logical, select: Lapack_logical, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, alphar: FixedArray[Double], alphai: FixedArray[Double], beta: FixedArray[Double], q: FixedArray[Double], ldq: Lapack_int, z: FixedArray[Double], ldz: Lapack_int, m: FixedArray[Int], pl: FixedArray[Double], pr: FixedArray[Double], dif: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtgsen"

///|
///
/// ```c
/// lapack_int LAPACKE_ctgsen(
///     int matrix_layout, lapack_int ijob, lapack_logical wantq, lapack_logical wantz, 
///     const lapack_logical* select, lapack_int n, lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* alpha, lapack_complex_float* beta, 
///     lapack_complex_float* q, lapack_int ldq, lapack_complex_float* z, lapack_int ldz, 
///     lapack_int* m, float* pl, float* pr, float* dif
/// );
/// ```
pub extern "C" fn lapacke_ctgsen(matrix_layout: Int, ijob: Lapack_int, wantq: Lapack_logical, wantz: Lapack_logical, select: Lapack_logical, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, alpha: FixedArray[Float], beta: FixedArray[Float], q: FixedArray[Float], ldq: Lapack_int, z: FixedArray[Float], ldz: Lapack_int, m: FixedArray[Int], pl: FixedArray[Float], pr: FixedArray[Float], dif: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctgsen"

///|
///
/// ```c
/// lapack_int LAPACKE_ztgsen(
///     int matrix_layout, lapack_int ijob, lapack_logical wantq, lapack_logical wantz, 
///     const lapack_logical* select, lapack_int n, lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* alpha, lapack_complex_double* beta, 
///     lapack_complex_double* q, lapack_int ldq, lapack_complex_double* z, lapack_int ldz, 
///     lapack_int* m, double* pl, double* pr, double* dif
/// );
/// ```
pub extern "C" fn lapacke_ztgsen(matrix_layout: Int, ijob: Lapack_int, wantq: Lapack_logical, wantz: Lapack_logical, select: Lapack_logical, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, alpha: FixedArray[Double], beta: FixedArray[Double], q: FixedArray[Double], ldq: Lapack_int, z: FixedArray[Double], ldz: Lapack_int, m: FixedArray[Int], pl: FixedArray[Double], pr: FixedArray[Double], dif: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztgsen"

///|
///
/// ```c
/// lapack_int LAPACKE_stgsja(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_int k, 
///     lapack_int l, float* a, lapack_int lda, float* b, 
///     lapack_int ldb, float tola, float tolb, float* alpha, 
///     float* beta, float* u, lapack_int ldu, float* v, 
///     lapack_int ldv, float* q, lapack_int ldq, lapack_int* ncycle
/// );
/// ```
pub extern "C" fn lapacke_stgsja(matrix_layout: Int, jobu: Byte, jobv: Byte, jobq: Byte, m: Lapack_int, p: Lapack_int, n: Lapack_int, k: Lapack_int, l: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, tola: Float, tolb: Float, alpha: FixedArray[Float], beta: FixedArray[Float], u: FixedArray[Float], ldu: Lapack_int, v: FixedArray[Float], ldv: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, ncycle: FixedArray[Int]) -> Lapack_int = "LAPACKE_stgsja"

///|
///
/// ```c
/// lapack_int LAPACKE_dtgsja(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_int k, 
///     lapack_int l, double* a, lapack_int lda, double* b, 
///     lapack_int ldb, double tola, double tolb, double* alpha, 
///     double* beta, double* u, lapack_int ldu, double* v, 
///     lapack_int ldv, double* q, lapack_int ldq, lapack_int* ncycle
/// );
/// ```
pub extern "C" fn lapacke_dtgsja(matrix_layout: Int, jobu: Byte, jobv: Byte, jobq: Byte, m: Lapack_int, p: Lapack_int, n: Lapack_int, k: Lapack_int, l: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, tola: Double, tolb: Double, alpha: FixedArray[Double], beta: FixedArray[Double], u: FixedArray[Double], ldu: Lapack_int, v: FixedArray[Double], ldv: Lapack_int, q: FixedArray[Double], ldq: Lapack_int, ncycle: FixedArray[Int]) -> Lapack_int = "LAPACKE_dtgsja"

///|
///
/// ```c
/// lapack_int LAPACKE_ctgsja(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_int k, 
///     lapack_int l, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, 
///     lapack_int ldb, float tola, float tolb, float* alpha, 
///     float* beta, lapack_complex_float* u, lapack_int ldu, lapack_complex_float* v, 
///     lapack_int ldv, lapack_complex_float* q, lapack_int ldq, lapack_int* ncycle
/// );
/// ```
pub extern "C" fn lapacke_ctgsja(matrix_layout: Int, jobu: Byte, jobv: Byte, jobq: Byte, m: Lapack_int, p: Lapack_int, n: Lapack_int, k: Lapack_int, l: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, tola: Float, tolb: Float, alpha: FixedArray[Float], beta: FixedArray[Float], u: FixedArray[Float], ldu: Lapack_int, v: FixedArray[Float], ldv: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, ncycle: FixedArray[Int]) -> Lapack_int = "LAPACKE_ctgsja"

///|
///
/// ```c
/// lapack_int LAPACKE_ztgsja(
///     int matrix_layout, char jobu, char jobv, char jobq, 
///     lapack_int m, lapack_int p, lapack_int n, lapack_int k, 
///     lapack_int l, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, 
///     lapack_int ldb, double tola, double tolb, double* alpha, 
///     double* beta, lapack_complex_double* u, lapack_int ldu, lapack_complex_double* v, 
///     lapack_int ldv, lapack_complex_double* q, lapack_int ldq, lapack_int* ncycle
/// );
/// ```
pub extern "C" fn lapacke_ztgsja(matrix_layout: Int, jobu: Byte, jobv: Byte, jobq: Byte, m: Lapack_int, p: Lapack_int, n: Lapack_int, k: Lapack_int, l: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, tola: Double, tolb: Double, alpha: FixedArray[Double], beta: FixedArray[Double], u: FixedArray[Double], ldu: Lapack_int, v: FixedArray[Double], ldv: Lapack_int, q: FixedArray[Double], ldq: Lapack_int, ncycle: FixedArray[Int]) -> Lapack_int = "LAPACKE_ztgsja"

///|
///
/// ```c
/// lapack_int LAPACKE_stgsna(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const float* a, lapack_int lda, const float* b, 
///     lapack_int ldb, const float* vl, lapack_int ldvl, const float* vr, 
///     lapack_int ldvr, float* s, float* dif, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_stgsna(matrix_layout: Int, job: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int, s: FixedArray[Float], dif: FixedArray[Float], mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_stgsna"

///|
///
/// ```c
/// lapack_int LAPACKE_dtgsna(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const double* a, lapack_int lda, const double* b, 
///     lapack_int ldb, const double* vl, lapack_int ldvl, const double* vr, 
///     lapack_int ldvr, double* s, double* dif, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_dtgsna(matrix_layout: Int, job: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int, s: FixedArray[Double], dif: FixedArray[Double], mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_dtgsna"

///|
///
/// ```c
/// lapack_int LAPACKE_ctgsna(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* b, 
///     lapack_int ldb, const lapack_complex_float* vl, lapack_int ldvl, const lapack_complex_float* vr, 
///     lapack_int ldvr, float* s, float* dif, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_ctgsna(matrix_layout: Int, job: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int, s: FixedArray[Float], dif: FixedArray[Float], mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_ctgsna"

///|
///
/// ```c
/// lapack_int LAPACKE_ztgsna(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* b, 
///     lapack_int ldb, const lapack_complex_double* vl, lapack_int ldvl, const lapack_complex_double* vr, 
///     lapack_int ldvr, double* s, double* dif, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_ztgsna(matrix_layout: Int, job: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int, s: FixedArray[Double], dif: FixedArray[Double], mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_ztgsna"

///|
///
/// ```c
/// lapack_int LAPACKE_stgsyl(
///     int matrix_layout, char trans, lapack_int ijob, lapack_int m, 
///     lapack_int n, const float* a, lapack_int lda, const float* b, 
///     lapack_int ldb, float* c, lapack_int ldc, const float* d, 
///     lapack_int ldd, const float* e, lapack_int lde, float* f, 
///     lapack_int ldf, float* scale, float* dif
/// );
/// ```
pub extern "C" fn lapacke_stgsyl(matrix_layout: Int, trans: Byte, ijob: Lapack_int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, c: FixedArray[Float], ldc: Lapack_int, d: FixedArray[Float], ldd: Lapack_int, e: FixedArray[Float], lde: Lapack_int, f: FixedArray[Float], ldf: Lapack_int, scale: FixedArray[Float], dif: FixedArray[Float]) -> Lapack_int = "LAPACKE_stgsyl"

///|
///
/// ```c
/// lapack_int LAPACKE_dtgsyl(
///     int matrix_layout, char trans, lapack_int ijob, lapack_int m, 
///     lapack_int n, const double* a, lapack_int lda, const double* b, 
///     lapack_int ldb, double* c, lapack_int ldc, const double* d, 
///     lapack_int ldd, const double* e, lapack_int lde, double* f, 
///     lapack_int ldf, double* scale, double* dif
/// );
/// ```
pub extern "C" fn lapacke_dtgsyl(matrix_layout: Int, trans: Byte, ijob: Lapack_int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, c: FixedArray[Double], ldc: Lapack_int, d: FixedArray[Double], ldd: Lapack_int, e: FixedArray[Double], lde: Lapack_int, f: FixedArray[Double], ldf: Lapack_int, scale: FixedArray[Double], dif: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtgsyl"

///|
///
/// ```c
/// lapack_int LAPACKE_ctgsyl(
///     int matrix_layout, char trans, lapack_int ijob, lapack_int m, 
///     lapack_int n, const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* c, lapack_int ldc, const lapack_complex_float* d, 
///     lapack_int ldd, const lapack_complex_float* e, lapack_int lde, lapack_complex_float* f, 
///     lapack_int ldf, float* scale, float* dif
/// );
/// ```
pub extern "C" fn lapacke_ctgsyl(matrix_layout: Int, trans: Byte, ijob: Lapack_int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, c: FixedArray[Float], ldc: Lapack_int, d: FixedArray[Float], ldd: Lapack_int, e: FixedArray[Float], lde: Lapack_int, f: FixedArray[Float], ldf: Lapack_int, scale: FixedArray[Float], dif: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctgsyl"

///|
///
/// ```c
/// lapack_int LAPACKE_ztgsyl(
///     int matrix_layout, char trans, lapack_int ijob, lapack_int m, 
///     lapack_int n, const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* c, lapack_int ldc, const lapack_complex_double* d, 
///     lapack_int ldd, const lapack_complex_double* e, lapack_int lde, lapack_complex_double* f, 
///     lapack_int ldf, double* scale, double* dif
/// );
/// ```
pub extern "C" fn lapacke_ztgsyl(matrix_layout: Int, trans: Byte, ijob: Lapack_int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, c: FixedArray[Double], ldc: Lapack_int, d: FixedArray[Double], ldd: Lapack_int, e: FixedArray[Double], lde: Lapack_int, f: FixedArray[Double], ldf: Lapack_int, scale: FixedArray[Double], dif: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztgsyl"

///|
///
/// ```c
/// lapack_int LAPACKE_stpcon(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const float* ap, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_stpcon(matrix_layout: Int, norm: Byte, uplo: Byte, diag: Byte, n: Lapack_int, ap: FixedArray[Float], rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_stpcon"

///|
///
/// ```c
/// lapack_int LAPACKE_dtpcon(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const double* ap, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_dtpcon(matrix_layout: Int, norm: Byte, uplo: Byte, diag: Byte, n: Lapack_int, ap: FixedArray[Double], rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtpcon"

///|
///
/// ```c
/// lapack_int LAPACKE_ctpcon(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const lapack_complex_float* ap, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_ctpcon(matrix_layout: Int, norm: Byte, uplo: Byte, diag: Byte, n: Lapack_int, ap: FixedArray[Float], rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctpcon"

///|
///
/// ```c
/// lapack_int LAPACKE_ztpcon(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const lapack_complex_double* ap, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_ztpcon(matrix_layout: Int, norm: Byte, uplo: Byte, diag: Byte, n: Lapack_int, ap: FixedArray[Double], rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztpcon"

///|
///
/// ```c
/// lapack_int LAPACKE_stprfs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const float* ap, const float* b, 
///     lapack_int ldb, const float* x, lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_stprfs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_stprfs"

///|
///
/// ```c
/// lapack_int LAPACKE_dtprfs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const double* ap, const double* b, 
///     lapack_int ldb, const double* x, lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dtprfs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtprfs"

///|
///
/// ```c
/// lapack_int LAPACKE_ctprfs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_float* ap, const lapack_complex_float* b, 
///     lapack_int ldb, const lapack_complex_float* x, lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_ctprfs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctprfs"

///|
///
/// ```c
/// lapack_int LAPACKE_ztprfs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_double* ap, const lapack_complex_double* b, 
///     lapack_int ldb, const lapack_complex_double* x, lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_ztprfs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztprfs"

///|
///
/// ```c
/// lapack_int LAPACKE_stptri(
///     int matrix_layout, char uplo, char diag, lapack_int n, float* ap
/// );
/// ```
pub extern "C" fn lapacke_stptri(matrix_layout: Int, uplo: Byte, diag: Byte, n: Lapack_int, ap: FixedArray[Float]) -> Lapack_int = "LAPACKE_stptri"

///|
///
/// ```c
/// lapack_int LAPACKE_dtptri(
///     int matrix_layout, char uplo, char diag, lapack_int n, double* ap
/// );
/// ```
pub extern "C" fn lapacke_dtptri(matrix_layout: Int, uplo: Byte, diag: Byte, n: Lapack_int, ap: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtptri"

///|
///
/// ```c
/// lapack_int LAPACKE_ctptri(
///     int matrix_layout, char uplo, char diag, lapack_int n, lapack_complex_float* ap
/// );
/// ```
pub extern "C" fn lapacke_ctptri(matrix_layout: Int, uplo: Byte, diag: Byte, n: Lapack_int, ap: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctptri"

///|
///
/// ```c
/// lapack_int LAPACKE_ztptri(
///     int matrix_layout, char uplo, char diag, lapack_int n, lapack_complex_double* ap
/// );
/// ```
pub extern "C" fn lapacke_ztptri(matrix_layout: Int, uplo: Byte, diag: Byte, n: Lapack_int, ap: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztptri"

///|
///
/// ```c
/// lapack_int LAPACKE_stptrs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const float* ap, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_stptrs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_stptrs"

///|
///
/// ```c
/// lapack_int LAPACKE_dtptrs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const double* ap, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dtptrs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dtptrs"

///|
///
/// ```c
/// lapack_int LAPACKE_ctptrs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_float* ap, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_ctptrs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_ctptrs"

///|
///
/// ```c
/// lapack_int LAPACKE_ztptrs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_double* ap, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_ztptrs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_ztptrs"

///|
///
/// ```c
/// lapack_int LAPACKE_stpttf(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const float* ap, float* arf
/// );
/// ```
pub extern "C" fn lapacke_stpttf(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], arf: FixedArray[Float]) -> Lapack_int = "LAPACKE_stpttf"

///|
///
/// ```c
/// lapack_int LAPACKE_dtpttf(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const double* ap, double* arf
/// );
/// ```
pub extern "C" fn lapacke_dtpttf(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], arf: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtpttf"

///|
///
/// ```c
/// lapack_int LAPACKE_ctpttf(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_float* ap, lapack_complex_float* arf
/// );
/// ```
pub extern "C" fn lapacke_ctpttf(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], arf: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctpttf"

///|
///
/// ```c
/// lapack_int LAPACKE_ztpttf(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_double* ap, lapack_complex_double* arf
/// );
/// ```
pub extern "C" fn lapacke_ztpttf(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], arf: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztpttf"

///|
///
/// ```c
/// lapack_int LAPACKE_stpttr(
///     int matrix_layout, char uplo, lapack_int n, const float* ap, 
///     float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_stpttr(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_stpttr"

///|
///
/// ```c
/// lapack_int LAPACKE_dtpttr(
///     int matrix_layout, char uplo, lapack_int n, const double* ap, 
///     double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_dtpttr(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_dtpttr"

///|
///
/// ```c
/// lapack_int LAPACKE_ctpttr(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* ap, 
///     lapack_complex_float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_ctpttr(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_ctpttr"

///|
///
/// ```c
/// lapack_int LAPACKE_ztpttr(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* ap, 
///     lapack_complex_double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_ztpttr(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_ztpttr"

///|
///
/// ```c
/// lapack_int LAPACKE_strcon(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const float* a, lapack_int lda, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_strcon(matrix_layout: Int, norm: Byte, uplo: Byte, diag: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_strcon"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrcon(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const double* a, lapack_int lda, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_dtrcon(matrix_layout: Int, norm: Byte, uplo: Byte, diag: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtrcon"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrcon(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const lapack_complex_float* a, lapack_int lda, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_ctrcon(matrix_layout: Int, norm: Byte, uplo: Byte, diag: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctrcon"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrcon(
///     int matrix_layout, char norm, char uplo, char diag, 
///     lapack_int n, const lapack_complex_double* a, lapack_int lda, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_ztrcon(matrix_layout: Int, norm: Byte, uplo: Byte, diag: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztrcon"

///|
///
/// ```c
/// lapack_int LAPACKE_strevc(
///     int matrix_layout, char side, char howmny, lapack_logical* select, 
///     lapack_int n, const float* t, lapack_int ldt, float* vl, 
///     lapack_int ldvl, float* vr, lapack_int ldvr, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_strevc(matrix_layout: Int, side: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, t: FixedArray[Float], ldt: Lapack_int, vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int, mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_strevc"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrevc(
///     int matrix_layout, char side, char howmny, lapack_logical* select, 
///     lapack_int n, const double* t, lapack_int ldt, double* vl, 
///     lapack_int ldvl, double* vr, lapack_int ldvr, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_dtrevc(matrix_layout: Int, side: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, t: FixedArray[Double], ldt: Lapack_int, vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int, mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_dtrevc"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrevc(
///     int matrix_layout, char side, char howmny, const lapack_logical* select, 
///     lapack_int n, lapack_complex_float* t, lapack_int ldt, lapack_complex_float* vl, 
///     lapack_int ldvl, lapack_complex_float* vr, lapack_int ldvr, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_ctrevc(matrix_layout: Int, side: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, t: FixedArray[Float], ldt: Lapack_int, vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int, mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_ctrevc"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrevc(
///     int matrix_layout, char side, char howmny, const lapack_logical* select, 
///     lapack_int n, lapack_complex_double* t, lapack_int ldt, lapack_complex_double* vl, 
///     lapack_int ldvl, lapack_complex_double* vr, lapack_int ldvr, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_ztrevc(matrix_layout: Int, side: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, t: FixedArray[Double], ldt: Lapack_int, vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int, mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_ztrevc"

///|
///
/// ```c
/// lapack_int LAPACKE_strexc(
///     int matrix_layout, char compq, lapack_int n, float* t, 
///     lapack_int ldt, float* q, lapack_int ldq, lapack_int* ifst, lapack_int* ilst
/// );
/// ```
pub extern "C" fn lapacke_strexc(matrix_layout: Int, compq: Byte, n: Lapack_int, t: FixedArray[Float], ldt: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, ifst: FixedArray[Int], ilst: FixedArray[Int]) -> Lapack_int = "LAPACKE_strexc"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrexc(
///     int matrix_layout, char compq, lapack_int n, double* t, 
///     lapack_int ldt, double* q, lapack_int ldq, lapack_int* ifst, lapack_int* ilst
/// );
/// ```
pub extern "C" fn lapacke_dtrexc(matrix_layout: Int, compq: Byte, n: Lapack_int, t: FixedArray[Double], ldt: Lapack_int, q: FixedArray[Double], ldq: Lapack_int, ifst: FixedArray[Int], ilst: FixedArray[Int]) -> Lapack_int = "LAPACKE_dtrexc"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrexc(
///     int matrix_layout, char compq, lapack_int n, lapack_complex_float* t, 
///     lapack_int ldt, lapack_complex_float* q, lapack_int ldq, lapack_int ifst, lapack_int ilst
/// );
/// ```
pub extern "C" fn lapacke_ctrexc(matrix_layout: Int, compq: Byte, n: Lapack_int, t: FixedArray[Float], ldt: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, ifst: Lapack_int, ilst: Lapack_int) -> Lapack_int = "LAPACKE_ctrexc"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrexc(
///     int matrix_layout, char compq, lapack_int n, lapack_complex_double* t, 
///     lapack_int ldt, lapack_complex_double* q, lapack_int ldq, lapack_int ifst, lapack_int ilst
/// );
/// ```
pub extern "C" fn lapacke_ztrexc(matrix_layout: Int, compq: Byte, n: Lapack_int, t: FixedArray[Double], ldt: Lapack_int, q: FixedArray[Double], ldq: Lapack_int, ifst: Lapack_int, ilst: Lapack_int) -> Lapack_int = "LAPACKE_ztrexc"

///|
///
/// ```c
/// lapack_int LAPACKE_strrfs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const float* a, lapack_int lda, 
///     const float* b, lapack_int ldb, const float* x, lapack_int ldx, 
///     float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_strrfs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_strrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrrfs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const double* a, lapack_int lda, 
///     const double* b, lapack_int ldb, const double* x, lapack_int ldx, 
///     double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dtrrfs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtrrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrrfs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, 
///     const lapack_complex_float* b, lapack_int ldb, const lapack_complex_float* x, lapack_int ldx, 
///     float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_ctrrfs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctrrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrrfs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* b, lapack_int ldb, const lapack_complex_double* x, lapack_int ldx, 
///     double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_ztrrfs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztrrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_strsen(
///     int matrix_layout, char job, char compq, const lapack_logical* select, 
///     lapack_int n, float* t, lapack_int ldt, float* q, 
///     lapack_int ldq, float* wr, float* wi, lapack_int* m, 
///     float* s, float* sep
/// );
/// ```
pub extern "C" fn lapacke_strsen(matrix_layout: Int, job: Byte, compq: Byte, select: Lapack_logical, n: Lapack_int, t: FixedArray[Float], ldt: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, wr: FixedArray[Float], wi: FixedArray[Float], m: FixedArray[Int], s: FixedArray[Float], sep: FixedArray[Float]) -> Lapack_int = "LAPACKE_strsen"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrsen(
///     int matrix_layout, char job, char compq, const lapack_logical* select, 
///     lapack_int n, double* t, lapack_int ldt, double* q, 
///     lapack_int ldq, double* wr, double* wi, lapack_int* m, 
///     double* s, double* sep
/// );
/// ```
pub extern "C" fn lapacke_dtrsen(matrix_layout: Int, job: Byte, compq: Byte, select: Lapack_logical, n: Lapack_int, t: FixedArray[Double], ldt: Lapack_int, q: FixedArray[Double], ldq: Lapack_int, wr: FixedArray[Double], wi: FixedArray[Double], m: FixedArray[Int], s: FixedArray[Double], sep: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtrsen"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrsen(
///     int matrix_layout, char job, char compq, const lapack_logical* select, 
///     lapack_int n, lapack_complex_float* t, lapack_int ldt, lapack_complex_float* q, 
///     lapack_int ldq, lapack_complex_float* w, lapack_int* m, float* s, float* sep
/// );
/// ```
pub extern "C" fn lapacke_ctrsen(matrix_layout: Int, job: Byte, compq: Byte, select: Lapack_logical, n: Lapack_int, t: FixedArray[Float], ldt: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, w: FixedArray[Float], m: FixedArray[Int], s: FixedArray[Float], sep: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctrsen"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrsen(
///     int matrix_layout, char job, char compq, const lapack_logical* select, 
///     lapack_int n, lapack_complex_double* t, lapack_int ldt, lapack_complex_double* q, 
///     lapack_int ldq, lapack_complex_double* w, lapack_int* m, double* s, double* sep
/// );
/// ```
pub extern "C" fn lapacke_ztrsen(matrix_layout: Int, job: Byte, compq: Byte, select: Lapack_logical, n: Lapack_int, t: FixedArray[Double], ldt: Lapack_int, q: FixedArray[Double], ldq: Lapack_int, w: FixedArray[Double], m: FixedArray[Int], s: FixedArray[Double], sep: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztrsen"

///|
///
/// ```c
/// lapack_int LAPACKE_strsna(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const float* t, lapack_int ldt, const float* vl, 
///     lapack_int ldvl, const float* vr, lapack_int ldvr, float* s, 
///     float* sep, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_strsna(matrix_layout: Int, job: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, t: FixedArray[Float], ldt: Lapack_int, vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int, s: FixedArray[Float], sep: FixedArray[Float], mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_strsna"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrsna(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const double* t, lapack_int ldt, const double* vl, 
///     lapack_int ldvl, const double* vr, lapack_int ldvr, double* s, 
///     double* sep, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_dtrsna(matrix_layout: Int, job: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, t: FixedArray[Double], ldt: Lapack_int, vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int, s: FixedArray[Double], sep: FixedArray[Double], mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_dtrsna"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrsna(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const lapack_complex_float* t, lapack_int ldt, const lapack_complex_float* vl, 
///     lapack_int ldvl, const lapack_complex_float* vr, lapack_int ldvr, float* s, 
///     float* sep, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_ctrsna(matrix_layout: Int, job: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, t: FixedArray[Float], ldt: Lapack_int, vl: FixedArray[Float], ldvl: Lapack_int, vr: FixedArray[Float], ldvr: Lapack_int, s: FixedArray[Float], sep: FixedArray[Float], mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_ctrsna"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrsna(
///     int matrix_layout, char job, char howmny, const lapack_logical* select, 
///     lapack_int n, const lapack_complex_double* t, lapack_int ldt, const lapack_complex_double* vl, 
///     lapack_int ldvl, const lapack_complex_double* vr, lapack_int ldvr, double* s, 
///     double* sep, lapack_int mm, lapack_int* m
/// );
/// ```
pub extern "C" fn lapacke_ztrsna(matrix_layout: Int, job: Byte, howmny: Byte, select: Lapack_logical, n: Lapack_int, t: FixedArray[Double], ldt: Lapack_int, vl: FixedArray[Double], ldvl: Lapack_int, vr: FixedArray[Double], ldvr: Lapack_int, s: FixedArray[Double], sep: FixedArray[Double], mm: Lapack_int, m: FixedArray[Int]) -> Lapack_int = "LAPACKE_ztrsna"

///|
///
/// ```c
/// lapack_int LAPACKE_strsyl(
///     int matrix_layout, char trana, char tranb, lapack_int isgn, 
///     lapack_int m, lapack_int n, const float* a, lapack_int lda, 
///     const float* b, lapack_int ldb, float* c, lapack_int ldc, float* scale
/// );
/// ```
pub extern "C" fn lapacke_strsyl(matrix_layout: Int, trana: Byte, tranb: Byte, isgn: Lapack_int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, c: FixedArray[Float], ldc: Lapack_int, scale: FixedArray[Float]) -> Lapack_int = "LAPACKE_strsyl"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrsyl(
///     int matrix_layout, char trana, char tranb, lapack_int isgn, 
///     lapack_int m, lapack_int n, const double* a, lapack_int lda, 
///     const double* b, lapack_int ldb, double* c, lapack_int ldc, double* scale
/// );
/// ```
pub extern "C" fn lapacke_dtrsyl(matrix_layout: Int, trana: Byte, tranb: Byte, isgn: Lapack_int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, c: FixedArray[Double], ldc: Lapack_int, scale: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtrsyl"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrsyl(
///     int matrix_layout, char trana, char tranb, lapack_int isgn, 
///     lapack_int m, lapack_int n, const lapack_complex_float* a, lapack_int lda, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* c, lapack_int ldc, float* scale
/// );
/// ```
pub extern "C" fn lapacke_ctrsyl(matrix_layout: Int, trana: Byte, tranb: Byte, isgn: Lapack_int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, c: FixedArray[Float], ldc: Lapack_int, scale: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctrsyl"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrsyl(
///     int matrix_layout, char trana, char tranb, lapack_int isgn, 
///     lapack_int m, lapack_int n, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* c, lapack_int ldc, double* scale
/// );
/// ```
pub extern "C" fn lapacke_ztrsyl(matrix_layout: Int, trana: Byte, tranb: Byte, isgn: Lapack_int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, c: FixedArray[Double], ldc: Lapack_int, scale: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztrsyl"

///|
///
/// ```c
/// lapack_int LAPACKE_strsyl3(
///     int matrix_layout, char trana, char tranb, lapack_int isgn, 
///     lapack_int m, lapack_int n, const float* a, lapack_int lda, 
///     const float* b, lapack_int ldb, float* c, lapack_int ldc, float* scale
/// );
/// ```
pub extern "C" fn lapacke_strsyl3(matrix_layout: Int, trana: Byte, tranb: Byte, isgn: Lapack_int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, c: FixedArray[Float], ldc: Lapack_int, scale: FixedArray[Float]) -> Lapack_int = "LAPACKE_strsyl3"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrsyl3(
///     int matrix_layout, char trana, char tranb, lapack_int isgn, 
///     lapack_int m, lapack_int n, const double* a, lapack_int lda, 
///     const double* b, lapack_int ldb, double* c, lapack_int ldc, double* scale
/// );
/// ```
pub extern "C" fn lapacke_dtrsyl3(matrix_layout: Int, trana: Byte, tranb: Byte, isgn: Lapack_int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, c: FixedArray[Double], ldc: Lapack_int, scale: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtrsyl3"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrsyl3(
///     int matrix_layout, char trana, char tranb, lapack_int isgn, 
///     lapack_int m, lapack_int n, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* c, lapack_int ldc, double* scale
/// );
/// ```
pub extern "C" fn lapacke_ztrsyl3(matrix_layout: Int, trana: Byte, tranb: Byte, isgn: Lapack_int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, c: FixedArray[Double], ldc: Lapack_int, scale: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztrsyl3"

///|
///
/// ```c
/// lapack_int LAPACKE_strtri(
///     int matrix_layout, char uplo, char diag, lapack_int n, 
///     float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_strtri(matrix_layout: Int, uplo: Byte, diag: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_strtri"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrtri(
///     int matrix_layout, char uplo, char diag, lapack_int n, 
///     double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_dtrtri(matrix_layout: Int, uplo: Byte, diag: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_dtrtri"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrtri(
///     int matrix_layout, char uplo, char diag, lapack_int n, 
///     lapack_complex_float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_ctrtri(matrix_layout: Int, uplo: Byte, diag: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_ctrtri"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrtri(
///     int matrix_layout, char uplo, char diag, lapack_int n, 
///     lapack_complex_double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_ztrtri(matrix_layout: Int, uplo: Byte, diag: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_ztrtri"

///|
///
/// ```c
/// lapack_int LAPACKE_strtrs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const float* a, lapack_int lda, 
///     float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_strtrs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_strtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrtrs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const double* a, lapack_int lda, 
///     double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dtrtrs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dtrtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrtrs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, 
///     lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_ctrtrs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_ctrtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrtrs(
///     int matrix_layout, char uplo, char trans, char diag, 
///     lapack_int n, lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, 
///     lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_ztrtrs(matrix_layout: Int, uplo: Byte, trans: Byte, diag: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_ztrtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_strttf(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const float* a, lapack_int lda, float* arf
/// );
/// ```
pub extern "C" fn lapacke_strttf(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, arf: FixedArray[Float]) -> Lapack_int = "LAPACKE_strttf"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrttf(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const double* a, lapack_int lda, double* arf
/// );
/// ```
pub extern "C" fn lapacke_dtrttf(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, arf: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtrttf"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrttf(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_float* a, lapack_int lda, lapack_complex_float* arf
/// );
/// ```
pub extern "C" fn lapacke_ctrttf(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, arf: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctrttf"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrttf(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     const lapack_complex_double* a, lapack_int lda, lapack_complex_double* arf
/// );
/// ```
pub extern "C" fn lapacke_ztrttf(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, arf: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztrttf"

///|
///
/// ```c
/// lapack_int LAPACKE_strttp(
///     int matrix_layout, char uplo, lapack_int n, const float* a, 
///     lapack_int lda, float* ap
/// );
/// ```
pub extern "C" fn lapacke_strttp(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ap: FixedArray[Float]) -> Lapack_int = "LAPACKE_strttp"

///|
///
/// ```c
/// lapack_int LAPACKE_dtrttp(
///     int matrix_layout, char uplo, lapack_int n, const double* a, 
///     lapack_int lda, double* ap
/// );
/// ```
pub extern "C" fn lapacke_dtrttp(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ap: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtrttp"

///|
///
/// ```c
/// lapack_int LAPACKE_ctrttp(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* ap
/// );
/// ```
pub extern "C" fn lapacke_ctrttp(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, ap: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctrttp"

///|
///
/// ```c
/// lapack_int LAPACKE_ztrttp(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* ap
/// );
/// ```
pub extern "C" fn lapacke_ztrttp(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, ap: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztrttp"

///|
///
/// ```c
/// lapack_int LAPACKE_stzrzf(
///     int matrix_layout, lapack_int m, lapack_int n, float* a, 
///     lapack_int lda, float* tau
/// );
/// ```
pub extern "C" fn lapacke_stzrzf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_stzrzf"

///|
///
/// ```c
/// lapack_int LAPACKE_dtzrzf(
///     int matrix_layout, lapack_int m, lapack_int n, double* a, 
///     lapack_int lda, double* tau
/// );
/// ```
pub extern "C" fn lapacke_dtzrzf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_dtzrzf"

///|
///
/// ```c
/// lapack_int LAPACKE_ctzrzf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_ctzrzf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_ctzrzf"

///|
///
/// ```c
/// lapack_int LAPACKE_ztzrzf(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_ztzrzf(matrix_layout: Int, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_ztzrzf"

///|
///
/// ```c
/// lapack_int LAPACKE_cungbr(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int k, lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cungbr(matrix_layout: Int, vect: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cungbr"

///|
///
/// ```c
/// lapack_int LAPACKE_zungbr(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int k, lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zungbr(matrix_layout: Int, vect: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zungbr"

///|
///
/// ```c
/// lapack_int LAPACKE_cunghr(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cunghr(matrix_layout: Int, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cunghr"

///|
///
/// ```c
/// lapack_int LAPACKE_zunghr(
///     int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, 
///     lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zunghr(matrix_layout: Int, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zunghr"

///|
///
/// ```c
/// lapack_int LAPACKE_cunglq(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cunglq(matrix_layout: Int, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cunglq"

///|
///
/// ```c
/// lapack_int LAPACKE_zunglq(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zunglq(matrix_layout: Int, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zunglq"

///|
///
/// ```c
/// lapack_int LAPACKE_cungql(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cungql(matrix_layout: Int, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cungql"

///|
///
/// ```c
/// lapack_int LAPACKE_zungql(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zungql(matrix_layout: Int, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zungql"

///|
///
/// ```c
/// lapack_int LAPACKE_cungqr(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cungqr(matrix_layout: Int, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cungqr"

///|
///
/// ```c
/// lapack_int LAPACKE_zungqr(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zungqr(matrix_layout: Int, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zungqr"

///|
///
/// ```c
/// lapack_int LAPACKE_cungrq(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cungrq(matrix_layout: Int, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cungrq"

///|
///
/// ```c
/// lapack_int LAPACKE_zungrq(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int k, 
///     lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zungrq(matrix_layout: Int, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zungrq"

///|
///
/// ```c
/// lapack_int LAPACKE_cungtr(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, const lapack_complex_float* tau
/// );
/// ```
pub extern "C" fn lapacke_cungtr(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_cungtr"

///|
///
/// ```c
/// lapack_int LAPACKE_zungtr(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, const lapack_complex_double* tau
/// );
/// ```
pub extern "C" fn lapacke_zungtr(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_zungtr"

///|
///
/// ```c
/// lapack_int LAPACKE_cungtsqr_row(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int mb, 
///     lapack_int nb, lapack_complex_float* a, lapack_int lda, const lapack_complex_float* t, lapack_int ldt
/// );
/// ```
pub extern "C" fn lapacke_cungtsqr_row(matrix_layout: Int, m: Lapack_int, n: Lapack_int, mb: Lapack_int, nb: Lapack_int, a: FixedArray[Float], lda: Lapack_int, t: FixedArray[Float], ldt: Lapack_int) -> Lapack_int = "LAPACKE_cungtsqr_row"

///|
///
/// ```c
/// lapack_int LAPACKE_zungtsqr_row(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int mb, 
///     lapack_int nb, lapack_complex_double* a, lapack_int lda, const lapack_complex_double* t, lapack_int ldt
/// );
/// ```
pub extern "C" fn lapacke_zungtsqr_row(matrix_layout: Int, m: Lapack_int, n: Lapack_int, mb: Lapack_int, nb: Lapack_int, a: FixedArray[Double], lda: Lapack_int, t: FixedArray[Double], ldt: Lapack_int) -> Lapack_int = "LAPACKE_zungtsqr_row"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmbr(
///     int matrix_layout, char vect, char side, char trans, 
///     lapack_int m, lapack_int n, lapack_int k, const lapack_complex_float* a, 
///     lapack_int lda, const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_cunmbr(matrix_layout: Int, vect: Byte, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_cunmbr"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmbr(
///     int matrix_layout, char vect, char side, char trans, 
///     lapack_int m, lapack_int n, lapack_int k, const lapack_complex_double* a, 
///     lapack_int lda, const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_zunmbr(matrix_layout: Int, vect: Byte, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_zunmbr"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmhr(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, const lapack_complex_float* a, 
///     lapack_int lda, const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_cunmhr(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_cunmhr"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmhr(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, const lapack_complex_double* a, 
///     lapack_int lda, const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_zunmhr(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_zunmhr"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmlq(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_float* a, lapack_int lda, 
///     const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_cunmlq(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_cunmlq"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmlq(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_zunmlq(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_zunmlq"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmql(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_float* a, lapack_int lda, 
///     const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_cunmql(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_cunmql"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmql(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_zunmql(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_zunmql"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmqr(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_float* a, lapack_int lda, 
///     const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_cunmqr(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_cunmqr"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmqr(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_zunmqr(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_zunmqr"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmrq(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_float* a, lapack_int lda, 
///     const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_cunmrq(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_cunmrq"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmrq(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_zunmrq(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_zunmrq"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmrz(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, const lapack_complex_float* a, 
///     lapack_int lda, const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_cunmrz(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, l: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_cunmrz"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmrz(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, const lapack_complex_double* a, 
///     lapack_int lda, const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_zunmrz(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, l: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_zunmrz"

///|
///
/// ```c
/// lapack_int LAPACKE_cunmtr(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const lapack_complex_float* a, lapack_int lda, 
///     const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_cunmtr(matrix_layout: Int, side: Byte, uplo: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_cunmtr"

///|
///
/// ```c
/// lapack_int LAPACKE_zunmtr(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const lapack_complex_double* a, lapack_int lda, 
///     const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_zunmtr(matrix_layout: Int, side: Byte, uplo: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_zunmtr"

///|
///
/// ```c
/// lapack_int LAPACKE_cupgtr(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* ap, 
///     const lapack_complex_float* tau, lapack_complex_float* q, lapack_int ldq
/// );
/// ```
pub extern "C" fn lapacke_cupgtr(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], tau: FixedArray[Float], q: FixedArray[Float], ldq: Lapack_int) -> Lapack_int = "LAPACKE_cupgtr"

///|
///
/// ```c
/// lapack_int LAPACKE_zupgtr(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* ap, 
///     const lapack_complex_double* tau, lapack_complex_double* q, lapack_int ldq
/// );
/// ```
pub extern "C" fn lapacke_zupgtr(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], tau: FixedArray[Double], q: FixedArray[Double], ldq: Lapack_int) -> Lapack_int = "LAPACKE_zupgtr"

///|
///
/// ```c
/// lapack_int LAPACKE_cupmtr(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const lapack_complex_float* ap, const lapack_complex_float* tau, 
///     lapack_complex_float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_cupmtr(matrix_layout: Int, side: Byte, uplo: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, ap: FixedArray[Float], tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_cupmtr"

///|
///
/// ```c
/// lapack_int LAPACKE_zupmtr(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const lapack_complex_double* ap, const lapack_complex_double* tau, 
///     lapack_complex_double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_zupmtr(matrix_layout: Int, side: Byte, uplo: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, ap: FixedArray[Double], tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_zupmtr"

///|
///
/// ```c
/// lapack_int LAPACKE_sbdsdc_work(
///     int matrix_layout, char uplo, char compq, lapack_int n, 
///     float* d, float* e, float* u, lapack_int ldu, 
///     float* vt, lapack_int ldvt, float* q, lapack_int* iq, 
///     float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sbdsdc_work(matrix_layout: Int, uplo: Byte, compq: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], u: FixedArray[Float], ldu: Lapack_int, vt: FixedArray[Float], ldvt: Lapack_int, q: FixedArray[Float], iq: FixedArray[Int], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_sbdsdc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dbdsdc_work(
///     int matrix_layout, char uplo, char compq, lapack_int n, 
///     double* d, double* e, double* u, lapack_int ldu, 
///     double* vt, lapack_int ldvt, double* q, lapack_int* iq, 
///     double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dbdsdc_work(matrix_layout: Int, uplo: Byte, compq: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], u: FixedArray[Double], ldu: Lapack_int, vt: FixedArray[Double], ldvt: Lapack_int, q: FixedArray[Double], iq: FixedArray[Int], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dbdsdc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sbdsvdx_work(
///     int matrix_layout, char uplo, char jobz, char range, 
///     lapack_int n, float* d, float* e, float vl, 
///     float vu, lapack_int il, lapack_int iu, lapack_int* ns, 
///     float* s, float* z, lapack_int ldz, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sbdsvdx_work(matrix_layout: Int, uplo: Byte, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, ns: FixedArray[Int], s: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_sbdsvdx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dbdsvdx_work(
///     int matrix_layout, char uplo, char jobz, char range, 
///     lapack_int n, double* d, double* e, double vl, 
///     double vu, lapack_int il, lapack_int iu, lapack_int* ns, 
///     double* s, double* z, lapack_int ldz, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dbdsvdx_work(matrix_layout: Int, uplo: Byte, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, ns: FixedArray[Int], s: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dbdsvdx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sbdsqr_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int ncvt, 
///     lapack_int nru, lapack_int ncc, float* d, float* e, 
///     float* vt, lapack_int ldvt, float* u, lapack_int ldu, 
///     float* c, lapack_int ldc, float* work
/// );
/// ```
pub extern "C" fn lapacke_sbdsqr_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ncvt: Lapack_int, nru: Lapack_int, ncc: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vt: FixedArray[Float], ldvt: Lapack_int, u: FixedArray[Float], ldu: Lapack_int, c: FixedArray[Float], ldc: Lapack_int, work: FixedArray[Float]) -> Lapack_int = "LAPACKE_sbdsqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dbdsqr_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int ncvt, 
///     lapack_int nru, lapack_int ncc, double* d, double* e, 
///     double* vt, lapack_int ldvt, double* u, lapack_int ldu, 
///     double* c, lapack_int ldc, double* work
/// );
/// ```
pub extern "C" fn lapacke_dbdsqr_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ncvt: Lapack_int, nru: Lapack_int, ncc: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vt: FixedArray[Double], ldvt: Lapack_int, u: FixedArray[Double], ldu: Lapack_int, c: FixedArray[Double], ldc: Lapack_int, work: FixedArray[Double]) -> Lapack_int = "LAPACKE_dbdsqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cbdsqr_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int ncvt, 
///     lapack_int nru, lapack_int ncc, float* d, float* e, 
///     lapack_complex_float* vt, lapack_int ldvt, lapack_complex_float* u, lapack_int ldu, 
///     lapack_complex_float* c, lapack_int ldc, float* work
/// );
/// ```
pub extern "C" fn lapacke_cbdsqr_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ncvt: Lapack_int, nru: Lapack_int, ncc: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vt: FixedArray[Float], ldvt: Lapack_int, u: FixedArray[Float], ldu: Lapack_int, c: FixedArray[Float], ldc: Lapack_int, work: FixedArray[Float]) -> Lapack_int = "LAPACKE_cbdsqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zbdsqr_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int ncvt, 
///     lapack_int nru, lapack_int ncc, double* d, double* e, 
///     lapack_complex_double* vt, lapack_int ldvt, lapack_complex_double* u, lapack_int ldu, 
///     lapack_complex_double* c, lapack_int ldc, double* work
/// );
/// ```
pub extern "C" fn lapacke_zbdsqr_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ncvt: Lapack_int, nru: Lapack_int, ncc: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vt: FixedArray[Double], ldvt: Lapack_int, u: FixedArray[Double], ldu: Lapack_int, c: FixedArray[Double], ldc: Lapack_int, work: FixedArray[Double]) -> Lapack_int = "LAPACKE_zbdsqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sdisna_work(
///     char job, lapack_int m, lapack_int n, const float* d, float* sep
/// );
/// ```
pub extern "C" fn lapacke_sdisna_work(job: Byte, m: Lapack_int, n: Lapack_int, d: FixedArray[Float], sep: FixedArray[Float]) -> Lapack_int = "LAPACKE_sdisna_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ddisna_work(
///     char job, lapack_int m, lapack_int n, const double* d, double* sep
/// );
/// ```
pub extern "C" fn lapacke_ddisna_work(job: Byte, m: Lapack_int, n: Lapack_int, d: FixedArray[Double], sep: FixedArray[Double]) -> Lapack_int = "LAPACKE_ddisna_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbbrd_work(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int ncc, lapack_int kl, lapack_int ku, float* ab, 
///     lapack_int ldab, float* d, float* e, float* q, 
///     lapack_int ldq, float* pt, lapack_int ldpt, float* c, 
///     lapack_int ldc, float* work
/// );
/// ```
pub extern "C" fn lapacke_sgbbrd_work(matrix_layout: Int, vect: Byte, m: Lapack_int, n: Lapack_int, ncc: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], q: FixedArray[Float], ldq: Lapack_int, pt: FixedArray[Float], ldpt: Lapack_int, c: FixedArray[Float], ldc: Lapack_int, work: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgbbrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbbrd_work(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int ncc, lapack_int kl, lapack_int ku, double* ab, 
///     lapack_int ldab, double* d, double* e, double* q, 
///     lapack_int ldq, double* pt, lapack_int ldpt, double* c, 
///     lapack_int ldc, double* work
/// );
/// ```
pub extern "C" fn lapacke_dgbbrd_work(matrix_layout: Int, vect: Byte, m: Lapack_int, n: Lapack_int, ncc: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], q: FixedArray[Double], ldq: Lapack_int, pt: FixedArray[Double], ldpt: Lapack_int, c: FixedArray[Double], ldc: Lapack_int, work: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgbbrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbbrd_work(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int ncc, lapack_int kl, lapack_int ku, lapack_complex_float* ab, 
///     lapack_int ldab, float* d, float* e, lapack_complex_float* q, 
///     lapack_int ldq, lapack_complex_float* pt, lapack_int ldpt, lapack_complex_float* c, 
///     lapack_int ldc, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cgbbrd_work(matrix_layout: Int, vect: Byte, m: Lapack_int, n: Lapack_int, ncc: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], q: FixedArray[Float], ldq: Lapack_int, pt: FixedArray[Float], ldpt: Lapack_int, c: FixedArray[Float], ldc: Lapack_int, work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgbbrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbbrd_work(
///     int matrix_layout, char vect, lapack_int m, lapack_int n, 
///     lapack_int ncc, lapack_int kl, lapack_int ku, lapack_complex_double* ab, 
///     lapack_int ldab, double* d, double* e, lapack_complex_double* q, 
///     lapack_int ldq, lapack_complex_double* pt, lapack_int ldpt, lapack_complex_double* c, 
///     lapack_int ldc, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zgbbrd_work(matrix_layout: Int, vect: Byte, m: Lapack_int, n: Lapack_int, ncc: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], q: FixedArray[Double], ldq: Lapack_int, pt: FixedArray[Double], ldpt: Lapack_int, c: FixedArray[Double], ldc: Lapack_int, work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgbbrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbcon_work(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const float* ab, lapack_int ldab, const lapack_int* ipiv, 
///     float anorm, float* rcond, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sgbcon_work(matrix_layout: Int, norm: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, ipiv: FixedArray[Int], anorm: Float, rcond: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_sgbcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbcon_work(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const double* ab, lapack_int ldab, const lapack_int* ipiv, 
///     double anorm, double* rcond, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dgbcon_work(matrix_layout: Int, norm: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, ipiv: FixedArray[Int], anorm: Double, rcond: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dgbcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbcon_work(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_float* ab, lapack_int ldab, const lapack_int* ipiv, 
///     float anorm, float* rcond, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cgbcon_work(matrix_layout: Int, norm: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, ipiv: FixedArray[Int], anorm: Float, rcond: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgbcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbcon_work(
///     int matrix_layout, char norm, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_double* ab, lapack_int ldab, const lapack_int* ipiv, 
///     double anorm, double* rcond, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zgbcon_work(matrix_layout: Int, norm: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, ipiv: FixedArray[Int], anorm: Double, rcond: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgbcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbequ_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const float* ab, lapack_int ldab, float* r, 
///     float* c, float* rowcnd, float* colcnd, float* amax
/// );
/// ```
pub extern "C" fn lapacke_sgbequ_work(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, r: FixedArray[Float], c: FixedArray[Float], rowcnd: FixedArray[Float], colcnd: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgbequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbequ_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const double* ab, lapack_int ldab, double* r, 
///     double* c, double* rowcnd, double* colcnd, double* amax
/// );
/// ```
pub extern "C" fn lapacke_dgbequ_work(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, r: FixedArray[Double], c: FixedArray[Double], rowcnd: FixedArray[Double], colcnd: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgbequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbequ_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_float* ab, lapack_int ldab, float* r, 
///     float* c, float* rowcnd, float* colcnd, float* amax
/// );
/// ```
pub extern "C" fn lapacke_cgbequ_work(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, r: FixedArray[Float], c: FixedArray[Float], rowcnd: FixedArray[Float], colcnd: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgbequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbequ_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_double* ab, lapack_int ldab, double* r, 
///     double* c, double* rowcnd, double* colcnd, double* amax
/// );
/// ```
pub extern "C" fn lapacke_zgbequ_work(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, r: FixedArray[Double], c: FixedArray[Double], rowcnd: FixedArray[Double], colcnd: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgbequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbequb_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const float* ab, lapack_int ldab, float* r, 
///     float* c, float* rowcnd, float* colcnd, float* amax
/// );
/// ```
pub extern "C" fn lapacke_sgbequb_work(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, r: FixedArray[Float], c: FixedArray[Float], rowcnd: FixedArray[Float], colcnd: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_sgbequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbequb_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const double* ab, lapack_int ldab, double* r, 
///     double* c, double* rowcnd, double* colcnd, double* amax
/// );
/// ```
pub extern "C" fn lapacke_dgbequb_work(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, r: FixedArray[Double], c: FixedArray[Double], rowcnd: FixedArray[Double], colcnd: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_dgbequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbequb_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_float* ab, lapack_int ldab, float* r, 
///     float* c, float* rowcnd, float* colcnd, float* amax
/// );
/// ```
pub extern "C" fn lapacke_cgbequb_work(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, r: FixedArray[Float], c: FixedArray[Float], rowcnd: FixedArray[Float], colcnd: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgbequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbequb_work(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, 
///     lapack_int ku, const lapack_complex_double* ab, lapack_int ldab, double* r, 
///     double* c, double* rowcnd, double* colcnd, double* amax
/// );
/// ```
pub extern "C" fn lapacke_zgbequb_work(matrix_layout: Int, m: Lapack_int, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, r: FixedArray[Double], c: FixedArray[Double], rowcnd: FixedArray[Double], colcnd: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgbequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sgbrfs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const float* ab, lapack_int ldab, 
///     const float* afb, lapack_int ldafb, const lapack_int* ipiv, const float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* ferr, 
///     float* berr, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sgbrfs_work(matrix_layout: Int, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_sgbrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dgbrfs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const double* ab, lapack_int ldab, 
///     const double* afb, lapack_int ldafb, const lapack_int* ipiv, const double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* ferr, 
///     double* berr, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dgbrfs_work(matrix_layout: Int, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dgbrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cgbrfs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const lapack_complex_float* ab, lapack_int ldab, 
///     const lapack_complex_float* afb, lapack_int ldafb, const lapack_int* ipiv, const lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* ferr, 
///     float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cgbrfs_work(matrix_layout: Int, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cgbrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zgbrfs_work(
///     int matrix_layout, char trans, lapack_int n, lapack_int kl, 
///     lapack_int ku, lapack_int nrhs, const lapack_complex_double* ab, lapack_int ldab, 
///     const lapack_complex_double* afb, lapack_int ldafb, const lapack_int* ipiv, const lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* ferr, 
///     double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zgbrfs_work(matrix_layout: Int, trans: Byte, n: Lapack_int, kl: Lapack_int, ku: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zgbrfs_work"

