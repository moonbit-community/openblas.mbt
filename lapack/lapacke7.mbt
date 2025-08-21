///|
///
/// ```c
/// lapack_int LAPACKE_sormbr_work(
///     int matrix_layout, char vect, char side, char trans, 
///     lapack_int m, lapack_int n, lapack_int k, const float* a, 
///     lapack_int lda, const float* tau, float* c, lapack_int ldc, 
///     float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sormbr_work(matrix_layout: Int, vect: Byte, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int, work: FixedArray[Float], lwork: Lapack_int) -> Lapack_int = "LAPACKE_sormbr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dormbr_work(
///     int matrix_layout, char vect, char side, char trans, 
///     lapack_int m, lapack_int n, lapack_int k, const double* a, 
///     lapack_int lda, const double* tau, double* c, lapack_int ldc, 
///     double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dormbr_work(matrix_layout: Int, vect: Byte, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int, work: FixedArray[Double], lwork: Lapack_int) -> Lapack_int = "LAPACKE_dormbr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sormhr_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, const float* a, 
///     lapack_int lda, const float* tau, float* c, lapack_int ldc, 
///     float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sormhr_work(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int, work: FixedArray[Float], lwork: Lapack_int) -> Lapack_int = "LAPACKE_sormhr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dormhr_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, const double* a, 
///     lapack_int lda, const double* tau, double* c, lapack_int ldc, 
///     double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dormhr_work(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int, work: FixedArray[Double], lwork: Lapack_int) -> Lapack_int = "LAPACKE_dormhr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sormlq_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const float* a, lapack_int lda, 
///     const float* tau, float* c, lapack_int ldc, float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sormlq_work(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int, work: FixedArray[Float], lwork: Lapack_int) -> Lapack_int = "LAPACKE_sormlq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dormlq_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const double* a, lapack_int lda, 
///     const double* tau, double* c, lapack_int ldc, double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dormlq_work(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int, work: FixedArray[Double], lwork: Lapack_int) -> Lapack_int = "LAPACKE_dormlq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sormql_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const float* a, lapack_int lda, 
///     const float* tau, float* c, lapack_int ldc, float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sormql_work(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int, work: FixedArray[Float], lwork: Lapack_int) -> Lapack_int = "LAPACKE_sormql_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dormql_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const double* a, lapack_int lda, 
///     const double* tau, double* c, lapack_int ldc, double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dormql_work(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int, work: FixedArray[Double], lwork: Lapack_int) -> Lapack_int = "LAPACKE_dormql_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sormqr_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const float* a, lapack_int lda, 
///     const float* tau, float* c, lapack_int ldc, float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sormqr_work(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int, work: FixedArray[Float], lwork: Lapack_int) -> Lapack_int = "LAPACKE_sormqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dormqr_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const double* a, lapack_int lda, 
///     const double* tau, double* c, lapack_int ldc, double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dormqr_work(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int, work: FixedArray[Double], lwork: Lapack_int) -> Lapack_int = "LAPACKE_dormqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sormrq_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const float* a, lapack_int lda, 
///     const float* tau, float* c, lapack_int ldc, float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sormrq_work(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int, work: FixedArray[Float], lwork: Lapack_int) -> Lapack_int = "LAPACKE_sormrq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dormrq_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const double* a, lapack_int lda, 
///     const double* tau, double* c, lapack_int ldc, double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dormrq_work(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int, work: FixedArray[Double], lwork: Lapack_int) -> Lapack_int = "LAPACKE_dormrq_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sormrz_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, const float* a, 
///     lapack_int lda, const float* tau, float* c, lapack_int ldc, 
///     float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sormrz_work(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, l: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int, work: FixedArray[Float], lwork: Lapack_int) -> Lapack_int = "LAPACKE_sormrz_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dormrz_work(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, const double* a, 
///     lapack_int lda, const double* tau, double* c, lapack_int ldc, 
///     double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dormrz_work(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, l: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int, work: FixedArray[Double], lwork: Lapack_int) -> Lapack_int = "LAPACKE_dormrz_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sormtr_work(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const float* a, lapack_int lda, 
///     const float* tau, float* c, lapack_int ldc, float* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_sormtr_work(matrix_layout: Int, side: Byte, uplo: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int, work: FixedArray[Float], lwork: Lapack_int) -> Lapack_int = "LAPACKE_sormtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dormtr_work(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const double* a, lapack_int lda, 
///     const double* tau, double* c, lapack_int ldc, double* work, lapack_int lwork
/// );
/// ```
pub extern "C" fn lapacke_dormtr_work(matrix_layout: Int, side: Byte, uplo: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int, work: FixedArray[Double], lwork: Lapack_int) -> Lapack_int = "LAPACKE_dormtr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spbcon_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const float* ab, lapack_int ldab, float anorm, float* rcond, 
///     float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_spbcon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, anorm: Float, rcond: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_spbcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbcon_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const double* ab, lapack_int ldab, double anorm, double* rcond, 
///     double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dpbcon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, anorm: Double, rcond: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dpbcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbcon_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const lapack_complex_float* ab, lapack_int ldab, float anorm, float* rcond, 
///     lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cpbcon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, anorm: Float, rcond: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpbcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbcon_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const lapack_complex_double* ab, lapack_int ldab, double anorm, double* rcond, 
///     lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zpbcon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, anorm: Double, rcond: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpbcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spbequ_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const float* ab, lapack_int ldab, float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_spbequ_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_spbequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbequ_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const double* ab, lapack_int ldab, double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_dpbequ_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpbequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbequ_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const lapack_complex_float* ab, lapack_int ldab, float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_cpbequ_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpbequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbequ_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const lapack_complex_double* ab, lapack_int ldab, double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_zpbequ_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpbequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spbrfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const float* ab, lapack_int ldab, const float* afb, 
///     lapack_int ldafb, const float* b, lapack_int ldb, float* x, 
///     lapack_int ldx, float* ferr, float* berr, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_spbrfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_spbrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbrfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const double* ab, lapack_int ldab, const double* afb, 
///     lapack_int ldafb, const double* b, lapack_int ldb, double* x, 
///     lapack_int ldx, double* ferr, double* berr, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dpbrfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dpbrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbrfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const lapack_complex_float* ab, lapack_int ldab, const lapack_complex_float* afb, 
///     lapack_int ldafb, const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* ferr, float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cpbrfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpbrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbrfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const lapack_complex_double* ab, lapack_int ldab, const lapack_complex_double* afb, 
///     lapack_int ldafb, const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* ferr, double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zpbrfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpbrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spbstf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kb, 
///     float* bb, lapack_int ldbb
/// );
/// ```
pub extern "C" fn lapacke_spbstf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kb: Lapack_int, bb: FixedArray[Float], ldbb: Lapack_int) -> Lapack_int = "LAPACKE_spbstf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbstf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kb, 
///     double* bb, lapack_int ldbb
/// );
/// ```
pub extern "C" fn lapacke_dpbstf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kb: Lapack_int, bb: FixedArray[Double], ldbb: Lapack_int) -> Lapack_int = "LAPACKE_dpbstf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbstf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kb, 
///     lapack_complex_float* bb, lapack_int ldbb
/// );
/// ```
pub extern "C" fn lapacke_cpbstf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kb: Lapack_int, bb: FixedArray[Float], ldbb: Lapack_int) -> Lapack_int = "LAPACKE_cpbstf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbstf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kb, 
///     lapack_complex_double* bb, lapack_int ldbb
/// );
/// ```
pub extern "C" fn lapacke_zpbstf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kb: Lapack_int, bb: FixedArray[Double], ldbb: Lapack_int) -> Lapack_int = "LAPACKE_zpbstf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spbsv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, float* ab, lapack_int ldab, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_spbsv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_spbsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbsv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, double* ab, lapack_int ldab, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dpbsv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dpbsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbsv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, lapack_complex_float* ab, lapack_int ldab, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cpbsv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cpbsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbsv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, lapack_complex_double* ab, lapack_int ldab, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zpbsv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zpbsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spbsvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int kd, lapack_int nrhs, float* ab, lapack_int ldab, 
///     float* afb, lapack_int ldafb, char* equed, float* s, 
///     float* b, lapack_int ldb, float* x, lapack_int ldx, 
///     float* rcond, float* ferr, float* berr, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_spbsvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_spbsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbsvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int kd, lapack_int nrhs, double* ab, lapack_int ldab, 
///     double* afb, lapack_int ldafb, char* equed, double* s, 
///     double* b, lapack_int ldb, double* x, lapack_int ldx, 
///     double* rcond, double* ferr, double* berr, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dpbsvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dpbsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbsvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int kd, lapack_int nrhs, lapack_complex_float* ab, lapack_int ldab, 
///     lapack_complex_float* afb, lapack_int ldafb, char* equed, float* s, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* rcond, float* ferr, float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cpbsvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpbsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbsvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int kd, lapack_int nrhs, lapack_complex_double* ab, lapack_int ldab, 
///     lapack_complex_double* afb, lapack_int ldafb, char* equed, double* s, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* rcond, double* ferr, double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zpbsvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpbsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spbtrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     float* ab, lapack_int ldab
/// );
/// ```
pub extern "C" fn lapacke_spbtrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int) -> Lapack_int = "LAPACKE_spbtrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbtrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     double* ab, lapack_int ldab
/// );
/// ```
pub extern "C" fn lapacke_dpbtrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int) -> Lapack_int = "LAPACKE_dpbtrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbtrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_complex_float* ab, lapack_int ldab
/// );
/// ```
pub extern "C" fn lapacke_cpbtrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int) -> Lapack_int = "LAPACKE_cpbtrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbtrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_complex_double* ab, lapack_int ldab
/// );
/// ```
pub extern "C" fn lapacke_zpbtrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int) -> Lapack_int = "LAPACKE_zpbtrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spbtrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const float* ab, lapack_int ldab, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_spbtrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_spbtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbtrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const double* ab, lapack_int ldab, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dpbtrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dpbtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbtrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const lapack_complex_float* ab, lapack_int ldab, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cpbtrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cpbtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbtrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const lapack_complex_double* ab, lapack_int ldab, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zpbtrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zpbtrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spftrf_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, float* a
/// );
/// ```
pub extern "C" fn lapacke_spftrf_work(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float]) -> Lapack_int = "LAPACKE_spftrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpftrf_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, double* a
/// );
/// ```
pub extern "C" fn lapacke_dpftrf_work(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpftrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpftrf_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, lapack_complex_float* a
/// );
/// ```
pub extern "C" fn lapacke_cpftrf_work(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpftrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpftrf_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, lapack_complex_double* a
/// );
/// ```
pub extern "C" fn lapacke_zpftrf_work(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpftrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spftri_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, float* a
/// );
/// ```
pub extern "C" fn lapacke_spftri_work(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float]) -> Lapack_int = "LAPACKE_spftri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpftri_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, double* a
/// );
/// ```
pub extern "C" fn lapacke_dpftri_work(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpftri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpftri_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, lapack_complex_float* a
/// );
/// ```
pub extern "C" fn lapacke_cpftri_work(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpftri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpftri_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, lapack_complex_double* a
/// );
/// ```
pub extern "C" fn lapacke_zpftri_work(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpftri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spftrs_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     lapack_int nrhs, const float* a, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_spftrs_work(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_spftrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpftrs_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     lapack_int nrhs, const double* a, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dpftrs_work(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dpftrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpftrs_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* a, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cpftrs_work(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cpftrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpftrs_work(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* a, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zpftrs_work(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zpftrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spocon_work(
///     int matrix_layout, char uplo, lapack_int n, const float* a, 
///     lapack_int lda, float anorm, float* rcond, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_spocon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, anorm: Float, rcond: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_spocon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpocon_work(
///     int matrix_layout, char uplo, lapack_int n, const double* a, 
///     lapack_int lda, double anorm, double* rcond, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dpocon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, anorm: Double, rcond: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dpocon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpocon_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, float anorm, float* rcond, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cpocon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, anorm: Float, rcond: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpocon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpocon_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, double anorm, double* rcond, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zpocon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, anorm: Double, rcond: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpocon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spoequ_work(
///     int matrix_layout, lapack_int n, const float* a, lapack_int lda, 
///     float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_spoequ_work(matrix_layout: Int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_spoequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpoequ_work(
///     int matrix_layout, lapack_int n, const double* a, lapack_int lda, 
///     double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_dpoequ_work(matrix_layout: Int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpoequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpoequ_work(
///     int matrix_layout, lapack_int n, const lapack_complex_float* a, lapack_int lda, 
///     float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_cpoequ_work(matrix_layout: Int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpoequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpoequ_work(
///     int matrix_layout, lapack_int n, const lapack_complex_double* a, lapack_int lda, 
///     double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_zpoequ_work(matrix_layout: Int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpoequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spoequb_work(
///     int matrix_layout, lapack_int n, const float* a, lapack_int lda, 
///     float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_spoequb_work(matrix_layout: Int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_spoequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpoequb_work(
///     int matrix_layout, lapack_int n, const double* a, lapack_int lda, 
///     double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_dpoequb_work(matrix_layout: Int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpoequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpoequb_work(
///     int matrix_layout, lapack_int n, const lapack_complex_float* a, lapack_int lda, 
///     float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_cpoequb_work(matrix_layout: Int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpoequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpoequb_work(
///     int matrix_layout, lapack_int n, const lapack_complex_double* a, lapack_int lda, 
///     double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_zpoequb_work(matrix_layout: Int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpoequb_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sporfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const float* af, lapack_int ldaf, 
///     const float* b, lapack_int ldb, float* x, lapack_int ldx, 
///     float* ferr, float* berr, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sporfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_sporfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dporfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const double* af, lapack_int ldaf, 
///     const double* b, lapack_int ldb, double* x, lapack_int ldx, 
///     double* ferr, double* berr, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dporfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dporfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cporfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, lapack_int ldaf, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* ferr, float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cporfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cporfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zporfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, lapack_int ldaf, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* ferr, double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zporfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zporfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sporfsx_work(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const float* a, lapack_int lda, const float* af, 
///     lapack_int ldaf, const float* s, const float* b, lapack_int ldb, 
///     float* x, lapack_int ldx, float* rcond, float* berr, 
///     lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, 
///     float* params, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sporfsx_work(matrix_layout: Int, uplo: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_sporfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dporfsx_work(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const double* a, lapack_int lda, const double* af, 
///     lapack_int ldaf, const double* s, const double* b, lapack_int ldb, 
///     double* x, lapack_int ldx, double* rcond, double* berr, 
///     lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, 
///     double* params, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dporfsx_work(matrix_layout: Int, uplo: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dporfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cporfsx_work(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, 
///     lapack_int ldaf, const float* s, const lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* x, lapack_int ldx, float* rcond, float* berr, 
///     lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, 
///     float* params, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cporfsx_work(matrix_layout: Int, uplo: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cporfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zporfsx_work(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, 
///     lapack_int ldaf, const double* s, const lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* x, lapack_int ldx, double* rcond, double* berr, 
///     lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, 
///     double* params, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zporfsx_work(matrix_layout: Int, uplo: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zporfsx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sposv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_sposv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_sposv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dposv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dposv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dposv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cposv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cposv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cposv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zposv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zposv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zposv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsposv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* x, lapack_int ldx, double* work, float* swork, lapack_int* iter
/// );
/// ```
pub extern "C" fn lapacke_dsposv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, work: FixedArray[Double], swork: FixedArray[Float], iter: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsposv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zcposv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* x, lapack_int ldx, lapack_complex_double* work, lapack_complex_float* swork, 
///     double* rwork, lapack_int* iter
/// );
/// ```
pub extern "C" fn lapacke_zcposv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, work: FixedArray[Double], swork: FixedArray[Float], rwork: FixedArray[Double], iter: FixedArray[Int]) -> Lapack_int = "LAPACKE_zcposv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sposvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, float* a, lapack_int lda, float* af, 
///     lapack_int ldaf, char* equed, float* s, float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* rcond, 
///     float* ferr, float* berr, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sposvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_sposvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dposvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, double* a, lapack_int lda, double* af, 
///     lapack_int ldaf, char* equed, double* s, double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* rcond, 
///     double* ferr, double* berr, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dposvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dposvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cposvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, char* equed, float* s, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond, 
///     float* ferr, float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cposvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cposvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zposvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, char* equed, double* s, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond, 
///     double* ferr, double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zposvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zposvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sposvxx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, float* a, lapack_int lda, float* af, 
///     lapack_int ldaf, char* equed, float* s, float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* rcond, 
///     float* rpvgrw, float* berr, lapack_int n_err_bnds, float* err_bnds_norm, 
///     float* err_bnds_comp, lapack_int nparams, float* params, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sposvxx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], rpvgrw: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_sposvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dposvxx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, double* a, lapack_int lda, double* af, 
///     lapack_int ldaf, char* equed, double* s, double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* rcond, 
///     double* rpvgrw, double* berr, lapack_int n_err_bnds, double* err_bnds_norm, 
///     double* err_bnds_comp, lapack_int nparams, double* params, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dposvxx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], rpvgrw: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dposvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cposvxx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, char* equed, float* s, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond, 
///     float* rpvgrw, float* berr, lapack_int n_err_bnds, float* err_bnds_norm, 
///     float* err_bnds_comp, lapack_int nparams, float* params, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cposvxx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], rpvgrw: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cposvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zposvxx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, char* equed, double* s, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond, 
///     double* rpvgrw, double* berr, lapack_int n_err_bnds, double* err_bnds_norm, 
///     double* err_bnds_comp, lapack_int nparams, double* params, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zposvxx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], rpvgrw: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zposvxx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spotrf2_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_spotrf2_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_spotrf2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpotrf2_work(
///     int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_dpotrf2_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_dpotrf2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpotrf2_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_cpotrf2_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_cpotrf2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpotrf2_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_zpotrf2_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_zpotrf2_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spotrf_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_spotrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_spotrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpotrf_work(
///     int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_dpotrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_dpotrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpotrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_cpotrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_cpotrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpotrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_zpotrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_zpotrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spotri_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_spotri_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_spotri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpotri_work(
///     int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_dpotri_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_dpotri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpotri_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_cpotri_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_cpotri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpotri_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_zpotri_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_zpotri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spotrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_spotrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_spotrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpotrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dpotrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dpotrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpotrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cpotrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cpotrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpotrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zpotrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zpotrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sppcon_work(
///     int matrix_layout, char uplo, lapack_int n, const float* ap, 
///     float anorm, float* rcond, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sppcon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], anorm: Float, rcond: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_sppcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dppcon_work(
///     int matrix_layout, char uplo, lapack_int n, const double* ap, 
///     double anorm, double* rcond, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dppcon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], anorm: Double, rcond: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dppcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cppcon_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* ap, 
///     float anorm, float* rcond, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cppcon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], anorm: Float, rcond: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cppcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zppcon_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* ap, 
///     double anorm, double* rcond, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zppcon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], anorm: Double, rcond: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zppcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sppequ_work(
///     int matrix_layout, char uplo, lapack_int n, const float* ap, 
///     float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_sppequ_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_sppequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dppequ_work(
///     int matrix_layout, char uplo, lapack_int n, const double* ap, 
///     double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_dppequ_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_dppequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cppequ_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* ap, 
///     float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_cppequ_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_cppequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zppequ_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* ap, 
///     double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_zppequ_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_zppequ_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spprfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* ap, const float* afp, const float* b, lapack_int ldb, 
///     float* x, lapack_int ldx, float* ferr, float* berr, 
///     float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_spprfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_spprfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpprfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* ap, const double* afp, const double* b, lapack_int ldb, 
///     double* x, lapack_int ldx, double* ferr, double* berr, 
///     double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dpprfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dpprfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpprfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* ap, const lapack_complex_float* afp, const lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* x, lapack_int ldx, float* ferr, float* berr, 
///     lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cpprfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpprfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpprfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* ap, const lapack_complex_double* afp, const lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* x, lapack_int ldx, double* ferr, double* berr, 
///     lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zpprfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpprfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sppsv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     float* ap, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_sppsv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_sppsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dppsv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* ap, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dppsv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dppsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cppsv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* ap, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cppsv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cppsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zppsv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* ap, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zppsv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zppsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sppsvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, float* ap, float* afp, char* equed, 
///     float* s, float* b, lapack_int ldb, float* x, 
///     lapack_int ldx, float* rcond, float* ferr, float* berr, 
///     float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sppsvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_sppsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dppsvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, double* ap, double* afp, char* equed, 
///     double* s, double* b, lapack_int ldb, double* x, 
///     lapack_int ldx, double* rcond, double* ferr, double* berr, 
///     double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dppsvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dppsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cppsvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* ap, lapack_complex_float* afp, char* equed, 
///     float* s, lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* rcond, float* ferr, float* berr, 
///     lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cppsvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cppsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zppsvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* ap, lapack_complex_double* afp, char* equed, 
///     double* s, lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* rcond, double* ferr, double* berr, 
///     lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zppsvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zppsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spptrf_work(
///     int matrix_layout, char uplo, lapack_int n, float* ap
/// );
/// ```
pub extern "C" fn lapacke_spptrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float]) -> Lapack_int = "LAPACKE_spptrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpptrf_work(
///     int matrix_layout, char uplo, lapack_int n, double* ap
/// );
/// ```
pub extern "C" fn lapacke_dpptrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpptrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpptrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* ap
/// );
/// ```
pub extern "C" fn lapacke_cpptrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpptrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpptrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* ap
/// );
/// ```
pub extern "C" fn lapacke_zpptrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpptrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spptri_work(
///     int matrix_layout, char uplo, lapack_int n, float* ap
/// );
/// ```
pub extern "C" fn lapacke_spptri_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float]) -> Lapack_int = "LAPACKE_spptri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpptri_work(
///     int matrix_layout, char uplo, lapack_int n, double* ap
/// );
/// ```
pub extern "C" fn lapacke_dpptri_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpptri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpptri_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* ap
/// );
/// ```
pub extern "C" fn lapacke_cpptri_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpptri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpptri_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* ap
/// );
/// ```
pub extern "C" fn lapacke_zpptri_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpptri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spptrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* ap, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_spptrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_spptrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpptrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* ap, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dpptrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dpptrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpptrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* ap, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cpptrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cpptrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpptrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* ap, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zpptrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zpptrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spstrf_work(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* piv, lapack_int* rank, float tol, float* work
/// );
/// ```
pub extern "C" fn lapacke_spstrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, piv: FixedArray[Int], rank: FixedArray[Int], tol: Float, work: FixedArray[Float]) -> Lapack_int = "LAPACKE_spstrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpstrf_work(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* piv, lapack_int* rank, double tol, double* work
/// );
/// ```
pub extern "C" fn lapacke_dpstrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, piv: FixedArray[Int], rank: FixedArray[Int], tol: Double, work: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpstrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpstrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* piv, lapack_int* rank, float tol, float* work
/// );
/// ```
pub extern "C" fn lapacke_cpstrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, piv: FixedArray[Int], rank: FixedArray[Int], tol: Float, work: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpstrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpstrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* piv, lapack_int* rank, double tol, double* work
/// );
/// ```
pub extern "C" fn lapacke_zpstrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, piv: FixedArray[Int], rank: FixedArray[Int], tol: Double, work: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpstrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sptcon_work(
///     lapack_int n, const float* d, const float* e, float anorm, 
///     float* rcond, float* work
/// );
/// ```
pub extern "C" fn lapacke_sptcon_work(n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], anorm: Float, rcond: FixedArray[Float], work: FixedArray[Float]) -> Lapack_int = "LAPACKE_sptcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dptcon_work(
///     lapack_int n, const double* d, const double* e, double anorm, 
///     double* rcond, double* work
/// );
/// ```
pub extern "C" fn lapacke_dptcon_work(n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], anorm: Double, rcond: FixedArray[Double], work: FixedArray[Double]) -> Lapack_int = "LAPACKE_dptcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cptcon_work(
///     lapack_int n, const float* d, const lapack_complex_float* e, float anorm, 
///     float* rcond, float* work
/// );
/// ```
pub extern "C" fn lapacke_cptcon_work(n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], anorm: Float, rcond: FixedArray[Float], work: FixedArray[Float]) -> Lapack_int = "LAPACKE_cptcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zptcon_work(
///     lapack_int n, const double* d, const lapack_complex_double* e, double anorm, 
///     double* rcond, double* work
/// );
/// ```
pub extern "C" fn lapacke_zptcon_work(n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], anorm: Double, rcond: FixedArray[Double], work: FixedArray[Double]) -> Lapack_int = "LAPACKE_zptcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spteqr_work(
///     int matrix_layout, char compz, lapack_int n, float* d, 
///     float* e, float* z, lapack_int ldz, float* work
/// );
/// ```
pub extern "C" fn lapacke_spteqr_work(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float]) -> Lapack_int = "LAPACKE_spteqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpteqr_work(
///     int matrix_layout, char compz, lapack_int n, double* d, 
///     double* e, double* z, lapack_int ldz, double* work
/// );
/// ```
pub extern "C" fn lapacke_dpteqr_work(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpteqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpteqr_work(
///     int matrix_layout, char compz, lapack_int n, float* d, 
///     float* e, lapack_complex_float* z, lapack_int ldz, float* work
/// );
/// ```
pub extern "C" fn lapacke_cpteqr_work(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpteqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpteqr_work(
///     int matrix_layout, char compz, lapack_int n, double* d, 
///     double* e, lapack_complex_double* z, lapack_int ldz, double* work
/// );
/// ```
pub extern "C" fn lapacke_zpteqr_work(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpteqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sptrfs_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, const float* d, 
///     const float* e, const float* df, const float* ef, const float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* ferr, 
///     float* berr, float* work
/// );
/// ```
pub extern "C" fn lapacke_sptrfs_work(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], df: FixedArray[Float], ef: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float]) -> Lapack_int = "LAPACKE_sptrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dptrfs_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, const double* d, 
///     const double* e, const double* df, const double* ef, const double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* ferr, 
///     double* berr, double* work
/// );
/// ```
pub extern "C" fn lapacke_dptrfs_work(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], df: FixedArray[Double], ef: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double]) -> Lapack_int = "LAPACKE_dptrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cptrfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* d, const lapack_complex_float* e, const float* df, const lapack_complex_float* ef, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* ferr, float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cptrfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], df: FixedArray[Float], ef: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cptrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zptrfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* d, const lapack_complex_double* e, const double* df, const lapack_complex_double* ef, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* ferr, double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zptrfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], df: FixedArray[Double], ef: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zptrfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sptsv_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, float* d, 
///     float* e, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_sptsv_work(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_sptsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dptsv_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, double* d, 
///     double* e, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dptsv_work(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dptsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cptsv_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, float* d, 
///     lapack_complex_float* e, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cptsv_work(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cptsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zptsv_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, double* d, 
///     lapack_complex_double* e, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zptsv_work(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zptsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sptsvx_work(
///     int matrix_layout, char fact, lapack_int n, lapack_int nrhs, 
///     const float* d, const float* e, float* df, float* ef, 
///     const float* b, lapack_int ldb, float* x, lapack_int ldx, 
///     float* rcond, float* ferr, float* berr, float* work
/// );
/// ```
pub extern "C" fn lapacke_sptsvx_work(matrix_layout: Int, fact: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], df: FixedArray[Float], ef: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float]) -> Lapack_int = "LAPACKE_sptsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dptsvx_work(
///     int matrix_layout, char fact, lapack_int n, lapack_int nrhs, 
///     const double* d, const double* e, double* df, double* ef, 
///     const double* b, lapack_int ldb, double* x, lapack_int ldx, 
///     double* rcond, double* ferr, double* berr, double* work
/// );
/// ```
pub extern "C" fn lapacke_dptsvx_work(matrix_layout: Int, fact: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], df: FixedArray[Double], ef: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double]) -> Lapack_int = "LAPACKE_dptsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cptsvx_work(
///     int matrix_layout, char fact, lapack_int n, lapack_int nrhs, 
///     const float* d, const lapack_complex_float* e, float* df, lapack_complex_float* ef, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* rcond, float* ferr, float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cptsvx_work(matrix_layout: Int, fact: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], df: FixedArray[Float], ef: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cptsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zptsvx_work(
///     int matrix_layout, char fact, lapack_int n, lapack_int nrhs, 
///     const double* d, const lapack_complex_double* e, double* df, lapack_complex_double* ef, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* rcond, double* ferr, double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zptsvx_work(matrix_layout: Int, fact: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], df: FixedArray[Double], ef: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zptsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spttrf_work(lapack_int n, float* d, float* e);
/// ```
pub extern "C" fn lapacke_spttrf_work(n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float]) -> Lapack_int = "LAPACKE_spttrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpttrf_work(lapack_int n, double* d, double* e);
/// ```
pub extern "C" fn lapacke_dpttrf_work(n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpttrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpttrf_work(lapack_int n, float* d, lapack_complex_float* e);
/// ```
pub extern "C" fn lapacke_cpttrf_work(n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpttrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpttrf_work(lapack_int n, double* d, lapack_complex_double* e);
/// ```
pub extern "C" fn lapacke_zpttrf_work(n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpttrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_spttrs_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, const float* d, 
///     const float* e, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_spttrs_work(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_spttrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dpttrs_work(
///     int matrix_layout, lapack_int n, lapack_int nrhs, const double* d, 
///     const double* e, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dpttrs_work(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dpttrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cpttrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* d, const lapack_complex_float* e, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cpttrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cpttrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zpttrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* d, const lapack_complex_double* e, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zpttrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zpttrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbev_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, float* ab, lapack_int ldab, float* w, 
///     float* z, lapack_int ldz, float* work
/// );
/// ```
pub extern "C" fn lapacke_ssbev_work(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssbev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbev_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, double* ab, lapack_int ldab, double* w, 
///     double* z, lapack_int ldz, double* work
/// );
/// ```
pub extern "C" fn lapacke_dsbev_work(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsbev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbevd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, float* ab, lapack_int ldab, float* w, 
///     float* z, lapack_int ldz, float* work, lapack_int lwork, 
///     lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_ssbevd_work(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_ssbevd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbevd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, double* ab, lapack_int ldab, double* w, 
///     double* z, lapack_int ldz, double* work, lapack_int lwork, 
///     lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_dsbevd_work(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_dsbevd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbevx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int kd, float* ab, lapack_int ldab, 
///     float* q, lapack_int ldq, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, float* z, lapack_int ldz, float* work, 
///     lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_ssbevx_work(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float], iwork: FixedArray[Int], ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_ssbevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbevx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int kd, double* ab, lapack_int ldab, 
///     double* q, lapack_int ldq, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, double* z, lapack_int ldz, double* work, 
///     lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_dsbevx_work(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, q: FixedArray[Double], ldq: Lapack_int, vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double], iwork: FixedArray[Int], ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsbevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbgst_work(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, float* ab, lapack_int ldab, 
///     const float* bb, lapack_int ldbb, float* x, lapack_int ldx, float* work
/// );
/// ```
pub extern "C" fn lapacke_ssbgst_work(matrix_layout: Int, vect: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, bb: FixedArray[Float], ldbb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, work: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssbgst_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbgst_work(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, double* ab, lapack_int ldab, 
///     const double* bb, lapack_int ldbb, double* x, lapack_int ldx, double* work
/// );
/// ```
pub extern "C" fn lapacke_dsbgst_work(matrix_layout: Int, vect: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, bb: FixedArray[Double], ldbb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, work: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsbgst_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbgv_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, float* ab, lapack_int ldab, 
///     float* bb, lapack_int ldbb, float* w, float* z, 
///     lapack_int ldz, float* work
/// );
/// ```
pub extern "C" fn lapacke_ssbgv_work(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, bb: FixedArray[Float], ldbb: Lapack_int, w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssbgv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbgv_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, double* ab, lapack_int ldab, 
///     double* bb, lapack_int ldbb, double* w, double* z, 
///     lapack_int ldz, double* work
/// );
/// ```
pub extern "C" fn lapacke_dsbgv_work(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, bb: FixedArray[Double], ldbb: Lapack_int, w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsbgv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbgvd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, float* ab, lapack_int ldab, 
///     float* bb, lapack_int ldbb, float* w, float* z, 
///     lapack_int ldz, float* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_ssbgvd_work(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, bb: FixedArray[Float], ldbb: Lapack_int, w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_ssbgvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbgvd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, double* ab, lapack_int ldab, 
///     double* bb, lapack_int ldbb, double* w, double* z, 
///     lapack_int ldz, double* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_dsbgvd_work(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, bb: FixedArray[Double], ldbb: Lapack_int, w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_dsbgvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbgvx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int ka, lapack_int kb, float* ab, 
///     lapack_int ldab, float* bb, lapack_int ldbb, float* q, 
///     lapack_int ldq, float vl, float vu, lapack_int il, 
///     lapack_int iu, float abstol, lapack_int* m, float* w, 
///     float* z, lapack_int ldz, float* work, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_ssbgvx_work(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, bb: FixedArray[Float], ldbb: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float], iwork: FixedArray[Int], ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_ssbgvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbgvx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int ka, lapack_int kb, double* ab, 
///     lapack_int ldab, double* bb, lapack_int ldbb, double* q, 
///     lapack_int ldq, double vl, double vu, lapack_int il, 
///     lapack_int iu, double abstol, lapack_int* m, double* w, 
///     double* z, lapack_int ldz, double* work, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_dsbgvx_work(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, bb: FixedArray[Double], ldbb: Lapack_int, q: FixedArray[Double], ldq: Lapack_int, vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double], iwork: FixedArray[Int], ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsbgvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbtrd_work(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int kd, float* ab, lapack_int ldab, float* d, 
///     float* e, float* q, lapack_int ldq, float* work
/// );
/// ```
pub extern "C" fn lapacke_ssbtrd_work(matrix_layout: Int, vect: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], q: FixedArray[Float], ldq: Lapack_int, work: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssbtrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbtrd_work(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int kd, double* ab, lapack_int ldab, double* d, 
///     double* e, double* q, lapack_int ldq, double* work
/// );
/// ```
pub extern "C" fn lapacke_dsbtrd_work(matrix_layout: Int, vect: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], q: FixedArray[Double], ldq: Lapack_int, work: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsbtrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssfrk_work(
///     int matrix_layout, char transr, char uplo, char trans, 
///     lapack_int n, lapack_int k, float alpha, const float* a, 
///     lapack_int lda, float beta, float* c
/// );
/// ```
pub extern "C" fn lapacke_ssfrk_work(matrix_layout: Int, transr: Byte, uplo: Byte, trans: Byte, n: Lapack_int, k: Lapack_int, alpha: Float, a: FixedArray[Float], lda: Lapack_int, beta: Float, c: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssfrk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsfrk_work(
///     int matrix_layout, char transr, char uplo, char trans, 
///     lapack_int n, lapack_int k, double alpha, const double* a, 
///     lapack_int lda, double beta, double* c
/// );
/// ```
pub extern "C" fn lapacke_dsfrk_work(matrix_layout: Int, transr: Byte, uplo: Byte, trans: Byte, n: Lapack_int, k: Lapack_int, alpha: Double, a: FixedArray[Double], lda: Lapack_int, beta: Double, c: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsfrk_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sspcon_work(
///     int matrix_layout, char uplo, lapack_int n, const float* ap, 
///     const lapack_int* ipiv, float anorm, float* rcond, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sspcon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int], anorm: Float, rcond: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_sspcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dspcon_work(
///     int matrix_layout, char uplo, lapack_int n, const double* ap, 
///     const lapack_int* ipiv, double anorm, double* rcond, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dspcon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int], anorm: Double, rcond: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dspcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cspcon_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* ap, 
///     const lapack_int* ipiv, float anorm, float* rcond, lapack_complex_float* work
/// );
/// ```
pub extern "C" fn lapacke_cspcon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int], anorm: Float, rcond: FixedArray[Float], work: FixedArray[Float]) -> Lapack_int = "LAPACKE_cspcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zspcon_work(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* ap, 
///     const lapack_int* ipiv, double anorm, double* rcond, lapack_complex_double* work
/// );
/// ```
pub extern "C" fn lapacke_zspcon_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int], anorm: Double, rcond: FixedArray[Double], work: FixedArray[Double]) -> Lapack_int = "LAPACKE_zspcon_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sspev_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     float* ap, float* w, float* z, lapack_int ldz, float* work
/// );
/// ```
pub extern "C" fn lapacke_sspev_work(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float]) -> Lapack_int = "LAPACKE_sspev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dspev_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     double* ap, double* w, double* z, lapack_int ldz, double* work
/// );
/// ```
pub extern "C" fn lapacke_dspev_work(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double]) -> Lapack_int = "LAPACKE_dspev_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sspevd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     float* ap, float* w, float* z, lapack_int ldz, 
///     float* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_sspevd_work(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_sspevd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dspevd_work(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     double* ap, double* w, double* z, lapack_int ldz, 
///     double* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_dspevd_work(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_dspevd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sspevx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, float* ap, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, float* z, lapack_int ldz, float* work, 
///     lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_sspevx_work(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float], iwork: FixedArray[Int], ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_sspevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dspevx_work(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, double* ap, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, double* z, lapack_int ldz, double* work, 
///     lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_dspevx_work(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double], iwork: FixedArray[Int], ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_dspevx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sspgst_work(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     float* ap, const float* bp
/// );
/// ```
pub extern "C" fn lapacke_sspgst_work(matrix_layout: Int, itype: Lapack_int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], bp: FixedArray[Float]) -> Lapack_int = "LAPACKE_sspgst_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dspgst_work(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     double* ap, const double* bp
/// );
/// ```
pub extern "C" fn lapacke_dspgst_work(matrix_layout: Int, itype: Lapack_int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], bp: FixedArray[Double]) -> Lapack_int = "LAPACKE_dspgst_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sspgv_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, float* ap, float* bp, float* w, 
///     float* z, lapack_int ldz, float* work
/// );
/// ```
pub extern "C" fn lapacke_sspgv_work(matrix_layout: Int, itype: Lapack_int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], bp: FixedArray[Float], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float]) -> Lapack_int = "LAPACKE_sspgv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dspgv_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, double* ap, double* bp, double* w, 
///     double* z, lapack_int ldz, double* work
/// );
/// ```
pub extern "C" fn lapacke_dspgv_work(matrix_layout: Int, itype: Lapack_int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], bp: FixedArray[Double], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double]) -> Lapack_int = "LAPACKE_dspgv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sspgvd_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, float* ap, float* bp, float* w, 
///     float* z, lapack_int ldz, float* work, lapack_int lwork, 
///     lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_sspgvd_work(matrix_layout: Int, itype: Lapack_int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], bp: FixedArray[Float], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_sspgvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dspgvd_work(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, double* ap, double* bp, double* w, 
///     double* z, lapack_int ldz, double* work, lapack_int lwork, 
///     lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_dspgvd_work(matrix_layout: Int, itype: Lapack_int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], bp: FixedArray[Double], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_dspgvd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sspgvx_work(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, float* ap, float* bp, 
///     float vl, float vu, lapack_int il, lapack_int iu, 
///     float abstol, lapack_int* m, float* w, float* z, 
///     lapack_int ldz, float* work, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_sspgvx_work(matrix_layout: Int, itype: Lapack_int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], bp: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float], iwork: FixedArray[Int], ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_sspgvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dspgvx_work(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, double* ap, double* bp, 
///     double vl, double vu, lapack_int il, lapack_int iu, 
///     double abstol, lapack_int* m, double* w, double* z, 
///     lapack_int ldz, double* work, lapack_int* iwork, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_dspgvx_work(matrix_layout: Int, itype: Lapack_int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], bp: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double], iwork: FixedArray[Int], ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_dspgvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssprfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* ap, const float* afp, const lapack_int* ipiv, const float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* ferr, 
///     float* berr, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_ssprfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_ssprfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsprfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* ap, const double* afp, const lapack_int* ipiv, const double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* ferr, 
///     double* berr, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dsprfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsprfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csprfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* ap, const lapack_complex_float* afp, const lapack_int* ipiv, const lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* ferr, 
///     float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_csprfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_csprfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsprfs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* ap, const lapack_complex_double* afp, const lapack_int* ipiv, const lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* ferr, 
///     double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zsprfs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zsprfs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sspsv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     float* ap, lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_sspsv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_sspsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dspsv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* ap, lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dspsv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dspsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cspsv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* ap, lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cspsv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cspsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zspsv_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* ap, lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zspsv_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zspsv_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sspsvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const float* ap, float* afp, lapack_int* ipiv, 
///     const float* b, lapack_int ldb, float* x, lapack_int ldx, 
///     float* rcond, float* ferr, float* berr, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sspsvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_sspsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dspsvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const double* ap, double* afp, lapack_int* ipiv, 
///     const double* b, lapack_int ldb, double* x, lapack_int ldx, 
///     double* rcond, double* ferr, double* berr, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dspsvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dspsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cspsvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* ap, lapack_complex_float* afp, lapack_int* ipiv, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* rcond, float* ferr, float* berr, lapack_complex_float* work, float* rwork
/// );
/// ```
pub extern "C" fn lapacke_cspsvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float], work: FixedArray[Float], rwork: FixedArray[Float]) -> Lapack_int = "LAPACKE_cspsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zspsvx_work(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* ap, lapack_complex_double* afp, lapack_int* ipiv, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* rcond, double* ferr, double* berr, lapack_complex_double* work, double* rwork
/// );
/// ```
pub extern "C" fn lapacke_zspsvx_work(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double], work: FixedArray[Double], rwork: FixedArray[Double]) -> Lapack_int = "LAPACKE_zspsvx_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssptrd_work(
///     int matrix_layout, char uplo, lapack_int n, float* ap, 
///     float* d, float* e, float* tau
/// );
/// ```
pub extern "C" fn lapacke_ssptrd_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], d: FixedArray[Float], e: FixedArray[Float], tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssptrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsptrd_work(
///     int matrix_layout, char uplo, lapack_int n, double* ap, 
///     double* d, double* e, double* tau
/// );
/// ```
pub extern "C" fn lapacke_dsptrd_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], d: FixedArray[Double], e: FixedArray[Double], tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsptrd_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssptrf_work(
///     int matrix_layout, char uplo, lapack_int n, float* ap, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_ssptrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_ssptrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsptrf_work(
///     int matrix_layout, char uplo, lapack_int n, double* ap, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_dsptrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsptrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csptrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* ap, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_csptrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_csptrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsptrf_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* ap, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_zsptrf_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_zsptrf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssptri_work(
///     int matrix_layout, char uplo, lapack_int n, float* ap, 
///     const lapack_int* ipiv, float* work
/// );
/// ```
pub extern "C" fn lapacke_ssptri_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int], work: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssptri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsptri_work(
///     int matrix_layout, char uplo, lapack_int n, double* ap, 
///     const lapack_int* ipiv, double* work
/// );
/// ```
pub extern "C" fn lapacke_dsptri_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int], work: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsptri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csptri_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* ap, 
///     const lapack_int* ipiv, lapack_complex_float* work
/// );
/// ```
pub extern "C" fn lapacke_csptri_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int], work: FixedArray[Float]) -> Lapack_int = "LAPACKE_csptri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsptri_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* ap, 
///     const lapack_int* ipiv, lapack_complex_double* work
/// );
/// ```
pub extern "C" fn lapacke_zsptri_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int], work: FixedArray[Double]) -> Lapack_int = "LAPACKE_zsptri_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssptrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* ap, const lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_ssptrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_ssptrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsptrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* ap, const lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dsptrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dsptrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csptrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* ap, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_csptrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_csptrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsptrs_work(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* ap, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zsptrs_work(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zsptrs_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sstebz_work(
///     char range, char order, lapack_int n, float vl, 
///     float vu, lapack_int il, lapack_int iu, float abstol, 
///     const float* d, const float* e, lapack_int* m, lapack_int* nsplit, 
///     float* w, lapack_int* iblock, lapack_int* isplit, float* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_sstebz_work(range: Byte, order: Byte, n: Lapack_int, vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, d: FixedArray[Float], e: FixedArray[Float], m: FixedArray[Int], nsplit: FixedArray[Int], w: FixedArray[Float], iblock: FixedArray[Int], isplit: FixedArray[Int], work: FixedArray[Float], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_sstebz_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dstebz_work(
///     char range, char order, lapack_int n, double vl, 
///     double vu, lapack_int il, lapack_int iu, double abstol, 
///     const double* d, const double* e, lapack_int* m, lapack_int* nsplit, 
///     double* w, lapack_int* iblock, lapack_int* isplit, double* work, lapack_int* iwork
/// );
/// ```
pub extern "C" fn lapacke_dstebz_work(range: Byte, order: Byte, n: Lapack_int, vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, d: FixedArray[Double], e: FixedArray[Double], m: FixedArray[Int], nsplit: FixedArray[Int], w: FixedArray[Double], iblock: FixedArray[Int], isplit: FixedArray[Int], work: FixedArray[Double], iwork: FixedArray[Int]) -> Lapack_int = "LAPACKE_dstebz_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sstedc_work(
///     int matrix_layout, char compz, lapack_int n, float* d, 
///     float* e, float* z, lapack_int ldz, float* work, 
///     lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_sstedc_work(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_sstedc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dstedc_work(
///     int matrix_layout, char compz, lapack_int n, double* d, 
///     double* e, double* z, lapack_int ldz, double* work, 
///     lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_dstedc_work(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_dstedc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cstedc_work(
///     int matrix_layout, char compz, lapack_int n, float* d, 
///     float* e, lapack_complex_float* z, lapack_int ldz, lapack_complex_float* work, 
///     lapack_int lwork, float* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_cstedc_work(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float], lwork: Lapack_int, rwork: FixedArray[Float], lrwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_cstedc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zstedc_work(
///     int matrix_layout, char compz, lapack_int n, double* d, 
///     double* e, lapack_complex_double* z, lapack_int ldz, lapack_complex_double* work, 
///     lapack_int lwork, double* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_zstedc_work(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double], lwork: Lapack_int, rwork: FixedArray[Double], lrwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_zstedc_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sstegr_work(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     float* d, float* e, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, float* z, lapack_int ldz, lapack_int* isuppz, 
///     float* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_sstegr_work(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, isuppz: FixedArray[Int], work: FixedArray[Float], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_sstegr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dstegr_work(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     double* d, double* e, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, double* z, lapack_int ldz, lapack_int* isuppz, 
///     double* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_dstegr_work(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, isuppz: FixedArray[Int], work: FixedArray[Double], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_dstegr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cstegr_work(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     float* d, float* e, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, lapack_complex_float* z, lapack_int ldz, lapack_int* isuppz, 
///     float* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_cstegr_work(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, isuppz: FixedArray[Int], work: FixedArray[Float], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_cstegr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zstegr_work(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     double* d, double* e, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, lapack_complex_double* z, lapack_int ldz, lapack_int* isuppz, 
///     double* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_zstegr_work(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, isuppz: FixedArray[Int], work: FixedArray[Double], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_zstegr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sstein_work(
///     int matrix_layout, lapack_int n, const float* d, const float* e, 
///     lapack_int m, const float* w, const lapack_int* iblock, const lapack_int* isplit, 
///     float* z, lapack_int ldz, float* work, lapack_int* iwork, lapack_int* ifailv
/// );
/// ```
pub extern "C" fn lapacke_sstein_work(matrix_layout: Int, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], m: Lapack_int, w: FixedArray[Float], iblock: FixedArray[Int], isplit: FixedArray[Int], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float], iwork: FixedArray[Int], ifailv: FixedArray[Int]) -> Lapack_int = "LAPACKE_sstein_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dstein_work(
///     int matrix_layout, lapack_int n, const double* d, const double* e, 
///     lapack_int m, const double* w, const lapack_int* iblock, const lapack_int* isplit, 
///     double* z, lapack_int ldz, double* work, lapack_int* iwork, lapack_int* ifailv
/// );
/// ```
pub extern "C" fn lapacke_dstein_work(matrix_layout: Int, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], m: Lapack_int, w: FixedArray[Double], iblock: FixedArray[Int], isplit: FixedArray[Int], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double], iwork: FixedArray[Int], ifailv: FixedArray[Int]) -> Lapack_int = "LAPACKE_dstein_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cstein_work(
///     int matrix_layout, lapack_int n, const float* d, const float* e, 
///     lapack_int m, const float* w, const lapack_int* iblock, const lapack_int* isplit, 
///     lapack_complex_float* z, lapack_int ldz, float* work, lapack_int* iwork, lapack_int* ifailv
/// );
/// ```
pub extern "C" fn lapacke_cstein_work(matrix_layout: Int, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], m: Lapack_int, w: FixedArray[Float], iblock: FixedArray[Int], isplit: FixedArray[Int], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float], iwork: FixedArray[Int], ifailv: FixedArray[Int]) -> Lapack_int = "LAPACKE_cstein_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zstein_work(
///     int matrix_layout, lapack_int n, const double* d, const double* e, 
///     lapack_int m, const double* w, const lapack_int* iblock, const lapack_int* isplit, 
///     lapack_complex_double* z, lapack_int ldz, double* work, lapack_int* iwork, lapack_int* ifailv
/// );
/// ```
pub extern "C" fn lapacke_zstein_work(matrix_layout: Int, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], m: Lapack_int, w: FixedArray[Double], iblock: FixedArray[Int], isplit: FixedArray[Int], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double], iwork: FixedArray[Int], ifailv: FixedArray[Int]) -> Lapack_int = "LAPACKE_zstein_work"

///|
///
/// ```c
/// lapack_int LAPACKE_sstemr_work(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     float* d, float* e, float vl, float vu, 
///     lapack_int il, lapack_int iu, lapack_int* m, float* w, 
///     float* z, lapack_int ldz, lapack_int nzc, lapack_int* isuppz, 
///     lapack_logical* tryrac, float* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_sstemr_work(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, nzc: Lapack_int, isuppz: FixedArray[Int], tryrac: Lapack_logical, work: FixedArray[Float], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_sstemr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dstemr_work(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     double* d, double* e, double vl, double vu, 
///     lapack_int il, lapack_int iu, lapack_int* m, double* w, 
///     double* z, lapack_int ldz, lapack_int nzc, lapack_int* isuppz, 
///     lapack_logical* tryrac, double* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_dstemr_work(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, nzc: Lapack_int, isuppz: FixedArray[Int], tryrac: Lapack_logical, work: FixedArray[Double], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_dstemr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_cstemr_work(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     float* d, float* e, float vl, float vu, 
///     lapack_int il, lapack_int iu, lapack_int* m, float* w, 
///     lapack_complex_float* z, lapack_int ldz, lapack_int nzc, lapack_int* isuppz, 
///     lapack_logical* tryrac, float* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_cstemr_work(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, nzc: Lapack_int, isuppz: FixedArray[Int], tryrac: Lapack_logical, work: FixedArray[Float], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_cstemr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zstemr_work(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     double* d, double* e, double vl, double vu, 
///     lapack_int il, lapack_int iu, lapack_int* m, double* w, 
///     lapack_complex_double* z, lapack_int ldz, lapack_int nzc, lapack_int* isuppz, 
///     lapack_logical* tryrac, double* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork
/// );
/// ```
pub extern "C" fn lapacke_zstemr_work(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, nzc: Lapack_int, isuppz: FixedArray[Int], tryrac: Lapack_logical, work: FixedArray[Double], lwork: Lapack_int, iwork: FixedArray[Int], liwork: Lapack_int) -> Lapack_int = "LAPACKE_zstemr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssteqr_work(
///     int matrix_layout, char compz, lapack_int n, float* d, 
///     float* e, float* z, lapack_int ldz, float* work
/// );
/// ```
pub extern "C" fn lapacke_ssteqr_work(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssteqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsteqr_work(
///     int matrix_layout, char compz, lapack_int n, double* d, 
///     double* e, double* z, lapack_int ldz, double* work
/// );
/// ```
pub extern "C" fn lapacke_dsteqr_work(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsteqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_csteqr_work(
///     int matrix_layout, char compz, lapack_int n, float* d, 
///     float* e, lapack_complex_float* z, lapack_int ldz, float* work
/// );
/// ```
pub extern "C" fn lapacke_csteqr_work(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, work: FixedArray[Float]) -> Lapack_int = "LAPACKE_csteqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_zsteqr_work(
///     int matrix_layout, char compz, lapack_int n, double* d, 
///     double* e, lapack_complex_double* z, lapack_int ldz, double* work
/// );
/// ```
pub extern "C" fn lapacke_zsteqr_work(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, work: FixedArray[Double]) -> Lapack_int = "LAPACKE_zsteqr_work"

///|
///
/// ```c
/// lapack_int LAPACKE_ssterf_work(lapack_int n, float* d, float* e);
/// ```
pub extern "C" fn lapacke_ssterf_work(n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssterf_work"

///|
///
/// ```c
/// lapack_int LAPACKE_dsterf_work(lapack_int n, double* d, double* e);
/// ```
pub extern "C" fn lapacke_dsterf_work(n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsterf_work"

