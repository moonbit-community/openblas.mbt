///|
///
/// ```c
/// lapack_int LAPACKE_dorgtsqr_row(
///     int matrix_layout, lapack_int m, lapack_int n, lapack_int mb, 
///     lapack_int nb, double* a, lapack_int lda, const double* t, lapack_int ldt
/// );
/// ```
pub extern "C" fn lapacke_dorgtsqr_row(matrix_layout: Int, m: Lapack_int, n: Lapack_int, mb: Lapack_int, nb: Lapack_int, a: FixedArray[Double], lda: Lapack_int, t: FixedArray[Double], ldt: Lapack_int) -> Lapack_int = "LAPACKE_dorgtsqr_row"

///|
///
/// ```c
/// lapack_int LAPACKE_sormbr(
///     int matrix_layout, char vect, char side, char trans, 
///     lapack_int m, lapack_int n, lapack_int k, const float* a, 
///     lapack_int lda, const float* tau, float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_sormbr(matrix_layout: Int, vect: Byte, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_sormbr"

///|
///
/// ```c
/// lapack_int LAPACKE_dormbr(
///     int matrix_layout, char vect, char side, char trans, 
///     lapack_int m, lapack_int n, lapack_int k, const double* a, 
///     lapack_int lda, const double* tau, double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_dormbr(matrix_layout: Int, vect: Byte, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_dormbr"

///|
///
/// ```c
/// lapack_int LAPACKE_sormhr(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, const float* a, 
///     lapack_int lda, const float* tau, float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_sormhr(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_sormhr"

///|
///
/// ```c
/// lapack_int LAPACKE_dormhr(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int ilo, lapack_int ihi, const double* a, 
///     lapack_int lda, const double* tau, double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_dormhr(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, ilo: Lapack_int, ihi: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_dormhr"

///|
///
/// ```c
/// lapack_int LAPACKE_sormlq(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const float* a, lapack_int lda, 
///     const float* tau, float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_sormlq(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_sormlq"

///|
///
/// ```c
/// lapack_int LAPACKE_dormlq(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const double* a, lapack_int lda, 
///     const double* tau, double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_dormlq(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_dormlq"

///|
///
/// ```c
/// lapack_int LAPACKE_sormql(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const float* a, lapack_int lda, 
///     const float* tau, float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_sormql(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_sormql"

///|
///
/// ```c
/// lapack_int LAPACKE_dormql(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const double* a, lapack_int lda, 
///     const double* tau, double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_dormql(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_dormql"

///|
///
/// ```c
/// lapack_int LAPACKE_sormqr(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const float* a, lapack_int lda, 
///     const float* tau, float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_sormqr(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_sormqr"

///|
///
/// ```c
/// lapack_int LAPACKE_dormqr(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const double* a, lapack_int lda, 
///     const double* tau, double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_dormqr(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_dormqr"

///|
///
/// ```c
/// lapack_int LAPACKE_sormrq(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const float* a, lapack_int lda, 
///     const float* tau, float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_sormrq(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_sormrq"

///|
///
/// ```c
/// lapack_int LAPACKE_dormrq(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, const double* a, lapack_int lda, 
///     const double* tau, double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_dormrq(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_dormrq"

///|
///
/// ```c
/// lapack_int LAPACKE_sormrz(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, const float* a, 
///     lapack_int lda, const float* tau, float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_sormrz(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, l: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_sormrz"

///|
///
/// ```c
/// lapack_int LAPACKE_dormrz(
///     int matrix_layout, char side, char trans, lapack_int m, 
///     lapack_int n, lapack_int k, lapack_int l, const double* a, 
///     lapack_int lda, const double* tau, double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_dormrz(matrix_layout: Int, side: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, k: Lapack_int, l: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_dormrz"

///|
///
/// ```c
/// lapack_int LAPACKE_sormtr(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const float* a, lapack_int lda, 
///     const float* tau, float* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_sormtr(matrix_layout: Int, side: Byte, uplo: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, tau: FixedArray[Float], c: FixedArray[Float], ldc: Lapack_int) -> Lapack_int = "LAPACKE_sormtr"

///|
///
/// ```c
/// lapack_int LAPACKE_dormtr(
///     int matrix_layout, char side, char uplo, char trans, 
///     lapack_int m, lapack_int n, const double* a, lapack_int lda, 
///     const double* tau, double* c, lapack_int ldc
/// );
/// ```
pub extern "C" fn lapacke_dormtr(matrix_layout: Int, side: Byte, uplo: Byte, trans: Byte, m: Lapack_int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, tau: FixedArray[Double], c: FixedArray[Double], ldc: Lapack_int) -> Lapack_int = "LAPACKE_dormtr"

///|
///
/// ```c
/// lapack_int LAPACKE_spbcon(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const float* ab, lapack_int ldab, float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_spbcon(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_spbcon"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbcon(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const double* ab, lapack_int ldab, double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_dpbcon(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpbcon"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbcon(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const lapack_complex_float* ab, lapack_int ldab, float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_cpbcon(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpbcon"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbcon(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const lapack_complex_double* ab, lapack_int ldab, double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_zpbcon(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpbcon"

///|
///
/// ```c
/// lapack_int LAPACKE_spbequ(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const float* ab, lapack_int ldab, float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_spbequ(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_spbequ"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbequ(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const double* ab, lapack_int ldab, double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_dpbequ(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpbequ"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbequ(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const lapack_complex_float* ab, lapack_int ldab, float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_cpbequ(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpbequ"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbequ(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     const lapack_complex_double* ab, lapack_int ldab, double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_zpbequ(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpbequ"

///|
///
/// ```c
/// lapack_int LAPACKE_spbrfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const float* ab, lapack_int ldab, const float* afb, 
///     lapack_int ldafb, const float* b, lapack_int ldb, float* x, 
///     lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_spbrfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_spbrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbrfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const double* ab, lapack_int ldab, const double* afb, 
///     lapack_int ldafb, const double* b, lapack_int ldb, double* x, 
///     lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dpbrfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpbrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbrfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const lapack_complex_float* ab, lapack_int ldab, const lapack_complex_float* afb, 
///     lapack_int ldafb, const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_cpbrfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpbrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbrfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const lapack_complex_double* ab, lapack_int ldab, const lapack_complex_double* afb, 
///     lapack_int ldafb, const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_zpbrfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpbrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_spbstf(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kb, 
///     float* bb, lapack_int ldbb
/// );
/// ```
pub extern "C" fn lapacke_spbstf(matrix_layout: Int, uplo: Byte, n: Lapack_int, kb: Lapack_int, bb: FixedArray[Float], ldbb: Lapack_int) -> Lapack_int = "LAPACKE_spbstf"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbstf(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kb, 
///     double* bb, lapack_int ldbb
/// );
/// ```
pub extern "C" fn lapacke_dpbstf(matrix_layout: Int, uplo: Byte, n: Lapack_int, kb: Lapack_int, bb: FixedArray[Double], ldbb: Lapack_int) -> Lapack_int = "LAPACKE_dpbstf"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbstf(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kb, 
///     lapack_complex_float* bb, lapack_int ldbb
/// );
/// ```
pub extern "C" fn lapacke_cpbstf(matrix_layout: Int, uplo: Byte, n: Lapack_int, kb: Lapack_int, bb: FixedArray[Float], ldbb: Lapack_int) -> Lapack_int = "LAPACKE_cpbstf"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbstf(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kb, 
///     lapack_complex_double* bb, lapack_int ldbb
/// );
/// ```
pub extern "C" fn lapacke_zpbstf(matrix_layout: Int, uplo: Byte, n: Lapack_int, kb: Lapack_int, bb: FixedArray[Double], ldbb: Lapack_int) -> Lapack_int = "LAPACKE_zpbstf"

///|
///
/// ```c
/// lapack_int LAPACKE_spbsv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, float* ab, lapack_int ldab, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_spbsv(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_spbsv"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbsv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, double* ab, lapack_int ldab, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dpbsv(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dpbsv"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbsv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, lapack_complex_float* ab, lapack_int ldab, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cpbsv(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cpbsv"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbsv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, lapack_complex_double* ab, lapack_int ldab, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zpbsv(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zpbsv"

///|
///
/// ```c
/// lapack_int LAPACKE_spbsvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int kd, lapack_int nrhs, float* ab, lapack_int ldab, 
///     float* afb, lapack_int ldafb, char* equed, float* s, 
///     float* b, lapack_int ldb, float* x, lapack_int ldx, 
///     float* rcond, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_spbsvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_spbsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbsvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int kd, lapack_int nrhs, double* ab, lapack_int ldab, 
///     double* afb, lapack_int ldafb, char* equed, double* s, 
///     double* b, lapack_int ldb, double* x, lapack_int ldx, 
///     double* rcond, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dpbsvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpbsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbsvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int kd, lapack_int nrhs, lapack_complex_float* ab, lapack_int ldab, 
///     lapack_complex_float* afb, lapack_int ldafb, char* equed, float* s, 
///     lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* rcond, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_cpbsvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, afb: FixedArray[Float], ldafb: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpbsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbsvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int kd, lapack_int nrhs, lapack_complex_double* ab, lapack_int ldab, 
///     lapack_complex_double* afb, lapack_int ldafb, char* equed, double* s, 
///     lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* rcond, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_zpbsvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, afb: FixedArray[Double], ldafb: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpbsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_spbtrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     float* ab, lapack_int ldab
/// );
/// ```
pub extern "C" fn lapacke_spbtrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int) -> Lapack_int = "LAPACKE_spbtrf"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbtrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     double* ab, lapack_int ldab
/// );
/// ```
pub extern "C" fn lapacke_dpbtrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int) -> Lapack_int = "LAPACKE_dpbtrf"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbtrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_complex_float* ab, lapack_int ldab
/// );
/// ```
pub extern "C" fn lapacke_cpbtrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int) -> Lapack_int = "LAPACKE_cpbtrf"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbtrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_complex_double* ab, lapack_int ldab
/// );
/// ```
pub extern "C" fn lapacke_zpbtrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int) -> Lapack_int = "LAPACKE_zpbtrf"

///|
///
/// ```c
/// lapack_int LAPACKE_spbtrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const float* ab, lapack_int ldab, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_spbtrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_spbtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_dpbtrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const double* ab, lapack_int ldab, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dpbtrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dpbtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_cpbtrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const lapack_complex_float* ab, lapack_int ldab, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cpbtrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cpbtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_zpbtrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int kd, 
///     lapack_int nrhs, const lapack_complex_double* ab, lapack_int ldab, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zpbtrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, kd: Lapack_int, nrhs: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zpbtrs"

///|
///
/// ```c
/// lapack_int LAPACKE_spftrf(
///     int matrix_layout, char transr, char uplo, lapack_int n, float* a
/// );
/// ```
pub extern "C" fn lapacke_spftrf(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float]) -> Lapack_int = "LAPACKE_spftrf"

///|
///
/// ```c
/// lapack_int LAPACKE_dpftrf(
///     int matrix_layout, char transr, char uplo, lapack_int n, double* a
/// );
/// ```
pub extern "C" fn lapacke_dpftrf(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpftrf"

///|
///
/// ```c
/// lapack_int LAPACKE_cpftrf(
///     int matrix_layout, char transr, char uplo, lapack_int n, lapack_complex_float* a
/// );
/// ```
pub extern "C" fn lapacke_cpftrf(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpftrf"

///|
///
/// ```c
/// lapack_int LAPACKE_zpftrf(
///     int matrix_layout, char transr, char uplo, lapack_int n, lapack_complex_double* a
/// );
/// ```
pub extern "C" fn lapacke_zpftrf(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpftrf"

///|
///
/// ```c
/// lapack_int LAPACKE_spftri(
///     int matrix_layout, char transr, char uplo, lapack_int n, float* a
/// );
/// ```
pub extern "C" fn lapacke_spftri(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float]) -> Lapack_int = "LAPACKE_spftri"

///|
///
/// ```c
/// lapack_int LAPACKE_dpftri(
///     int matrix_layout, char transr, char uplo, lapack_int n, double* a
/// );
/// ```
pub extern "C" fn lapacke_dpftri(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpftri"

///|
///
/// ```c
/// lapack_int LAPACKE_cpftri(
///     int matrix_layout, char transr, char uplo, lapack_int n, lapack_complex_float* a
/// );
/// ```
pub extern "C" fn lapacke_cpftri(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpftri"

///|
///
/// ```c
/// lapack_int LAPACKE_zpftri(
///     int matrix_layout, char transr, char uplo, lapack_int n, lapack_complex_double* a
/// );
/// ```
pub extern "C" fn lapacke_zpftri(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, a: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpftri"

///|
///
/// ```c
/// lapack_int LAPACKE_spftrs(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     lapack_int nrhs, const float* a, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_spftrs(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_spftrs"

///|
///
/// ```c
/// lapack_int LAPACKE_dpftrs(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     lapack_int nrhs, const double* a, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dpftrs(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dpftrs"

///|
///
/// ```c
/// lapack_int LAPACKE_cpftrs(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* a, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cpftrs(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cpftrs"

///|
///
/// ```c
/// lapack_int LAPACKE_zpftrs(
///     int matrix_layout, char transr, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* a, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zpftrs(matrix_layout: Int, transr: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zpftrs"

///|
///
/// ```c
/// lapack_int LAPACKE_spocon(
///     int matrix_layout, char uplo, lapack_int n, const float* a, 
///     lapack_int lda, float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_spocon(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_spocon"

///|
///
/// ```c
/// lapack_int LAPACKE_dpocon(
///     int matrix_layout, char uplo, lapack_int n, const double* a, 
///     lapack_int lda, double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_dpocon(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpocon"

///|
///
/// ```c
/// lapack_int LAPACKE_cpocon(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* a, 
///     lapack_int lda, float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_cpocon(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpocon"

///|
///
/// ```c
/// lapack_int LAPACKE_zpocon(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* a, 
///     lapack_int lda, double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_zpocon(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpocon"

///|
///
/// ```c
/// lapack_int LAPACKE_spoequ(
///     int matrix_layout, lapack_int n, const float* a, lapack_int lda, 
///     float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_spoequ(matrix_layout: Int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_spoequ"

///|
///
/// ```c
/// lapack_int LAPACKE_dpoequ(
///     int matrix_layout, lapack_int n, const double* a, lapack_int lda, 
///     double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_dpoequ(matrix_layout: Int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpoequ"

///|
///
/// ```c
/// lapack_int LAPACKE_cpoequ(
///     int matrix_layout, lapack_int n, const lapack_complex_float* a, lapack_int lda, 
///     float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_cpoequ(matrix_layout: Int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpoequ"

///|
///
/// ```c
/// lapack_int LAPACKE_zpoequ(
///     int matrix_layout, lapack_int n, const lapack_complex_double* a, lapack_int lda, 
///     double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_zpoequ(matrix_layout: Int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpoequ"

///|
///
/// ```c
/// lapack_int LAPACKE_spoequb(
///     int matrix_layout, lapack_int n, const float* a, lapack_int lda, 
///     float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_spoequb(matrix_layout: Int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_spoequb"

///|
///
/// ```c
/// lapack_int LAPACKE_dpoequb(
///     int matrix_layout, lapack_int n, const double* a, lapack_int lda, 
///     double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_dpoequb(matrix_layout: Int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpoequb"

///|
///
/// ```c
/// lapack_int LAPACKE_cpoequb(
///     int matrix_layout, lapack_int n, const lapack_complex_float* a, lapack_int lda, 
///     float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_cpoequb(matrix_layout: Int, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpoequb"

///|
///
/// ```c
/// lapack_int LAPACKE_zpoequb(
///     int matrix_layout, lapack_int n, const lapack_complex_double* a, lapack_int lda, 
///     double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_zpoequb(matrix_layout: Int, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpoequb"

///|
///
/// ```c
/// lapack_int LAPACKE_sporfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, const float* af, lapack_int ldaf, 
///     const float* b, lapack_int ldb, float* x, lapack_int ldx, 
///     float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_sporfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_sporfs"

///|
///
/// ```c
/// lapack_int LAPACKE_dporfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, const double* af, lapack_int ldaf, 
///     const double* b, lapack_int ldb, double* x, lapack_int ldx, 
///     double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dporfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dporfs"

///|
///
/// ```c
/// lapack_int LAPACKE_cporfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, lapack_int ldaf, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_cporfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_cporfs"

///|
///
/// ```c
/// lapack_int LAPACKE_zporfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, lapack_int ldaf, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_zporfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_zporfs"

///|
///
/// ```c
/// lapack_int LAPACKE_sporfsx(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const float* a, lapack_int lda, const float* af, 
///     lapack_int ldaf, const float* s, const float* b, lapack_int ldb, 
///     float* x, lapack_int ldx, float* rcond, float* berr, 
///     lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_sporfsx(matrix_layout: Int, uplo: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_sporfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_dporfsx(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const double* a, lapack_int lda, const double* af, 
///     lapack_int ldaf, const double* s, const double* b, lapack_int ldb, 
///     double* x, lapack_int ldx, double* rcond, double* berr, 
///     lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_dporfsx(matrix_layout: Int, uplo: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_dporfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_cporfsx(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* af, 
///     lapack_int ldaf, const float* s, const lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* x, lapack_int ldx, float* rcond, float* berr, 
///     lapack_int n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_cporfsx(matrix_layout: Int, uplo: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_cporfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_zporfsx(
///     int matrix_layout, char uplo, char equed, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* a, lapack_int lda, const lapack_complex_double* af, 
///     lapack_int ldaf, const double* s, const lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* x, lapack_int ldx, double* rcond, double* berr, 
///     lapack_int n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_zporfsx(matrix_layout: Int, uplo: Byte, equed: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_zporfsx"

///|
///
/// ```c
/// lapack_int LAPACKE_sposv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     float* a, lapack_int lda, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_sposv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_sposv"

///|
///
/// ```c
/// lapack_int LAPACKE_dposv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dposv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dposv"

///|
///
/// ```c
/// lapack_int LAPACKE_cposv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cposv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cposv"

///|
///
/// ```c
/// lapack_int LAPACKE_zposv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zposv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zposv"

///|
///
/// ```c
/// lapack_int LAPACKE_dsposv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* a, lapack_int lda, double* b, lapack_int ldb, 
///     double* x, lapack_int ldx, lapack_int* iter
/// );
/// ```
pub extern "C" fn lapacke_dsposv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, iter: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsposv"

///|
///
/// ```c
/// lapack_int LAPACKE_zcposv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* x, lapack_int ldx, lapack_int* iter
/// );
/// ```
pub extern "C" fn lapacke_zcposv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, iter: FixedArray[Int]) -> Lapack_int = "LAPACKE_zcposv"

///|
///
/// ```c
/// lapack_int LAPACKE_sposvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, float* a, lapack_int lda, float* af, 
///     lapack_int ldaf, char* equed, float* s, float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* rcond, 
///     float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_sposvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_sposvx"

///|
///
/// ```c
/// lapack_int LAPACKE_dposvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, double* a, lapack_int lda, double* af, 
///     lapack_int ldaf, char* equed, double* s, double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* rcond, 
///     double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dposvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dposvx"

///|
///
/// ```c
/// lapack_int LAPACKE_cposvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, char* equed, float* s, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond, 
///     float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_cposvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_cposvx"

///|
///
/// ```c
/// lapack_int LAPACKE_zposvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, char* equed, double* s, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond, 
///     double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_zposvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_zposvx"

///|
///
/// ```c
/// lapack_int LAPACKE_sposvxx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, float* a, lapack_int lda, float* af, 
///     lapack_int ldaf, char* equed, float* s, float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* rcond, 
///     float* rpvgrw, float* berr, lapack_int n_err_bnds, float* err_bnds_norm, 
///     float* err_bnds_comp, lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_sposvxx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], rpvgrw: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_sposvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_dposvxx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, double* a, lapack_int lda, double* af, 
///     lapack_int ldaf, char* equed, double* s, double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* rcond, 
///     double* rpvgrw, double* berr, lapack_int n_err_bnds, double* err_bnds_norm, 
///     double* err_bnds_comp, lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_dposvxx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], rpvgrw: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_dposvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_cposvxx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* af, 
///     lapack_int ldaf, char* equed, float* s, lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond, 
///     float* rpvgrw, float* berr, lapack_int n_err_bnds, float* err_bnds_norm, 
///     float* err_bnds_comp, lapack_int nparams, float* params
/// );
/// ```
pub extern "C" fn lapacke_cposvxx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, af: FixedArray[Float], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], rpvgrw: FixedArray[Float], berr: FixedArray[Float], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Float], err_bnds_comp: FixedArray[Float], nparams: Lapack_int, params: FixedArray[Float]) -> Lapack_int = "LAPACKE_cposvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_zposvxx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* af, 
///     lapack_int ldaf, char* equed, double* s, lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond, 
///     double* rpvgrw, double* berr, lapack_int n_err_bnds, double* err_bnds_norm, 
///     double* err_bnds_comp, lapack_int nparams, double* params
/// );
/// ```
pub extern "C" fn lapacke_zposvxx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, af: FixedArray[Double], ldaf: Lapack_int, equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], rpvgrw: FixedArray[Double], berr: FixedArray[Double], n_err_bnds: Lapack_int, err_bnds_norm: FixedArray[Double], err_bnds_comp: FixedArray[Double], nparams: Lapack_int, params: FixedArray[Double]) -> Lapack_int = "LAPACKE_zposvxx"

///|
///
/// ```c
/// lapack_int LAPACKE_spotrf2(
///     int matrix_layout, char uplo, lapack_int n, float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_spotrf2(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_spotrf2"

///|
///
/// ```c
/// lapack_int LAPACKE_dpotrf2(
///     int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_dpotrf2(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_dpotrf2"

///|
///
/// ```c
/// lapack_int LAPACKE_cpotrf2(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_cpotrf2(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_cpotrf2"

///|
///
/// ```c
/// lapack_int LAPACKE_zpotrf2(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_zpotrf2(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_zpotrf2"

///|
///
/// ```c
/// lapack_int LAPACKE_spotrf(
///     int matrix_layout, char uplo, lapack_int n, float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_spotrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_spotrf"

///|
///
/// ```c
/// lapack_int LAPACKE_dpotrf(
///     int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_dpotrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_dpotrf"

///|
///
/// ```c
/// lapack_int LAPACKE_cpotrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_cpotrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_cpotrf"

///|
///
/// ```c
/// lapack_int LAPACKE_zpotrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_zpotrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_zpotrf"

///|
///
/// ```c
/// lapack_int LAPACKE_spotri(
///     int matrix_layout, char uplo, lapack_int n, float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_spotri(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_spotri"

///|
///
/// ```c
/// lapack_int LAPACKE_dpotri(
///     int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_dpotri(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_dpotri"

///|
///
/// ```c
/// lapack_int LAPACKE_cpotri(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_cpotri(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int) -> Lapack_int = "LAPACKE_cpotri"

///|
///
/// ```c
/// lapack_int LAPACKE_zpotri(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda
/// );
/// ```
pub extern "C" fn lapacke_zpotri(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int) -> Lapack_int = "LAPACKE_zpotri"

///|
///
/// ```c
/// lapack_int LAPACKE_spotrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* a, lapack_int lda, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_spotrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_spotrs"

///|
///
/// ```c
/// lapack_int LAPACKE_dpotrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* a, lapack_int lda, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dpotrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dpotrs"

///|
///
/// ```c
/// lapack_int LAPACKE_cpotrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cpotrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Float], lda: Lapack_int, b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cpotrs"

///|
///
/// ```c
/// lapack_int LAPACKE_zpotrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zpotrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, a: FixedArray[Double], lda: Lapack_int, b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zpotrs"

///|
///
/// ```c
/// lapack_int LAPACKE_sppcon(
///     int matrix_layout, char uplo, lapack_int n, const float* ap, 
///     float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_sppcon(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_sppcon"

///|
///
/// ```c
/// lapack_int LAPACKE_dppcon(
///     int matrix_layout, char uplo, lapack_int n, const double* ap, 
///     double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_dppcon(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_dppcon"

///|
///
/// ```c
/// lapack_int LAPACKE_cppcon(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* ap, 
///     float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_cppcon(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_cppcon"

///|
///
/// ```c
/// lapack_int LAPACKE_zppcon(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* ap, 
///     double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_zppcon(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_zppcon"

///|
///
/// ```c
/// lapack_int LAPACKE_sppequ(
///     int matrix_layout, char uplo, lapack_int n, const float* ap, 
///     float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_sppequ(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_sppequ"

///|
///
/// ```c
/// lapack_int LAPACKE_dppequ(
///     int matrix_layout, char uplo, lapack_int n, const double* ap, 
///     double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_dppequ(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_dppequ"

///|
///
/// ```c
/// lapack_int LAPACKE_cppequ(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* ap, 
///     float* s, float* scond, float* amax
/// );
/// ```
pub extern "C" fn lapacke_cppequ(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], s: FixedArray[Float], scond: FixedArray[Float], amax: FixedArray[Float]) -> Lapack_int = "LAPACKE_cppequ"

///|
///
/// ```c
/// lapack_int LAPACKE_zppequ(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* ap, 
///     double* s, double* scond, double* amax
/// );
/// ```
pub extern "C" fn lapacke_zppequ(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], s: FixedArray[Double], scond: FixedArray[Double], amax: FixedArray[Double]) -> Lapack_int = "LAPACKE_zppequ"

///|
///
/// ```c
/// lapack_int LAPACKE_spprfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* ap, const float* afp, const float* b, lapack_int ldb, 
///     float* x, lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_spprfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_spprfs"

///|
///
/// ```c
/// lapack_int LAPACKE_dpprfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* ap, const double* afp, const double* b, lapack_int ldb, 
///     double* x, lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dpprfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpprfs"

///|
///
/// ```c
/// lapack_int LAPACKE_cpprfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* ap, const lapack_complex_float* afp, const lapack_complex_float* b, lapack_int ldb, 
///     lapack_complex_float* x, lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_cpprfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpprfs"

///|
///
/// ```c
/// lapack_int LAPACKE_zpprfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* ap, const lapack_complex_double* afp, const lapack_complex_double* b, lapack_int ldb, 
///     lapack_complex_double* x, lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_zpprfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpprfs"

///|
///
/// ```c
/// lapack_int LAPACKE_sppsv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     float* ap, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_sppsv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_sppsv"

///|
///
/// ```c
/// lapack_int LAPACKE_dppsv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* ap, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dppsv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dppsv"

///|
///
/// ```c
/// lapack_int LAPACKE_cppsv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* ap, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cppsv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cppsv"

///|
///
/// ```c
/// lapack_int LAPACKE_zppsv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* ap, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zppsv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zppsv"

///|
///
/// ```c
/// lapack_int LAPACKE_sppsvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, float* ap, float* afp, char* equed, 
///     float* s, float* b, lapack_int ldb, float* x, 
///     lapack_int ldx, float* rcond, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_sppsvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_sppsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_dppsvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, double* ap, double* afp, char* equed, 
///     double* s, double* b, lapack_int ldb, double* x, 
///     lapack_int ldx, double* rcond, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dppsvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dppsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_cppsvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_float* ap, lapack_complex_float* afp, char* equed, 
///     float* s, lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, 
///     lapack_int ldx, float* rcond, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_cppsvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], equed: FixedArray[Byte], s: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_cppsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_zppsvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, lapack_complex_double* ap, lapack_complex_double* afp, char* equed, 
///     double* s, lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, 
///     lapack_int ldx, double* rcond, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_zppsvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], equed: FixedArray[Byte], s: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_zppsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_spptrf(
///     int matrix_layout, char uplo, lapack_int n, float* ap
/// );
/// ```
pub extern "C" fn lapacke_spptrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float]) -> Lapack_int = "LAPACKE_spptrf"

///|
///
/// ```c
/// lapack_int LAPACKE_dpptrf(
///     int matrix_layout, char uplo, lapack_int n, double* ap
/// );
/// ```
pub extern "C" fn lapacke_dpptrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpptrf"

///|
///
/// ```c
/// lapack_int LAPACKE_cpptrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* ap
/// );
/// ```
pub extern "C" fn lapacke_cpptrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpptrf"

///|
///
/// ```c
/// lapack_int LAPACKE_zpptrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* ap
/// );
/// ```
pub extern "C" fn lapacke_zpptrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpptrf"

///|
///
/// ```c
/// lapack_int LAPACKE_spptri(
///     int matrix_layout, char uplo, lapack_int n, float* ap
/// );
/// ```
pub extern "C" fn lapacke_spptri(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float]) -> Lapack_int = "LAPACKE_spptri"

///|
///
/// ```c
/// lapack_int LAPACKE_dpptri(
///     int matrix_layout, char uplo, lapack_int n, double* ap
/// );
/// ```
pub extern "C" fn lapacke_dpptri(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpptri"

///|
///
/// ```c
/// lapack_int LAPACKE_cpptri(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* ap
/// );
/// ```
pub extern "C" fn lapacke_cpptri(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpptri"

///|
///
/// ```c
/// lapack_int LAPACKE_zpptri(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* ap
/// );
/// ```
pub extern "C" fn lapacke_zpptri(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpptri"

///|
///
/// ```c
/// lapack_int LAPACKE_spptrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* ap, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_spptrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_spptrs"

///|
///
/// ```c
/// lapack_int LAPACKE_dpptrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* ap, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dpptrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dpptrs"

///|
///
/// ```c
/// lapack_int LAPACKE_cpptrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* ap, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cpptrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cpptrs"

///|
///
/// ```c
/// lapack_int LAPACKE_zpptrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* ap, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zpptrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zpptrs"

///|
///
/// ```c
/// lapack_int LAPACKE_spstrf(
///     int matrix_layout, char uplo, lapack_int n, float* a, 
///     lapack_int lda, lapack_int* piv, lapack_int* rank, float tol
/// );
/// ```
pub extern "C" fn lapacke_spstrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, piv: FixedArray[Int], rank: FixedArray[Int], tol: Float) -> Lapack_int = "LAPACKE_spstrf"

///|
///
/// ```c
/// lapack_int LAPACKE_dpstrf(
///     int matrix_layout, char uplo, lapack_int n, double* a, 
///     lapack_int lda, lapack_int* piv, lapack_int* rank, double tol
/// );
/// ```
pub extern "C" fn lapacke_dpstrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, piv: FixedArray[Int], rank: FixedArray[Int], tol: Double) -> Lapack_int = "LAPACKE_dpstrf"

///|
///
/// ```c
/// lapack_int LAPACKE_cpstrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, 
///     lapack_int lda, lapack_int* piv, lapack_int* rank, float tol
/// );
/// ```
pub extern "C" fn lapacke_cpstrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Float], lda: Lapack_int, piv: FixedArray[Int], rank: FixedArray[Int], tol: Float) -> Lapack_int = "LAPACKE_cpstrf"

///|
///
/// ```c
/// lapack_int LAPACKE_zpstrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, 
///     lapack_int lda, lapack_int* piv, lapack_int* rank, double tol
/// );
/// ```
pub extern "C" fn lapacke_zpstrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, a: FixedArray[Double], lda: Lapack_int, piv: FixedArray[Int], rank: FixedArray[Int], tol: Double) -> Lapack_int = "LAPACKE_zpstrf"

///|
///
/// ```c
/// lapack_int LAPACKE_sptcon(
///     lapack_int n, const float* d, const float* e, float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_sptcon(n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_sptcon"

///|
///
/// ```c
/// lapack_int LAPACKE_dptcon(
///     lapack_int n, const double* d, const double* e, double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_dptcon(n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_dptcon"

///|
///
/// ```c
/// lapack_int LAPACKE_cptcon(
///     lapack_int n, const float* d, const lapack_complex_float* e, float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_cptcon(n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_cptcon"

///|
///
/// ```c
/// lapack_int LAPACKE_zptcon(
///     lapack_int n, const double* d, const lapack_complex_double* e, double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_zptcon(n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_zptcon"

///|
///
/// ```c
/// lapack_int LAPACKE_spteqr(
///     int matrix_layout, char compz, lapack_int n, float* d, 
///     float* e, float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_spteqr(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_spteqr"

///|
///
/// ```c
/// lapack_int LAPACKE_dpteqr(
///     int matrix_layout, char compz, lapack_int n, double* d, 
///     double* e, double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_dpteqr(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_dpteqr"

///|
///
/// ```c
/// lapack_int LAPACKE_cpteqr(
///     int matrix_layout, char compz, lapack_int n, float* d, 
///     float* e, lapack_complex_float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_cpteqr(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_cpteqr"

///|
///
/// ```c
/// lapack_int LAPACKE_zpteqr(
///     int matrix_layout, char compz, lapack_int n, double* d, 
///     double* e, lapack_complex_double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_zpteqr(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_zpteqr"

///|
///
/// ```c
/// lapack_int LAPACKE_sptrfs(
///     int matrix_layout, lapack_int n, lapack_int nrhs, const float* d, 
///     const float* e, const float* df, const float* ef, const float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_sptrfs(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], df: FixedArray[Float], ef: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_sptrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_dptrfs(
///     int matrix_layout, lapack_int n, lapack_int nrhs, const double* d, 
///     const double* e, const double* df, const double* ef, const double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dptrfs(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], df: FixedArray[Double], ef: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dptrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_cptrfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* d, const lapack_complex_float* e, const float* df, const lapack_complex_float* ef, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_cptrfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], df: FixedArray[Float], ef: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_cptrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_zptrfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* d, const lapack_complex_double* e, const double* df, const lapack_complex_double* ef, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_zptrfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], df: FixedArray[Double], ef: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_zptrfs"

///|
///
/// ```c
/// lapack_int LAPACKE_sptsv(
///     int matrix_layout, lapack_int n, lapack_int nrhs, float* d, 
///     float* e, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_sptsv(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_sptsv"

///|
///
/// ```c
/// lapack_int LAPACKE_dptsv(
///     int matrix_layout, lapack_int n, lapack_int nrhs, double* d, 
///     double* e, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dptsv(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dptsv"

///|
///
/// ```c
/// lapack_int LAPACKE_cptsv(
///     int matrix_layout, lapack_int n, lapack_int nrhs, float* d, 
///     lapack_complex_float* e, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cptsv(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cptsv"

///|
///
/// ```c
/// lapack_int LAPACKE_zptsv(
///     int matrix_layout, lapack_int n, lapack_int nrhs, double* d, 
///     lapack_complex_double* e, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zptsv(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zptsv"

///|
///
/// ```c
/// lapack_int LAPACKE_sptsvx(
///     int matrix_layout, char fact, lapack_int n, lapack_int nrhs, 
///     const float* d, const float* e, float* df, float* ef, 
///     const float* b, lapack_int ldb, float* x, lapack_int ldx, 
///     float* rcond, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_sptsvx(matrix_layout: Int, fact: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], df: FixedArray[Float], ef: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_sptsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_dptsvx(
///     int matrix_layout, char fact, lapack_int n, lapack_int nrhs, 
///     const double* d, const double* e, double* df, double* ef, 
///     const double* b, lapack_int ldb, double* x, lapack_int ldx, 
///     double* rcond, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dptsvx(matrix_layout: Int, fact: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], df: FixedArray[Double], ef: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dptsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_cptsvx(
///     int matrix_layout, char fact, lapack_int n, lapack_int nrhs, 
///     const float* d, const lapack_complex_float* e, float* df, lapack_complex_float* ef, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* rcond, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_cptsvx(matrix_layout: Int, fact: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], df: FixedArray[Float], ef: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_cptsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_zptsvx(
///     int matrix_layout, char fact, lapack_int n, lapack_int nrhs, 
///     const double* d, const lapack_complex_double* e, double* df, lapack_complex_double* ef, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* rcond, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_zptsvx(matrix_layout: Int, fact: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], df: FixedArray[Double], ef: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_zptsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_spttrf(lapack_int n, float* d, float* e);
/// ```
pub extern "C" fn lapacke_spttrf(n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float]) -> Lapack_int = "LAPACKE_spttrf"

///|
///
/// ```c
/// lapack_int LAPACKE_dpttrf(lapack_int n, double* d, double* e);
/// ```
pub extern "C" fn lapacke_dpttrf(n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double]) -> Lapack_int = "LAPACKE_dpttrf"

///|
///
/// ```c
/// lapack_int LAPACKE_cpttrf(lapack_int n, float* d, lapack_complex_float* e);
/// ```
pub extern "C" fn lapacke_cpttrf(n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float]) -> Lapack_int = "LAPACKE_cpttrf"

///|
///
/// ```c
/// lapack_int LAPACKE_zpttrf(lapack_int n, double* d, lapack_complex_double* e);
/// ```
pub extern "C" fn lapacke_zpttrf(n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double]) -> Lapack_int = "LAPACKE_zpttrf"

///|
///
/// ```c
/// lapack_int LAPACKE_spttrs(
///     int matrix_layout, lapack_int n, lapack_int nrhs, const float* d, 
///     const float* e, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_spttrs(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_spttrs"

///|
///
/// ```c
/// lapack_int LAPACKE_dpttrs(
///     int matrix_layout, lapack_int n, lapack_int nrhs, const double* d, 
///     const double* e, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dpttrs(matrix_layout: Int, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dpttrs"

///|
///
/// ```c
/// lapack_int LAPACKE_cpttrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* d, const lapack_complex_float* e, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cpttrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cpttrs"

///|
///
/// ```c
/// lapack_int LAPACKE_zpttrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* d, const lapack_complex_double* e, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zpttrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zpttrs"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbev(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, float* ab, lapack_int ldab, float* w, 
///     float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_ssbev(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_ssbev"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbev(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, double* ab, lapack_int ldab, double* w, 
///     double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_dsbev(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_dsbev"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbevd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, float* ab, lapack_int ldab, float* w, 
///     float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_ssbevd(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_ssbevd"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbevd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int kd, double* ab, lapack_int ldab, double* w, 
///     double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_dsbevd(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_dsbevd"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbevx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int kd, float* ab, lapack_int ldab, 
///     float* q, lapack_int ldq, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, float* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_ssbevx(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_ssbevx"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbevx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int kd, double* ab, lapack_int ldab, 
///     double* q, lapack_int ldq, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, double* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_dsbevx(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, q: FixedArray[Double], ldq: Lapack_int, vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsbevx"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbgst(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, float* ab, lapack_int ldab, 
///     const float* bb, lapack_int ldbb, float* x, lapack_int ldx
/// );
/// ```
pub extern "C" fn lapacke_ssbgst(matrix_layout: Int, vect: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, bb: FixedArray[Float], ldbb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int) -> Lapack_int = "LAPACKE_ssbgst"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbgst(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, double* ab, lapack_int ldab, 
///     const double* bb, lapack_int ldbb, double* x, lapack_int ldx
/// );
/// ```
pub extern "C" fn lapacke_dsbgst(matrix_layout: Int, vect: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, bb: FixedArray[Double], ldbb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int) -> Lapack_int = "LAPACKE_dsbgst"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbgv(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, float* ab, lapack_int ldab, 
///     float* bb, lapack_int ldbb, float* w, float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_ssbgv(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, bb: FixedArray[Float], ldbb: Lapack_int, w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_ssbgv"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbgv(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, double* ab, lapack_int ldab, 
///     double* bb, lapack_int ldbb, double* w, double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_dsbgv(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, bb: FixedArray[Double], ldbb: Lapack_int, w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_dsbgv"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbgvd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, float* ab, lapack_int ldab, 
///     float* bb, lapack_int ldbb, float* w, float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_ssbgvd(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, bb: FixedArray[Float], ldbb: Lapack_int, w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_ssbgvd"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbgvd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     lapack_int ka, lapack_int kb, double* ab, lapack_int ldab, 
///     double* bb, lapack_int ldbb, double* w, double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_dsbgvd(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, bb: FixedArray[Double], ldbb: Lapack_int, w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_dsbgvd"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbgvx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int ka, lapack_int kb, float* ab, 
///     lapack_int ldab, float* bb, lapack_int ldbb, float* q, 
///     lapack_int ldq, float vl, float vu, lapack_int il, 
///     lapack_int iu, float abstol, lapack_int* m, float* w, 
///     float* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_ssbgvx(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, bb: FixedArray[Float], ldbb: Lapack_int, q: FixedArray[Float], ldq: Lapack_int, vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_ssbgvx"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbgvx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, lapack_int ka, lapack_int kb, double* ab, 
///     lapack_int ldab, double* bb, lapack_int ldbb, double* q, 
///     lapack_int ldq, double vl, double vu, lapack_int il, 
///     lapack_int iu, double abstol, lapack_int* m, double* w, 
///     double* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_dsbgvx(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, ka: Lapack_int, kb: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, bb: FixedArray[Double], ldbb: Lapack_int, q: FixedArray[Double], ldq: Lapack_int, vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsbgvx"

///|
///
/// ```c
/// lapack_int LAPACKE_ssbtrd(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int kd, float* ab, lapack_int ldab, float* d, 
///     float* e, float* q, lapack_int ldq
/// );
/// ```
pub extern "C" fn lapacke_ssbtrd(matrix_layout: Int, vect: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Float], ldab: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], q: FixedArray[Float], ldq: Lapack_int) -> Lapack_int = "LAPACKE_ssbtrd"

///|
///
/// ```c
/// lapack_int LAPACKE_dsbtrd(
///     int matrix_layout, char vect, char uplo, lapack_int n, 
///     lapack_int kd, double* ab, lapack_int ldab, double* d, 
///     double* e, double* q, lapack_int ldq
/// );
/// ```
pub extern "C" fn lapacke_dsbtrd(matrix_layout: Int, vect: Byte, uplo: Byte, n: Lapack_int, kd: Lapack_int, ab: FixedArray[Double], ldab: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], q: FixedArray[Double], ldq: Lapack_int) -> Lapack_int = "LAPACKE_dsbtrd"

///|
///
/// ```c
/// lapack_int LAPACKE_ssfrk(
///     int matrix_layout, char transr, char uplo, char trans, 
///     lapack_int n, lapack_int k, float alpha, const float* a, 
///     lapack_int lda, float beta, float* c
/// );
/// ```
pub extern "C" fn lapacke_ssfrk(matrix_layout: Int, transr: Byte, uplo: Byte, trans: Byte, n: Lapack_int, k: Lapack_int, alpha: Float, a: FixedArray[Float], lda: Lapack_int, beta: Float, c: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssfrk"

///|
///
/// ```c
/// lapack_int LAPACKE_dsfrk(
///     int matrix_layout, char transr, char uplo, char trans, 
///     lapack_int n, lapack_int k, double alpha, const double* a, 
///     lapack_int lda, double beta, double* c
/// );
/// ```
pub extern "C" fn lapacke_dsfrk(matrix_layout: Int, transr: Byte, uplo: Byte, trans: Byte, n: Lapack_int, k: Lapack_int, alpha: Double, a: FixedArray[Double], lda: Lapack_int, beta: Double, c: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsfrk"

///|
///
/// ```c
/// lapack_int LAPACKE_sspcon(
///     int matrix_layout, char uplo, lapack_int n, const float* ap, 
///     const lapack_int* ipiv, float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_sspcon(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int], anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_sspcon"

///|
///
/// ```c
/// lapack_int LAPACKE_dspcon(
///     int matrix_layout, char uplo, lapack_int n, const double* ap, 
///     const lapack_int* ipiv, double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_dspcon(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int], anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_dspcon"

///|
///
/// ```c
/// lapack_int LAPACKE_cspcon(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* ap, 
///     const lapack_int* ipiv, float anorm, float* rcond
/// );
/// ```
pub extern "C" fn lapacke_cspcon(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int], anorm: Float, rcond: FixedArray[Float]) -> Lapack_int = "LAPACKE_cspcon"

///|
///
/// ```c
/// lapack_int LAPACKE_zspcon(
///     int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* ap, 
///     const lapack_int* ipiv, double anorm, double* rcond
/// );
/// ```
pub extern "C" fn lapacke_zspcon(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int], anorm: Double, rcond: FixedArray[Double]) -> Lapack_int = "LAPACKE_zspcon"

///|
///
/// ```c
/// lapack_int LAPACKE_sspev(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     float* ap, float* w, float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_sspev(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_sspev"

///|
///
/// ```c
/// lapack_int LAPACKE_dspev(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     double* ap, double* w, double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_dspev(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_dspev"

///|
///
/// ```c
/// lapack_int LAPACKE_sspevd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     float* ap, float* w, float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_sspevd(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_sspevd"

///|
///
/// ```c
/// lapack_int LAPACKE_dspevd(
///     int matrix_layout, char jobz, char uplo, lapack_int n, 
///     double* ap, double* w, double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_dspevd(matrix_layout: Int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_dspevd"

///|
///
/// ```c
/// lapack_int LAPACKE_sspevx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, float* ap, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, float* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_sspevx(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_sspevx"

///|
///
/// ```c
/// lapack_int LAPACKE_dspevx(
///     int matrix_layout, char jobz, char range, char uplo, 
///     lapack_int n, double* ap, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, double* z, lapack_int ldz, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_dspevx(matrix_layout: Int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_dspevx"

///|
///
/// ```c
/// lapack_int LAPACKE_sspgst(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     float* ap, const float* bp
/// );
/// ```
pub extern "C" fn lapacke_sspgst(matrix_layout: Int, itype: Lapack_int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], bp: FixedArray[Float]) -> Lapack_int = "LAPACKE_sspgst"

///|
///
/// ```c
/// lapack_int LAPACKE_dspgst(
///     int matrix_layout, lapack_int itype, char uplo, lapack_int n, 
///     double* ap, const double* bp
/// );
/// ```
pub extern "C" fn lapacke_dspgst(matrix_layout: Int, itype: Lapack_int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], bp: FixedArray[Double]) -> Lapack_int = "LAPACKE_dspgst"

///|
///
/// ```c
/// lapack_int LAPACKE_sspgv(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, float* ap, float* bp, float* w, 
///     float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_sspgv(matrix_layout: Int, itype: Lapack_int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], bp: FixedArray[Float], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_sspgv"

///|
///
/// ```c
/// lapack_int LAPACKE_dspgv(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, double* ap, double* bp, double* w, 
///     double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_dspgv(matrix_layout: Int, itype: Lapack_int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], bp: FixedArray[Double], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_dspgv"

///|
///
/// ```c
/// lapack_int LAPACKE_sspgvd(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, float* ap, float* bp, float* w, 
///     float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_sspgvd(matrix_layout: Int, itype: Lapack_int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], bp: FixedArray[Float], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_sspgvd"

///|
///
/// ```c
/// lapack_int LAPACKE_dspgvd(
///     int matrix_layout, lapack_int itype, char jobz, char uplo, 
///     lapack_int n, double* ap, double* bp, double* w, 
///     double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_dspgvd(matrix_layout: Int, itype: Lapack_int, jobz: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], bp: FixedArray[Double], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_dspgvd"

///|
///
/// ```c
/// lapack_int LAPACKE_sspgvx(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, float* ap, float* bp, 
///     float vl, float vu, lapack_int il, lapack_int iu, 
///     float abstol, lapack_int* m, float* w, float* z, 
///     lapack_int ldz, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_sspgvx(matrix_layout: Int, itype: Lapack_int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], bp: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_sspgvx"

///|
///
/// ```c
/// lapack_int LAPACKE_dspgvx(
///     int matrix_layout, lapack_int itype, char jobz, char range, 
///     char uplo, lapack_int n, double* ap, double* bp, 
///     double vl, double vu, lapack_int il, lapack_int iu, 
///     double abstol, lapack_int* m, double* w, double* z, 
///     lapack_int ldz, lapack_int* ifail
/// );
/// ```
pub extern "C" fn lapacke_dspgvx(matrix_layout: Int, itype: Lapack_int, jobz: Byte, range: Byte, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], bp: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, ifail: FixedArray[Int]) -> Lapack_int = "LAPACKE_dspgvx"

///|
///
/// ```c
/// lapack_int LAPACKE_ssprfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* ap, const float* afp, const lapack_int* ipiv, const float* b, 
///     lapack_int ldb, float* x, lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_ssprfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssprfs"

///|
///
/// ```c
/// lapack_int LAPACKE_dsprfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* ap, const double* afp, const lapack_int* ipiv, const double* b, 
///     lapack_int ldb, double* x, lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dsprfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsprfs"

///|
///
/// ```c
/// lapack_int LAPACKE_csprfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* ap, const lapack_complex_float* afp, const lapack_int* ipiv, const lapack_complex_float* b, 
///     lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_csprfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_csprfs"

///|
///
/// ```c
/// lapack_int LAPACKE_zsprfs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* ap, const lapack_complex_double* afp, const lapack_int* ipiv, const lapack_complex_double* b, 
///     lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_zsprfs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_zsprfs"

///|
///
/// ```c
/// lapack_int LAPACKE_sspsv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     float* ap, lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_sspsv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_sspsv"

///|
///
/// ```c
/// lapack_int LAPACKE_dspsv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     double* ap, lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dspsv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dspsv"

///|
///
/// ```c
/// lapack_int LAPACKE_cspsv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_float* ap, lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_cspsv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_cspsv"

///|
///
/// ```c
/// lapack_int LAPACKE_zspsv(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     lapack_complex_double* ap, lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zspsv(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zspsv"

///|
///
/// ```c
/// lapack_int LAPACKE_sspsvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const float* ap, float* afp, lapack_int* ipiv, 
///     const float* b, lapack_int ldb, float* x, lapack_int ldx, 
///     float* rcond, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_sspsvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_sspsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_dspsvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const double* ap, double* afp, lapack_int* ipiv, 
///     const double* b, lapack_int ldb, double* x, lapack_int ldx, 
///     double* rcond, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_dspsvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_dspsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_cspsvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_float* ap, lapack_complex_float* afp, lapack_int* ipiv, 
///     const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x, lapack_int ldx, 
///     float* rcond, float* ferr, float* berr
/// );
/// ```
pub extern "C" fn lapacke_cspsvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], afp: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int, x: FixedArray[Float], ldx: Lapack_int, rcond: FixedArray[Float], ferr: FixedArray[Float], berr: FixedArray[Float]) -> Lapack_int = "LAPACKE_cspsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_zspsvx(
///     int matrix_layout, char fact, char uplo, lapack_int n, 
///     lapack_int nrhs, const lapack_complex_double* ap, lapack_complex_double* afp, lapack_int* ipiv, 
///     const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x, lapack_int ldx, 
///     double* rcond, double* ferr, double* berr
/// );
/// ```
pub extern "C" fn lapacke_zspsvx(matrix_layout: Int, fact: Byte, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], afp: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int, x: FixedArray[Double], ldx: Lapack_int, rcond: FixedArray[Double], ferr: FixedArray[Double], berr: FixedArray[Double]) -> Lapack_int = "LAPACKE_zspsvx"

///|
///
/// ```c
/// lapack_int LAPACKE_ssptrd(
///     int matrix_layout, char uplo, lapack_int n, float* ap, 
///     float* d, float* e, float* tau
/// );
/// ```
pub extern "C" fn lapacke_ssptrd(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], d: FixedArray[Float], e: FixedArray[Float], tau: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssptrd"

///|
///
/// ```c
/// lapack_int LAPACKE_dsptrd(
///     int matrix_layout, char uplo, lapack_int n, double* ap, 
///     double* d, double* e, double* tau
/// );
/// ```
pub extern "C" fn lapacke_dsptrd(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], d: FixedArray[Double], e: FixedArray[Double], tau: FixedArray[Double]) -> Lapack_int = "LAPACKE_dsptrd"

///|
///
/// ```c
/// lapack_int LAPACKE_ssptrf(
///     int matrix_layout, char uplo, lapack_int n, float* ap, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_ssptrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_ssptrf"

///|
///
/// ```c
/// lapack_int LAPACKE_dsptrf(
///     int matrix_layout, char uplo, lapack_int n, double* ap, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_dsptrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsptrf"

///|
///
/// ```c
/// lapack_int LAPACKE_csptrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* ap, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_csptrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_csptrf"

///|
///
/// ```c
/// lapack_int LAPACKE_zsptrf(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* ap, lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_zsptrf(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_zsptrf"

///|
///
/// ```c
/// lapack_int LAPACKE_ssptri(
///     int matrix_layout, char uplo, lapack_int n, float* ap, const lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_ssptri(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_ssptri"

///|
///
/// ```c
/// lapack_int LAPACKE_dsptri(
///     int matrix_layout, char uplo, lapack_int n, double* ap, const lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_dsptri(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_dsptri"

///|
///
/// ```c
/// lapack_int LAPACKE_csptri(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_float* ap, const lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_csptri(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_csptri"

///|
///
/// ```c
/// lapack_int LAPACKE_zsptri(
///     int matrix_layout, char uplo, lapack_int n, lapack_complex_double* ap, const lapack_int* ipiv
/// );
/// ```
pub extern "C" fn lapacke_zsptri(matrix_layout: Int, uplo: Byte, n: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int]) -> Lapack_int = "LAPACKE_zsptri"

///|
///
/// ```c
/// lapack_int LAPACKE_ssptrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const float* ap, const lapack_int* ipiv, float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_ssptrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_ssptrs"

///|
///
/// ```c
/// lapack_int LAPACKE_dsptrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const double* ap, const lapack_int* ipiv, double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_dsptrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_dsptrs"

///|
///
/// ```c
/// lapack_int LAPACKE_csptrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_float* ap, const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_csptrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Float], ipiv: FixedArray[Int], b: FixedArray[Float], ldb: Lapack_int) -> Lapack_int = "LAPACKE_csptrs"

///|
///
/// ```c
/// lapack_int LAPACKE_zsptrs(
///     int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, 
///     const lapack_complex_double* ap, const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb
/// );
/// ```
pub extern "C" fn lapacke_zsptrs(matrix_layout: Int, uplo: Byte, n: Lapack_int, nrhs: Lapack_int, ap: FixedArray[Double], ipiv: FixedArray[Int], b: FixedArray[Double], ldb: Lapack_int) -> Lapack_int = "LAPACKE_zsptrs"

///|
///
/// ```c
/// lapack_int LAPACKE_sstebz(
///     char range, char order, lapack_int n, float vl, 
///     float vu, lapack_int il, lapack_int iu, float abstol, 
///     const float* d, const float* e, lapack_int* m, lapack_int* nsplit, 
///     float* w, lapack_int* iblock, lapack_int* isplit
/// );
/// ```
pub extern "C" fn lapacke_sstebz(range: Byte, order: Byte, n: Lapack_int, vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, d: FixedArray[Float], e: FixedArray[Float], m: FixedArray[Int], nsplit: FixedArray[Int], w: FixedArray[Float], iblock: FixedArray[Int], isplit: FixedArray[Int]) -> Lapack_int = "LAPACKE_sstebz"

///|
///
/// ```c
/// lapack_int LAPACKE_dstebz(
///     char range, char order, lapack_int n, double vl, 
///     double vu, lapack_int il, lapack_int iu, double abstol, 
///     const double* d, const double* e, lapack_int* m, lapack_int* nsplit, 
///     double* w, lapack_int* iblock, lapack_int* isplit
/// );
/// ```
pub extern "C" fn lapacke_dstebz(range: Byte, order: Byte, n: Lapack_int, vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, d: FixedArray[Double], e: FixedArray[Double], m: FixedArray[Int], nsplit: FixedArray[Int], w: FixedArray[Double], iblock: FixedArray[Int], isplit: FixedArray[Int]) -> Lapack_int = "LAPACKE_dstebz"

///|
///
/// ```c
/// lapack_int LAPACKE_sstedc(
///     int matrix_layout, char compz, lapack_int n, float* d, 
///     float* e, float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_sstedc(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_sstedc"

///|
///
/// ```c
/// lapack_int LAPACKE_dstedc(
///     int matrix_layout, char compz, lapack_int n, double* d, 
///     double* e, double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_dstedc(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_dstedc"

///|
///
/// ```c
/// lapack_int LAPACKE_cstedc(
///     int matrix_layout, char compz, lapack_int n, float* d, 
///     float* e, lapack_complex_float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_cstedc(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_cstedc"

///|
///
/// ```c
/// lapack_int LAPACKE_zstedc(
///     int matrix_layout, char compz, lapack_int n, double* d, 
///     double* e, lapack_complex_double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_zstedc(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_zstedc"

///|
///
/// ```c
/// lapack_int LAPACKE_sstegr(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     float* d, float* e, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, float* z, lapack_int ldz, lapack_int* isuppz
/// );
/// ```
pub extern "C" fn lapacke_sstegr(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, isuppz: FixedArray[Int]) -> Lapack_int = "LAPACKE_sstegr"

///|
///
/// ```c
/// lapack_int LAPACKE_dstegr(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     double* d, double* e, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, double* z, lapack_int ldz, lapack_int* isuppz
/// );
/// ```
pub extern "C" fn lapacke_dstegr(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, isuppz: FixedArray[Int]) -> Lapack_int = "LAPACKE_dstegr"

///|
///
/// ```c
/// lapack_int LAPACKE_cstegr(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     float* d, float* e, float vl, float vu, 
///     lapack_int il, lapack_int iu, float abstol, lapack_int* m, 
///     float* w, lapack_complex_float* z, lapack_int ldz, lapack_int* isuppz
/// );
/// ```
pub extern "C" fn lapacke_cstegr(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, abstol: Float, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, isuppz: FixedArray[Int]) -> Lapack_int = "LAPACKE_cstegr"

///|
///
/// ```c
/// lapack_int LAPACKE_zstegr(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     double* d, double* e, double vl, double vu, 
///     lapack_int il, lapack_int iu, double abstol, lapack_int* m, 
///     double* w, lapack_complex_double* z, lapack_int ldz, lapack_int* isuppz
/// );
/// ```
pub extern "C" fn lapacke_zstegr(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, abstol: Double, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, isuppz: FixedArray[Int]) -> Lapack_int = "LAPACKE_zstegr"

///|
///
/// ```c
/// lapack_int LAPACKE_sstein(
///     int matrix_layout, lapack_int n, const float* d, const float* e, 
///     lapack_int m, const float* w, const lapack_int* iblock, const lapack_int* isplit, 
///     float* z, lapack_int ldz, lapack_int* ifailv
/// );
/// ```
pub extern "C" fn lapacke_sstein(matrix_layout: Int, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], m: Lapack_int, w: FixedArray[Float], iblock: FixedArray[Int], isplit: FixedArray[Int], z: FixedArray[Float], ldz: Lapack_int, ifailv: FixedArray[Int]) -> Lapack_int = "LAPACKE_sstein"

///|
///
/// ```c
/// lapack_int LAPACKE_dstein(
///     int matrix_layout, lapack_int n, const double* d, const double* e, 
///     lapack_int m, const double* w, const lapack_int* iblock, const lapack_int* isplit, 
///     double* z, lapack_int ldz, lapack_int* ifailv
/// );
/// ```
pub extern "C" fn lapacke_dstein(matrix_layout: Int, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], m: Lapack_int, w: FixedArray[Double], iblock: FixedArray[Int], isplit: FixedArray[Int], z: FixedArray[Double], ldz: Lapack_int, ifailv: FixedArray[Int]) -> Lapack_int = "LAPACKE_dstein"

///|
///
/// ```c
/// lapack_int LAPACKE_cstein(
///     int matrix_layout, lapack_int n, const float* d, const float* e, 
///     lapack_int m, const float* w, const lapack_int* iblock, const lapack_int* isplit, 
///     lapack_complex_float* z, lapack_int ldz, lapack_int* ifailv
/// );
/// ```
pub extern "C" fn lapacke_cstein(matrix_layout: Int, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], m: Lapack_int, w: FixedArray[Float], iblock: FixedArray[Int], isplit: FixedArray[Int], z: FixedArray[Float], ldz: Lapack_int, ifailv: FixedArray[Int]) -> Lapack_int = "LAPACKE_cstein"

///|
///
/// ```c
/// lapack_int LAPACKE_zstein(
///     int matrix_layout, lapack_int n, const double* d, const double* e, 
///     lapack_int m, const double* w, const lapack_int* iblock, const lapack_int* isplit, 
///     lapack_complex_double* z, lapack_int ldz, lapack_int* ifailv
/// );
/// ```
pub extern "C" fn lapacke_zstein(matrix_layout: Int, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], m: Lapack_int, w: FixedArray[Double], iblock: FixedArray[Int], isplit: FixedArray[Int], z: FixedArray[Double], ldz: Lapack_int, ifailv: FixedArray[Int]) -> Lapack_int = "LAPACKE_zstein"

///|
///
/// ```c
/// lapack_int LAPACKE_sstemr(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     float* d, float* e, float vl, float vu, 
///     lapack_int il, lapack_int iu, lapack_int* m, float* w, 
///     float* z, lapack_int ldz, lapack_int nzc, lapack_int* isuppz, lapack_logical* tryrac
/// );
/// ```
pub extern "C" fn lapacke_sstemr(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, nzc: Lapack_int, isuppz: FixedArray[Int], tryrac: Lapack_logical) -> Lapack_int = "LAPACKE_sstemr"

///|
///
/// ```c
/// lapack_int LAPACKE_dstemr(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     double* d, double* e, double vl, double vu, 
///     lapack_int il, lapack_int iu, lapack_int* m, double* w, 
///     double* z, lapack_int ldz, lapack_int nzc, lapack_int* isuppz, lapack_logical* tryrac
/// );
/// ```
pub extern "C" fn lapacke_dstemr(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, nzc: Lapack_int, isuppz: FixedArray[Int], tryrac: Lapack_logical) -> Lapack_int = "LAPACKE_dstemr"

///|
///
/// ```c
/// lapack_int LAPACKE_cstemr(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     float* d, float* e, float vl, float vu, 
///     lapack_int il, lapack_int iu, lapack_int* m, float* w, 
///     lapack_complex_float* z, lapack_int ldz, lapack_int nzc, lapack_int* isuppz, lapack_logical* tryrac
/// );
/// ```
pub extern "C" fn lapacke_cstemr(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], vl: Float, vu: Float, il: Lapack_int, iu: Lapack_int, m: FixedArray[Int], w: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int, nzc: Lapack_int, isuppz: FixedArray[Int], tryrac: Lapack_logical) -> Lapack_int = "LAPACKE_cstemr"

///|
///
/// ```c
/// lapack_int LAPACKE_zstemr(
///     int matrix_layout, char jobz, char range, lapack_int n, 
///     double* d, double* e, double vl, double vu, 
///     lapack_int il, lapack_int iu, lapack_int* m, double* w, 
///     lapack_complex_double* z, lapack_int ldz, lapack_int nzc, lapack_int* isuppz, lapack_logical* tryrac
/// );
/// ```
pub extern "C" fn lapacke_zstemr(matrix_layout: Int, jobz: Byte, range: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], vl: Double, vu: Double, il: Lapack_int, iu: Lapack_int, m: FixedArray[Int], w: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int, nzc: Lapack_int, isuppz: FixedArray[Int], tryrac: Lapack_logical) -> Lapack_int = "LAPACKE_zstemr"

///|
///
/// ```c
/// lapack_int LAPACKE_ssteqr(
///     int matrix_layout, char compz, lapack_int n, float* d, 
///     float* e, float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_ssteqr(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_ssteqr"

///|
///
/// ```c
/// lapack_int LAPACKE_dsteqr(
///     int matrix_layout, char compz, lapack_int n, double* d, 
///     double* e, double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_dsteqr(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_dsteqr"

///|
///
/// ```c
/// lapack_int LAPACKE_csteqr(
///     int matrix_layout, char compz, lapack_int n, float* d, 
///     float* e, lapack_complex_float* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_csteqr(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float], z: FixedArray[Float], ldz: Lapack_int) -> Lapack_int = "LAPACKE_csteqr"

///|
///
/// ```c
/// lapack_int LAPACKE_zsteqr(
///     int matrix_layout, char compz, lapack_int n, double* d, 
///     double* e, lapack_complex_double* z, lapack_int ldz
/// );
/// ```
pub extern "C" fn lapacke_zsteqr(matrix_layout: Int, compz: Byte, n: Lapack_int, d: FixedArray[Double], e: FixedArray[Double], z: FixedArray[Double], ldz: Lapack_int) -> Lapack_int = "LAPACKE_zsteqr"

///|
///
/// ```c
/// lapack_int LAPACKE_ssterf(lapack_int n, float* d, float* e);
/// ```
pub extern "C" fn lapacke_ssterf(n: Lapack_int, d: FixedArray[Float], e: FixedArray[Float]) -> Lapack_int = "LAPACKE_ssterf"

