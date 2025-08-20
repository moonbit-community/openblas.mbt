#external
pub type VoidPtr

fn VoidPtr::to_uint64(p: Self) -> UInt64 = "%identity"

fn[T] to_void_ptr(p: T) -> VoidPtr = "%identity"

extern "C" fn cfree(p: VoidPtr) = "free"

fn[T] free(p: T) -> Unit {
  let ptr = to_void_ptr(p)
  if ptr.to_uint64() != 0 {
    cfree(ptr)
  }
}

#external
priv type CStr

extern "C" fn CStr::to_string(self: Self) -> String = "cstr_to_moonbit_str";

typealias UInt64 as OpenBlasComplexFloat

///| Set the number of threads on runtime.*/
/// 
/// ```c
/// void openblas_set_num_threads(int num_threads);
/// ```
pub extern "C" fn openblas_set_num_threads(num_threads: Int) = "openblas_set_num_threads";

///|
/// void goto_set_num_threads(int num_threads);
pub extern "C" fn goto_set_num_threads(num_threads: Int) = "goto_set_num_threads";

//int openblas_set_num_threads_local(int num_threads);
pub extern "C" fn openblas_set_num_threads_local(num_threads: Int) -> Int = "openblas_set_num_threads_local";

///*Get the number of threads on runtime.*/
//int openblas_get_num_threads(void);
pub extern "C" fn openblas_get_num_threads() -> Int = "openblas_get_num_threads";
//
///*Get the number of physical processors (cores).*/
//int openblas_get_num_procs(void);
pub extern "C" fn openblas_get_num_procs() -> Int = "openblas_get_num_procs";

///*Get the build configure on runtime.*/
//char* openblas_get_config(void);
extern "C" fn __openblas_get_config() -> CStr = "openblas_get_config";
pub fn openblas_get_config() -> String {
  let cstr = __openblas_get_config()
  let str = cstr.to_string()
  free(cstr)
  str
}

///| Get the CPU corename on runtime.*/
///
/// ```c
/// char* openblas_get_corename(void);
/// ```
extern "C" fn __openblas_get_corename() -> CStr = "openblas_get_corename";
pub fn openblas_get_corename() -> String {
  let cstr = __openblas_get_corename()
  let str = cstr.to_string()
  free(cstr)
  str
}


///| Set the threading backend to a custom callback.*/
///
/// ```c
/// typedef void (*openblas_dojob_callback)(int thread_num, void *jobdata, int dojob_data);
/// ```
pub typealias FuncRef[(Int, VoidPtr, Int) -> Unit] as Openblas_dojob_callback ;

///|
///
/// ```c
/// typedef void (*openblas_threads_callback)(int sync, openblas_dojob_callback dojob, int numjobs, size_t jobdata_elsize, void *jobdata, int dojob_data);
/// ``` 
pub typealias FuncRef[(Int, Openblas_dojob_callback, Int, Int, VoidPtr, Int) -> Unit] as Openblas_threads_callback;

///|
///
/// ```c
/// void openblas_set_threads_callback_function(openblas_threads_callback callback);
/// ```
pub extern "C" fn openblas_set_threads_callback_function(callback: Openblas_threads_callback) = "openblas_set_threads_callback_function";

//
//#ifdef OPENBLAS_OS_LINUX
///* Sets thread affinity for OpenBLAS threads. `thread_idx` is in [0, openblas_get_num_threads()-1]. */
//int openblas_setaffinity(int thread_idx, size_t cpusetsize, cpu_set_t* cpu_set);
///* Queries thread affinity for OpenBLAS threads. `thread_idx` is in [0, openblas_get_num_threads()-1]. */
//int openblas_getaffinity(int thread_idx, size_t cpusetsize, cpu_set_t* cpu_set);
//#endif


///* Get the parallelization type which is used by OpenBLAS */
//int openblas_get_parallel(void);
pub extern "C" fn openblas_get_parallel() -> Int = "openblas_get_parallel";


///* OpenBLAS is compiled for sequential use  */
//#define OPENBLAS_SEQUENTIAL  0
pub const OPENBLAS_SEQUENTIAL: Int = 0;

///* OpenBLAS is compiled using normal threading model */
//#define OPENBLAS_THREAD  1
pub const OPENBLAS_THREAD: Int = 1;

///* OpenBLAS is compiled using OpenMP threading model */
//#define OPENBLAS_OPENMP 2
pub const OPENBLAS_OPENMP: Int = 2;

//typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
pub(all) enum CBLAS_ORDER {
  CblasRowMajor = 101;
  CblasColMajor = 102;
}

//typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
pub(all) enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111;
  CblasTrans = 112;
  CblasConjTrans = 113;
  CblasConjNoTrans = 114;
}

//typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
pub(all) enum CBLAS_UPLO {
  CblasUpper = 121;
  CblasLower = 122;
}

//typedef enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
pub(all) enum CBLAS_DIAG {
  CblasNonUnit = 131;
  CblasUnit = 132;
}

//typedef enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
pub(all) enum CBLAS_SIDE {
  CblasLeft = 141;
  CblasRight = 142;
}

//typedef CBLAS_ORDER CBLAS_LAYOUT;
typealias CBLAS_ORDER as CBLAS_LAYOUT;

//float  cblas_sdsdot(const blasint n, const float alpha, const float *x, const blasint incx, const float *y, const blasint incy);
pub extern "C" fn cblas_sdsdot(n: Int, alpha: Float, x: FixedArray[Float], incx: Int, y: FixedArray[Float], incy: Int) -> Float = "cblas_sdsdot";

//double cblas_dsdot (const blasint n, const float *x, const blasint incx, const float *y, const blasint incy);
pub extern "C" fn cblas_dsdot(n: Int, x: FixedArray[Float], incx: Int, y: FixedArray[Float], incy: Int) -> Double = "cblas_dsdot";

//float  cblas_sdot(const blasint n, const float  *x, const blasint incx, const float  *y, const blasint incy);
pub extern "C" fn cblas_sdot(n: Int, x: FixedArray[Float], incx: Int, y: FixedArray[Float], incy: Int) -> Float = "cblas_sdot";

//double cblas_ddot(const blasint n, const double *x, const blasint incx, const double *y, const blasint incy);
pub extern "C" fn cblas_ddot(n: Int, x: FixedArray[Double], incx: Int, y: FixedArray[Double], incy: Int) -> Double = "cblas_ddot";

//openblas_complex_float  cblas_cdotu(const blasint n, const void  *x, const blasint incx, const void  *y, const blasint incy);
pub extern "C" fn cblas_cdotu(n: Int, x: FixedArray[OpenBlasComplexFloat], incx: Int, y: FixedArray[OpenBlasComplexFloat], incy: Int) -> OpenBlasComplexFloat = "cblas_cdotu";
//openblas_complex_float  cblas_cdotc(const blasint n, const void  *x, const blasint incx, const void  *y, const blasint incy);
pub extern "C" fn cblas_cdotc(n: Int, x: FixedArray[OpenBlasComplexFloat], incx: Int, y: FixedArray[OpenBlasComplexFloat], incy: Int) -> OpenBlasComplexFloat = "cblas_cdotc";
//openblas_complex_double cblas_zdotu(const blasint n, const void *x, const blasint incx, const void *y, const blasint incy);
pub extern "C" fn cblas_zdotu(n: Int, x: FixedArray[OpenBlasComplexFloat], incx: Int, y: FixedArray[OpenBlasComplexFloat], incy: Int) -> OpenBlasComplexFloat = "cblas_zdotu";
//openblas_complex_double cblas_zdotc(const blasint n, const void *x, const blasint incx, const void *y, const blasint incy);
pub extern "C" fn cblas_zdotc(n: Int, x: FixedArray[OpenBlasComplexFloat], incx: Int, y: FixedArray[OpenBlasComplexFloat], incy: Int) -> OpenBlasComplexFloat = "cblas_zdotc";

//void  cblas_cdotu_sub(const blasint n, const void  *x, const blasint incx, const void  *y, const blasint incy, void  *ret);
pub extern "C" fn cblas_cdotu_sub(n: Int, x: FixedArray[OpenBlasComplexFloat], incx: Int, y: FixedArray[OpenBlasComplexFloat], incy: Int, ret: VoidPtr) = "cblas_cdotu_sub";
//void  cblas_cdotc_sub(const blasint n, const void  *x, const blasint incx, const void  *y, const blasint incy, void  *ret);
pub extern "C" fn cblas_cdotc_sub(n: Int, x: FixedArray[OpenBlasComplexFloat], incx: Int, y: FixedArray[OpenBlasComplexFloat], incy: Int, ret: VoidPtr) = "cblas_cdotc_sub";
//void  cblas_zdotu_sub(const blasint n, const void *x, const blasint incx, const void *y, const blasint incy, void *ret);
pub extern "C" fn cblas_zdotu_sub(n: Int, x: FixedArray[OpenBlasComplexFloat], incx: Int, y: FixedArray[OpenBlasComplexFloat], incy: Int, ret: VoidPtr) = "cblas_zdotu_sub";
//void  cblas_zdotc_sub(const blasint n, const void *x, const blasint incx, const void *y, const blasint incy, void *ret);
pub extern "C" fn cblas_zdotc_sub(n: Int, x: FixedArray[OpenBlasComplexFloat], incx: Int, y: FixedArray[OpenBlasComplexFloat], incy: Int, ret: VoidPtr) = "cblas_zdotc_sub";

//float  cblas_sasum (const blasint n, const float  *x, const blasint incx);
pub extern "C" fn cblas_sasum(n: Int, x: FixedArray[Float], incx: Int) -> Float = "cblas_sasum";
//double cblas_dasum (const blasint n, const double *x, const blasint incx);
pub extern "C" fn cblas_dasum(n: Int, x: FixedArray[Double], incx: Int) -> Double = "cblas_dasum";
//float  cblas_scasum(const blasint n, const void  *x, const blasint incx);
pub extern "C" fn cblas_scasum(n: Int, x: FixedArray[OpenBlasComplexFloat], incx: Int) -> Float = "cblas_scasum";
//double cblas_dzasum(const blasint n, const void *x, const blasint incx);
pub extern "C" fn cblas_dzasum(n: Int, x: FixedArray[OpenBlasComplexFloat], incx: Int) -> Double = "cblas_dzasum";


//float  cblas_ssum (const blasint n, const float  *x, const blasint incx);
pub extern "C" fn cblas_ssum(n: Int, x: FixedArray[Float], incx: Int) -> Float = "cblas_ssum";
//double cblas_dsum (const blasint n, const double *x, const blasint incx);
pub extern "C" fn cblas_dsum(n: Int, x: FixedArray[Double], incx: Int) -> Double = "cblas_dsum";
//float  cblas_scsum(const blasint n, const void  *x, const blasint incx);
pub extern "C" fn cblas_scsum(n: Int, x: VoidPtr, incx: Int) -> Float = "cblas_scsum";
//double cblas_dzsum(const blasint n, const void *x, const blasint incx);
pub extern "C" fn cblas_dzsum(n: Int, x: VoidPtr, incx: Int) -> Double = "cblas_dzsum";

//float  cblas_snrm2 (const blasint N, const float  *X, const blasint incX);
pub extern "C" fn cblas_snrm2(n: Int, x: FixedArray[Float], incx: Int) -> Float = "cblas_snrm2";
//double cblas_dnrm2 (const blasint N, const double *X, const blasint incX);
pub extern "C" fn cblas_dnrm2(n: Int, x: FixedArray[Double], incx: Int) -> Double = "cblas_dnrm2";
//float  cblas_scnrm2(const blasint N, const void  *X, const blasint incX);
pub extern "C" fn cblas_scnrm2(n: Int, x: FixedArray[OpenBlasComplexFloat], incx: Int) -> Float = "cblas_scnrm2";
//double cblas_dznrm2(const blasint N, const void *X, const blasint incX);
pub extern "C" fn cblas_dznrm2(n: Int, x: FixedArray[OpenBlasComplexFloat], incx: Int) -> Double = "cblas_dznrm2";

//UInt64 cblas_isamax(const blasint n, const float  *x, const blasint incx);
pub extern "C" fn cblas_isamax(n: Int, x: FixedArray[Float], incx: Int) -> UInt64 = "cblas_isamax";
//UInt64 cblas_idamax(const blasint n, const double *x, const blasint incx);
pub extern "C" fn cblas_idamax(n: Int, x: FixedArray[Double], incx: Int) -> UInt64 = "cblas_idamax";
//UInt64 cblas_icamax(const blasint n, const void  *x, const blasint incx);
pub extern "C" fn cblas_icamax(n: Int, x: FixedArray[OpenBlasComplexFloat], incx: Int) -> UInt64 = "cblas_icamax";
//UInt64 cblas_izamax(const blasint n, const void *x, const blasint incx);
pub extern "C" fn cblas_izamax(n: Int, x: FixedArray[OpenBlasComplexFloat], incx: Int) -> UInt64 = "cblas_izamax";

//UInt64 cblas_isamin(const blasint n, const float  *x, const blasint incx);
//UInt64 cblas_idamin(const blasint n, const double *x, const blasint incx);
//UInt64 cblas_icamin(const blasint n, const void  *x, const blasint incx);
//UInt64 cblas_izamin(const blasint n, const void *x, const blasint incx);
//
//float cblas_samax(const blasint n, const float  *x, const blasint incx);
//double cblas_damax(const blasint n, const double *x, const blasint incx);
//float cblas_scamax(const blasint n, const void  *x, const blasint incx);
//double cblas_dzamax(const blasint n, const void *x, const blasint incx);
//
//float cblas_samin(const blasint n, const float  *x, const blasint incx);
//double cblas_damin(const blasint n, const double *x, const blasint incx);
//float cblas_scamin(const blasint n, const void  *x, const blasint incx);
//double cblas_dzamin(const blasint n, const void *x, const blasint incx);
//
//UInt64 cblas_ismax(const blasint n, const float  *x, const blasint incx);
//UInt64 cblas_idmax(const blasint n, const double *x, const blasint incx);
//UInt64 cblas_icmax(const blasint n, const void  *x, const blasint incx);
//UInt64 cblas_izmax(const blasint n, const void *x, const blasint incx);
//
//UInt64 cblas_ismin(const blasint n, const float  *x, const blasint incx);
//UInt64 cblas_idmin(const blasint n, const double *x, const blasint incx);
//UInt64 cblas_icmin(const blasint n, const void  *x, const blasint incx);
//UInt64 cblas_izmin(const blasint n, const void *x, const blasint incx);
//
//void cblas_saxpy(const blasint n, const float alpha, const float *x, const blasint incx, float *y, const blasint incy);
//void cblas_daxpy(const blasint n, const double alpha, const double *x, const blasint incx, double *y, const blasint incy);
//void cblas_caxpy(const blasint n, const void *alpha, const void *x, const blasint incx, void *y, const blasint incy);
//void cblas_zaxpy(const blasint n, const void *alpha, const void *x, const blasint incx, void *y, const blasint incy);
//
//void cblas_caxpyc(const blasint n, const void *alpha, const void *x, const blasint incx, void *y, const blasint incy);
//void cblas_zaxpyc(const blasint n, const void *alpha, const void *x, const blasint incx, void *y, const blasint incy);
//
//void cblas_scopy(const blasint n, const float *x, const blasint incx, float *y, const blasint incy);
//void cblas_dcopy(const blasint n, const double *x, const blasint incx, double *y, const blasint incy);
//void cblas_ccopy(const blasint n, const void *x, const blasint incx, void *y, const blasint incy);
//void cblas_zcopy(const blasint n, const void *x, const blasint incx, void *y, const blasint incy);
//
//void cblas_sswap(const blasint n, float *x, const blasint incx, float *y, const blasint incy);
//void cblas_dswap(const blasint n, double *x, const blasint incx, double *y, const blasint incy);
//void cblas_cswap(const blasint n, void *x, const blasint incx, void *y, const blasint incy);
//void cblas_zswap(const blasint n, void *x, const blasint incx, void *y, const blasint incy);
//
//void cblas_srot(const blasint N, float *X, const blasint incX, float *Y, const blasint incY, const float c, const float s);
//void cblas_drot(const blasint N, double *X, const blasint incX, double *Y, const blasint incY, const double c, const double  s);
//void cblas_csrot(const blasint n, const void *x, const blasint incx, void *y, const blasint incY, const float c, const float s);
//void cblas_zdrot(const blasint n, const void *x, const blasint incx, void *y, const blasint incY, const double c, const double s);
//
//void cblas_srotg(float *a, float *b, float *c, float *s);
//void cblas_drotg(double *a, double *b, double *c, double *s);
//void cblas_crotg(void *a, void *b, float *c, void *s);
//void cblas_zrotg(void *a, void *b, double *c, void *s);
//
//
//void cblas_srotm(const blasint N, float *X, const blasint incX, float *Y, const blasint incY, const float *P);
//void cblas_drotm(const blasint N, double *X, const blasint incX, double *Y, const blasint incY, const double *P);
//
//void cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
//void cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P);
//
//void cblas_sscal(const blasint N, const float alpha, float *X, const blasint incX);
//void cblas_dscal(const blasint N, const double alpha, double *X, const blasint incX);
//void cblas_cscal(const blasint N, const void *alpha, void *X, const blasint incX);
//void cblas_zscal(const blasint N, const void *alpha, void *X, const blasint incX);
//void cblas_csscal(const blasint N, const float alpha, void *X, const blasint incX);
//void cblas_zdscal(const blasint N, const double alpha, void *X, const blasint incX);
//
//void cblas_sgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const blasint m, const blasint n,
//		 const float alpha, const float  *a, const blasint lda,  const float  *x, const blasint incx,  const float beta,  float  *y, const blasint incy);
//void cblas_dgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const blasint m, const blasint n,
//		 const double alpha, const double  *a, const blasint lda,  const double  *x, const blasint incx,  const double beta,  double  *y, const blasint incy);
//void cblas_cgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const blasint m, const blasint n,
//		 const void *alpha, const void  *a, const blasint lda,  const void  *x, const blasint incx,  const void *beta,  void  *y, const blasint incy);
//void cblas_zgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const blasint m, const blasint n,
//		 const void *alpha, const void  *a, const blasint lda,  const void  *x, const blasint incx,  const void *beta,  void  *y, const blasint incy);
//
//void cblas_sger (const enum CBLAS_ORDER order, const blasint M, const blasint N, const float   alpha, const float  *X, const blasint incX, const float  *Y, const blasint incY, float  *A, const blasint lda);
//void cblas_dger (const enum CBLAS_ORDER order, const blasint M, const blasint N, const double  alpha, const double *X, const blasint incX, const double *Y, const blasint incY, double *A, const blasint lda);
//void cblas_cgeru(const enum CBLAS_ORDER order, const blasint M, const blasint N, const void  *alpha, const void  *X, const blasint incX, const void  *Y, const blasint incY, void  *A, const blasint lda);
//void cblas_cgerc(const enum CBLAS_ORDER order, const blasint M, const blasint N, const void  *alpha, const void  *X, const blasint incX, const void  *Y, const blasint incY, void  *A, const blasint lda);
//void cblas_zgeru(const enum CBLAS_ORDER order, const blasint M, const blasint N, const void *alpha, const void *X, const blasint incX, const void *Y, const blasint incY, void *A, const blasint lda);
//void cblas_zgerc(const enum CBLAS_ORDER order, const blasint M, const blasint N, const void *alpha, const void *X, const blasint incX, const void *Y, const blasint incY, void *A, const blasint lda);
//
//void cblas_strsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const float *A, const blasint lda, float *X, const blasint incX);
//void cblas_dtrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const double *A, const blasint lda, double *X, const blasint incX);
//void cblas_ctrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const void *A, const blasint lda, void *X, const blasint incX);
//void cblas_ztrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const void *A, const blasint lda, void *X, const blasint incX);
//
//void cblas_strmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const float *A, const blasint lda, float *X, const blasint incX);
//void cblas_dtrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const double *A, const blasint lda, double *X, const blasint incX);
//void cblas_ctrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const void *A, const blasint lda, void *X, const blasint incX);
//void cblas_ztrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const void *A, const blasint lda, void *X, const blasint incX);
//
//void cblas_ssyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const float alpha, const float *X, const blasint incX, float *A, const blasint lda);
//void cblas_dsyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const double *X, const blasint incX, double *A, const blasint lda);
//void cblas_cher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const float alpha, const void *X, const blasint incX, void *A, const blasint lda);
//void cblas_zher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const void *X, const blasint incX, void *A, const blasint lda);
//
//void cblas_ssyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,const blasint N, const float alpha, const float *X,
//                const blasint incX, const float *Y, const blasint incY, float *A, const blasint lda);
//void cblas_dsyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const double *X,
//                const blasint incX, const double *Y, const blasint incY, double *A, const blasint lda);
//void cblas_cher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const void *alpha, const void *X, const blasint incX,
//                const void *Y, const blasint incY, void *A, const blasint lda);
//void cblas_zher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const void *alpha, const void *X, const blasint incX,
//                const void *Y, const blasint incY, void *A, const blasint lda);
//
//void cblas_sgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const blasint M, const blasint N,
//                 const blasint KL, const blasint KU, const float alpha, const float *A, const blasint lda, const float *X, const blasint incX, const float beta, float *Y, const blasint incY);
//void cblas_dgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const blasint M, const blasint N,
//                 const blasint KL, const blasint KU, const double alpha, const double *A, const blasint lda, const double *X, const blasint incX, const double beta, double *Y, const blasint incY);
//void cblas_cgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const blasint M, const blasint N,
//                 const blasint KL, const blasint KU, const void *alpha, const void *A, const blasint lda, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
//void cblas_zgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const blasint M, const blasint N,
//                 const blasint KL, const blasint KU, const void *alpha, const void *A, const blasint lda, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
//
//void cblas_ssbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const blasint K, const float alpha, const float *A,
//                 const blasint lda, const float *X, const blasint incX, const float beta, float *Y, const blasint incY);
//void cblas_dsbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const blasint K, const double alpha, const double *A,
//                 const blasint lda, const double *X, const blasint incX, const double beta, double *Y, const blasint incY);
//
//
//void cblas_stbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const blasint K, const float *A, const blasint lda, float *X, const blasint incX);
//void cblas_dtbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const blasint K, const double *A, const blasint lda, double *X, const blasint incX);
//void cblas_ctbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const blasint K, const void *A, const blasint lda, void *X, const blasint incX);
//void cblas_ztbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const blasint K, const void *A, const blasint lda, void *X, const blasint incX);
//
//void cblas_stbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const blasint K, const float *A, const blasint lda, float *X, const blasint incX);
//void cblas_dtbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const blasint K, const double *A, const blasint lda, double *X, const blasint incX);
//void cblas_ctbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const blasint K, const void *A, const blasint lda, void *X, const blasint incX);
//void cblas_ztbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const blasint K, const void *A, const blasint lda, void *X, const blasint incX);
//
//void cblas_stpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const float *Ap, float *X, const blasint incX);
//void cblas_dtpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const double *Ap, double *X, const blasint incX);
//void cblas_ctpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const void *Ap, void *X, const blasint incX);
//void cblas_ztpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const void *Ap, void *X, const blasint incX);
//
//void cblas_stpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const float *Ap, float *X, const blasint incX);
//void cblas_dtpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const double *Ap, double *X, const blasint incX);
//void cblas_ctpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const void *Ap, void *X, const blasint incX);
//void cblas_ztpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
//                 const blasint N, const void *Ap, void *X, const blasint incX);
//
//void cblas_ssymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const float alpha, const float *A,
//                 const blasint lda, const float *X, const blasint incX, const float beta, float *Y, const blasint incY);
//void cblas_dsymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const double *A,
//                 const blasint lda, const double *X, const blasint incX, const double beta, double *Y, const blasint incY);
//void cblas_chemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const void *alpha, const void *A,
//                 const blasint lda, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
//void cblas_zhemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const void *alpha, const void *A,
//                 const blasint lda, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
//
//
//void cblas_sspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const float alpha, const float *Ap,
//                 const float *X, const blasint incX, const float beta, float *Y, const blasint incY);
//void cblas_dspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const double *Ap,
//                 const double *X, const blasint incX, const double beta, double *Y, const blasint incY);
//
//void cblas_sspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const float alpha, const float *X, const blasint incX, float *Ap);
//void cblas_dspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const double *X, const blasint incX, double *Ap);
//
//void cblas_chpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const float alpha, const void *X, const blasint incX, void *A);
//void cblas_zhpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const void *X,const blasint incX, void *A);
//
//void cblas_sspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const float alpha, const float *X, const blasint incX, const float *Y, const blasint incY, float *A);
//void cblas_dspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const double *X, const blasint incX, const double *Y, const blasint incY, double *A);
//void cblas_chpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const void *alpha, const void *X, const blasint incX, const void *Y, const blasint incY, void *Ap);
//void cblas_zhpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const void *alpha, const void *X, const blasint incX, const void *Y, const blasint incY, void *Ap);
//
//void cblas_chbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const blasint K,
//		 const void *alpha, const void *A, const blasint lda, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
//void cblas_zhbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const blasint K,
//		 const void *alpha, const void *A, const blasint lda, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
//
//void cblas_chpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N,
//		 const void *alpha, const void *Ap, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
//void cblas_zhpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N,
//		 const void *alpha, const void *Ap, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
//
//void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
//		 const float alpha, const float *A, const blasint lda, const float *B, const blasint ldb, const float beta, float *C, const blasint ldc);
//void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
//		 const double alpha, const double *A, const blasint lda, const double *B, const blasint ldb, const double beta, double *C, const blasint ldc);
//void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
//		 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
//void cblas_cgemm3m(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
//		 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
//void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
//		 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
//void cblas_zgemm3m(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
//		 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
//
//void cblas_sgemmt(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint K,
//		 const float alpha, const float *A, const blasint lda, const float *B, const blasint ldb, const float beta, float *C, const blasint ldc);
//void cblas_dgemmt(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint K,
//		 const double alpha, const double *A, const blasint lda, const double *B, const blasint ldb, const double beta, double *C, const blasint ldc);
//void cblas_cgemmt(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint K,
//		 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
//void cblas_zgemmt(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint K,
//		 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
//
//void cblas_ssymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const blasint M, const blasint N,
//                 const float alpha, const float *A, const blasint lda, const float *B, const blasint ldb, const float beta, float *C, const blasint ldc);
//void cblas_dsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const blasint M, const blasint N,
//                 const double alpha, const double *A, const blasint lda, const double *B, const blasint ldb, const double beta, double *C, const blasint ldc);
//void cblas_csymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const blasint M, const blasint N,
//                 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
//void cblas_zsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const blasint M, const blasint N,
//                 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
//
//void cblas_ssyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
//		 const blasint N, const blasint K, const float alpha, const float *A, const blasint lda, const float beta, float *C, const blasint ldc);
//void cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
//		 const blasint N, const blasint K, const double alpha, const double *A, const blasint lda, const double beta, double *C, const blasint ldc);
//void cblas_csyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
//		 const blasint N, const blasint K, const void *alpha, const void *A, const blasint lda, const void *beta, void *C, const blasint ldc);
//void cblas_zsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
//		 const blasint N, const blasint K, const void *alpha, const void *A, const blasint lda, const void *beta, void *C, const blasint ldc);
//
//void cblas_ssyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
//		  const blasint N, const blasint K, const float alpha, const float *A, const blasint lda, const float *B, const blasint ldb, const float beta, float *C, const blasint ldc);
//void cblas_dsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
//		  const blasint N, const blasint K, const double alpha, const double *A, const blasint lda, const double *B, const blasint ldb, const double beta, double *C, const blasint ldc);
//void cblas_csyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
//		  const blasint N, const blasint K, const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
//void cblas_zsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
//		  const blasint N, const blasint K, const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
//
//void cblas_strmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
//                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const float alpha, const float *A, const blasint lda, float *B, const blasint ldb);
//void cblas_dtrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
//                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const double alpha, const double *A, const blasint lda, double *B, const blasint ldb);
//void cblas_ctrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
//                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const void *alpha, const void *A, const blasint lda, void *B, const blasint ldb);
//void cblas_ztrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
//                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const void *alpha, const void *A, const blasint lda, void *B, const blasint ldb);
//
//void cblas_strsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
//                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const float alpha, const float *A, const blasint lda, float *B, const blasint ldb);
//void cblas_dtrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
//                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const double alpha, const double *A, const blasint lda, double *B, const blasint ldb);
//void cblas_ctrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
//                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const void *alpha, const void *A, const blasint lda, void *B, const blasint ldb);
//void cblas_ztrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
//                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const void *alpha, const void *A, const blasint lda, void *B, const blasint ldb);
//
//void cblas_chemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const blasint M, const blasint N,
//                 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
//void cblas_zhemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const blasint M, const blasint N,
//                 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
//
//void cblas_cherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const blasint N, const blasint K,
//                 const float alpha, const void *A, const blasint lda, const float beta, void *C, const blasint ldc);
//void cblas_zherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const blasint N, const blasint K,
//                 const double alpha, const void *A, const blasint lda, const double beta, void *C, const blasint ldc);
//
//void cblas_cher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const blasint N, const blasint K,
//                  const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const float beta, void *C, const blasint ldc);
//void cblas_zher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const blasint N, const blasint K,
//                  const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const double beta, void *C, const blasint ldc);
//
//void cblas_xerbla(blasint p, const char *rout, const char *form, ...);
//
///*** BLAS extensions ***/
//
//void cblas_saxpby(const blasint n, const float alpha, const float *x, const blasint incx,const float beta, float *y, const blasint incy);
//
//void cblas_daxpby(const blasint n, const double alpha, const double *x, const blasint incx,const double beta, double *y, const blasint incy);
//
//void cblas_caxpby(const blasint n, const void *alpha, const void *x, const blasint incx,const void *beta, void *y, const blasint incy);
//
//void cblas_zaxpby(const blasint n, const void *alpha, const void *x, const blasint incx,const void *beta, void *y, const blasint incy);
//
//void cblas_somatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const float calpha, const float *a, 
//		     const blasint clda, float *b, const blasint cldb); 
//void cblas_domatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const double calpha, const double *a,
//		     const blasint clda, double *b, const blasint cldb); 
//void cblas_comatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const float* calpha, const float* a, 
//		     const blasint clda, float*b, const blasint cldb); 
//void cblas_zomatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const double* calpha, const double* a, 
//		     const blasint clda,  double *b, const blasint cldb); 
//
//void cblas_simatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const float calpha, float *a, 
//		     const blasint clda, const blasint cldb); 
//void cblas_dimatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const double calpha, double *a,
//		     const blasint clda, const blasint cldb); 
//void cblas_cimatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const float* calpha, float* a, 
//		     const blasint clda, const blasint cldb); 
//void cblas_zimatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const double* calpha, double* a, 
//		     const blasint clda, const blasint cldb); 
//
//void cblas_sgeadd(const enum CBLAS_ORDER CORDER,const blasint crows, const blasint ccols, const float calpha, const float *a, const blasint clda, const float cbeta, 
//		  float *c, const blasint cldc); 
//void cblas_dgeadd(const enum CBLAS_ORDER CORDER,const blasint crows, const blasint ccols, const double calpha, const double *a, const blasint clda, const double cbeta, 
//		  double *c, const blasint cldc); 
//void cblas_cgeadd(const enum CBLAS_ORDER CORDER,const blasint crows, const blasint ccols, const float *calpha, const float *a, const blasint clda, const float *cbeta, 
//		  float *c, const blasint cldc); 
//void cblas_zgeadd(const enum CBLAS_ORDER CORDER,const blasint crows, const blasint ccols, const double *calpha, const double *a, const blasint clda, const double *cbeta, 
//		  double *c, const blasint cldc); 
//
//void cblas_sgemm_batch(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE * TransA_array, const enum CBLAS_TRANSPOSE * TransB_array, const blasint * M_array, const blasint * N_array, const blasint * K_array,
//		       const float * alpha_array, const float ** A_array, const blasint * lda_array, const float ** B_array, const blasint * ldb_array, const float * beta_array, float ** C_array, const blasint * ldc_array, const blasint group_count, const blasint * group_size);
//
//void cblas_dgemm_batch(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE * TransA_array, const enum CBLAS_TRANSPOSE * TransB_array, const blasint * M_array, const blasint * N_array, const blasint * K_array,
//		       const double * alpha_array, const double ** A_array, const blasint * lda_array, const double ** B_array, const blasint * ldb_array, const double * beta_array, double ** C_array, const blasint * ldc_array, const blasint group_count, const blasint * group_size);
//
//void cblas_cgemm_batch(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE * TransA_array, const enum CBLAS_TRANSPOSE * TransB_array, const blasint * M_array, const blasint * N_array, const blasint * K_array,
//		       const void * alpha_array, const void ** A_array, const blasint * lda_array, const void ** B_array, const blasint * ldb_array, const void * beta_array, void ** C_array, const blasint * ldc_array, const blasint group_count, const blasint * group_size);
//
//void cblas_zgemm_batch(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE * TransA_array, const enum CBLAS_TRANSPOSE * TransB_array, const blasint * M_array, const blasint * N_array, const blasint * K_array,
//		       const void * alpha_array, const void ** A_array, const blasint * lda_array, const void ** B_array, const blasint * ldb_array, const void * beta_array, void ** C_array, const blasint * ldc_array, const blasint group_count, const blasint * group_size);
//
///*** BFLOAT16 and INT8 extensions ***/
///* convert float array to BFLOAT16 array by rounding */
//void   cblas_sbstobf16(const blasint n, const float  *in, const blasint incin, bfloat16 *out, const blasint incout);
///* convert double array to BFLOAT16 array by rounding */
//void   cblas_sbdtobf16(const blasint n, const double *in, const blasint incin, bfloat16 *out, const blasint incout);
///* convert BFLOAT16 array to float array */
//void   cblas_sbf16tos(const blasint n, const bfloat16 *in, const blasint incin, float  *out, const blasint incout);
///* convert BFLOAT16 array to double array */
//void   cblas_dbf16tod(const blasint n, const bfloat16 *in, const blasint incin, double *out, const blasint incout);
///* dot production of BFLOAT16 input arrays, and output as float */
//float  cblas_sbdot(const blasint n, const bfloat16 *x, const blasint incx, const bfloat16 *y, const blasint incy);
//void   cblas_sbgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const blasint m, const blasint n, const float alpha, const bfloat16 *a, const blasint lda, const bfloat16 *x, const blasint incx, const float beta, float *y, const blasint incy);
//
//void   cblas_sbgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
//		    const float alpha, const bfloat16 *A, const blasint lda, const bfloat16 *B, const blasint ldb, const float beta, float *C, const blasint ldc);
//void cblas_sbgemm_batch(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE * TransA_array, const enum CBLAS_TRANSPOSE * TransB_array, const blasint * M_array, const blasint * N_array, const blasint * K_array,
//		       const float * alpha_array, const bfloat16 ** A_array, const blasint * lda_array, const bfloat16 ** B_array, const blasint * ldb_array, const float * beta_array, float ** C_array, const blasint * ldc_array, const blasint group_count, const blasint * group_size);
