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

///| Set the number of threads on runtime.*/
/// 
/// ```c
/// void openblas_set_num_threads(int num_threads);
/// ```
pub extern "C" fn openblas_set_num_threads(num_threads: Int) = "openblas_set_num_threads";

///|
/// 
/// ```c
/// void goto_set_num_threads(int num_threads);
/// ```
pub extern "C" fn goto_set_num_threads(num_threads: Int) = "goto_set_num_threads";

///|
///
/// ```c
/// int openblas_set_num_threads_local(int num_threads);
/// ```
pub extern "C" fn openblas_set_num_threads_local(num_threads: Int) -> Int = "openblas_set_num_threads_local";

///| Get the number of threads on runtime.*/
///
/// ```c
/// int openblas_get_num_threads(void);
/// ```
pub extern "C" fn openblas_get_num_threads() -> Int = "openblas_get_num_threads";

///| Get the number of physical processors (cores).*/
///
/// ```c
/// int openblas_get_num_procs(void);
/// ```
pub extern "C" fn openblas_get_num_procs() -> Int = "openblas_get_num_procs";

///| Get the build configure on runtime.*/
///
/// ```c
/// char* openblas_get_config(void);
/// ```
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


///| Get the parallelization type which is used by OpenBLAS */
///
/// ```c
/// int openblas_get_parallel(void);
/// ```
pub extern "C" fn openblas_get_parallel() -> Int = "openblas_get_parallel";


///| 
///
/// ```c
/// #define OPENBLAS_SEQUENTIAL  0
/// ```
pub const OPENBLAS_SEQUENTIAL: Int = 0;

///| OpenBLAS is compiled using normal threading model
///
/// ```c
/// #define OPENBLAS_THREAD  1
/// ```
pub const OPENBLAS_THREAD: Int = 1;

///| OpenBLAS is compiled using OpenMP threading model
///
/// ```c
/// #define OPENBLAS_OPENMP 2
/// ```
pub const OPENBLAS_OPENMP: Int = 2;

///|
///
/// ```c
/// typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
/// ```
pub(all) enum CBLAS_ORDER {
  CblasRowMajor = 101;
  CblasColMajor = 102;
}

///|
///
/// ```c
/// typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
/// ```
pub(all) enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111;
  CblasTrans = 112;
  CblasConjTrans = 113;
  CblasConjNoTrans = 114;
}

///|
///
/// ```c
/// typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
/// ```
pub(all) enum CBLAS_UPLO {
  CblasUpper = 121;
  CblasLower = 122;
}

///|
///
/// ```c
/// typedef enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
/// ```
pub(all) enum CBLAS_DIAG {
  CblasNonUnit = 131;
  CblasUnit = 132;
}

///|
///
/// ```c
/// typedef enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
/// ```
pub(all) enum CBLAS_SIDE {
  CblasLeft = 141;
  CblasRight = 142;
}

///|
///
/// ```c
/// typedef CBLAS_ORDER CBLAS_LAYOUT;
/// ```
pub typealias CBLAS_ORDER as CBLAS_LAYOUT;

///|
///
/// ```c
/// float  cblas_sdsdot(const blasint n, const float alpha, const float *x, const blasint incx, const float *y, const blasint incy);
/// ```
pub extern "C" fn cblas_sdsdot(n: Int, alpha: Float, x: FixedArray[Float], incx: Int, y: FixedArray[Float], incy: Int) -> Float = "cblas_sdsdot";

///|
///
/// ```c
/// double cblas_dsdot (const blasint n, const float *x, const blasint incx, const float *y, const blasint incy);
/// ```
pub extern "C" fn cblas_dsdot(n: Int, x: FixedArray[Float], incx: Int, y: FixedArray[Float], incy: Int) -> Double = "cblas_dsdot";

///|
///
/// ```c
/// float  cblas_sdot(const blasint n, const float  *x, const blasint incx, const float  *y, const blasint incy);
/// ```
pub extern "C" fn cblas_sdot(n: Int, x: FixedArray[Float], incx: Int, y: FixedArray[Float], incy: Int) -> Float = "cblas_sdot";

///|
///
/// ```c
/// double cblas_ddot(const blasint n, const double *x, const blasint incx, const double *y, const blasint incy);
/// ```
pub extern "C" fn cblas_ddot(n: Int, x: FixedArray[Double], incx: Int, y: FixedArray[Double], incy: Int) -> Double = "cblas_ddot";

///|
///
/// ```c
/// openblas_complex_float  cblas_cdotu(const blasint n, const void  *x, const blasint incx, const void  *y, const blasint incy);
/// ```
pub extern "C" fn cblas_cdotu(n: Int, x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int) -> ComplexFloat = "cblas_cdotu";

///|
///
/// ```c
/// openblas_complex_float cblas_cdotc(const blasint n, const void *x, const blasint incx, const void *y, const blasint incy);
/// ```
pub extern "C" fn cblas_cdotc(n: Int, x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int) -> ComplexFloat = "cblas_cdotc";

///|
///
/// ```c
/// openblas_complex_double cblas_zdotu(const blasint n, const void *x, const blasint incx, const void *y, const blasint incy);
/// ```
pub extern "C" fn cblas_zdotu(n: Int, x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int) -> ComplexFloat = "cblas_zdotu";

///|
///
/// ```c
/// openblas_complex_double cblas_zdotc(const blasint n, const void *x, const blasint incx, const void *y, const blasint incy);
/// ```
pub extern "C" fn cblas_zdotc(n: Int, x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int) -> ComplexFloat = "cblas_zdotc";

///|
///
/// ```c
/// void cblas_cdotu_sub(const blasint n, const void *x, const blasint incx, const void *y, const blasint incy, void *ret);
/// ```
pub extern "C" fn cblas_cdotu_sub(n: Int, x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int, ret: VoidPtr) = "cblas_cdotu_sub";

///|
///
/// ```c
/// void cblas_cdotc_sub(const blasint n, const void *x, const blasint incx, const void *y, const blasint incy, void *ret);
/// ```
pub extern "C" fn cblas_cdotc_sub(n: Int, x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int, ret: VoidPtr) = "cblas_cdotc_sub";

///|
///
/// ```c
/// void cblas_zdotu_sub(const blasint n, const void *x, const blasint incx, const void *y, const blasint incy, void *ret);
/// ```
pub extern "C" fn cblas_zdotu_sub(n: Int, x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int, ret: VoidPtr) = "cblas_zdotu_sub";

///|
///
/// ```c
/// void cblas_zdotc_sub(const blasint n, const void *x, const blasint incx, const void *y, const blasint incy, void *ret);
/// ```
pub extern "C" fn cblas_zdotc_sub(n: Int, x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int, ret: VoidPtr) = "cblas_zdotc_sub";

///|
///
/// ```c
/// float cblas_sasum(const blasint n, const float *x, const blasint incx);
/// ```
pub extern "C" fn cblas_sasum(n: Int, x: FixedArray[Float], incx: Int) -> Float = "cblas_sasum";

///|
///
/// ```c
/// double cblas_dasum(const blasint n, const double *x, const blasint incx);
/// ```
pub extern "C" fn cblas_dasum(n: Int, x: FixedArray[Double], incx: Int) -> Double = "cblas_dasum";

///|
///
/// ```c
/// float cblas_scasum(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_scasum(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> Float = "cblas_scasum";

///|
///
/// ```c
/// double cblas_dzasum(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_dzasum(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> Double = "cblas_dzasum";


///|
///
/// ```c
/// float cblas_ssum(const blasint n, const float *x, const blasint incx);
/// ```
pub extern "C" fn cblas_ssum(n: Int, x: FixedArray[Float], incx: Int) -> Float = "cblas_ssum";

///|
///
/// ```c
/// double cblas_dsum(const blasint n, const double *x, const blasint incx);
/// ```
pub extern "C" fn cblas_dsum(n: Int, x: FixedArray[Double], incx: Int) -> Double = "cblas_dsum";

///|
///
/// ```c
/// float cblas_scsum(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_scsum(n: Int, x: VoidPtr, incx: Int) -> Float = "cblas_scsum";

///|
///
/// ```c
/// double cblas_dzsum(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_dzsum(n: Int, x: VoidPtr, incx: Int) -> Double = "cblas_dzsum";

///|
///
/// ```c
/// float cblas_snrm2(const blasint N, const float *X, const blasint incX);
/// ```
pub extern "C" fn cblas_snrm2(n: Int, x: FixedArray[Float], incx: Int) -> Float = "cblas_snrm2";

///|
///
/// ```c
/// double cblas_dnrm2(const blasint N, const double *X, const blasint incX);
/// ```
pub extern "C" fn cblas_dnrm2(n: Int, x: FixedArray[Double], incx: Int) -> Double = "cblas_dnrm2";

///|
///
/// ```c
/// float cblas_scnrm2(const blasint N, const void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_scnrm2(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> Float = "cblas_scnrm2";

///|
///
/// ```c
/// double cblas_dznrm2(const blasint N, const void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_dznrm2(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> Double = "cblas_dznrm2";

///|
///
/// ```c
/// UInt64 cblas_isamax(const blasint n, const float *x, const blasint incx);
/// ```
pub extern "C" fn cblas_isamax(n: Int, x: FixedArray[Float], incx: Int) -> UInt64 = "cblas_isamax";

///|
///
/// ```c
/// UInt64 cblas_idamax(const blasint n, const double *x, const blasint incx);
/// ```
pub extern "C" fn cblas_idamax(n: Int, x: FixedArray[Double], incx: Int) -> UInt64 = "cblas_idamax";

///|
///
/// ```c
/// UInt64 cblas_icamax(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_icamax(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> UInt64 = "cblas_icamax";

///|
///
/// ```c
/// UInt64 cblas_izamax(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_izamax(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> UInt64 = "cblas_izamax";

///|
///
/// ```c
/// UInt64 cblas_isamin(const blasint n, const float *x, const blasint incx);
/// ```
pub extern "C" fn cblas_isamin(n: Int, x: FixedArray[Float], incx: Int) -> UInt64 = "cblas_isamin";

///|
///
/// ```c
/// UInt64 cblas_idamin(const blasint n, const double *x, const blasint incx);
/// ```
pub extern "C" fn cblas_idamin(n: Int, x: FixedArray[Double], incx: Int) -> UInt64 = "cblas_idamin";

///|
///
/// ```c
/// UInt64 cblas_icamin(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_icamin(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> UInt64 = "cblas_icamin";

///|
///
/// ```c
/// UInt64 cblas_izamin(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_izamin(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> UInt64 = "cblas_izamin";

///|
///
/// ```c
/// float cblas_samax(const blasint n, const float *x, const blasint incx);
/// ```
pub extern "C" fn cblas_samax(n: Int, x: FixedArray[Float], incx: Int) -> Float = "cblas_samax";

///|
///
/// ```c
/// double cblas_damax(const blasint n, const double *x, const blasint incx);
/// ```
pub extern "C" fn cblas_damax(n: Int, x: FixedArray[Double], incx: Int) -> Double = "cblas_damax";

///|
///
/// ```c
/// float cblas_scamax(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_scamax(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> Float = "cblas_scamax";

///|
///
/// ```c
/// double cblas_dzamax(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_dzamax(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> Double = "cblas_dzamax";

///|
///
/// ```c
/// float cblas_samin(const blasint n, const float *x, const blasint incx);
/// ```
pub extern "C" fn cblas_samin(n: Int, x: FixedArray[Float], incx: Int) -> Float = "cblas_samin";

///|
///
/// ```c
/// double cblas_damin(const blasint n, const double *x, const blasint incx);
/// ```
pub extern "C" fn cblas_damin(n: Int, x: FixedArray[Double], incx: Int) -> Double = "cblas_damin";

///|
///
/// ```c
/// float cblas_scamin(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_scamin(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> Float = "cblas_scamin";

///|
///
/// ```c
/// double cblas_dzamin(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_dzamin(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> Double = "cblas_dzamin";

///|
///
/// ```c
/// UInt64 cblas_ismax(const blasint n, const float *x, const blasint incx);
/// ```
pub extern "C" fn cblas_ismax(n: Int, x: FixedArray[Float], incx: Int) -> UInt64 = "cblas_ismax";

///|
///
/// ```c
/// UInt64 cblas_idmax(const blasint n, const double *x, const blasint incx);
/// ```
pub extern "C" fn cblas_idmax(n: Int, x: FixedArray[Double], incx: Int) -> UInt64 = "cblas_idmax";

///|
///
/// ```c
/// UInt64 cblas_icmax(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_icmax(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> UInt64 = "cblas_icmax";

///|
///
/// ```c
/// UInt64 cblas_izmax(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_izmax(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> UInt64 = "cblas_izmax";

///|
///
/// ```c
/// UInt64 cblas_ismin(const blasint n, const float *x, const blasint incx);
/// ```
pub extern "C" fn cblas_ismin(n: Int, x: FixedArray[Float], incx: Int) -> UInt64 = "cblas_ismin";

///|
///
/// ```c
/// UInt64 cblas_idmin(const blasint n, const double *x, const blasint incx);
/// ```
pub extern "C" fn cblas_idmin(n: Int, x: FixedArray[Double], incx: Int) -> UInt64 = "cblas_idmin";

///|
///
/// ```c
/// UInt64 cblas_icmin(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_icmin(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> UInt64 = "cblas_icmin";

///|
///
/// ```c
/// UInt64 cblas_izmin(const blasint n, const void *x, const blasint incx);
/// ```
pub extern "C" fn cblas_izmin(n: Int, x: FixedArray[ComplexFloat], incx: Int) -> UInt64 = "cblas_izmin";

///|
///
/// ```c
/// void cblas_saxpy(const blasint n, const float alpha, const float *x, const blasint incx, float *y, const blasint incy);
/// ```
pub extern "C" fn cblas_saxpy(n: Int, alpha: Float, x: FixedArray[Float], incx: Int, y: FixedArray[Float], incy: Int) = "cblas_saxpy";

///|
///
/// ```c
/// void cblas_daxpy(const blasint n, const double alpha, const double *x, const blasint incx, double *y, const blasint incy);
/// ```
pub extern "C" fn cblas_daxpy(n: Int, alpha: Double, x: FixedArray[Double], incx: Int, y: FixedArray[Double], incy: Int) = "cblas_daxpy";

///|
///
/// ```c
/// void cblas_caxpy(const blasint n, const void *alpha, const void *x, const blasint incx, void *y, const blasint incy);
/// ```
pub extern "C" fn cblas_caxpy(n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int) = "cblas_caxpy";

///|
///
/// ```c
/// void cblas_zaxpy(const blasint n, const void *alpha, const void *x, const blasint incx, void *y, const blasint incy);
/// ```
pub extern "C" fn cblas_zaxpy(n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int) = "cblas_zaxpy";

///|
///
/// ```c
/// void cblas_caxpyc(const blasint n, const void *alpha, const void *x, const blasint incx, void *y, const blasint incy);
/// ```
pub extern "C" fn cblas_caxpyc(n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int) = "cblas_caxpyc";

///|
///
/// ```c
/// void cblas_zaxpyc(const blasint n, const void *alpha, const void *x, const blasint incx, void *y, const blasint incy);
/// ```
pub extern "C" fn cblas_zaxpyc(n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int) = "cblas_zaxpyc";
//
///|
///
/// ```c
/// void cblas_scopy(const blasint n, const float *x, const blasint incx, float *y, const blasint incy);
/// ```
pub extern "C" fn cblas_scopy(n: Int, x: FixedArray[Float], incx: Int, y: FixedArray[Float], incy: Int) = "cblas_scopy";

///|
///
/// ```c
/// void cblas_dcopy(const blasint n, const double *x, const blasint incx, double *y, const blasint incy);
/// ```
pub extern "C" fn cblas_dcopy(n: Int, x: FixedArray[Double], incx: Int, y: FixedArray[Double], incy: Int) = "cblas_dcopy";

///|
///
/// ```c
/// void cblas_ccopy(const blasint n, const void *x, const blasint incx, void *y, const blasint incy);
/// ```
pub extern "C" fn cblas_ccopy(n: Int, x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int) = "cblas_ccopy";

///|
///
/// ```c
/// void cblas_zcopy(const blasint n, const void *x, const blasint incx, void *y, const blasint incy);
/// ```
pub extern "C" fn cblas_zcopy(n: Int, x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int) = "cblas_zcopy";

///|
///
/// ```c
/// void cblas_sswap(const blasint n, float *x, const blasint incx, float *y, const blasint incy);
/// ```
pub extern "C" fn cblas_sswap(n: Int, x: FixedArray[Float], incx: Int, y: FixedArray[Float], incy: Int) = "cblas_sswap";

///|
///
/// ```c
/// void cblas_dswap(const blasint n, double *x, const blasint incx, double *y, const blasint incy);
/// ```
pub extern "C" fn cblas_dswap(n: Int, x: FixedArray[Double], incx: Int, y: FixedArray[Double], incy: Int) = "cblas_dswap";

///|
///
/// ```c
/// void cblas_cswap(const blasint n, void *x, const blasint incx, void *y, const blasint incy);
/// ```
pub extern "C" fn cblas_cswap(n: Int, x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int) = "cblas_cswap";

///|
///
/// ```c
/// void cblas_zswap(const blasint n, void *x, const blasint incx, void *y, const blasint incy);
/// ```
pub extern "C" fn cblas_zswap(n: Int, x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int) = "cblas_zswap";

///|
///
/// ```c
/// void cblas_srot(const blasint N, float *X, const blasint incX, float *Y, const blasint incY, const float c, const float s);
/// ```
pub extern "C" fn cblas_srot(n: Int, x: FixedArray[Float], incx: Int, y: FixedArray[Float], incy: Int, c: Float, s: Float) = "cblas_srot";

///|
///
/// ```c
/// void cblas_drot(const blasint N, double *X, const blasint incX, double *Y, const blasint incY, const double c, const double s);
/// ```
pub extern "C" fn cblas_drot(n: Int, x: FixedArray[Double], incx: Int, y: FixedArray[Double], incy: Int, c: Double, s: Double) = "cblas_drot";

///|
///
/// ```c
/// void cblas_csrot(const blasint n, const void *x, const blasint incx, void *y, const blasint incY, const float c, const float s);
/// ```
pub extern "C" fn cblas_csrot(n: Int, x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int, c: Float, s: Float) = "cblas_csrot";

///|
///
/// ```c
/// void cblas_zdrot(const blasint n, const void *x, const blasint incx, void *y, const blasint incY, const double c, const double s);
/// ```
pub extern "C" fn cblas_zdrot(n: Int, x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int, c: Double, s: Double) = "cblas_zdrot";

///|
///
/// ```c
/// void cblas_srotg(float *a, float *b, float *c, float *s);
/// ```
pub extern "C" fn cblas_srotg(a: FixedArray[Float], b: FixedArray[Float], c: FixedArray[Float], s: FixedArray[Float]) = "cblas_srotg";

///|
///
/// ```c
/// void cblas_drotg(double *a, double *b, double *c, double *s);
/// ```
pub extern "C" fn cblas_drotg(a: FixedArray[Double], b: FixedArray[Double], c: FixedArray[Double], s: FixedArray[Double]) = "cblas_drotg";

///|
///
/// ```c
/// void cblas_crotg(void *a, void *b, float *c, void *s);
/// ```
pub extern "C" fn cblas_crotg(a: FixedArray[ComplexFloat], b: FixedArray[ComplexFloat], c: FixedArray[Float], s: FixedArray[ComplexFloat]) = "cblas_crotg";

///|
///
/// ```c
/// void cblas_zrotg(void *a, void *b, double *c, void *s);
/// ```
pub extern "C" fn cblas_zrotg(a: FixedArray[ComplexFloat], b: FixedArray[ComplexFloat], c: FixedArray[Double], s: FixedArray[ComplexFloat]) = "cblas_zrotg";

///|
///
/// ```c
/// void cblas_srotm(const blasint N, float *X, const blasint incX, float *Y, const blasint incY, const float *P);
/// ```
pub extern "C" fn cblas_srotm(n: Int, x: FixedArray[Float], incx: Int, y: FixedArray[Float], incy: Int, p: FixedArray[Float]) = "cblas_srotm";

///|
///
/// ```c
/// void cblas_drotm(const blasint N, double *X, const blasint incX, double *Y, const blasint incY, const double *P);
/// ```
pub extern "C" fn cblas_drotm(n: Int, x: FixedArray[Double], incx: Int, y: FixedArray[Double], incy: Int, p: FixedArray[Double]) = "cblas_drotm";

///|
///
/// ```c
/// void cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
/// ```
pub extern "C" fn cblas_srotmg(d1: FixedArray[Float], d2: FixedArray[Float], b1: FixedArray[Float], b2: Float, p: FixedArray[Float]) = "cblas_srotmg";

///|
///
/// ```c
/// void cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P);
/// ```
pub extern "C" fn cblas_drotmg(d1: FixedArray[Double], d2: FixedArray[Double], b1: FixedArray[Double], b2: Double, p: FixedArray[Double]) = "cblas_drotmg";
//
///|
///
/// ```c
/// void cblas_sscal(const blasint N, const float alpha, float *X, const blasint incX);
/// ```
pub extern "C" fn cblas_sscal(n: Int, alpha: Float, x: FixedArray[Float], incx: Int) = "cblas_sscal";

///|
///
/// ```c
/// void cblas_dscal(const blasint N, const double alpha, double *X, const blasint incX);
/// ```
pub extern "C" fn cblas_dscal(n: Int, alpha: Double, x: FixedArray[Double], incx: Int) = "cblas_dscal";

///|
///
/// ```c
/// void cblas_cscal(const blasint N, const void *alpha, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_cscal(n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int) = "cblas_cscal";

///|
///
/// ```c
/// void cblas_zscal(const blasint N, const void *alpha, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_zscal(n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int) = "cblas_zscal";

///|
///
/// ```c
/// void cblas_csscal(const blasint N, const float alpha, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_csscal(n: Int, alpha: Float, x: FixedArray[ComplexFloat], incx: Int) = "cblas_csscal";

///|
///
/// ```c
/// void cblas_zdscal(const blasint N, const double alpha, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_zdscal(n: Int, alpha: Double, x: FixedArray[ComplexFloat], incx: Int) = "cblas_zdscal";

///|
///
/// ```c
/// void cblas_sgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, const blasint m, const blasint n,
///                  const float alpha, const float *a, const blasint lda, const float *x, const blasint incx, const float beta, float *y, const blasint incy);
/// ```
pub extern "C" fn cblas_sgemv(order: CBLAS_ORDER, trans: CBLAS_TRANSPOSE, m: Int, n: Int, alpha: Float, a: FixedArray[Float], lda: Int, x: FixedArray[Float], incx: Int, beta: Float, y: FixedArray[Float], incy: Int) = "cblas_sgemv";

///|
///
/// ```c
/// void cblas_dgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const blasint m, const blasint n,
////	 const double alpha, const double  *a, const blasint lda,  const double  *x, const blasint incx,  const double beta,  double  *y, const blasint incy);
/// ```
pub extern "C" fn cblas_dgemv(order: CBLAS_ORDER, trans: CBLAS_TRANSPOSE, m: Int, n: Int, alpha: Double, a: FixedArray[Double], lda: Int, x: FixedArray[Double], incx: Int, beta: Double, y: FixedArray[Double], incy: Int) = "cblas_dgemv";

///|
///
/// ```c
/// void cblas_cgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const blasint m, const blasint n,
////	 const void *alpha, const void  *a, const blasint lda,  const void  *x, const blasint incx,  const void *beta,  void  *y, const blasint incy);
/// ```
pub extern "C" fn cblas_cgemv(order: CBLAS_ORDER, trans: CBLAS_TRANSPOSE, m: Int, n: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int, beta: FixedArray[ComplexFloat], y: FixedArray[ComplexFloat], incy: Int) = "cblas_cgemv";

///|
///
/// ```c
/// void cblas_zgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const blasint m, const blasint n,
////	 const void *alpha, const void  *a, const blasint lda,  const void  *x, const blasint incx,  const void *beta,  void  *y, const blasint incy);
/// ```
pub extern "C" fn cblas_zgemv(order: CBLAS_ORDER, trans: CBLAS_TRANSPOSE, m: Int, n: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int, beta: FixedArray[ComplexFloat], y: FixedArray[ComplexFloat], incy: Int) = "cblas_zgemv";

///|
///
/// ```c
/// void cblas_sger(const enum CBLAS_ORDER order, const blasint M, const blasint N, const float alpha, const float *X, const blasint incX, const float *Y, const blasint incY, float *A, const blasint lda);
/// ```
pub extern "C" fn cblas_sger(order: CBLAS_ORDER, m: Int, n: Int, alpha: Float, x: FixedArray[Float], incx: Int, y: FixedArray[Float], incy: Int, a: FixedArray[Float], lda: Int) = "cblas_sger";

///|
///
/// ```c
/// void cblas_dger(const enum CBLAS_ORDER order, const blasint M, const blasint N, const double alpha, const double *X, const blasint incX, const double *Y, const blasint incY, double *A, const blasint lda);
/// ```
pub extern "C" fn cblas_dger(order: CBLAS_ORDER, m: Int, n: Int, alpha: Double, x: FixedArray[Double], incx: Int, y: FixedArray[Double], incy: Int, a: FixedArray[Double], lda: Int) = "cblas_dger";

///|
///
/// ```c
/// void cblas_cgeru(const enum CBLAS_ORDER order, const blasint M, const blasint N, const void *alpha, const void *X, const blasint incX, const void *Y, const blasint incY, void *A, const blasint lda);
/// ```
pub extern "C" fn cblas_cgeru(order: CBLAS_ORDER, m: Int, n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int, a: FixedArray[ComplexFloat], lda: Int) = "cblas_cgeru";

///|
///
/// ```c
/// void cblas_cgerc(const enum CBLAS_ORDER order, const blasint M, const blasint N, const void *alpha, const void *X, const blasint incX, const void *Y, const blasint incY, void *A, const blasint lda);
/// ```
pub extern "C" fn cblas_cgerc(order: CBLAS_ORDER, m: Int, n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int, a: FixedArray[ComplexFloat], lda: Int) = "cblas_cgerc";

///|
///
/// ```c
/// void cblas_zgeru(const enum CBLAS_ORDER order, const blasint M, const blasint N, const void *alpha, const void *X, const blasint incX, const void *Y, const blasint incY, void *A, const blasint lda);
/// ```
pub extern "C" fn cblas_zgeru(order: CBLAS_ORDER, m: Int, n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int, a: FixedArray[ComplexFloat], lda: Int) = "cblas_zgeru";

///|
///
/// ```c
/// void cblas_zgerc(const enum CBLAS_ORDER order, const blasint M, const blasint N, const void *alpha, const void *X, const blasint incX, const void *Y, const blasint incY, void *A, const blasint lda);
/// ```
pub extern "C" fn cblas_zgerc(order: CBLAS_ORDER, m: Int, n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int, a: FixedArray[ComplexFloat], lda: Int) = "cblas_zgerc";

///|
///
/// ```c
/// void cblas_strsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const float *A, const blasint lda, float *X, const blasint incX);
/// ```
pub extern "C" fn cblas_strsv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, a: FixedArray[Float], lda: Int, x: FixedArray[Float], incx: Int) = "cblas_strsv";
///|
///
/// ```c
/// void cblas_dtrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const double *A, const blasint lda, double *X, const blasint incX);
/// ```
pub extern "C" fn cblas_dtrsv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, a: FixedArray[Double], lda: Int, x: FixedArray[Double], incx: Int) = "cblas_dtrsv";

///|
///
/// ```c
/// void cblas_ctrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const void *A, const blasint lda, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_ctrsv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int) = "cblas_ctrsv";

///|
///
/// ```c
/// void cblas_ztrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const void *A, const blasint lda, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_ztrsv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int) = "cblas_ztrsv";

///|
///
/// ```c
/// void cblas_strmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const float *A, const blasint lda, float *X, const blasint incX);
/// ```
pub extern "C" fn cblas_strmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, a: FixedArray[Float], lda: Int, x: FixedArray[Float], incx: Int) = "cblas_strmv";

///|
///
/// ```c
/// void cblas_dtrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const double *A, const blasint lda, double *X, const blasint incX);
/// ```
pub extern "C" fn cblas_dtrmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, a: FixedArray[Double], lda: Int, x: FixedArray[Double], incx: Int) = "cblas_dtrmv";

///|
///
/// ```c
/// void cblas_ctrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const void *A, const blasint lda, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_ctrmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int) = "cblas_ctrmv";

///|
///
/// ```c
/// void cblas_ztrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const blasint N, const void *A, const blasint lda, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_ztrmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int) = "cblas_ztrmv";

///|
///
/// ```c
/// void cblas_ssyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const float alpha, const float *X, const blasint incX, float *A, const blasint lda);
/// ```
pub extern "C" fn cblas_ssyr(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Float, x: FixedArray[Float], incx: Int, a: FixedArray[Float], lda: Int) = "cblas_ssyr";

///|
///
/// ```c
/// void cblas_dsyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const double *X, const blasint incX, double *A, const blasint lda);
/// ```
pub extern "C" fn cblas_dsyr(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Double, x: FixedArray[Double], incx: Int, a: FixedArray[Double], lda: Int) = "cblas_dsyr";

///|
///
/// ```c
/// void cblas_cher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const float alpha, const void *X, const blasint incX, void *A, const blasint lda);
/// ```
pub extern "C" fn cblas_cher(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Float, x: FixedArray[ComplexFloat], incx: Int, a: FixedArray[ComplexFloat], lda: Int) = "cblas_cher";

///|
///
/// ```c
/// void cblas_zher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const void *X, const blasint incX, void *A, const blasint lda);
/// ```
pub extern "C" fn cblas_zher(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Double, x: FixedArray[ComplexFloat], incx: Int, a: FixedArray[ComplexFloat], lda: Int) = "cblas_zher";

///|
///
/// ```c
/// void cblas_ssyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,const blasint N, const float alpha, const float *X,
///                const blasint incX, const float *Y, const blasint incY, float *A, const blasint lda);
/// ```
pub extern "C" fn cblas_ssyr2(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Float, x: FixedArray[Float], incx: Int, y: FixedArray[Float], incy: Int, a: FixedArray[Float], lda: Int) = "cblas_ssyr2";
///|
///
/// ```c
/// void cblas_dsyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const double *X,
///                const blasint incX, const double *Y, const blasint incY, double *A, const blasint lda);
/// ```
pub extern "C" fn cblas_dsyr2(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Double, x: FixedArray[Double], incx: Int, y: FixedArray[Double], incy: Int, a: FixedArray[Double], lda: Int) = "cblas_dsyr2";
///|
///
/// ```c
/// void cblas_cher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const void *alpha, const void *X, const blasint incX,
///                const void *Y, const blasint incY, void *A, const blasint lda);
/// ```
pub extern "C" fn cblas_cher2(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int, a: FixedArray[ComplexFloat], lda: Int) = "cblas_cher2";
///|
///
/// ```c
/// void cblas_zher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const void *alpha, const void *X, const blasint incX,
///                const void *Y, const blasint incY, void *A, const blasint lda);
/// ```
pub extern "C" fn cblas_zher2(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int, a: FixedArray[ComplexFloat], lda: Int) = "cblas_zher2";
//
///|
///
/// ```c
/// void cblas_sgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const blasint M, const blasint N,
///                 const blasint KL, const blasint KU, const float alpha, const float *A, const blasint lda, const float *X, const blasint incX, const float beta, float *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_sgbmv(order: CBLAS_ORDER, trans_a: CBLAS_TRANSPOSE, m: Int, n: Int, kl: Int, ku: Int, alpha: Float, a: FixedArray[Float], lda: Int, x: FixedArray[Float], incx: Int, beta: Float, y: FixedArray[Float], incy: Int) = "cblas_sgbmv";
///|
///
/// ```c
/// void cblas_dgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const blasint M, const blasint N,
///                 const blasint KL, const blasint KU, const double alpha, const double *A, const blasint lda, const double *X, const blasint incX, const double beta, double *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_dgbmv(order: CBLAS_ORDER, trans_a: CBLAS_TRANSPOSE, m: Int, n: Int, kl: Int, ku: Int, alpha: Double, a: FixedArray[Double], lda: Int, x: FixedArray[Double], incx: Int, beta: Double, y: FixedArray[Double], incy: Int) = "cblas_dgbmv";
///|
///
/// ```c
/// void cblas_cgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const blasint M, const blasint N,
///                 const blasint KL, const blasint KU, const void *alpha, const void *A, const blasint lda, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_cgbmv(order: CBLAS_ORDER, trans_a: CBLAS_TRANSPOSE, m: Int, n: Int, kl: Int, ku: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int, beta: FixedArray[ComplexFloat], y: FixedArray[ComplexFloat], incy: Int) = "cblas_cgbmv";
///|
///
/// ```c
/// void cblas_zgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const blasint M, const blasint N,
///                 const blasint KL, const blasint KU, const void *alpha, const void *A, const blasint lda, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_zgbmv(order: CBLAS_ORDER, trans_a: CBLAS_TRANSPOSE, m: Int, n: Int, kl: Int, ku: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int, beta: FixedArray[ComplexFloat], y: FixedArray[ComplexFloat], incy: Int) = "cblas_zgbmv";

///|
///
/// ```c
/// void cblas_ssbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const blasint K, const float alpha, const float *A,
///                 const blasint lda, const float *X, const blasint incX, const float beta, float *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_ssbmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, k: Int, alpha: Float, a: FixedArray[Float], lda: Int, x: FixedArray[Float], incx: Int, beta: Float, y: FixedArray[Float], incy: Int) = "cblas_ssbmv";
///|
///
/// ```c
/// void cblas_dsbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const blasint K, const double alpha, const double *A,
///                 const blasint lda, const double *X, const blasint incX, const double beta, double *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_dsbmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, k: Int, alpha: Double, a: FixedArray[Double], lda: Int, x: FixedArray[Double], incx: Int, beta: Double, y: FixedArray[Double], incy: Int) = "cblas_dsbmv";
//
//
///|
///
/// ```c
/// void cblas_stbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const blasint K, const float *A, const blasint lda, float *X, const blasint incX);
/// ```
pub extern "C" fn cblas_stbmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, k: Int, a: FixedArray[Float], lda: Int, x: FixedArray[Float], incx: Int) = "cblas_stbmv";
///|
///
/// ```c
/// void cblas_dtbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const blasint K, const double *A, const blasint lda, double *X, const blasint incX);
/// ```
pub extern "C" fn cblas_dtbmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, k: Int, a: FixedArray[Double], lda: Int, x: FixedArray[Double], incx: Int) = "cblas_dtbmv";
///|
///
/// ```c
/// void cblas_ctbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const blasint K, const void *A, const blasint lda, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_ctbmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, k: Int, a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int) = "cblas_ctbmv";
///|
///
/// ```c
/// void cblas_ztbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const blasint K, const void *A, const blasint lda, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_ztbmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, k: Int, a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int) = "cblas_ztbmv";

///|
///
/// ```c
/// void cblas_stbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const blasint K, const float *A, const blasint lda, float *X, const blasint incX);
/// ```
pub extern "C" fn cblas_stbsv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, k: Int, a: FixedArray[Float], lda: Int, x: FixedArray[Float], incx: Int) = "cblas_stbsv";
///|
///
/// ```c
/// void cblas_dtbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const blasint K, const double *A, const blasint lda, double *X, const blasint incX);
/// ```
pub extern "C" fn cblas_dtbsv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, k: Int, a: FixedArray[Double], lda: Int, x: FixedArray[Double], incx: Int) = "cblas_dtbsv";
///|
///
/// ```c
/// void cblas_ctbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const blasint K, const void *A, const blasint lda, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_ctbsv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, k: Int, a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int) = "cblas_ctbsv";
///|
///
/// ```c
/// void cblas_ztbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const blasint K, const void *A, const blasint lda, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_ztbsv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, k: Int, a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int) = "cblas_ztbsv";
//
///|
///
/// ```c
/// void cblas_stpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const float *Ap, float *X, const blasint incX);
/// ```
pub extern "C" fn cblas_stpmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, ap: FixedArray[Float], x: FixedArray[Float], incx: Int) = "cblas_stpmv";
///|
///
/// ```c
/// void cblas_dtpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const double *Ap, double *X, const blasint incX);
/// ```
pub extern "C" fn cblas_dtpmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, ap: FixedArray[Double], x: FixedArray[Double], incx: Int) = "cblas_dtpmv";
///|
///
/// ```c
/// void cblas_ctpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const void *Ap, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_ctpmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, ap: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int) = "cblas_ctpmv";
///|
///
/// ```c
/// void cblas_ztpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const void *Ap, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_ztpmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, ap: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int) = "cblas_ztpmv";

///|
///
/// ```c
/// void cblas_stpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const float *Ap, float *X, const blasint incX);
/// ```
pub extern "C" fn cblas_stpsv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, ap: FixedArray[Float], x: FixedArray[Float], incx: Int) = "cblas_stpsv";
///|
///
/// ```c
/// void cblas_dtpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const double *Ap, double *X, const blasint incX);
/// ```
pub extern "C" fn cblas_dtpsv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, ap: FixedArray[Double], x: FixedArray[Double], incx: Int) = "cblas_dtpsv";
///|
///
/// ```c
/// void cblas_ctpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const void *Ap, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_ctpsv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, ap: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int) = "cblas_ctpsv";
///|
///
/// ```c
/// void cblas_ztpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
///                 const blasint N, const void *Ap, void *X, const blasint incX);
/// ```
pub extern "C" fn cblas_ztpsv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, n: Int, ap: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int) = "cblas_ztpsv";
//
///|
///
/// ```c
/// void cblas_ssymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const float alpha, const float *A,
///                 const blasint lda, const float *X, const blasint incX, const float beta, float *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_ssymv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Float, a: FixedArray[Float], lda: Int, x: FixedArray[Float], incx: Int, beta: Float, y: FixedArray[Float], incy: Int) = "cblas_ssymv";
///|
///
/// ```c
/// void cblas_dsymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const double *A,
///                 const blasint lda, const double *X, const blasint incX, const double beta, double *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_dsymv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Double, a: FixedArray[Double], lda: Int, x: FixedArray[Double], incx: Int, beta: Double, y: FixedArray[Double], incy: Int) = "cblas_dsymv";
///|
///
/// ```c
/// void cblas_chemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const void *alpha, const void *A,
///                 const blasint lda, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_chemv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int, beta: FixedArray[ComplexFloat], y: FixedArray[ComplexFloat], incy: Int) = "cblas_chemv";
///|
///
/// ```c
/// void cblas_zhemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const void *alpha, const void *A,
///                 const blasint lda, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_zhemv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int, beta: FixedArray[ComplexFloat], y: FixedArray[ComplexFloat], incy: Int) = "cblas_zhemv";

///|
///
/// ```c
/// void cblas_sspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const float alpha, const float *Ap,
///                 const float *X, const blasint incX, const float beta, float *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_sspmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Float, ap: FixedArray[Float], x: FixedArray[Float], incx: Int, beta: Float, y: FixedArray[Float], incy: Int) = "cblas_sspmv";
///|
///
/// ```c
/// void cblas_dspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const double *Ap,
///                 const double *X, const blasint incX, const double beta, double *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_dspmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Double, ap: FixedArray[Double], x: FixedArray[Double], incx: Int, beta: Double, y: FixedArray[Double], incy: Int) = "cblas_dspmv";

///|
///
/// ```c
/// void cblas_sspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const float alpha, const float *X, const blasint incX, float *Ap);
/// ```
pub extern "C" fn cblas_sspr(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Float, x: FixedArray[Float], incx: Int, ap: FixedArray[Float]) = "cblas_sspr";
///|
///
/// ```c
/// void cblas_dspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const double *X, const blasint incX, double *Ap);
/// ```
pub extern "C" fn cblas_dspr(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Double, x: FixedArray[Double], incx: Int, ap: FixedArray[Double]) = "cblas_dspr";

///|
///
/// ```c
/// void cblas_chpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const float alpha, const void *X, const blasint incX, void *A);
/// ```
pub extern "C" fn cblas_chpr(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Float, x: FixedArray[ComplexFloat], incx: Int, a: FixedArray[ComplexFloat]) = "cblas_chpr";
///|
///
/// ```c
/// void cblas_zhpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const void *X,const blasint incX, void *A);
/// ```
pub extern "C" fn cblas_zhpr(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Double, x: FixedArray[ComplexFloat], incx: Int, a: FixedArray[ComplexFloat]) = "cblas_zhpr";

///|
///
/// ```c
/// void cblas_sspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const float alpha, const float *X, const blasint incX, const float *Y, const blasint incY, float *A);
/// ```
pub extern "C" fn cblas_sspr2(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Float, x: FixedArray[Float], incx: Int, y: FixedArray[Float], incy: Int, a: FixedArray[Float]) = "cblas_sspr2";
///|
///
/// ```c
/// void cblas_dspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const double alpha, const double *X, const blasint incX, const double *Y, const blasint incY, double *A);
/// ```
pub extern "C" fn cblas_dspr2(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: Double, x: FixedArray[Double], incx: Int, y: FixedArray[Double], incy: Int, a: FixedArray[Double]) = "cblas_dspr2";
///|
///
/// ```c
/// void cblas_chpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const void *alpha, const void *X, const blasint incX, const void *Y, const blasint incY, void *Ap);
/// ```
pub extern "C" fn cblas_chpr2(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int, ap: FixedArray[ComplexFloat]) = "cblas_chpr2";
///|
///
/// ```c
/// void cblas_zhpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const void *alpha, const void *X, const blasint incX, const void *Y, const blasint incY, void *Ap);
/// ```
pub extern "C" fn cblas_zhpr2(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, y: FixedArray[ComplexFloat], incy: Int, ap: FixedArray[ComplexFloat]) = "cblas_zhpr2";
//
///|
///
/// ```c
/// void cblas_chbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const blasint K,
///	 const void *alpha, const void *A, const blasint lda, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_chbmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, k: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int, beta: FixedArray[ComplexFloat], y: FixedArray[ComplexFloat], incy: Int) = "cblas_chbmv";
///|
///
/// ```c
/// void cblas_zhbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N, const blasint K,
///	 const void *alpha, const void *A, const blasint lda, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_zhbmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, k: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, x: FixedArray[ComplexFloat], incx: Int, beta: FixedArray[ComplexFloat], y: FixedArray[ComplexFloat], incy: Int) = "cblas_zhbmv";

///|
///
/// ```c
/// void cblas_chpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N,
///	 const void *alpha, const void *Ap, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_chpmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: FixedArray[ComplexFloat], ap: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, beta: FixedArray[ComplexFloat], y: FixedArray[ComplexFloat], incy: Int) = "cblas_chpmv";
///|
///
/// ```c
/// void cblas_zhpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const blasint N,
///	 const void *alpha, const void *Ap, const void *X, const blasint incX, const void *beta, void *Y, const blasint incY);
/// ```
pub extern "C" fn cblas_zhpmv(order: CBLAS_ORDER, uplo: CBLAS_UPLO, n: Int, alpha: FixedArray[ComplexFloat], ap: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, beta: FixedArray[ComplexFloat], y: FixedArray[ComplexFloat], incy: Int) = "cblas_zhpmv";
//
///|
///
/// ```c
/// void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
///	 const float alpha, const float *A, const blasint lda, const float *B, const blasint ldb, const float beta, float *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_sgemm(order: CBLAS_ORDER, trans_a: CBLAS_TRANSPOSE, trans_b: CBLAS_TRANSPOSE, m: Int, n: Int, k: Int, alpha: Float, a: FixedArray[Float], lda: Int, b: FixedArray[Float], ldb: Int, beta: Float, c: FixedArray[Float], ldc: Int) = "cblas_sgemm";
///|
///
/// ```c
/// void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
///	 const double alpha, const double *A, const blasint lda, const double *B, const blasint ldb, const double beta, double *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_dgemm(order: CBLAS_ORDER, trans_a: CBLAS_TRANSPOSE, trans_b: CBLAS_TRANSPOSE, m: Int, n: Int, k: Int, alpha: Double, a: FixedArray[Double], lda: Int, b: FixedArray[Double], ldb: Int, beta: Double, c: FixedArray[Double], ldc: Int) = "cblas_dgemm";
///|
///
/// ```c
/// void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
///	 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_cgemm(order: CBLAS_ORDER, trans_a: CBLAS_TRANSPOSE, trans_b: CBLAS_TRANSPOSE, m: Int, n: Int, k: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int, beta: FixedArray[ComplexFloat], c: FixedArray[ComplexFloat], ldc: Int) = "cblas_cgemm";
///|
///
/// ```c
/// void cblas_cgemm3m(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
///	 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_cgemm3m(order: CBLAS_ORDER, trans_a: CBLAS_TRANSPOSE, trans_b: CBLAS_TRANSPOSE, m: Int, n: Int, k: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int, beta: FixedArray[ComplexFloat], c: FixedArray[ComplexFloat], ldc: Int) = "cblas_cgemm3m";
///|
///
/// ```c
/// void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
///	 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_zgemm(order: CBLAS_ORDER, trans_a: CBLAS_TRANSPOSE, trans_b: CBLAS_TRANSPOSE, m: Int, n: Int, k: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int, beta: FixedArray[ComplexFloat], c: FixedArray[ComplexFloat], ldc: Int) = "cblas_zgemm";
///|
///
/// ```c
/// void cblas_zgemm3m(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
///	 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_zgemm3m(order: CBLAS_ORDER, trans_a: CBLAS_TRANSPOSE, trans_b: CBLAS_TRANSPOSE, m: Int, n: Int, k: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int, beta: FixedArray[ComplexFloat], c: FixedArray[ComplexFloat], ldc: Int) = "cblas_zgemm3m";
//
///|
///
/// ```c
/// void cblas_sgemmt(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint K,
///	 const float alpha, const float *A, const blasint lda, const float *B, const blasint ldb, const float beta, float *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_sgemmt(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, trans_b: CBLAS_TRANSPOSE, m: Int, k: Int, alpha: Float, a: FixedArray[Float], lda: Int, b: FixedArray[Float], ldb: Int, beta: Float, c: FixedArray[Float], ldc: Int) = "cblas_sgemmt";
///|
///
/// ```c
/// void cblas_dgemmt(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint K,
///	 const double alpha, const double *A, const blasint lda, const double *B, const blasint ldb, const double beta, double *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_dgemmt(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, trans_b: CBLAS_TRANSPOSE, m: Int, k: Int, alpha: Double, a: FixedArray[Double], lda: Int, b: FixedArray[Double], ldb: Int, beta: Double, c: FixedArray[Double], ldc: Int) = "cblas_dgemmt";
///|
///
/// ```c
/// void cblas_cgemmt(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint K,
///	 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_cgemmt(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, trans_b: CBLAS_TRANSPOSE, m: Int, k: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int, beta: FixedArray[ComplexFloat], c: FixedArray[ComplexFloat], ldc: Int) = "cblas_cgemmt";
///|
///
/// ```c
/// void cblas_zgemmt(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint K,
///	 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_zgemmt(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, trans_b: CBLAS_TRANSPOSE, m: Int, k: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int, beta: FixedArray[ComplexFloat], c: FixedArray[ComplexFloat], ldc: Int) = "cblas_zgemmt";
//
///|
///
/// ```c
/// void cblas_ssymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const blasint M, const blasint N,
///                 const float alpha, const float *A, const blasint lda, const float *B, const blasint ldb, const float beta, float *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_ssymm(order: CBLAS_ORDER, side: CBLAS_SIDE, uplo: CBLAS_UPLO, m: Int, n: Int, alpha: Float, a: FixedArray[Float], lda: Int, b: FixedArray[Float], ldb: Int, beta: Float, c: FixedArray[Float], ldc: Int) = "cblas_ssymm";
///|
///
/// ```c
/// void cblas_dsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const blasint M, const blasint N,
///                 const double alpha, const double *A, const blasint lda, const double *B, const blasint ldb, const double beta, double *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_dsymm(order: CBLAS_ORDER, side: CBLAS_SIDE, uplo: CBLAS_UPLO, m: Int, n: Int, alpha: Double, a: FixedArray[Double], lda: Int, b: FixedArray[Double], ldb: Int, beta: Double, c: FixedArray[Double], ldc: Int) = "cblas_dsymm";
///|
///
/// ```c
/// void cblas_csymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const blasint M, const blasint N,
///                 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_csymm(order: CBLAS_ORDER, side: CBLAS_SIDE, uplo: CBLAS_UPLO, m: Int, n: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int, beta: FixedArray[ComplexFloat], c: FixedArray[ComplexFloat], ldc: Int) = "cblas_csymm";
///|
///
/// ```c
/// void cblas_zsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const blasint M, const blasint N,
///                 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_zsymm(order: CBLAS_ORDER, side: CBLAS_SIDE, uplo: CBLAS_UPLO, m: Int, n: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int, beta: FixedArray[ComplexFloat], c: FixedArray[ComplexFloat], ldc: Int) = "cblas_zsymm";

///|
///
/// ```c
/// void cblas_ssyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
///	 const blasint N, const blasint K, const float alpha, const float *A, const blasint lda, const float beta, float *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_ssyrk(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: Int, k: Int, alpha: Float, a: FixedArray[Float], lda: Int, beta: Float, c: FixedArray[Float], ldc: Int) = "cblas_ssyrk";
///|
///
/// ```c
/// void cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
///	 const blasint N, const blasint K, const double alpha, const double *A, const blasint lda, const double beta, double *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_dsyrk(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: Int, k: Int, alpha: Double, a: FixedArray[Double], lda: Int, beta: Double, c: FixedArray[Double], ldc: Int) = "cblas_dsyrk";
///|
///
/// ```c
/// void cblas_csyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
///	 const blasint N, const blasint K, const void *alpha, const void *A, const blasint lda, const void *beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_csyrk(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: Int, k: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, beta: FixedArray[ComplexFloat], c: FixedArray[ComplexFloat], ldc: Int) = "cblas_csyrk";
///|
///
/// ```c
/// void cblas_zsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
///	 const blasint N, const blasint K, const void *alpha, const void *A, const blasint lda, const void *beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_zsyrk(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: Int, k: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, beta: FixedArray[ComplexFloat], c: FixedArray[ComplexFloat], ldc: Int) = "cblas_zsyrk";

///|
///
/// ```c
/// void cblas_ssyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
///	  const blasint N, const blasint K, const float alpha, const float *A, const blasint lda, const float *B, const blasint ldb, const float beta, float *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_ssyr2k(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: Int, k: Int, alpha: Float, a: FixedArray[Float], lda: Int, b: FixedArray[Float], ldb: Int, beta: Float, c: FixedArray[Float], ldc: Int) = "cblas_ssyr2k";
///|
///
/// ```c
/// void cblas_dsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
///	  const blasint N, const blasint K, const double alpha, const double *A, const blasint lda, const double *B, const blasint ldb, const double beta, double *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_dsyr2k(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: Int, k: Int, alpha: Double, a: FixedArray[Double], lda: Int, b: FixedArray[Double], ldb: Int, beta: Double, c: FixedArray[Double], ldc: Int) = "cblas_dsyr2k";
///|
///
/// ```c
/// void cblas_csyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
///	  const blasint N, const blasint K, const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_csyr2k(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: Int, k: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int, beta: FixedArray[ComplexFloat], c: FixedArray[ComplexFloat], ldc: Int) = "cblas_csyr2k";
///|
///
/// ```c
/// void cblas_zsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
///	  const blasint N, const blasint K, const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_zsyr2k(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: Int, k: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int, beta: FixedArray[ComplexFloat], c: FixedArray[ComplexFloat], ldc: Int) = "cblas_zsyr2k";
//
///|
///
/// ```c
/// void cblas_strmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
///                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const float alpha, const float *A, const blasint lda, float *B, const blasint ldb);
/// ```
pub extern "C" fn cblas_strmm(order: CBLAS_ORDER, side: CBLAS_SIDE, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: Int, n: Int, alpha: Float, a: FixedArray[Float], lda: Int, b: FixedArray[Float], ldb: Int) = "cblas_strmm";
///|
///
/// ```c
/// void cblas_dtrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
///                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const double alpha, const double *A, const blasint lda, double *B, const blasint ldb);
/// ```
pub extern "C" fn cblas_dtrmm(order: CBLAS_ORDER, side: CBLAS_SIDE, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: Int, n: Int, alpha: Double, a: FixedArray[Double], lda: Int, b: FixedArray[Double], ldb: Int) = "cblas_dtrmm";
///|
///
/// ```c
/// void cblas_ctrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
///                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const void *alpha, const void *A, const blasint lda, void *B, const blasint ldb);
/// ```
pub extern "C" fn cblas_ctrmm(order: CBLAS_ORDER, side: CBLAS_SIDE, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: Int, n: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int) = "cblas_ctrmm";
///|
///
/// ```c
/// void cblas_ztrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
///                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const void *alpha, const void *A, const blasint lda, void *B, const blasint ldb);
/// ```
pub extern "C" fn cblas_ztrmm(order: CBLAS_ORDER, side: CBLAS_SIDE, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: Int, n: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int) = "cblas_ztrmm";

///|
///
/// ```c
/// void cblas_strsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
///                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const float alpha, const float *A, const blasint lda, float *B, const blasint ldb);
/// ```
pub extern "C" fn cblas_strsm(order: CBLAS_ORDER, side: CBLAS_SIDE, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: Int, n: Int, alpha: Float, a: FixedArray[Float], lda: Int, b: FixedArray[Float], ldb: Int) = "cblas_strsm";
///|
///
/// ```c
/// void cblas_dtrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
///                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const double alpha, const double *A, const blasint lda, double *B, const blasint ldb);
/// ```
pub extern "C" fn cblas_dtrsm(order: CBLAS_ORDER, side: CBLAS_SIDE, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: Int, n: Int, alpha: Double, a: FixedArray[Double], lda: Int, b: FixedArray[Double], ldb: Int) = "cblas_dtrsm";
///|
///
/// ```c
/// void cblas_ctrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
///                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const void *alpha, const void *A, const blasint lda, void *B, const blasint ldb);
/// ```
pub extern "C" fn cblas_ctrsm(order: CBLAS_ORDER, side: CBLAS_SIDE, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: Int, n: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int) = "cblas_ctrsm";
///|
///
/// ```c
/// void cblas_ztrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
///                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N, const void *alpha, const void *A, const blasint lda, void *B, const blasint ldb);
/// ```
pub extern "C" fn cblas_ztrsm(order: CBLAS_ORDER, side: CBLAS_SIDE, uplo: CBLAS_UPLO, trans_a: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: Int, n: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int) = "cblas_ztrsm";
//
///|
///
/// ```c
/// void cblas_chemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const blasint M, const blasint N,
///                 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_chemm(order: CBLAS_ORDER, side: CBLAS_SIDE, uplo: CBLAS_UPLO, m: Int, n: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int, beta: FixedArray[ComplexFloat], c: FixedArray[ComplexFloat], ldc: Int) = "cblas_chemm";
///|
///
/// ```c
/// void cblas_zhemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const blasint M, const blasint N,
///                 const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const void *beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_zhemm(order: CBLAS_ORDER, side: CBLAS_SIDE, uplo: CBLAS_UPLO, m: Int, n: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int, beta: FixedArray[ComplexFloat], c: FixedArray[ComplexFloat], ldc: Int) = "cblas_zhemm";

///|
///
/// ```c
/// void cblas_cherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const blasint N, const blasint K,
///                 const float alpha, const void *A, const blasint lda, const float beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_cherk(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: Int, k: Int, alpha: Float, a: FixedArray[ComplexFloat], lda: Int, beta: Float, c: FixedArray[ComplexFloat], ldc: Int) = "cblas_cherk";
///|
///
/// ```c
/// void cblas_zherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const blasint N, const blasint K,
///                 const double alpha, const void *A, const blasint lda, const double beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_zherk(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: Int, k: Int, alpha: Double, a: FixedArray[ComplexFloat], lda: Int, beta: Double, c: FixedArray[ComplexFloat], ldc: Int) = "cblas_zherk";

///|
///
/// ```c
/// void cblas_cher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const blasint N, const blasint K,
///                  const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const float beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_cher2k(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: Int, k: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int, beta: Float, c: FixedArray[ComplexFloat], ldc: Int) = "cblas_cher2k";
///|
///
/// ```c
/// void cblas_zher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const blasint N, const blasint K,
///                  const void *alpha, const void *A, const blasint lda, const void *B, const blasint ldb, const double beta, void *C, const blasint ldc);
/// ```
pub extern "C" fn cblas_zher2k(order: CBLAS_ORDER, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: Int, k: Int, alpha: FixedArray[ComplexFloat], a: FixedArray[ComplexFloat], lda: Int, b: FixedArray[ComplexFloat], ldb: Int, beta: Double, c: FixedArray[ComplexFloat], ldc: Int) = "cblas_zher2k";
//
//void cblas_xerbla(blasint p, const char *rout, const char *form, ...);
///// Note: cblas_xerbla skipped due to variadic arguments

///*** BLAS extensions ***/

///|
///
/// ```c
/// void cblas_saxpby(const blasint n, const float alpha, const float *x, const blasint incx,const float beta, float *y, const blasint incy);
/// ```
pub extern "C" fn cblas_saxpby(n: Int, alpha: Float, x: FixedArray[Float], incx: Int, beta: Float, y: FixedArray[Float], incy: Int) = "cblas_saxpby";

///|
///
/// ```c
/// void cblas_daxpby(const blasint n, const double alpha, const double *x, const blasint incx,const double beta, double *y, const blasint incy);
/// ```
pub extern "C" fn cblas_daxpby(n: Int, alpha: Double, x: FixedArray[Double], incx: Int, beta: Double, y: FixedArray[Double], incy: Int) = "cblas_daxpby";

///|
///
/// ```c
/// void cblas_caxpby(const blasint n, const void *alpha, const void *x, const blasint incx,const void *beta, void *y, const blasint incy);
/// ```
pub extern "C" fn cblas_caxpby(n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, beta: FixedArray[ComplexFloat], y: FixedArray[ComplexFloat], incy: Int) = "cblas_caxpby";

///|
///
/// ```c
/// void cblas_zaxpby(const blasint n, const void *alpha, const void *x, const blasint incx,const void *beta, void *y, const blasint incy);
/// ```
pub extern "C" fn cblas_zaxpby(n: Int, alpha: FixedArray[ComplexFloat], x: FixedArray[ComplexFloat], incx: Int, beta: FixedArray[ComplexFloat], y: FixedArray[ComplexFloat], incy: Int) = "cblas_zaxpby";
//
///|
///
/// ```c
/// void cblas_somatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const float calpha, const float *a, 
///	     const blasint clda, float *b, const blasint cldb);
/// ```
pub extern "C" fn cblas_somatcopy(corder: CBLAS_ORDER, ctrans: CBLAS_TRANSPOSE, crows: Int, ccols: Int, calpha: Float, a: FixedArray[Float], clda: Int, b: FixedArray[Float], cldb: Int) = "cblas_somatcopy";
///|
///
/// ```c
/// void cblas_domatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const double calpha, const double *a,
///	     const blasint clda, double *b, const blasint cldb);
/// ```
pub extern "C" fn cblas_domatcopy(corder: CBLAS_ORDER, ctrans: CBLAS_TRANSPOSE, crows: Int, ccols: Int, calpha: Double, a: FixedArray[Double], clda: Int, b: FixedArray[Double], cldb: Int) = "cblas_domatcopy";
///|
///
/// ```c
/// void cblas_comatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const float* calpha, const float* a, 
///	     const blasint clda, float*b, const blasint cldb);
/// ```
pub extern "C" fn cblas_comatcopy(corder: CBLAS_ORDER, ctrans: CBLAS_TRANSPOSE, crows: Int, ccols: Int, calpha: FixedArray[Float], a: FixedArray[Float], clda: Int, b: FixedArray[Float], cldb: Int) = "cblas_comatcopy";
///|
///
/// ```c
/// void cblas_zomatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const double* calpha, const double* a, 
///	     const blasint clda,  double *b, const blasint cldb);
/// ```
pub extern "C" fn cblas_zomatcopy(corder: CBLAS_ORDER, ctrans: CBLAS_TRANSPOSE, crows: Int, ccols: Int, calpha: FixedArray[Double], a: FixedArray[Double], clda: Int, b: FixedArray[Double], cldb: Int) = "cblas_zomatcopy"; 
//
///|
///
/// ```c
/// void cblas_simatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const float calpha, float *a, 
///	     const blasint clda, const blasint cldb);
/// ```
pub extern "C" fn cblas_simatcopy(corder: CBLAS_ORDER, ctrans: CBLAS_TRANSPOSE, crows: Int, ccols: Int, calpha: Float, a: FixedArray[Float], clda: Int, cldb: Int) = "cblas_simatcopy";
///|
///
/// ```c
/// void cblas_dimatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const double calpha, double *a,
///	     const blasint clda, const blasint cldb);
/// ```
pub extern "C" fn cblas_dimatcopy(corder: CBLAS_ORDER, ctrans: CBLAS_TRANSPOSE, crows: Int, ccols: Int, calpha: Double, a: FixedArray[Double], clda: Int, cldb: Int) = "cblas_dimatcopy";
///|
///
/// ```c
/// void cblas_cimatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const float* calpha, float* a, 
///	     const blasint clda, const blasint cldb);
/// ```
pub extern "C" fn cblas_cimatcopy(corder: CBLAS_ORDER, ctrans: CBLAS_TRANSPOSE, crows: Int, ccols: Int, calpha: FixedArray[Float], a: FixedArray[Float], clda: Int, cldb: Int) = "cblas_cimatcopy";
///|
///
/// ```c
/// void cblas_zimatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const blasint crows, const blasint ccols, const double* calpha, double* a, 
///	     const blasint clda, const blasint cldb);
/// ```
pub extern "C" fn cblas_zimatcopy(corder: CBLAS_ORDER, ctrans: CBLAS_TRANSPOSE, crows: Int, ccols: Int, calpha: FixedArray[Double], a: FixedArray[Double], clda: Int, cldb: Int) = "cblas_zimatcopy"; 
//
///|
///
/// ```c
/// void cblas_sgeadd(const enum CBLAS_ORDER CORDER,const blasint crows, const blasint ccols, const float calpha, const float *a, const blasint clda, const float cbeta, 
///	  float *c, const blasint cldc);
/// ```
pub extern "C" fn cblas_sgeadd(corder: CBLAS_ORDER, crows: Int, ccols: Int, calpha: Float, a: FixedArray[Float], clda: Int, cbeta: Float, c: FixedArray[Float], cldc: Int) = "cblas_sgeadd";
///|
///
/// ```c
/// void cblas_dgeadd(const enum CBLAS_ORDER CORDER,const blasint crows, const blasint ccols, const double calpha, const double *a, const blasint clda, const double cbeta, 
///	  double *c, const blasint cldc);
/// ```
pub extern "C" fn cblas_dgeadd(corder: CBLAS_ORDER, crows: Int, ccols: Int, calpha: Double, a: FixedArray[Double], clda: Int, cbeta: Double, c: FixedArray[Double], cldc: Int) = "cblas_dgeadd";
///|
///
/// ```c
/// void cblas_cgeadd(const enum CBLAS_ORDER CORDER,const blasint crows, const blasint ccols, const float *calpha, const float *a, const blasint clda, const float *cbeta, 
///	  float *c, const blasint cldc);
/// ```
pub extern "C" fn cblas_cgeadd(corder: CBLAS_ORDER, crows: Int, ccols: Int, calpha: FixedArray[Float], a: FixedArray[Float], clda: Int, cbeta: FixedArray[Float], c: FixedArray[Float], cldc: Int) = "cblas_cgeadd";
///|
///
/// ```c
/// void cblas_zgeadd(const enum CBLAS_ORDER CORDER,const blasint crows, const blasint ccols, const double *calpha, const double *a, const blasint clda, const double *cbeta, 
///	  double *c, const blasint cldc);
/// ```
pub extern "C" fn cblas_zgeadd(corder: CBLAS_ORDER, crows: Int, ccols: Int, calpha: FixedArray[Double], a: FixedArray[Double], clda: Int, cbeta: FixedArray[Double], c: FixedArray[Double], cldc: Int) = "cblas_zgeadd"; 
//
///|
///
/// ```c
/// void cblas_sgemm_batch(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE * TransA_array, const enum CBLAS_TRANSPOSE * TransB_array, const blasint * M_array, const blasint * N_array, const blasint * K_array,
///	       const float * alpha_array, const float ** A_array, const blasint * lda_array, const float ** B_array, const blasint * ldb_array, const float * beta_array, float ** C_array, const blasint * ldc_array, const blasint group_count, const blasint * group_size);
/// ```
pub extern "C" fn cblas_sgemm_batch(order: CBLAS_ORDER, trans_a_array: VoidPtr, trans_b_array: VoidPtr, m_array: FixedArray[Int], n_array: FixedArray[Int], k_array: FixedArray[Int], alpha_array: FixedArray[Float], a_array: VoidPtr, lda_array: FixedArray[Int], b_array: VoidPtr, ldb_array: FixedArray[Int], beta_array: FixedArray[Float], c_array: VoidPtr, ldc_array: FixedArray[Int], group_count: Int, group_size: FixedArray[Int]) = "cblas_sgemm_batch";

///|
///
/// ```c
/// void cblas_dgemm_batch(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE * TransA_array, const enum CBLAS_TRANSPOSE * TransB_array, const blasint * M_array, const blasint * N_array, const blasint * K_array,
///	       const double * alpha_array, const double ** A_array, const blasint * lda_array, const double ** B_array, const blasint * ldb_array, const double * beta_array, double ** C_array, const blasint * ldc_array, const blasint group_count, const blasint * group_size);
/// ```
pub extern "C" fn cblas_dgemm_batch(order: CBLAS_ORDER, trans_a_array: VoidPtr, trans_b_array: VoidPtr, m_array: FixedArray[Int], n_array: FixedArray[Int], k_array: FixedArray[Int], alpha_array: FixedArray[Double], a_array: VoidPtr, lda_array: FixedArray[Int], b_array: VoidPtr, ldb_array: FixedArray[Int], beta_array: FixedArray[Double], c_array: VoidPtr, ldc_array: FixedArray[Int], group_count: Int, group_size: FixedArray[Int]) = "cblas_dgemm_batch";

///|
///
/// ```c
/// void cblas_cgemm_batch(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE * TransA_array, const enum CBLAS_TRANSPOSE * TransB_array, const blasint * M_array, const blasint * N_array, const blasint * K_array,
///	       const void * alpha_array, const void ** A_array, const blasint * lda_array, const void ** B_array, const blasint * ldb_array, const void * beta_array, void ** C_array, const blasint * ldc_array, const blasint group_count, const blasint * group_size);
/// ```
pub extern "C" fn cblas_cgemm_batch(order: CBLAS_ORDER, trans_a_array: VoidPtr, trans_b_array: VoidPtr, m_array: FixedArray[Int], n_array: FixedArray[Int], k_array: FixedArray[Int], alpha_array: VoidPtr, a_array: VoidPtr, lda_array: FixedArray[Int], b_array: VoidPtr, ldb_array: FixedArray[Int], beta_array: VoidPtr, c_array: VoidPtr, ldc_array: FixedArray[Int], group_count: Int, group_size: FixedArray[Int]) = "cblas_cgemm_batch";

///|
///
/// ```c
/// void cblas_zgemm_batch(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE * TransA_array, const enum CBLAS_TRANSPOSE * TransB_array, const blasint * M_array, const blasint * N_array, const blasint * K_array,
///	       const void * alpha_array, const void ** A_array, const blasint * lda_array, const void ** B_array, const blasint * ldb_array, const void * beta_array, void ** C_array, const blasint * ldc_array, const blasint group_count, const blasint * group_size);
/// ```
pub extern "C" fn cblas_zgemm_batch(order: CBLAS_ORDER, trans_a_array: VoidPtr, trans_b_array: VoidPtr, m_array: FixedArray[Int], n_array: FixedArray[Int], k_array: FixedArray[Int], alpha_array: VoidPtr, a_array: VoidPtr, lda_array: FixedArray[Int], b_array: VoidPtr, ldb_array: FixedArray[Int], beta_array: VoidPtr, c_array: VoidPtr, ldc_array: FixedArray[Int], group_count: Int, group_size: FixedArray[Int]) = "cblas_zgemm_batch";
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
///	    const float alpha, const bfloat16 *A, const blasint lda, const bfloat16 *B, const blasint ldb, const float beta, float *C, const blasint ldc);
//void cblas_sbgemm_batch(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE * TransA_array, const enum CBLAS_TRANSPOSE * TransB_array, const blasint * M_array, const blasint * N_array, const blasint * K_array,
///	       const float * alpha_array, const bfloat16 ** A_array, const blasint * lda_array, const bfloat16 ** B_array, const blasint * ldb_array, const float * beta_array, float ** C_array, const blasint * ldc_array, const blasint group_count, const blasint * group_size);
