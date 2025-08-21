#include "cblas_test.h"

// Note: Most complex double precision functions are skipped due to complex double precision binding issues

// int test_cblas_zdotu() - skipped due to complex double precision binding issues
// int test_cblas_zdotc() - skipped due to complex double precision binding issues

// int test_cblas_dzasum() - skipped due to complex double precision binding issues
// int test_cblas_dzsum() - skipped due to VoidPtr parameter
// int test_cblas_dznrm2() - skipped due to complex double precision binding issues

// int test_cblas_izamax() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_izamin() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_dzamax() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_dzamin() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_izmax() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_izmin() - skipped due to OpenBlasComplexFloat parameter

// int test_cblas_zaxpy() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_zaxpyc() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_zcopy() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_zswap() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_zscal() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_zdscal() - skipped due to OpenBlasComplexFloat parameter

// int test_cblas_zgemv() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_zgemm() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_zgemm3m() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_zgemmt() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_zsymm() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_zsyrk() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_zsyr2k() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_ztrmm() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_ztrsm() - skipped due to OpenBlasComplexFloat parameter

// Hermitian functions are also skipped due to complex parameters
// int test_cblas_zhemm() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_zherk() - skipped due to OpenBlasComplexFloat parameter
// int test_cblas_zher2k() - skipped due to OpenBlasComplexFloat parameter

// BLAS extensions for complex double are also skipped
// int test_cblas_zaxpby() - skipped due to OpenBlasComplexFloat parameter

// Currently no double precision complex tests are implemented due to binding limitations
