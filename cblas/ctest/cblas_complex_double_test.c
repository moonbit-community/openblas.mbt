#include "cblas_test.h"

int test_cblas_zdotu() {
    int n = 3;
    openblas_complex_double x[] = {openblas_make_complex_double(1.0, 2.0),
                                  openblas_make_complex_double(3.0, 4.0),
                                  openblas_make_complex_double(5.0, 6.0)};
    openblas_complex_double y[] = {openblas_make_complex_double(7.0, 8.0),
                                  openblas_make_complex_double(9.0, 10.0),
                                  openblas_make_complex_double(11.0, 12.0)};
    openblas_complex_double result = cblas_zdotu(n, x, 1, y, 1);
    openblas_complex_double expected = {-39.0, 214.0};
    int failed = 0;
    failed += assert_eq(openblas_complex_double_real(result), openblas_complex_double_real(expected), "cblas_zdotu real part");
    failed += assert_eq(openblas_complex_double_imag(result), openblas_complex_double_imag(expected), "cblas_zdotu imag part");
    return failed;
}
// int test_cblas_zdotc()

// int test_cblas_dzasum()
// int test_cblas_dzsum()
// int test_cblas_dznrm2()

// int test_cblas_izamax()
// int test_cblas_izamin()
// int test_cblas_dzamax()
// int test_cblas_dzamin()
// int test_cblas_izmax()
// int test_cblas_izmin()

// int test_cblas_zaxpy()
// int test_cblas_zaxpyc()
// int test_cblas_zcopy()
// int test_cblas_zswap()
// int test_cblas_zscal()
// int test_cblas_zdscal()

// int test_cblas_zgemv()
// int test_cblas_zgemm()
// int test_cblas_zgemm3m()
// int test_cblas_zgemmt()
// int test_cblas_zsymm()
// int test_cblas_zsyrk()
// int test_cblas_zsyr2k()
// int test_cblas_ztrmm()
// int test_cblas_ztrsm()

// int test_cblas_zhemm()
// int test_cblas_zherk()
// int test_cblas_zher2k()

// int test_cblas_zaxpby()
