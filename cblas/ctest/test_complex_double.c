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

int test_cblas_zdotc() {
    int n = 3;
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // Complex numbers as real,imag pairs
    double y[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    
    // cblas_zdotc returns a complex number - using zdotc_sub for testing
    double result[2] = {0.0, 0.0};
    cblas_zdotc_sub(n, x, 1, y, 1, result);
    
    // Expected result for conjugate dot product: 217-18i
    return assert_eq(result[0], 217.0, "cblas_zdotc real") && 
           assert_eq(result[1], -18.0, "cblas_zdotc imag");
}

int test_cblas_zaxpy() {
    int n = 2;
    double alpha[] = {2.0, 1.0}; // 2+1i
    double x[] = {1.0, 2.0, 3.0, 4.0}; // Complex numbers
    double y[] = {5.0, 6.0, 7.0, 8.0}; // Complex numbers
    
    cblas_zaxpy(n, alpha, x, 1, y, 1);
    
    // Expected: y = alpha*x + y = [(2+i)*(1+2i) + 5+6i, (2+i)*(3+4i) + 7+8i] = [5+11i, 9+19i]
    int result = 1;
    result &= assert_eq(y[0], 5.0, "cblas_zaxpy y[0] real");
    result &= assert_eq(y[1], 11.0, "cblas_zaxpy y[0] imag");
    result &= assert_eq(y[2], 9.0, "cblas_zaxpy y[1] real");
    result &= assert_eq(y[3], 19.0, "cblas_zaxpy y[1] imag");
    return result;
}

int test_cblas_zaxpyc() {
    int n = 2;
    double alpha[] = {1.0, 1.0}; // 1+i
    double x[] = {1.0, 2.0, 3.0, 4.0}; // Complex numbers
    double y[] = {5.0, 6.0, 7.0, 8.0}; // Complex numbers
    
    cblas_zaxpyc(n, alpha, x, 1, y, 1);
    
    // Expected: y = alpha*conj(x) + y = [(1+i)*(1-2i) + 5+6i, (1+i)*(3-4i) + 7+8i] = [8+5i, 14+7i]
    int result = 1;
    result &= assert_eq(y[0], 8.0, "cblas_zaxpyc y[0] real");
    result &= assert_eq(y[1], 5.0, "cblas_zaxpyc y[0] imag");
    result &= assert_eq(y[2], 14.0, "cblas_zaxpyc y[1] real");
    result &= assert_eq(y[3], 7.0, "cblas_zaxpyc y[1] imag");
    return result;
}

int test_cblas_zcopy() {
    int n = 3;
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // Complex numbers
    double y[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // Complex numbers
    
    cblas_zcopy(n, x, 1, y, 1);
    
    // Expected: y = x
    int result = 1;
    for (int i = 0; i < 6; i++) {
        result &= assert_eq(y[i], x[i], "cblas_zcopy");
    }
    return result;
}

int test_cblas_zswap() {
    int n = 2;
    double x[] = {1.0, 2.0, 3.0, 4.0}; // Complex numbers
    double y[] = {5.0, 6.0, 7.0, 8.0}; // Complex numbers
    double x_orig[] = {1.0, 2.0, 3.0, 4.0}; // Store original x
    double y_orig[] = {5.0, 6.0, 7.0, 8.0}; // Store original y
    
    cblas_zswap(n, x, 1, y, 1);
    
    // Expected: x and y are swapped
    int result = 1;
    for (int i = 0; i < 4; i++) {
        result &= assert_eq(x[i], y_orig[i], "cblas_zswap x");
        result &= assert_eq(y[i], x_orig[i], "cblas_zswap y");
    }
    return result;
}

int test_cblas_zscal() {
    int n = 2;
    double alpha[] = {2.0, 1.0}; // 2+i
    double x[] = {1.0, 2.0, 3.0, 4.0}; // Complex numbers
    
    cblas_zscal(n, alpha, x, 1);
    
    // Expected: x = alpha*x = [(2+i)*(1+2i), (2+i)*(3+4i)] = [0+5i, 2+11i]
    int result = 1;
    result &= assert_eq(x[0], 0.0, "cblas_zscal x[0] real");
    result &= assert_eq(x[1], 5.0, "cblas_zscal x[0] imag");
    result &= assert_eq(x[2], 2.0, "cblas_zscal x[1] real");
    result &= assert_eq(x[3], 11.0, "cblas_zscal x[1] imag");
    return result;
}

int test_cblas_zdscal() {
    int n = 2;
    double alpha = 2.0; // Real scalar
    double x[] = {1.0, 2.0, 3.0, 4.0}; // Complex numbers
    
    cblas_zdscal(n, alpha, x, 1);
    
    // Expected: x = alpha*x = [2*(1+2i), 2*(3+4i)] = [2+4i, 6+8i]
    int result = 1;
    result &= assert_eq(x[0], 2.0, "cblas_zdscal x[0] real");
    result &= assert_eq(x[1], 4.0, "cblas_zdscal x[0] imag");
    result &= assert_eq(x[2], 6.0, "cblas_zdscal x[1] real");
    result &= assert_eq(x[3], 8.0, "cblas_zdscal x[1] imag");
    return result;
}

int test_cblas_zrotg() {
    double a[] = {3.0, 4.0}; // 3+4i
    double b[] = {1.0, 2.0}; // 1+2i
    double c[1] = {0.0}; // cosine (real)
    double s[] = {0.0, 0.0}; // sine (complex)
    
    cblas_zrotg(a, b, c, s);
    
    // Check that the function doesn't crash and produces reasonable values
    return (c[0] >= 0.0 && c[0] <= 1.0) ? 0 : 1; // c should be between 0 and 1
}
int test_cblas_zgemv() {
    int m = 2, n = 2;
    double alpha[] = {1.0, 0.0}; // 1+0i
    double beta[] = {0.0, 0.0}; // 0+0i
    // Matrix A: [[1+0i, 2+0i], [3+0i, 4+0i]]
    double a[] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0};
    double x[] = {1.0, 1.0, 2.0, 2.0}; // [1+i, 2+2i]
    double y[] = {0.0, 0.0, 0.0, 0.0}; // Result vector
    
    cblas_zgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, n, x, 1, beta, y, 1);
    
    // Expected result: [5+5i, 11+11i]
    int result = 1;
    result &= assert_eq(y[0], 5.0, "cblas_zgemv y[0] real");
    result &= assert_eq(y[1], 5.0, "cblas_zgemv y[0] imag");
    result &= assert_eq(y[2], 11.0, "cblas_zgemv y[1] real");
    result &= assert_eq(y[3], 11.0, "cblas_zgemv y[1] imag");
    return result;
}

int test_cblas_zgeru() {
    int m = 2, n = 2;
    double alpha[] = {1.0, 0.0}; // 1+0i
    double x[] = {1.0, 1.0, 2.0, 0.0}; // [1+i, 2+0i]
    double y[] = {3.0, 0.0, 1.0, 1.0}; // [3+0i, 1+i]
    double a[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // 2x2 matrix
    
    cblas_zgeru(CblasRowMajor, m, n, alpha, x, 1, y, 1, a, n);
    
    // Expected result: [[3+3i, 0+2i], [6+0i, 2+2i]]
    int result = 1;
    result &= assert_eq(a[0], 3.0, "cblas_zgeru a[0,0] real");
    result &= assert_eq(a[1], 3.0, "cblas_zgeru a[0,0] imag");
    result &= assert_eq(a[2], 0.0, "cblas_zgeru a[0,1] real");
    result &= assert_eq(a[3], 2.0, "cblas_zgeru a[0,1] imag");
    result &= assert_eq(a[4], 6.0, "cblas_zgeru a[1,0] real");
    result &= assert_eq(a[5], 0.0, "cblas_zgeru a[1,0] imag");
    result &= assert_eq(a[6], 2.0, "cblas_zgeru a[1,1] real");
    result &= assert_eq(a[7], 2.0, "cblas_zgeru a[1,1] imag");
    return result;
}

int test_cblas_zgerc() {
    int m = 2, n = 2;
    double alpha[] = {1.0, 0.0}; // 1+0i
    double x[] = {1.0, 1.0, 2.0, 0.0}; // [1+i, 2+0i]
    double y[] = {3.0, 1.0, 1.0, 1.0}; // [3+i, 1+i]
    double a[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // 2x2 matrix
    
    cblas_zgerc(CblasRowMajor, m, n, alpha, x, 1, y, 1, a, n);
    
    // Expected result: [[4+2i, 2+0i], [6-2i, 2-2i]]
    int result = 1;
    result &= assert_eq(a[0], 4.0, "cblas_zgerc a[0,0] real");
    result &= assert_eq(a[1], 2.0, "cblas_zgerc a[0,0] imag");
    result &= assert_eq(a[2], 2.0, "cblas_zgerc a[0,1] real");
    result &= assert_eq(a[3], 0.0, "cblas_zgerc a[0,1] imag");
    result &= assert_eq(a[4], 6.0, "cblas_zgerc a[1,0] real");
    result &= assert_eq(a[5], -2.0, "cblas_zgerc a[1,0] imag");
    result &= assert_eq(a[6], 2.0, "cblas_zgerc a[1,1] real");
    result &= assert_eq(a[7], -2.0, "cblas_zgerc a[1,1] imag");
    return result;
}