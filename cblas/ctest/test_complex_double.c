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

int test_cblas_zhemm() {
    int order = CblasRowMajor;
    int side = CblasLeft;
    int uplo = CblasUpper;
    int m = 2, n = 2;
    openblas_complex_double alpha = openblas_make_complex_double(1.0, 0.0);
    openblas_complex_double beta = openblas_make_complex_double(0.0, 0.0);
    
    // Hermitian matrix A: 2x2 upper triangular stored
    openblas_complex_double a[] = {openblas_make_complex_double(2.0, 0.0), openblas_make_complex_double(1.0, 1.0),
                                  openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(3.0, 0.0)};
    
    // Matrix B: 2x2
    openblas_complex_double b[] = {openblas_make_complex_double(1.0, 0.0), openblas_make_complex_double(0.0, 1.0),
                                  openblas_make_complex_double(2.0, 0.0), openblas_make_complex_double(1.0, 0.0)};
    
    // Matrix C: 2x2 initialized to zero
    openblas_complex_double c[] = {openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0),
                                  openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0)};
    
    cblas_zhemm(order, side, uplo, m, n, &alpha, a, m, b, n, &beta, c, n);
    
    // Just verify function executes without error
    return 0; // No specific checks, just ensure no crash
}

int test_cblas_zherk() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int trans = CblasNoTrans;
    int n = 2, k = 2;
    double alpha = 1.0, beta = 0.0;
    
    // Matrix A: 2x2
    openblas_complex_double a[] = {openblas_make_complex_double(1.0, 1.0), openblas_make_complex_double(2.0, 0.0),
                                  openblas_make_complex_double(0.0, 1.0), openblas_make_complex_double(1.0, 0.0)};
    
    // Matrix C: 2x2 initialized to zero
    openblas_complex_double c[] = {openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0),
                                  openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0)};
    
    cblas_zherk(order, uplo, trans, n, k, alpha, a, k, beta, c, n);
    
    // For Hermitian result, diagonal should be real
    int failed = 0;
    failed += assert_eq(openblas_complex_double_imag(c[0]), 0.0, "zherk C[0,0] imag should be 0");
    failed += assert_eq(openblas_complex_double_imag(c[3]), 0.0, "zherk C[1,1] imag should be 0");
    return failed;
}

int test_cblas_zsyrk() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int trans = CblasNoTrans;
    int n = 2, k = 2;
    openblas_complex_double alpha = openblas_make_complex_double(1.0, 0.0);
    openblas_complex_double beta = openblas_make_complex_double(0.0, 0.0);
    
    // Matrix A: 2x2
    openblas_complex_double a[] = {openblas_make_complex_double(1.0, 1.0), openblas_make_complex_double(2.0, 0.0),
                                  openblas_make_complex_double(0.0, 1.0), openblas_make_complex_double(1.0, 0.0)};
    
    // Matrix C: 2x2 initialized to zero
    openblas_complex_double c[] = {openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0),
                                  openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0)};
    
    cblas_zsyrk(order, uplo, trans, n, k, &alpha, a, k, &beta, c, n);
    
    // Just verify function executes without error
    return 0;
}

int test_cblas_ztrmm() {
    int order = CblasRowMajor;
    int side = CblasLeft;
    int uplo = CblasUpper;
    int transa = CblasNoTrans;
    int diag = CblasNonUnit;
    int m = 2, n = 2;
    openblas_complex_double alpha = openblas_make_complex_double(1.0, 0.0);
    
    // Triangular matrix A: 2x2 upper triangular
    openblas_complex_double a[] = {openblas_make_complex_double(1.0, 1.0), openblas_make_complex_double(2.0, 0.0),
                                  openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(1.0, 0.0)};
    
    // Matrix B: 2x2
    openblas_complex_double b[] = {openblas_make_complex_double(1.0, 0.0), openblas_make_complex_double(0.0, 1.0),
                                  openblas_make_complex_double(2.0, 0.0), openblas_make_complex_double(1.0, 0.0)};
    
    cblas_ztrmm(order, side, uplo, transa, diag, m, n, &alpha, a, m, b, n);
    
    // Just verify function executes without error
    return 0;
}

int test_cblas_ztrsm() {
    int order = CblasRowMajor;
    int side = CblasLeft;
    int uplo = CblasUpper;
    int transa = CblasNoTrans;
    int diag = CblasNonUnit;
    int m = 2, n = 2;
    openblas_complex_double alpha = openblas_make_complex_double(1.0, 0.0);
    
    // Triangular matrix A: 2x2 upper triangular
    openblas_complex_double a[] = {openblas_make_complex_double(2.0, 0.0), openblas_make_complex_double(1.0, 0.0),
                                  openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(2.0, 0.0)};
    
    // Matrix B: 2x2
    openblas_complex_double b[] = {openblas_make_complex_double(4.0, 0.0), openblas_make_complex_double(2.0, 0.0),
                                  openblas_make_complex_double(2.0, 0.0), openblas_make_complex_double(4.0, 0.0)};
    
    cblas_ztrsm(order, side, uplo, transa, diag, m, n, &alpha, a, m, b, n);
    
    // Just verify function executes without error
    return 0;
}

int test_cblas_zgemmt() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int transa = CblasNoTrans;
    int transb = CblasNoTrans;
    int n = 2, k = 2;
    openblas_complex_double alpha = openblas_make_complex_double(1.0, 0.0);
    openblas_complex_double beta = openblas_make_complex_double(0.0, 0.0);
    
    // Matrix A: 2x2
    openblas_complex_double a[] = {openblas_make_complex_double(1.0, 1.0), openblas_make_complex_double(2.0, 0.0),
                                  openblas_make_complex_double(0.0, 1.0), openblas_make_complex_double(1.0, 0.0)};
    
    // Matrix B: 2x2
    openblas_complex_double b[] = {openblas_make_complex_double(2.0, 0.0), openblas_make_complex_double(1.0, 1.0),
                                  openblas_make_complex_double(1.0, 0.0), openblas_make_complex_double(2.0, 1.0)};
    
    // Matrix C: 2x2 initialized to zero
    openblas_complex_double c[] = {openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0),
                                  openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0)};
    
    cblas_zgemmt(order, uplo, transa, transb, n, k, &alpha, a, k, b, k, &beta, c, n);
    
    // Just verify function executes without error
    return 0;
}

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
int test_cblas_zdotu_sub() {
    int n = 2;
    double x[] = {1.0, 2.0, 3.0, 4.0}; // [1+2i, 3+4i]
    double y[] = {5.0, 6.0, 7.0, 8.0}; // [5+6i, 7+8i]
    double result[] = {0.0, 0.0}; // result as flattened complex number
    
    cblas_zdotu_sub(n, x, 1, y, 1, result);
    
    // Expected result: (-18+68i)
    int failed = 0;
    failed += assert_eq(result[0], -18.0, "cblas_zdotu_sub real");
    failed += assert_eq(result[1], 68.0, "cblas_zdotu_sub imag");
    return failed;
}

int test_cblas_zdotc_sub() {
    int n = 2;
    double x[] = {1.0, 2.0, 3.0, 4.0}; // [1+2i, 3+4i]
    double y[] = {5.0, 6.0, 7.0, 8.0}; // [5+6i, 7+8i]
    double result[] = {0.0, 0.0}; // result as flattened complex number
    
    cblas_zdotc_sub(n, x, 1, y, 1, result);
    
    // Expected result: (70-8i)
    int failed = 0;
    failed += assert_eq(result[0], 70.0, "cblas_zdotc_sub real");
    failed += assert_eq(result[1], -8.0, "cblas_zdotc_sub imag");
    return failed;
}

int test_cblas_zdrot() {
    int n = 3;
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // [1+2i, 3+4i, 5+6i]
    double y[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0}; // [7+8i, 9+10i, 11+12i]
    double c = 0.6, s = 0.8;
    
    cblas_zdrot(n, x, 1, y, 1, c, s);
    
    // Expected result for first element: x'[0] = 6.2+7.6i, y'[0] = 3.4+3.2i
    int failed = 0;
    if (fabs(x[0] - 6.2) > 0.001) {
        printf("cblas_zdrot x[0] real Test Failed: actual: %f, expect: %f\n", x[0], 6.2);
        failed++;
    }
    if (fabs(x[1] - 7.6) > 0.001) {
        printf("cblas_zdrot x[0] imag Test Failed: actual: %f, expect: %f\n", x[1], 7.6);
        failed++;
    }
    if (fabs(y[0] - 3.4) > 0.001) {
        printf("cblas_zdrot y[0] real Test Failed: actual: %f, expect: %f\n", y[0], 3.4);
        failed++;
    }
    if (fabs(y[1] - 3.2) > 0.001) {
        printf("cblas_zdrot y[0] imag Test Failed: actual: %f, expect: %f\n", y[1], 3.2);
        failed++;
    }
    return failed;
}

int test_cblas_zaxpby() {
    int n = 2;
    float alpha[] = {2.0, 1.0}; // 2+1i (using Float as per interface)
    float beta[] = {1.0, 1.0}; // 1+1i
    float x[] = {1.0, 2.0, 3.0, 4.0}; // [1+2i, 3+4i] (using Float)
    float y[] = {5.0, 6.0, 7.0, 8.0}; // [5+6i, 7+8i] (using Float)
    
    cblas_zaxpby(n, alpha, x, 1, beta, y, 1);
    
    // Expected result: y = alpha*x + beta*y = [-1+16i, 1+26i]
    int failed = 0;
    failed += assert_eq(y[0], -1.0, "cblas_zaxpby y[0] real");
    failed += assert_eq(y[1], 16.0, "cblas_zaxpby y[0] imag");
    failed += assert_eq(y[2], 1.0, "cblas_zaxpby y[1] real");
    failed += assert_eq(y[3], 26.0, "cblas_zaxpby y[1] imag");
    return failed;
}

int test_cblas_ztrsv() {
    int n = 3;
    // Upper triangular matrix: [[2+0i, 1+0i, 1+0i], [0+0i, 2+0i, 1+0i], [0+0i, 0+0i, 2+0i]]
    double a[] = {2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0};
    double x[] = {6.0, 0.0, 4.0, 0.0, 2.0, 0.0}; // right-hand side [6+0i, 4+0i, 2+0i]
    
    cblas_ztrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1);
    
    // Expected solution: [1.75+0i, 1.5+0i, 1.0+0i]
    int failed = 0;
    if (fabs(x[0] - 1.75) > 0.001) {
        printf("cblas_ztrsv x[0] real Test Failed: actual: %f, expect: %f\n", x[0], 1.75);
        failed++;
    }
    if (fabs(x[1]) > 0.001) {
        printf("cblas_ztrsv x[0] imag Test Failed: actual: %f, expect: %f\n", x[1], 0.0);
        failed++;
    }
    if (fabs(x[2] - 1.5) > 0.001) {
        printf("cblas_ztrsv x[1] real Test Failed: actual: %f, expect: %f\n", x[2], 1.5);
        failed++;
    }
    if (fabs(x[3]) > 0.001) {
        printf("cblas_ztrsv x[1] imag Test Failed: actual: %f, expect: %f\n", x[3], 0.0);
        failed++;
    }
    if (fabs(x[4] - 1.0) > 0.001) {
        printf("cblas_ztrsv x[2] real Test Failed: actual: %f, expect: %f\n", x[4], 1.0);
        failed++;
    }
    if (fabs(x[5]) > 0.001) {
        printf("cblas_ztrsv x[2] imag Test Failed: actual: %f, expect: %f\n", x[5], 0.0);
        failed++;
    }
    return failed;
}

int test_cblas_ztrmv() {
    int n = 3;
    // Upper triangular matrix: [[1+0i, 2+0i, 3+0i], [0+0i, 4+0i, 5+0i], [0+0i, 0+0i, 6+0i]]
    double a[] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0};
    double x[] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0}; // [1+0i, 2+0i, 3+0i]
    
    cblas_ztrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1);
    
    // Expected result: [14+0i, 23+0i, 18+0i]
    int failed = 0;
    failed += assert_eq(x[0], 14.0, "cblas_ztrmv x[0] real");
    failed += assert_eq(x[1], 0.0, "cblas_ztrmv x[0] imag");
    failed += assert_eq(x[2], 23.0, "cblas_ztrmv x[1] real");
    failed += assert_eq(x[3], 0.0, "cblas_ztrmv x[1] imag");
    failed += assert_eq(x[4], 18.0, "cblas_ztrmv x[2] real");
    failed += assert_eq(x[5], 0.0, "cblas_ztrmv x[2] imag");
    return failed;
}

int test_cblas_zher() {
    int n = 2;
    double alpha = 2.0; // real scalar
    double x[] = {1.0, 1.0, 2.0, 0.0}; // [1+i, 2+0i]
    double a[] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}; // 2x2 identity matrix
    
    cblas_zher(CblasRowMajor, CblasUpper, n, alpha, x, 1, a, n);
    
    // Expected result: A = [[5+0i, 4+4i], [4-4i, 9+0i]] (only upper triangle updated)
    int failed = 0;
    failed += assert_eq(a[0], 5.0, "cblas_zher a[0,0] real");
    failed += assert_eq(a[1], 0.0, "cblas_zher a[0,0] imag");
    failed += assert_eq(a[2], 4.0, "cblas_zher a[0,1] real");
    failed += assert_eq(a[3], 4.0, "cblas_zher a[0,1] imag");
    failed += assert_eq(a[6], 9.0, "cblas_zher a[1,1] real");
    failed += assert_eq(a[7], 0.0, "cblas_zher a[1,1] imag");
    return failed;
}
int test_cblas_zgemm() {
    int m = 2, n = 2, k = 2;
    double alpha[] = {1.0, 0.0}; // 1+0i
    double beta[] = {0.0, 0.0}; // 0+0i
    
    // Matrix A: 2x2 = [[1+i, 2+0i], [0+i, 1+0i]]
    double a[] = {1.0, 1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0};
    
    // Matrix B: 2x2 = [[2+0i, 1+i], [1+0i, 2+i]]
    double b[] = {2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0};
    
    // Matrix C: 2x2 initialized to zero
    double c[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, n);
    
    // Expected result: C = A * B = [[4+2i, 4+4i], [1+2i, 1+2i]]
    int failed = 0;
    failed += assert_eq(c[0], 4.0, "cblas_zgemm c[0,0] real");
    failed += assert_eq(c[1], 2.0, "cblas_zgemm c[0,0] imag");
    failed += assert_eq(c[2], 4.0, "cblas_zgemm c[0,1] real");
    failed += assert_eq(c[3], 4.0, "cblas_zgemm c[0,1] imag");
    failed += assert_eq(c[4], 1.0, "cblas_zgemm c[1,0] real");
    failed += assert_eq(c[5], 2.0, "cblas_zgemm c[1,0] imag");
    failed += assert_eq(c[6], 1.0, "cblas_zgemm c[1,1] real");
    failed += assert_eq(c[7], 2.0, "cblas_zgemm c[1,1] imag");
    return failed;
}

int test_cblas_zher2() {
    int n = 2;
    double alpha[] = {1.0, 0.0}; // 1+0i
    double x[] = {1.0, 1.0, 2.0, 0.0}; // [1+i, 2+0i]
    double y[] = {2.0, 0.0, 1.0, 1.0}; // [2+0i, 1+i]
    double a[] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}; // 2x2 identity matrix
    
    cblas_zher2(CblasRowMajor, CblasUpper, n, alpha, x, 1, y, 1, a, n);
    
    // Expected result: A = A + alpha*x*conj(y)^H + conj(alpha)*y*conj(x)^H
    // For Hermitian matrices, diagonal elements should be real
    int failed = 0;
    if (fabs(a[1]) > 0.001) {
        printf("cblas_zher2 a[0,0] imag Test Failed: should be 0 for Hermitian\n");
        failed++;
    }
    if (fabs(a[7]) > 0.001) {
        printf("cblas_zher2 a[1,1] imag Test Failed: should be 0 for Hermitian\n");
        failed++;
    }
    return failed;
}

int test_cblas_zgbmv() {
    int m = 3, n = 3, kl = 1, ku = 1;
    double alpha[] = {1.0, 0.0}; // 1+0i
    double beta[] = {0.0, 0.0}; // 0+0i
    
    // Band matrix A stored in band format
    double a[] = {
        0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // row 0
        2.0, 0.0, 2.0, 0.0, 2.0, 0.0, // row 1
        1.0, 0.0, 1.0, 0.0, 0.0, 0.0, // row 2
    };
    
    double x[] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0}; // [1+0i, 2+0i, 3+0i]
    double y[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // Result vector
    
    cblas_zgbmv(CblasRowMajor, CblasNoTrans, m, n, kl, ku, alpha, a, kl + ku + 1, x, 1, beta, y, 1);
    
    // Just check that function executes and produces some result
    int failed = 0;
    if (y[0] == 0.0 && y[1] == 0.0 && y[2] == 0.0 && y[3] == 0.0) {
        printf("cblas_zgbmv Test Failed: result vector is all zeros\n");
        failed++;
    }
    return failed;
}
int test_cblas_zsymm() {
    // Test cblas_zsymm (complex double precision symmetric matrix multiplication)
    int m = 2, n = 2;
    openblas_complex_double alpha = openblas_make_complex_double(1.0, 0.0); // 1+0i
    openblas_complex_double beta = openblas_make_complex_double(0.0, 0.0); // 0+0i
    
    // Symmetric matrix A: [[1+i, 2+0i], [2+0i, 3+i]] (stored as upper triangular)
    openblas_complex_double a[] = {
        openblas_make_complex_double(1.0, 1.0), openblas_make_complex_double(2.0, 0.0),
        openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(3.0, 1.0)
    };
    
    // Matrix B: [[1+0i, 2+i], [3+0i, 4+i]]
    openblas_complex_double b[] = {
        openblas_make_complex_double(1.0, 0.0), openblas_make_complex_double(2.0, 1.0),
        openblas_make_complex_double(3.0, 0.0), openblas_make_complex_double(4.0, 1.0)
    };
    
    // Matrix C: 2x2 initialized to zero
    openblas_complex_double c[] = {
        openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0),
        openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0)
    };
    
    cblas_zsymm(CblasRowMajor, CblasLeft, CblasUpper, m, n, &alpha, a, m, b, n, &beta, c, n);
    
    // Expected result: C = A * B where A is symmetric
    // A * B = [[1+i, 2+0i], [2+0i, 3+i]] * [[1+0i, 2+i], [3+0i, 4+i]]
    // C[0,0] = (1+i)*1 + 2*3 = 1+i + 6 = 7+i
    // C[0,1] = (1+i)*(2+i) + 2*(4+i) = 1+3i-1 + 8+2i = 8+5i
    // C[1,0] = 2*1 + (3+i)*3 = 2 + 9+3i = 11+3i
    // C[1,1] = 2*(2+i) + (3+i)*(4+i) = 4+2i + 11+7i-1 = 14+9i
    int failed = 0;
    failed += assert_eq(openblas_complex_double_real(c[0]), 7.0, "cblas_zsymm C[0,0] real");
    failed += assert_eq(openblas_complex_double_imag(c[0]), 1.0, "cblas_zsymm C[0,0] imag");
    failed += assert_eq(openblas_complex_double_real(c[1]), 9.0, "cblas_zsymm C[0,1] real");
    failed += assert_eq(openblas_complex_double_imag(c[1]), 5.0, "cblas_zsymm C[0,1] imag");
    failed += assert_eq(openblas_complex_double_real(c[2]), 11.0, "cblas_zsymm C[1,0] real");
    failed += assert_eq(openblas_complex_double_imag(c[2]), 3.0, "cblas_zsymm C[1,0] imag");
    failed += assert_eq(openblas_complex_double_real(c[3]), 15.0, "cblas_zsymm C[1,1] real");
    failed += assert_eq(openblas_complex_double_imag(c[3]), 9.0, "cblas_zsymm C[1,1] imag");
    return failed;
}
int test_cblas_zhemv() {
    // Test cblas_zhemv (complex double precision Hermitian matrix vector multiplication)
    int n = 2;
    openblas_complex_double alpha = openblas_make_complex_double(1.0, 0.0);
    openblas_complex_double beta = openblas_make_complex_double(0.0, 0.0);
    
    // Hermitian matrix A: [[2+0i, 1+i], [1-i, 3+0i]] (stored as upper triangular)
    openblas_complex_double a[] = {
        openblas_make_complex_double(2.0, 0.0), openblas_make_complex_double(1.0, 1.0),
        openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(3.0, 0.0)
    };
    
    // Vector x: [1+i, 2+0i]
    openblas_complex_double x[] = {
        openblas_make_complex_double(1.0, 1.0), openblas_make_complex_double(2.0, 0.0)
    };
    
    // Vector y: [0+0i, 0+0i] (result vector)
    openblas_complex_double y[] = {
        openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0)
    };
    
    cblas_zhemv(CblasRowMajor, CblasUpper, n, &alpha, a, n, x, 1, &beta, y, 1);
    
    // Expected result: y = A * x where A is Hermitian
    // A * x = [[2+0i, 1+i], [1-i, 3+0i]] * [1+i, 2+0i]
    // y[0] = (2+0i)*(1+i) + (1+i)*2 = 2+2i + 2+2i = 4+4i
    // y[1] = (1-i)*(1+i) + (3+0i)*2 = 1+1 + 6 = 8+0i
    int failed = 0;
    failed += assert_eq(openblas_complex_double_real(y[0]), 4.0, "cblas_zhemv y[0] real");
    failed += assert_eq(openblas_complex_double_imag(y[0]), 4.0, "cblas_zhemv y[0] imag");
    failed += assert_eq(openblas_complex_double_real(y[1]), 8.0, "cblas_zhemv y[1] real");
    failed += assert_eq(openblas_complex_double_imag(y[1]), 0.0, "cblas_zhemv y[1] imag");
    return failed;
}
int test_cblas_zhbmv() {
    // Test cblas_zhbmv (complex double precision Hermitian band matrix vector multiplication)
    int n = 3, k = 1;
    openblas_complex_double alpha = openblas_make_complex_double(1.0, 0.0);
    openblas_complex_double beta = openblas_make_complex_double(0.0, 0.0);
    
    // Hermitian band matrix A stored in band format
    openblas_complex_double a[] = {
        openblas_make_complex_double(1.0, 1.0), openblas_make_complex_double(2.0, 1.0),
        openblas_make_complex_double(2.0, 0.0), openblas_make_complex_double(3.0, 0.0), openblas_make_complex_double(4.0, 0.0)
    };
    
    // Vector x: [1+0i, 2+0i, 1+i]
    openblas_complex_double x[] = {
        openblas_make_complex_double(1.0, 0.0), openblas_make_complex_double(2.0, 0.0), openblas_make_complex_double(1.0, 1.0)
    };
    
    // Vector y: [0+0i, 0+0i, 0+0i] (result vector)
    openblas_complex_double y[] = {
        openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0)
    };
    
    cblas_zhbmv(CblasRowMajor, CblasUpper, n, k, &alpha, a, k + 1, x, 1, &beta, y, 1);
    
    // Just check that function executes and produces some result
    int failed = 0;
    double y0_real = openblas_complex_double_real(y[0]);
    double y0_imag = openblas_complex_double_imag(y[0]);
    if (y0_real == 0.0 && y0_imag == 0.0 && 
        openblas_complex_double_real(y[1]) == 0.0 && openblas_complex_double_imag(y[1]) == 0.0) {
        printf("cblas_zhbmv Test Failed: result vector is all zeros\n");
        failed++;
    }
    return failed;
}
// test_cblas_zgemm3m removed - function not available in current OpenBLAS version
int test_cblas_zhpr() {
    // Test cblas_zhpr (complex double precision Hermitian packed rank-1 update)
    int n = 2;
    double alpha = 2.0; // real scalar
    
    // Vector x: [1+i, 2+0i]
    openblas_complex_double x[] = {
        openblas_make_complex_double(1.0, 1.0), openblas_make_complex_double(2.0, 0.0)
    };
    
    // Hermitian matrix A in packed format: [a00, a01, a11]
    openblas_complex_double a[] = {
        openblas_make_complex_double(1.0, 0.0), openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(1.0, 0.0)
    };
    
    cblas_zhpr(CblasRowMajor, CblasUpper, n, alpha, x, 1, a);
    
    // Expected result: A := alpha*x*conj(x)^T + A
    // Packed: [5+0i, 4+4i, 9+0i]
    int failed = 0;
    failed += assert_eq(openblas_complex_double_real(a[0]), 5.0, "cblas_zhpr a[0,0] real");
    failed += assert_eq(openblas_complex_double_imag(a[0]), 0.0, "cblas_zhpr a[0,0] imag");
    failed += assert_eq(openblas_complex_double_real(a[1]), 4.0, "cblas_zhpr a[0,1] real");
    failed += assert_eq(openblas_complex_double_imag(a[1]), 4.0, "cblas_zhpr a[0,1] imag");
    failed += assert_eq(openblas_complex_double_real(a[2]), 9.0, "cblas_zhpr a[1,1] real");
    failed += assert_eq(openblas_complex_double_imag(a[2]), 0.0, "cblas_zhpr a[1,1] imag");
    return failed;
}
int test_cblas_ztbmv() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int trans = CblasNoTrans;
    int diag = CblasNonUnit;
    int n = 3, k = 1;
    
    // Triangular band matrix A
    openblas_complex_double a[] = {
        openblas_make_complex_double(1.0, 0.0), openblas_make_complex_double(2.0, 0.0),
        openblas_make_complex_double(2.0, 0.0), openblas_make_complex_double(3.0, 0.0),
        openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(1.0, 0.0)
    };
    
    // Vector x
    openblas_complex_double x[] = {
        openblas_make_complex_double(1.0, 0.0),
        openblas_make_complex_double(1.0, 1.0),
        openblas_make_complex_double(2.0, 0.0)
    };
    
    cblas_ztbmv(order, uplo, trans, diag, n, k, a, k + 1, x, 1);
    
    // Just verify function executes without error
    return 0;
}

int test_cblas_zhpmv() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int n = 2;
    openblas_complex_double alpha = openblas_make_complex_double(1.0, 0.0);
    openblas_complex_double beta = openblas_make_complex_double(0.0, 0.0);
    
    // Hermitian matrix A in packed format
    openblas_complex_double ap[] = {
        openblas_make_complex_double(2.0, 0.0), // a00 - must be real
        openblas_make_complex_double(1.0, 1.0), // a01
        openblas_make_complex_double(3.0, 0.0)  // a11 - must be real
    };
    
    // Vector x
    openblas_complex_double x[] = {
        openblas_make_complex_double(1.0, 0.0),
        openblas_make_complex_double(1.0, 1.0)
    };
    
    // Vector y initialized to zero
    openblas_complex_double y[] = {
        openblas_make_complex_double(0.0, 0.0),
        openblas_make_complex_double(0.0, 0.0)
    };
    
    cblas_zhpmv(order, uplo, n, &alpha, ap, x, 1, &beta, y, 1);
    
    // Just verify function executes without error
    return 0;
}

int test_cblas_zsyr2k() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int trans = CblasNoTrans;
    int n = 2, k = 2;
    openblas_complex_double alpha = openblas_make_complex_double(1.0, 0.0);
    openblas_complex_double beta = openblas_make_complex_double(0.0, 0.0);
    
    // Matrix A: 2x2
    openblas_complex_double a[] = {
        openblas_make_complex_double(1.0, 1.0), openblas_make_complex_double(2.0, 0.0),
        openblas_make_complex_double(0.0, 1.0), openblas_make_complex_double(1.0, 0.0)
    };
    
    // Matrix B: 2x2
    openblas_complex_double b[] = {
        openblas_make_complex_double(2.0, 0.0), openblas_make_complex_double(1.0, 1.0),
        openblas_make_complex_double(1.0, 0.0), openblas_make_complex_double(2.0, 1.0)
    };
    
    // Matrix C: 2x2 initialized to zero
    openblas_complex_double c[] = {
        openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0),
        openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0)
    };
    
    cblas_zsyr2k(order, uplo, trans, n, k, &alpha, a, k, b, k, &beta, c, n);
    
    // Just verify function executes without error
    return 0;
}
int test_cblas_zher2k() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int trans = CblasNoTrans;
    int n = 2, k = 2;
    openblas_complex_double alpha = openblas_make_complex_double(1.0, 0.0);
    double beta = 0.0; // real scalar for Hermitian matrices
    
    // Matrix A: 2x2
    openblas_complex_double a[] = {
        openblas_make_complex_double(1.0, 1.0), openblas_make_complex_double(2.0, 0.0),
        openblas_make_complex_double(0.0, 1.0), openblas_make_complex_double(1.0, 0.0)
    };
    
    // Matrix B: 2x2
    openblas_complex_double b[] = {
        openblas_make_complex_double(2.0, 0.0), openblas_make_complex_double(1.0, 1.0),
        openblas_make_complex_double(1.0, 0.0), openblas_make_complex_double(2.0, 1.0)
    };
    
    // Matrix C: 2x2 initialized to zero (Hermitian)
    openblas_complex_double c[] = {
        openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0),
        openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(0.0, 0.0)
    };
    
    cblas_zher2k(order, uplo, trans, n, k, &alpha, a, k, b, k, beta, c, n);
    
    // For Hermitian matrices, diagonal elements should be real
    int failed = 0;
    failed += assert_eq(openblas_complex_double_imag(c[0]), 0.0, "zher2k C[0,0] imag should be 0");
    failed += assert_eq(openblas_complex_double_imag(c[3]), 0.0, "zher2k C[1,1] imag should be 0");
    return failed;
}
int test_cblas_ztbsv() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int trans = CblasNoTrans;
    int diag = CblasNonUnit;
    int n = 3, k = 1;
    
    // Triangular band matrix A: represents [[2, 1, 0], [0, 2, 1], [0, 0, 2]]
    openblas_complex_double a[] = {
        openblas_make_complex_double(1.0, 0.0), openblas_make_complex_double(2.0, 0.0), // [a01, a00]
        openblas_make_complex_double(1.0, 0.0), openblas_make_complex_double(2.0, 0.0), // [a12, a11]
        openblas_make_complex_double(0.0, 0.0), openblas_make_complex_double(2.0, 0.0)  // [0, a22]
    };
    
    // Right-hand side: [6, 4, 2]
    openblas_complex_double x[] = {
        openblas_make_complex_double(6.0, 0.0),
        openblas_make_complex_double(4.0, 0.0),
        openblas_make_complex_double(2.0, 0.0)
    };
    
    cblas_ztbsv(order, uplo, trans, diag, n, k, a, k + 1, x, 1);
    
    // Just verify function executes without error and modifies the vector
    int failed = 0;
    if (openblas_complex_double_real(x[0]) == 6.0 && 
        openblas_complex_double_real(x[1]) == 4.0 && 
        openblas_complex_double_real(x[2]) == 2.0) {
        printf("cblas_ztbsv Test Failed: vector was not modified\n");
        failed++;
    }
    return failed;
}
int test_cblas_ztpmv() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int trans = CblasNoTrans;
    int diag = CblasNonUnit;
    int n = 3;
    
    // Triangular matrix A in packed format (upper triangular)
    // For n=3: [a00, a01, a02, a11, a12, a22]
    openblas_complex_double ap[] = {
        openblas_make_complex_double(2.0, 0.0), // a00
        openblas_make_complex_double(1.0, 0.0), // a01
        openblas_make_complex_double(3.0, 0.0), // a02
        openblas_make_complex_double(4.0, 0.0), // a11
        openblas_make_complex_double(2.0, 0.0), // a12
        openblas_make_complex_double(1.0, 0.0)  // a22
    };
    
    // Vector x: [1+0i, 2+0i, 1+i]
    openblas_complex_double x[] = {
        openblas_make_complex_double(1.0, 0.0),
        openblas_make_complex_double(2.0, 0.0),
        openblas_make_complex_double(1.0, 1.0)
    };
    
    cblas_ztpmv(order, uplo, trans, diag, n, ap, x, 1);
    
    // Just verify function executes without error and modifies the vector
    int failed = 0;
    if (openblas_complex_double_real(x[0]) == 1.0 && 
        openblas_complex_double_real(x[1]) == 2.0) {
        printf("cblas_ztpmv Test Failed: vector was not modified\n");
        failed++;
    }
    return failed;
}
int test_cblas_ztpsv() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int trans = CblasNoTrans;
    int diag = CblasNonUnit;
    int n = 3;
    
    // Triangular matrix A in packed format (upper triangular)
    // For n=3: [a00, a01, a02, a11, a12, a22]
    // Represents the matrix: [[2, 1, 0], [0, 2, 1], [0, 0, 2]]
    openblas_complex_double ap[] = {
        openblas_make_complex_double(2.0, 0.0), // a00
        openblas_make_complex_double(1.0, 0.0), // a01
        openblas_make_complex_double(0.0, 0.0), // a02
        openblas_make_complex_double(2.0, 0.0), // a11
        openblas_make_complex_double(1.0, 0.0), // a12
        openblas_make_complex_double(2.0, 0.0)  // a22
    };
    
    // Right-hand side: [6, 4, 2]
    openblas_complex_double x[] = {
        openblas_make_complex_double(6.0, 0.0),
        openblas_make_complex_double(4.0, 0.0),
        openblas_make_complex_double(2.0, 0.0)
    };
    
    cblas_ztpsv(order, uplo, trans, diag, n, ap, x, 1);
    
    // Just verify function executes without error and modifies the vector
    int failed = 0;
    if (openblas_complex_double_real(x[0]) == 6.0 && 
        openblas_complex_double_real(x[1]) == 4.0 && 
        openblas_complex_double_real(x[2]) == 2.0) {
        printf("cblas_ztpsv Test Failed: vector was not modified\n");
        failed++;
    }
    return failed;
}
int test_cblas_zhpr2() {
    int uplo = CblasUpper;
    int n = 2;
    openblas_complex_double alpha = openblas_make_complex_double(1.0, 0.0);
    
    // Vector x: [1+i, 2+0i]
    openblas_complex_double x[] = {
        openblas_make_complex_double(1.0, 1.0),
        openblas_make_complex_double(2.0, 0.0)
    };
    
    // Vector y: [2+0i, 1+i]
    openblas_complex_double y[] = {
        openblas_make_complex_double(2.0, 0.0),
        openblas_make_complex_double(1.0, 1.0)
    };
    
    // Hermitian matrix A in packed format: [a00, a01, a11]
    openblas_complex_double a[] = {
        openblas_make_complex_double(1.0, 0.0), // a00 - must be real
        openblas_make_complex_double(0.0, 0.0), // a01
        openblas_make_complex_double(1.0, 0.0)  // a11 - must be real
    };
    
    cblas_zhpr2(CblasRowMajor, uplo, n, &alpha, x, 1, y, 1, a);
    
    // For Hermitian matrices, diagonal elements should be real
    int failed = 0;
    failed += assert_eq(openblas_complex_double_imag(a[0]), 0.0, "zhpr2 a[0,0] imag should be 0");
    failed += assert_eq(openblas_complex_double_imag(a[2]), 0.0, "zhpr2 a[1,1] imag should be 0");
    // Check that the diagonal elements changed
    if (openblas_complex_double_real(a[0]) == 1.0) {
        printf("cblas_zhpr2 Test Failed: diagonal element not modified\n");
        failed++;
    }
    return failed;
}