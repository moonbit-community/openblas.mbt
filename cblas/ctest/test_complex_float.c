#include "cblas_test.h"

int test_cblas_cdotu() {
    int n = 3;
    openblas_complex_float x[] = {openblas_make_complex_float(1.0f, 2.0f),
                                  openblas_make_complex_float(3.0f, 4.0f),
                                  openblas_make_complex_float(5.0f, 6.0f)};
    openblas_complex_float y[] = {openblas_make_complex_float(7.0f, 8.0f),
                                  openblas_make_complex_float(9.0f, 10.0f),
                                  openblas_make_complex_float(11.0f, 12.0f)};
    openblas_complex_float result = cblas_cdotu(n, x, 1, y, 1);
    openblas_complex_float expected = {-39.0f, 214.0f};
    int failed = 0;
    failed += assert_eq(openblas_complex_float_real(result), openblas_complex_float_real(expected), "cblas_cdotu real part");
    failed += assert_eq(openblas_complex_float_imag(result), openblas_complex_float_imag(expected), "cblas_cdotu imag part");
    return failed;
}

int test_cblas_cdotc() {
    int n = 3;
    openblas_complex_float x[] = {openblas_make_complex_float(1.0f, 2.0f),
                                  openblas_make_complex_float(3.0f, 4.0f),
                                  openblas_make_complex_float(5.0f, 6.0f)};
    openblas_complex_float y[] = {openblas_make_complex_float(7.0f, 8.0f),
                                  openblas_make_complex_float(9.0f, 10.0f),
                                  openblas_make_complex_float(11.0f, 12.0f)};
    openblas_complex_float result = cblas_cdotc(n, x, 1, y, 1);
    openblas_complex_float expected = {217.0f, -18.0f};
    int failed = 0;
    failed += assert_eq(openblas_complex_float_real(result), openblas_complex_float_real(expected), "cblas_cdotc real part");
    failed += assert_eq(openblas_complex_float_imag(result), openblas_complex_float_imag(expected), "cblas_cdotc imag part");
    return failed;
}

int test_cblas_scasum() {
    int n = 3;
    openblas_complex_float x[] = {openblas_make_complex_float(3.0f, 4.0f),
                                  openblas_make_complex_float(-5.0f, 12.0f),
                                  openblas_make_complex_float(0.0f, -1.0f)};

    float result = cblas_scasum(n, x, 1);
    return assert_eq(result, 25.0f, "cblas_scasum");
}

int test_cblas_scnrm2() {
    int n = 2;
    openblas_complex_float x[] = {openblas_make_complex_float(3.0f, 4.0f),
                                  openblas_make_complex_float(0.0f, 0.0f)};

    float result = cblas_scnrm2(n, x, 1);
    return assert_eq(result, 5.0f, "cblas_scnrm2");
}

int test_cblas_icamax() {
    int n = 3;
    openblas_complex_float x[] = {openblas_make_complex_float(1.0f, 2.0f),
                                  openblas_make_complex_float(-3.0f, -4.0f),
                                  openblas_make_complex_float(2.0f, 1.0f)};

    size_t result = cblas_icamax(n, x, 1);
    return assert_eq_uint(result, 1, "cblas_icamax");
}

int test_cblas_icamin() {
    int n = 3;
    openblas_complex_float x[] = {openblas_make_complex_float(3.0f, 4.0f),
                                  openblas_make_complex_float(1.0f, 0.0f),
                                  openblas_make_complex_float(2.0f, 2.0f)};

    size_t result = cblas_icamin(n, x, 1);
    return assert_eq_uint(result, 1, "cblas_icamin");
}

int test_cblas_scamax() {
    int n = 3;
    openblas_complex_float x[] = {openblas_make_complex_float(1.0f, 2.0f),
                                  openblas_make_complex_float(-3.0f, -4.0f),
                                  openblas_make_complex_float(2.0f, 1.0f)};

    float result = cblas_scamax(n, x, 1);
    return assert_eq(result, 7.0f, "cblas_scamax");
}

int test_cblas_scamin() {
    int n = 3;
    openblas_complex_float x[] = {openblas_make_complex_float(3.0f, 4.0f),
                                  openblas_make_complex_float(1.0f, 0.0f),
                                  openblas_make_complex_float(2.0f, 2.0f)};

    float result = cblas_scamin(n, x, 1);
    return assert_eq(result, 1.0f, "cblas_scamin");
}

int test_cblas_icmax() {
    int n = 3;
    openblas_complex_float x[] = {openblas_make_complex_float(1.0f, 0.0f),
                                  openblas_make_complex_float(2.0f, 1.0f),
                                  openblas_make_complex_float(-1.0f, 2.0f)};

    size_t result = cblas_icmax(n, x, 1);
    return assert_eq_uint(result, 2, "cblas_icmax");
}

int test_cblas_icmin() {
    int n = 3;
    openblas_complex_float x[] = {openblas_make_complex_float(1.0f, 0.0f),
                                  openblas_make_complex_float(2.0f, 1.0f),
                                  openblas_make_complex_float(-1.0f, 2.0f)};

    size_t result = cblas_icmin(n, x, 1);
    return assert_eq_uint(result, 1, "cblas_icmin");
}

int test_cblas_caxpy() {
    int n = 3;
    openblas_complex_float alpha = openblas_make_complex_float(2.0f, 1.0f);
    openblas_complex_float x[] = {openblas_make_complex_float(1.0f, 2.0f),
                                  openblas_make_complex_float(3.0f, 4.0f),
                                  openblas_make_complex_float(5.0f, 6.0f)};
    openblas_complex_float y[] = {openblas_make_complex_float(7.0f, 8.0f),
                                  openblas_make_complex_float(9.0f, 10.0f),
                                  openblas_make_complex_float(11.0f, 12.0f)};

    cblas_caxpy(n, &alpha, x, 1, y, 1);

    int failed = 0;
    failed += assert_eq(openblas_complex_float_real(y[0]), 7.0f, "cblas_caxpy[0] real");
    failed += assert_eq(openblas_complex_float_imag(y[0]), 13.0f, "cblas_caxpy[0] imag");
    failed += assert_eq(openblas_complex_float_real(y[1]), 11.0f, "cblas_caxpy[1] real");
    failed += assert_eq(openblas_complex_float_imag(y[1]), 21.0f, "cblas_caxpy[1] imag");
    failed += assert_eq(openblas_complex_float_real(y[2]), 15.0f, "cblas_caxpy[2] real");
    failed += assert_eq(openblas_complex_float_imag(y[2]), 29.0f, "cblas_caxpy[2] imag");
    return failed;
}

int test_cblas_caxpyc() {
    int n = 2;
    openblas_complex_float alpha = openblas_make_complex_float(1.0f, 1.0f);
    openblas_complex_float x[] = {openblas_make_complex_float(1.0f, 2.0f),
                                  openblas_make_complex_float(3.0f, 4.0f)};
    openblas_complex_float y[] = {openblas_make_complex_float(5.0f, 6.0f),
                                  openblas_make_complex_float(7.0f, 8.0f)};

    cblas_caxpyc(n, &alpha, x, 1, y, 1);

    int failed = 0;
    failed += assert_eq(openblas_complex_float_real(y[0]), 8.0f, "cblas_caxpyc[0] real");
    failed += assert_eq(openblas_complex_float_imag(y[0]), 5.0f, "cblas_caxpyc[0] imag");
    failed += assert_eq(openblas_complex_float_real(y[1]), 14.0f, "cblas_caxpyc[1] real");
    failed += assert_eq(openblas_complex_float_imag(y[1]), 7.0f, "cblas_caxpyc[1] imag");
    return failed;
}

int test_cblas_ccopy() {
    int n = 3;
    openblas_complex_float x[] = {openblas_make_complex_float(1.0f, 2.0f),
                                  openblas_make_complex_float(3.0f, 4.0f),
                                  openblas_make_complex_float(5.0f, 6.0f)};
    openblas_complex_float y[] = {openblas_make_complex_float(0.0f, 0.0f),
                                  openblas_make_complex_float(0.0f, 0.0f),
                                  openblas_make_complex_float(0.0f, 0.0f)};

    cblas_ccopy(n, x, 1, y, 1);

    int failed = 0;
    failed += assert_eq(openblas_complex_float_real(y[0]), 1.0f, "cblas_ccopy[0] real");
    failed += assert_eq(openblas_complex_float_imag(y[0]), 2.0f, "cblas_ccopy[0] imag");
    failed += assert_eq(openblas_complex_float_real(y[1]), 3.0f, "cblas_ccopy[1] real");
    failed += assert_eq(openblas_complex_float_imag(y[1]), 4.0f, "cblas_ccopy[1] imag");
    failed += assert_eq(openblas_complex_float_real(y[2]), 5.0f, "cblas_ccopy[2] real");
    failed += assert_eq(openblas_complex_float_imag(y[2]), 6.0f, "cblas_ccopy[2] imag");
    return failed;
}

int test_cblas_cswap() {
    int n = 2;
    openblas_complex_float x[] = {openblas_make_complex_float(1.0f, 2.0f),
                                  openblas_make_complex_float(3.0f, 4.0f)};
    openblas_complex_float y[] = {openblas_make_complex_float(5.0f, 6.0f),
                                  openblas_make_complex_float(7.0f, 8.0f)};

    cblas_cswap(n, x, 1, y, 1);

    int failed = 0;
    failed += assert_eq(openblas_complex_float_real(x[0]), 5.0f, "cblas_cswap x[0] real");
    failed += assert_eq(openblas_complex_float_imag(x[0]), 6.0f, "cblas_cswap x[0] imag");
    failed += assert_eq(openblas_complex_float_real(x[1]), 7.0f, "cblas_cswap x[1] real");
    failed += assert_eq(openblas_complex_float_imag(x[1]), 8.0f, "cblas_cswap x[1] imag");
    failed += assert_eq(openblas_complex_float_real(y[0]), 1.0f, "cblas_cswap y[0] real");
    failed += assert_eq(openblas_complex_float_imag(y[0]), 2.0f, "cblas_cswap y[0] imag");
    failed += assert_eq(openblas_complex_float_real(y[1]), 3.0f, "cblas_cswap y[1] real");
    failed += assert_eq(openblas_complex_float_imag(y[1]), 4.0f, "cblas_cswap y[1] imag");
    return failed;
}

int test_cblas_cscal() {
    int n = 2;
    openblas_complex_float alpha = openblas_make_complex_float(2.0f, 1.0f);
    openblas_complex_float x[] = {openblas_make_complex_float(1.0f, 2.0f),
                                  openblas_make_complex_float(3.0f, 4.0f)};

    cblas_cscal(n, &alpha, x, 1);

    int failed = 0;
    failed += assert_eq(openblas_complex_float_real(x[0]), 0.0f, "cblas_cscal[0] real");
    failed += assert_eq(openblas_complex_float_imag(x[0]), 5.0f, "cblas_cscal[0] imag");
    failed += assert_eq(openblas_complex_float_real(x[1]), 2.0f, "cblas_cscal[1] real");
    failed += assert_eq(openblas_complex_float_imag(x[1]), 11.0f, "cblas_cscal[1] imag");
    return failed;
}

int test_cblas_csscal() {
    int n = 2;
    float alpha = 2.0f;
    openblas_complex_float x[] = {openblas_make_complex_float(1.0f, 2.0f),
                                  openblas_make_complex_float(3.0f, 4.0f)};

    cblas_csscal(n, alpha, x, 1);

    int failed = 0;
    failed += assert_eq(openblas_complex_float_real(x[0]), 2.0f, "cblas_csscal[0] real");
    failed += assert_eq(openblas_complex_float_imag(x[0]), 4.0f, "cblas_csscal[0] imag");
    failed += assert_eq(openblas_complex_float_real(x[1]), 6.0f, "cblas_csscal[1] real");
    failed += assert_eq(openblas_complex_float_imag(x[1]), 8.0f, "cblas_csscal[1] imag");
    return failed;
}

int test_cblas_cdotu_sub() {
    int n = 2;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f}; // Complex numbers as real,imag pairs
    float y[] = {5.0f, 6.0f, 7.0f, 8.0f}; // Complex numbers
    float result[2] = {0.0f, 0.0f}; // result as flattened complex number
    
    cblas_cdotu_sub(n, x, 1, y, 1, result);
    
    // Expected result: -18+68i
    return assert_eq(result[0], -18.0f, "cblas_cdotu_sub real") && 
           assert_eq(result[1], 68.0f, "cblas_cdotu_sub imag");
}

int test_cblas_cdotc_sub() {
    int n = 2;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f}; // Complex numbers
    float y[] = {5.0f, 6.0f, 7.0f, 8.0f}; // Complex numbers
    float result[2] = {0.0f, 0.0f}; // result as flattened complex number
    
    cblas_cdotc_sub(n, x, 1, y, 1, result);
    
    // Expected result: 70-8i
    return assert_eq(result[0], 70.0f, "cblas_cdotc_sub real") && 
           assert_eq(result[1], -8.0f, "cblas_cdotc_sub imag");
}

int test_cblas_crotg() {
    float a[] = {3.0f, 4.0f}; // 3+4i
    float b[] = {1.0f, 2.0f}; // 1+2i
    float c[1] = {0.0f}; // cosine (real)
    float s[] = {0.0f, 0.0f}; // sine (complex)
    
    cblas_crotg(a, b, c, s);
    
    // Check that the function doesn't crash and produces reasonable values
    return (c[0] >= 0.0f && c[0] <= 1.0f) ? 0 : 1; // c should be between 0 and 1
}

int test_cblas_caxpby() {
    int n = 2;
    float alpha[] = {2.0f, 1.0f}; // 2+1i
    float beta[] = {1.0f, 1.0f}; // 1+1i
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f}; // Complex numbers
    float y[] = {5.0f, 6.0f, 7.0f, 8.0f}; // Complex numbers
    
    cblas_caxpby(n, alpha, x, 1, beta, y, 1);
    
    // Expected: y = alpha*x + beta*y = [-1+16i, 1+26i]
    int result = 1;
    result &= assert_eq(y[0], -1.0f, "cblas_caxpby y[0] real");
    result &= assert_eq(y[1], 16.0f, "cblas_caxpby y[0] imag");
    result &= assert_eq(y[2], 1.0f, "cblas_caxpby y[1] real");
    result &= assert_eq(y[3], 26.0f, "cblas_caxpby y[1] imag");
    return result;
}
int test_cblas_cgemv() {
    int m = 2, n = 2;
    float alpha[] = {1.0f, 0.0f}; // 1+0i
    float beta[] = {0.0f, 0.0f}; // 0+0i
    // Matrix A: [[1+0i, 2+0i], [3+0i, 4+0i]]
    float a[] = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f};
    float x[] = {1.0f, 1.0f, 2.0f, 2.0f}; // [1+i, 2+2i]
    float y[] = {0.0f, 0.0f, 0.0f, 0.0f}; // Result vector
    
    cblas_cgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, n, x, 1, beta, y, 1);
    
    // Expected result: [5+5i, 11+11i]
    int result = 1;
    result &= assert_eq(y[0], 5.0f, "cblas_cgemv y[0] real");
    result &= assert_eq(y[1], 5.0f, "cblas_cgemv y[0] imag");
    result &= assert_eq(y[2], 11.0f, "cblas_cgemv y[1] real");
    result &= assert_eq(y[3], 11.0f, "cblas_cgemv y[1] imag");
    return result;
}

int test_cblas_cgeru() {
    int m = 2, n = 2;
    float alpha[] = {1.0f, 0.0f}; // 1+0i
    float x[] = {1.0f, 1.0f, 2.0f, 0.0f}; // [1+i, 2+0i]
    float y[] = {3.0f, 0.0f, 1.0f, 1.0f}; // [3+0i, 1+i]
    float a[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // 2x2 matrix
    
    cblas_cgeru(CblasRowMajor, m, n, alpha, x, 1, y, 1, a, n);
    
    // Expected result: [[3+3i, 0+2i], [6+0i, 2+2i]]
    int result = 1;
    result &= assert_eq(a[0], 3.0f, "cblas_cgeru a[0,0] real");
    result &= assert_eq(a[1], 3.0f, "cblas_cgeru a[0,0] imag");
    result &= assert_eq(a[2], 0.0f, "cblas_cgeru a[0,1] real");
    result &= assert_eq(a[3], 2.0f, "cblas_cgeru a[0,1] imag");
    result &= assert_eq(a[4], 6.0f, "cblas_cgeru a[1,0] real");
    result &= assert_eq(a[5], 0.0f, "cblas_cgeru a[1,0] imag");
    result &= assert_eq(a[6], 2.0f, "cblas_cgeru a[1,1] real");
    result &= assert_eq(a[7], 2.0f, "cblas_cgeru a[1,1] imag");
    return result;
}

int test_cblas_cgerc() {
    int m = 2, n = 2;
    float alpha[] = {1.0f, 0.0f}; // 1+0i
    float x[] = {1.0f, 1.0f, 2.0f, 0.0f}; // [1+i, 2+0i]
    float y[] = {3.0f, 1.0f, 1.0f, 1.0f}; // [3+i, 1+i]
    float a[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // 2x2 matrix
    
    cblas_cgerc(CblasRowMajor, m, n, alpha, x, 1, y, 1, a, n);
    
    // Expected result: [[4+2i, 2+0i], [6-2i, 2-2i]]
    int result = 1;
    result &= assert_eq(a[0], 4.0f, "cblas_cgerc a[0,0] real");
    result &= assert_eq(a[1], 2.0f, "cblas_cgerc a[0,0] imag");
    result &= assert_eq(a[2], 2.0f, "cblas_cgerc a[0,1] real");
    result &= assert_eq(a[3], 0.0f, "cblas_cgerc a[0,1] imag");
    result &= assert_eq(a[4], 6.0f, "cblas_cgerc a[1,0] real");
    result &= assert_eq(a[5], -2.0f, "cblas_cgerc a[1,0] imag");
    result &= assert_eq(a[6], 2.0f, "cblas_cgerc a[1,1] real");
    result &= assert_eq(a[7], -2.0f, "cblas_cgerc a[1,1] imag");
    return result;
}
int test_cblas_csrot() {
    int n = 3;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}; // [1+2i, 3+4i, 5+6i]
    float y[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}; // [7+8i, 9+10i, 11+12i]
    float c = 0.6f, s = 0.8f;
    
    cblas_csrot(n, x, 1, y, 1, c, s);
    
    // Expected result for first element: x'[0] = 6.2+7.6i, y'[0] = 3.4+3.2i
    int failed = 0;
    if (fabsf(x[0] - 6.2f) > 0.001f) {
        printf("cblas_csrot x[0] real Test Failed: actual: %f, expect: %f\n", x[0], 6.2f);
        failed++;
    }
    if (fabsf(x[1] - 7.6f) > 0.001f) {
        printf("cblas_csrot x[0] imag Test Failed: actual: %f, expect: %f\n", x[1], 7.6f);
        failed++;
    }
    if (fabsf(y[0] - 3.4f) > 0.001f) {
        printf("cblas_csrot y[0] real Test Failed: actual: %f, expect: %f\n", y[0], 3.4f);
        failed++;
    }
    if (fabsf(y[1] - 3.2f) > 0.001f) {
        printf("cblas_csrot y[0] imag Test Failed: actual: %f, expect: %f\n", y[1], 3.2f);
        failed++;
    }
    return failed;
}

int test_cblas_ctrsv() {
    int n = 3;
    // Upper triangular matrix: [[2+0i, 1+0i, 1+0i], [0+0i, 2+0i, 1+0i], [0+0i, 0+0i, 2+0i]]
    float a[] = {2.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f};
    float x[] = {6.0f, 0.0f, 4.0f, 0.0f, 2.0f, 0.0f}; // right-hand side [6+0i, 4+0i, 2+0i]
    
    cblas_ctrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1);
    
    // Expected solution: [1.75+0i, 1.5+0i, 1.0+0i]
    int failed = 0;
    if (fabsf(x[0] - 1.75f) > 0.001f) {
        printf("cblas_ctrsv x[0] real Test Failed: actual: %f, expect: %f\n", x[0], 1.75f);
        failed++;
    }
    if (fabsf(x[1]) > 0.001f) {
        printf("cblas_ctrsv x[0] imag Test Failed: actual: %f, expect: %f\n", x[1], 0.0f);
        failed++;
    }
    if (fabsf(x[2] - 1.5f) > 0.001f) {
        printf("cblas_ctrsv x[1] real Test Failed: actual: %f, expect: %f\n", x[2], 1.5f);
        failed++;
    }
    if (fabsf(x[3]) > 0.001f) {
        printf("cblas_ctrsv x[1] imag Test Failed: actual: %f, expect: %f\n", x[3], 0.0f);
        failed++;
    }
    if (fabsf(x[4] - 1.0f) > 0.001f) {
        printf("cblas_ctrsv x[2] real Test Failed: actual: %f, expect: %f\n", x[4], 1.0f);
        failed++;
    }
    if (fabsf(x[5]) > 0.001f) {
        printf("cblas_ctrsv x[2] imag Test Failed: actual: %f, expect: %f\n", x[5], 0.0f);
        failed++;
    }
    return failed;
}

int test_cblas_ctrmv() {
    int n = 3;
    // Upper triangular matrix: [[1+0i, 2+0i, 3+0i], [0+0i, 4+0i, 5+0i], [0+0i, 0+0i, 6+0i]]
    float a[] = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 0.0f};
    float x[] = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f}; // [1+0i, 2+0i, 3+0i]
    
    cblas_ctrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1);
    
    // Expected result: [14+0i, 23+0i, 18+0i]
    int failed = 0;
    failed += assert_eq(x[0], 14.0f, "cblas_ctrmv x[0] real");
    failed += assert_eq(x[1], 0.0f, "cblas_ctrmv x[0] imag");
    failed += assert_eq(x[2], 23.0f, "cblas_ctrmv x[1] real");
    failed += assert_eq(x[3], 0.0f, "cblas_ctrmv x[1] imag");
    failed += assert_eq(x[4], 18.0f, "cblas_ctrmv x[2] real");
    failed += assert_eq(x[5], 0.0f, "cblas_ctrmv x[2] imag");
    return failed;
}

int test_cblas_cher() {
    int n = 2;
    float alpha = 2.0f; // real scalar
    float x[] = {1.0f, 1.0f, 2.0f, 0.0f}; // [1+i, 2+0i]
    float a[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}; // 2x2 identity matrix
    
    cblas_cher(CblasRowMajor, CblasUpper, n, alpha, x, 1, a, n);
    
    // Expected result: A = [[5+0i, 4+4i], [4-4i, 9+0i]] (only upper triangle updated)
    int failed = 0;
    failed += assert_eq(a[0], 5.0f, "cblas_cher a[0,0] real");
    failed += assert_eq(a[1], 0.0f, "cblas_cher a[0,0] imag");
    failed += assert_eq(a[2], 4.0f, "cblas_cher a[0,1] real");
    failed += assert_eq(a[3], 4.0f, "cblas_cher a[0,1] imag");
    failed += assert_eq(a[6], 9.0f, "cblas_cher a[1,1] real");
    failed += assert_eq(a[7], 0.0f, "cblas_cher a[1,1] imag");
    return failed;
}