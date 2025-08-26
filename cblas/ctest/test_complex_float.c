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
int test_cblas_cgemm() {
    // Test cblas_cgemm with a simple 2x2 matrix multiplication
    int m = 2, n = 2, k = 2;
    
    // Matrix A (2x2)
    openblas_complex_float a[] = {
        openblas_make_complex_float(1.0f, 2.0f), openblas_make_complex_float(3.0f, 4.0f),
        openblas_make_complex_float(2.0f, 1.0f), openblas_make_complex_float(4.0f, 3.0f)
    };
    
    // Matrix B (2x2)
    openblas_complex_float b[] = {
        openblas_make_complex_float(5.0f, 6.0f), openblas_make_complex_float(7.0f, 8.0f),
        openblas_make_complex_float(1.0f, 2.0f), openblas_make_complex_float(3.0f, 4.0f)
    };
    
    // Matrix C (2x2) initialized to zero
    openblas_complex_float c[4];
    for (int i = 0; i < 4; i++) {
        c[i] = openblas_make_complex_float(0.0f, 0.0f);
    }
    
    openblas_complex_float alpha = openblas_make_complex_float(1.0f, 0.0f);
    openblas_complex_float beta = openblas_make_complex_float(0.0f, 0.0f);
    
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, &alpha, a, k, b, n, &beta, c, n);
    
    int failed = 0;
    failed += assert_eq(openblas_complex_float_real(c[0]), -12.0f, "cblas_cgemm C[0,0] real");
    failed += assert_eq(openblas_complex_float_imag(c[0]), 26.0f, "cblas_cgemm C[0,0] imag");
    failed += assert_eq(openblas_complex_float_real(c[1]), -16.0f, "cblas_cgemm C[0,1] real");
    failed += assert_eq(openblas_complex_float_imag(c[1]), 46.0f, "cblas_cgemm C[0,1] imag");
    failed += assert_eq(openblas_complex_float_real(c[2]), 2.0f, "cblas_cgemm C[1,0] real");
    failed += assert_eq(openblas_complex_float_imag(c[2]), 28.0f, "cblas_cgemm C[1,0] imag");
    failed += assert_eq(openblas_complex_float_real(c[3]), 6.0f, "cblas_cgemm C[1,1] real");
    failed += assert_eq(openblas_complex_float_imag(c[3]), 48.0f, "cblas_cgemm C[1,1] imag");
    return failed;
}

int test_cblas_chemm() {
    // Test cblas_chemm with a simple Hermitian matrix multiplication
    int m = 2, n = 2;
    
    // Hermitian matrix A (2x2)
    openblas_complex_float a[] = {
        openblas_make_complex_float(2.0f, 0.0f), openblas_make_complex_float(1.0f, 2.0f),
        openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(3.0f, 0.0f)
    };
    
    // Matrix B (2x2)
    openblas_complex_float b[] = {
        openblas_make_complex_float(1.0f, 1.0f), openblas_make_complex_float(2.0f, 0.0f),
        openblas_make_complex_float(3.0f, 0.0f), openblas_make_complex_float(1.0f, -1.0f)
    };
    
    // Matrix C (2x2) initialized to zero
    openblas_complex_float c[4];
    for (int i = 0; i < 4; i++) {
        c[i] = openblas_make_complex_float(0.0f, 0.0f);
    }
    
    openblas_complex_float alpha = openblas_make_complex_float(1.0f, 0.0f);
    openblas_complex_float beta = openblas_make_complex_float(0.0f, 0.0f);
    
    cblas_chemm(CblasRowMajor, CblasLeft, CblasUpper,
                m, n, &alpha, a, m, b, n, &beta, c, n);
    
    int failed = 0;
    failed += assert_eq(openblas_complex_float_real(c[0]), 5.0f, "cblas_chemm C[0,0] real");
    failed += assert_eq(openblas_complex_float_imag(c[0]), 8.0f, "cblas_chemm C[0,0] imag");
    failed += assert_eq(openblas_complex_float_real(c[1]), 7.0f, "cblas_chemm C[0,1] real");
    failed += assert_eq(openblas_complex_float_imag(c[1]), 1.0f, "cblas_chemm C[0,1] imag");
    failed += assert_eq(openblas_complex_float_real(c[2]), 12.0f, "cblas_chemm C[1,0] real");
    failed += assert_eq(openblas_complex_float_imag(c[2]), -1.0f, "cblas_chemm C[1,0] imag");
    failed += assert_eq(openblas_complex_float_real(c[3]), 5.0f, "cblas_chemm C[1,1] real");
    failed += assert_eq(openblas_complex_float_imag(c[3]), -7.0f, "cblas_chemm C[1,1] imag");
    return failed;
}

int test_cblas_cherk() {
    // Test cblas_cherk with Hermitian rank-k update
    int n = 2, k = 2;
    
    // Matrix A (2x2)
    openblas_complex_float a[] = {
        openblas_make_complex_float(1.0f, 1.0f), openblas_make_complex_float(2.0f, 0.0f),
        openblas_make_complex_float(0.0f, 1.0f), openblas_make_complex_float(1.0f, 1.0f)
    };
    
    // Initial Hermitian matrix C
    openblas_complex_float c[] = {
        openblas_make_complex_float(1.0f, 0.0f), openblas_make_complex_float(1.0f, 1.0f),
        openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(2.0f, 0.0f)
    };
    
    float alpha = 1.0f;
    float beta = 1.0f;
    
    cblas_cherk(CblasRowMajor, CblasUpper, CblasNoTrans,
                n, k, alpha, a, k, beta, c, n);
    
    int failed = 0;
    failed += assert_eq(openblas_complex_float_real(c[0]), 7.0f, "cblas_cherk C[0,0] real");
    failed += assert_eq(openblas_complex_float_imag(c[0]), 0.0f, "cblas_cherk C[0,0] imag");
    failed += assert_eq(openblas_complex_float_real(c[1]), 4.0f, "cblas_cherk C[0,1] real");
    failed += assert_eq(openblas_complex_float_imag(c[1]), -2.0f, "cblas_cherk C[0,1] imag");
    failed += assert_eq(openblas_complex_float_real(c[3]), 5.0f, "cblas_cherk C[1,1] real");
    failed += assert_eq(openblas_complex_float_imag(c[3]), 0.0f, "cblas_cherk C[1,1] imag");
    return failed;
}

int test_cblas_cher2() {
    // Test cblas_cher2 with Hermitian rank-2 update
    int n = 3;
    
    // Vector x
    openblas_complex_float x[] = {
        openblas_make_complex_float(1.0f, 1.0f),
        openblas_make_complex_float(2.0f, 0.0f),
        openblas_make_complex_float(0.0f, 1.0f)
    };
    
    // Vector y
    openblas_complex_float y[] = {
        openblas_make_complex_float(1.0f, 0.0f),
        openblas_make_complex_float(1.0f, 1.0f),
        openblas_make_complex_float(2.0f, 0.0f)
    };
    
    // Initial Hermitian matrix A
    openblas_complex_float a[] = {
        openblas_make_complex_float(1.0f, 0.0f), openblas_make_complex_float(1.0f, 1.0f), openblas_make_complex_float(2.0f, 1.0f),
        openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(2.0f, 0.0f), openblas_make_complex_float(3.0f, 0.0f),
        openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(4.0f, 0.0f)
    };
    
    openblas_complex_float alpha = openblas_make_complex_float(0.5f, 0.5f);
    
    cblas_cher2(CblasRowMajor, CblasUpper, n, &alpha, x, 1, y, 1, a, n);
    
    int failed = 0;
    // Check that diagonal elements remain real
    failed += assert_eq(openblas_complex_float_imag(a[0]), 0.0f, "cblas_cher2 A[0,0] imag");
    failed += assert_eq(openblas_complex_float_imag(a[4]), 0.0f, "cblas_cher2 A[1,1] imag");
    failed += assert_eq(openblas_complex_float_imag(a[8]), 0.0f, "cblas_cher2 A[2,2] imag");
    return failed;
}
int test_cblas_csymm() {
    // Test cblas_csymm (complex single precision symmetric matrix multiplication)
    int m = 2, n = 2;
    openblas_complex_float alpha = openblas_make_complex_float(1.0f, 0.0f); // 1+0i
    openblas_complex_float beta = openblas_make_complex_float(0.0f, 0.0f); // 0+0i
    
    // Symmetric matrix A: [[1+i, 2+0i], [2+0i, 3+i]] (stored as upper triangular)
    openblas_complex_float a[] = {
        openblas_make_complex_float(1.0f, 1.0f), openblas_make_complex_float(2.0f, 0.0f),
        openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(3.0f, 1.0f)
    };
    
    // Matrix B: [[1+0i, 2+i], [3+0i, 4+i]]
    openblas_complex_float b[] = {
        openblas_make_complex_float(1.0f, 0.0f), openblas_make_complex_float(2.0f, 1.0f),
        openblas_make_complex_float(3.0f, 0.0f), openblas_make_complex_float(4.0f, 1.0f)
    };
    
    // Matrix C: 2x2 initialized to zero
    openblas_complex_float c[] = {
        openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(0.0f, 0.0f),
        openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(0.0f, 0.0f)
    };
    
    cblas_csymm(CblasRowMajor, CblasLeft, CblasUpper, m, n, &alpha, a, m, b, n, &beta, c, n);
    
    // Expected result: C = A * B where A is symmetric
    // A * B = [[1+i, 2+0i], [2+0i, 3+i]] * [[1+0i, 2+i], [3+0i, 4+i]]
    // C[0,0] = (1+i)*1 + 2*3 = 1+i + 6 = 7+i
    // C[0,1] = (1+i)*(2+i) + 2*(4+i) = 1+3i-1 + 8+2i = 8+5i
    // C[1,0] = 2*1 + (3+i)*3 = 2 + 9+3i = 11+3i
    // C[1,1] = 2*(2+i) + (3+i)*(4+i) = 4+2i + 11+7i-1 = 14+9i
    int failed = 0;
    failed += assert_eq(openblas_complex_float_real(c[0]), 7.0f, "cblas_csymm C[0,0] real");
    failed += assert_eq(openblas_complex_float_imag(c[0]), 1.0f, "cblas_csymm C[0,0] imag");
    failed += assert_eq(openblas_complex_float_real(c[1]), 9.0f, "cblas_csymm C[0,1] real");
    failed += assert_eq(openblas_complex_float_imag(c[1]), 5.0f, "cblas_csymm C[0,1] imag");
    failed += assert_eq(openblas_complex_float_real(c[2]), 11.0f, "cblas_csymm C[1,0] real");
    failed += assert_eq(openblas_complex_float_imag(c[2]), 3.0f, "cblas_csymm C[1,0] imag");
    failed += assert_eq(openblas_complex_float_real(c[3]), 15.0f, "cblas_csymm C[1,1] real");
    failed += assert_eq(openblas_complex_float_imag(c[3]), 9.0f, "cblas_csymm C[1,1] imag");
    return failed;
}
int test_cblas_chemv() {
    // Test cblas_chemv (complex single precision Hermitian matrix vector multiplication)
    int n = 2;
    openblas_complex_float alpha = openblas_make_complex_float(1.0f, 0.0f);
    openblas_complex_float beta = openblas_make_complex_float(0.0f, 0.0f);
    
    // Hermitian matrix A: [[2+0i, 1+i], [1-i, 3+0i]] (stored as upper triangular)
    openblas_complex_float a[] = {
        openblas_make_complex_float(2.0f, 0.0f), openblas_make_complex_float(1.0f, 1.0f),
        openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(3.0f, 0.0f)
    };
    
    // Vector x: [1+i, 2+0i]
    openblas_complex_float x[] = {
        openblas_make_complex_float(1.0f, 1.0f), openblas_make_complex_float(2.0f, 0.0f)
    };
    
    // Vector y: [0+0i, 0+0i] (result vector)
    openblas_complex_float y[] = {
        openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(0.0f, 0.0f)
    };
    
    cblas_chemv(CblasRowMajor, CblasUpper, n, &alpha, a, n, x, 1, &beta, y, 1);
    
    // Expected result: y = A * x where A is Hermitian
    // A * x = [[2+0i, 1+i], [1-i, 3+0i]] * [1+i, 2+0i]
    // y[0] = (2+0i)*(1+i) + (1+i)*2 = 2+2i + 2+2i = 4+4i
    // y[1] = (1-i)*(1+i) + (3+0i)*2 = 1+1 + 6 = 8+0i
    int failed = 0;
    failed += assert_eq(openblas_complex_float_real(y[0]), 4.0f, "cblas_chemv y[0] real");
    failed += assert_eq(openblas_complex_float_imag(y[0]), 4.0f, "cblas_chemv y[0] imag");
    failed += assert_eq(openblas_complex_float_real(y[1]), 8.0f, "cblas_chemv y[1] real");
    failed += assert_eq(openblas_complex_float_imag(y[1]), 0.0f, "cblas_chemv y[1] imag");
    return failed;
}
int test_cblas_chbmv() {
    // Test cblas_chbmv (complex single precision Hermitian band matrix vector multiplication)
    int n = 3, k = 1;
    openblas_complex_float alpha = openblas_make_complex_float(1.0f, 0.0f);
    openblas_complex_float beta = openblas_make_complex_float(0.0f, 0.0f);
    
    // Hermitian band matrix A stored in band format
    openblas_complex_float a[] = {
        openblas_make_complex_float(1.0f, 1.0f), openblas_make_complex_float(2.0f, 1.0f),
        openblas_make_complex_float(2.0f, 0.0f), openblas_make_complex_float(3.0f, 0.0f), openblas_make_complex_float(4.0f, 0.0f)
    };
    
    // Vector x: [1+0i, 2+0i, 1+i]
    openblas_complex_float x[] = {
        openblas_make_complex_float(1.0f, 0.0f), openblas_make_complex_float(2.0f, 0.0f), openblas_make_complex_float(1.0f, 1.0f)
    };
    
    // Vector y: [0+0i, 0+0i, 0+0i] (result vector)
    openblas_complex_float y[] = {
        openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(0.0f, 0.0f)
    };
    
    cblas_chbmv(CblasRowMajor, CblasUpper, n, k, &alpha, a, k + 1, x, 1, &beta, y, 1);
    
    // Just check that function executes and produces some result
    int failed = 0;
    float y0_real = openblas_complex_float_real(y[0]);
    float y0_imag = openblas_complex_float_imag(y[0]);
    if (y0_real == 0.0f && y0_imag == 0.0f && 
        openblas_complex_float_real(y[1]) == 0.0f && openblas_complex_float_imag(y[1]) == 0.0f) {
        printf("cblas_chbmv Test Failed: result vector is all zeros\n");
        failed++;
    }
    return failed;
}
// test_cblas_cgemm3m removed - function not available in current OpenBLAS version
int test_cblas_chpr() {
    // Test cblas_chpr (complex single precision Hermitian packed rank-1 update)
    int n = 2;
    float alpha = 2.0f; // real scalar
    
    // Vector x: [1+i, 2+0i]
    openblas_complex_float x[] = {
        openblas_make_complex_float(1.0f, 1.0f), openblas_make_complex_float(2.0f, 0.0f)
    };
    
    // Hermitian matrix A in packed format: [a00, a01, a11]
    openblas_complex_float a[] = {
        openblas_make_complex_float(1.0f, 0.0f), openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(1.0f, 0.0f)
    };
    
    cblas_chpr(CblasRowMajor, CblasUpper, n, alpha, x, 1, a);
    
    // Expected result: A := alpha*x*conj(x)^T + A
    // Packed: [5+0i, 4+4i, 9+0i]
    int failed = 0;
    failed += assert_eq(openblas_complex_float_real(a[0]), 5.0f, "cblas_chpr a[0,0] real");
    failed += assert_eq(openblas_complex_float_imag(a[0]), 0.0f, "cblas_chpr a[0,0] imag");
    failed += assert_eq(openblas_complex_float_real(a[1]), 4.0f, "cblas_chpr a[0,1] real");
    failed += assert_eq(openblas_complex_float_imag(a[1]), 4.0f, "cblas_chpr a[0,1] imag");
    failed += assert_eq(openblas_complex_float_real(a[2]), 9.0f, "cblas_chpr a[1,1] real");
    failed += assert_eq(openblas_complex_float_imag(a[2]), 0.0f, "cblas_chpr a[1,1] imag");
    return failed;
}
int test_cblas_ctrmm() {
    int order = CblasRowMajor;
    int side = CblasLeft;
    int uplo = CblasUpper;
    int transa = CblasNoTrans;
    int diag = CblasNonUnit;
    int m = 2, n = 2;
    openblas_complex_float alpha = openblas_make_complex_float(1.0f, 0.0f);
    
    // Triangular matrix A: 2x2 upper triangular
    openblas_complex_float a[] = {openblas_make_complex_float(1.0f, 1.0f), openblas_make_complex_float(2.0f, 0.0f),
                                  openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(1.0f, 0.0f)};
    
    // Matrix B: 2x2
    openblas_complex_float b[] = {openblas_make_complex_float(1.0f, 0.0f), openblas_make_complex_float(0.0f, 1.0f),
                                  openblas_make_complex_float(2.0f, 0.0f), openblas_make_complex_float(1.0f, 0.0f)};
    
    cblas_ctrmm(order, side, uplo, transa, diag, m, n, &alpha, a, m, b, n);
    
    // Just verify function executes without error
    return 0;
}

int test_cblas_ctrsm() {
    int order = CblasRowMajor;
    int side = CblasLeft;
    int uplo = CblasUpper;
    int transa = CblasNoTrans;
    int diag = CblasNonUnit;
    int m = 2, n = 2;
    openblas_complex_float alpha = openblas_make_complex_float(1.0f, 0.0f);
    
    // Triangular matrix A: 2x2 upper triangular
    openblas_complex_float a[] = {openblas_make_complex_float(2.0f, 0.0f), openblas_make_complex_float(1.0f, 0.0f),
                                  openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(2.0f, 0.0f)};
    
    // Matrix B: 2x2
    openblas_complex_float b[] = {openblas_make_complex_float(4.0f, 0.0f), openblas_make_complex_float(2.0f, 0.0f),
                                  openblas_make_complex_float(2.0f, 0.0f), openblas_make_complex_float(4.0f, 0.0f)};
    
    cblas_ctrsm(order, side, uplo, transa, diag, m, n, &alpha, a, m, b, n);
    
    // Just verify function executes without error
    return 0;
}

int test_cblas_cgemmt() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int transa = CblasNoTrans;
    int transb = CblasNoTrans;
    int n = 2, k = 2;
    openblas_complex_float alpha = openblas_make_complex_float(1.0f, 0.0f);
    openblas_complex_float beta = openblas_make_complex_float(0.0f, 0.0f);
    
    // Matrix A: 2x2
    openblas_complex_float a[] = {openblas_make_complex_float(1.0f, 1.0f), openblas_make_complex_float(2.0f, 0.0f),
                                  openblas_make_complex_float(0.0f, 1.0f), openblas_make_complex_float(1.0f, 0.0f)};
    
    // Matrix B: 2x2
    openblas_complex_float b[] = {openblas_make_complex_float(2.0f, 0.0f), openblas_make_complex_float(1.0f, 1.0f),
                                  openblas_make_complex_float(1.0f, 0.0f), openblas_make_complex_float(2.0f, 1.0f)};
    
    // Matrix C: 2x2 initialized to zero
    openblas_complex_float c[] = {openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(0.0f, 0.0f),
                                  openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(0.0f, 0.0f)};
    
    cblas_cgemmt(order, uplo, transa, transb, n, k, &alpha, a, k, b, k, &beta, c, n);
    
    // Just verify function executes without error
    return 0;
}

int test_cblas_csyrk() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int trans = CblasNoTrans;
    int n = 2, k = 2;
    openblas_complex_float alpha = openblas_make_complex_float(1.0f, 0.0f);
    openblas_complex_float beta = openblas_make_complex_float(0.0f, 0.0f);
    
    // Matrix A: 2x2
    openblas_complex_float a[] = {openblas_make_complex_float(1.0f, 1.0f), openblas_make_complex_float(2.0f, 0.0f),
                                  openblas_make_complex_float(0.0f, 1.0f), openblas_make_complex_float(1.0f, 0.0f)};
    
    // Matrix C: 2x2 initialized to zero
    openblas_complex_float c[] = {openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(0.0f, 0.0f),
                                  openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(0.0f, 0.0f)};
    
    cblas_csyrk(order, uplo, trans, n, k, &alpha, a, k, &beta, c, n);
    
    // Just verify function executes without error
    return 0;
}
int test_cblas_cgbmv() {
    int order = CblasRowMajor;
    int trans = CblasNoTrans;
    int m = 3, n = 3;
    int kl = 1, ku = 1; // bandwidth
    openblas_complex_float alpha = openblas_make_complex_float(1.0f, 0.0f);
    openblas_complex_float beta = openblas_make_complex_float(0.0f, 0.0f);
    
    // Band matrix A in band storage format
    openblas_complex_float a[] = {
        openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(1.0f, 1.0f), openblas_make_complex_float(2.0f, 0.0f),
        openblas_make_complex_float(3.0f, 1.0f), openblas_make_complex_float(4.0f, 0.0f), openblas_make_complex_float(1.0f, 0.0f),
        openblas_make_complex_float(2.0f, 0.0f), openblas_make_complex_float(5.0f, 0.0f), openblas_make_complex_float(0.0f, 0.0f)
    };
    
    // Vector x
    openblas_complex_float x[] = {
        openblas_make_complex_float(1.0f, 1.0f),
        openblas_make_complex_float(2.0f, 0.0f),
        openblas_make_complex_float(1.0f, 1.0f)
    };
    
    // Vector y initialized to zero
    openblas_complex_float y[] = {
        openblas_make_complex_float(0.0f, 0.0f),
        openblas_make_complex_float(0.0f, 0.0f),
        openblas_make_complex_float(0.0f, 0.0f)
    };
    
    cblas_cgbmv(order, trans, m, n, kl, ku, &alpha, a, kl + ku + 1, x, 1, &beta, y, 1);
    
    // Just verify function executes without error
    return 0;
}

int test_cblas_ctbmv() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int trans = CblasNoTrans;
    int diag = CblasNonUnit;
    int n = 3, k = 1;
    
    // Triangular band matrix A
    openblas_complex_float a[] = {
        openblas_make_complex_float(1.0f, 0.0f), openblas_make_complex_float(2.0f, 0.0f),
        openblas_make_complex_float(2.0f, 0.0f), openblas_make_complex_float(3.0f, 0.0f),
        openblas_make_complex_float(0.0f, 0.0f), openblas_make_complex_float(1.0f, 0.0f)
    };
    
    // Vector x
    openblas_complex_float x[] = {
        openblas_make_complex_float(1.0f, 0.0f),
        openblas_make_complex_float(1.0f, 1.0f),
        openblas_make_complex_float(2.0f, 0.0f)
    };
    
    cblas_ctbmv(order, uplo, trans, diag, n, k, a, k + 1, x, 1);
    
    // Just verify function executes without error
    return 0;
}

int test_cblas_chpmv() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int n = 2;
    openblas_complex_float alpha = openblas_make_complex_float(1.0f, 0.0f);
    openblas_complex_float beta = openblas_make_complex_float(0.0f, 0.0f);
    
    // Hermitian matrix A in packed format
    openblas_complex_float ap[] = {
        openblas_make_complex_float(2.0f, 0.0f), // a00 - must be real
        openblas_make_complex_float(1.0f, 1.0f), // a01
        openblas_make_complex_float(3.0f, 0.0f)  // a11 - must be real
    };
    
    // Vector x
    openblas_complex_float x[] = {
        openblas_make_complex_float(1.0f, 0.0f),
        openblas_make_complex_float(1.0f, 1.0f)
    };
    
    // Vector y initialized to zero
    openblas_complex_float y[] = {
        openblas_make_complex_float(0.0f, 0.0f),
        openblas_make_complex_float(0.0f, 0.0f)
    };
    
    cblas_chpmv(order, uplo, n, &alpha, ap, x, 1, &beta, y, 1);
    
    // Just verify function executes without error
    return 0;
}
int test_cblas_csyr2k() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int trans = CblasNoTrans;
    int n = 2, k = 2;
    openblas_complex_float alpha = openblas_make_complex_float(1.0, 0.0);
    openblas_complex_float beta = openblas_make_complex_float(0.0, 0.0);
    
    // Matrix A: 2x2
    openblas_complex_float a[] = {
        openblas_make_complex_float(1.0, 1.0), openblas_make_complex_float(2.0, 0.0),
        openblas_make_complex_float(0.0, 1.0), openblas_make_complex_float(1.0, 0.0)
    };
    
    // Matrix B: 2x2
    openblas_complex_float b[] = {
        openblas_make_complex_float(2.0, 0.0), openblas_make_complex_float(1.0, 1.0),
        openblas_make_complex_float(1.0, 0.0), openblas_make_complex_float(2.0, 1.0)
    };
    
    // Matrix C: 2x2 initialized to zero
    openblas_complex_float c[] = {
        openblas_make_complex_float(0.0, 0.0), openblas_make_complex_float(0.0, 0.0),
        openblas_make_complex_float(0.0, 0.0), openblas_make_complex_float(0.0, 0.0)
    };
    
    cblas_csyr2k(order, uplo, trans, n, k, &alpha, a, k, b, k, &beta, c, n);
    
    // Just verify function executes without error
    return 0;
}
int test_cblas_cher2k() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int trans = CblasNoTrans;
    int n = 2, k = 2;
    openblas_complex_float alpha = openblas_make_complex_float(1.0, 0.0);
    float beta = 0.0; // real scalar for Hermitian matrices
    
    // Matrix A: 2x2
    openblas_complex_float a[] = {
        openblas_make_complex_float(1.0, 1.0), openblas_make_complex_float(2.0, 0.0),
        openblas_make_complex_float(0.0, 1.0), openblas_make_complex_float(1.0, 0.0)
    };
    
    // Matrix B: 2x2
    openblas_complex_float b[] = {
        openblas_make_complex_float(2.0, 0.0), openblas_make_complex_float(1.0, 1.0),
        openblas_make_complex_float(1.0, 0.0), openblas_make_complex_float(2.0, 1.0)
    };
    
    // Matrix C: 2x2 initialized to zero (Hermitian)
    openblas_complex_float c[] = {
        openblas_make_complex_float(0.0, 0.0), openblas_make_complex_float(0.0, 0.0),
        openblas_make_complex_float(0.0, 0.0), openblas_make_complex_float(0.0, 0.0)
    };
    
    cblas_cher2k(order, uplo, trans, n, k, &alpha, a, k, b, k, beta, c, n);
    
    // For Hermitian matrices, diagonal elements should be real
    int failed = 0;
    failed += assert_eq(openblas_complex_float_imag(c[0]), 0.0, "cher2k C[0,0] imag should be 0");
    failed += assert_eq(openblas_complex_float_imag(c[3]), 0.0, "cher2k C[1,1] imag should be 0");
    return failed;
}
int test_cblas_ctbsv() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int trans = CblasNoTrans;
    int diag = CblasNonUnit;
    int n = 3, k = 1;
    
    // Triangular band matrix A: represents [[2, 1, 0], [0, 2, 1], [0, 0, 2]]
    openblas_complex_float a[] = {
        openblas_make_complex_float(1.0, 0.0), openblas_make_complex_float(2.0, 0.0), // [a01, a00]
        openblas_make_complex_float(1.0, 0.0), openblas_make_complex_float(2.0, 0.0), // [a12, a11]
        openblas_make_complex_float(0.0, 0.0), openblas_make_complex_float(2.0, 0.0)  // [0, a22]
    };
    
    // Right-hand side: [6, 4, 2]
    openblas_complex_float x[] = {
        openblas_make_complex_float(6.0, 0.0),
        openblas_make_complex_float(4.0, 0.0),
        openblas_make_complex_float(2.0, 0.0)
    };
    
    cblas_ctbsv(order, uplo, trans, diag, n, k, a, k + 1, x, 1);
    
    // Just verify function executes without error and modifies the vector
    int failed = 0;
    if (openblas_complex_float_real(x[0]) == 6.0 && 
        openblas_complex_float_real(x[1]) == 4.0 && 
        openblas_complex_float_real(x[2]) == 2.0) {
        printf("cblas_ctbsv Test Failed: vector was not modified\n");
        failed++;
    }
    return failed;
}
int test_cblas_ctpmv() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int trans = CblasNoTrans;
    int diag = CblasNonUnit;
    int n = 3;
    
    // Triangular matrix A in packed format (upper triangular)
    // For n=3: [a00, a01, a02, a11, a12, a22]
    openblas_complex_float ap[] = {
        openblas_make_complex_float(2.0, 0.0), // a00
        openblas_make_complex_float(1.0, 0.0), // a01
        openblas_make_complex_float(3.0, 0.0), // a02
        openblas_make_complex_float(4.0, 0.0), // a11
        openblas_make_complex_float(2.0, 0.0), // a12
        openblas_make_complex_float(1.0, 0.0)  // a22
    };
    
    // Vector x: [1+0i, 2+0i, 1+i]
    openblas_complex_float x[] = {
        openblas_make_complex_float(1.0, 0.0),
        openblas_make_complex_float(2.0, 0.0),
        openblas_make_complex_float(1.0, 1.0)
    };
    
    cblas_ctpmv(order, uplo, trans, diag, n, ap, x, 1);
    
    // Just verify function executes without error and modifies the vector
    int failed = 0;
    if (openblas_complex_float_real(x[0]) == 1.0 && 
        openblas_complex_float_real(x[1]) == 2.0) {
        printf("cblas_ctpmv Test Failed: vector was not modified\n");
        failed++;
    }
    return failed;
}
int test_cblas_ctpsv() {
    int order = CblasRowMajor;
    int uplo = CblasUpper;
    int trans = CblasNoTrans;
    int diag = CblasNonUnit;
    int n = 3;
    
    // Triangular matrix A in packed format (upper triangular)
    // For n=3: [a00, a01, a02, a11, a12, a22]
    // Represents the matrix: [[2, 1, 0], [0, 2, 1], [0, 0, 2]]
    openblas_complex_float ap[] = {
        openblas_make_complex_float(2.0, 0.0), // a00
        openblas_make_complex_float(1.0, 0.0), // a01
        openblas_make_complex_float(0.0, 0.0), // a02
        openblas_make_complex_float(2.0, 0.0), // a11
        openblas_make_complex_float(1.0, 0.0), // a12
        openblas_make_complex_float(2.0, 0.0)  // a22
    };
    
    // Right-hand side: [6, 4, 2]
    openblas_complex_float x[] = {
        openblas_make_complex_float(6.0, 0.0),
        openblas_make_complex_float(4.0, 0.0),
        openblas_make_complex_float(2.0, 0.0)
    };
    
    cblas_ctpsv(order, uplo, trans, diag, n, ap, x, 1);
    
    // Just verify function executes without error and modifies the vector
    int failed = 0;
    if (openblas_complex_float_real(x[0]) == 6.0 && 
        openblas_complex_float_real(x[1]) == 4.0 && 
        openblas_complex_float_real(x[2]) == 2.0) {
        printf("cblas_ctpsv Test Failed: vector was not modified\n");
        failed++;
    }
    return failed;
}
int test_cblas_chpr2() {
    int uplo = CblasUpper;
    int n = 2;
    openblas_complex_float alpha = openblas_make_complex_float(1.0, 0.0);
    
    // Vector x: [1+i, 2+0i]
    openblas_complex_float x[] = {
        openblas_make_complex_float(1.0, 1.0),
        openblas_make_complex_float(2.0, 0.0)
    };
    
    // Vector y: [2+0i, 1+i]
    openblas_complex_float y[] = {
        openblas_make_complex_float(2.0, 0.0),
        openblas_make_complex_float(1.0, 1.0)
    };
    
    // Hermitian matrix A in packed format: [a00, a01, a11]
    openblas_complex_float a[] = {
        openblas_make_complex_float(1.0, 0.0), // a00 - must be real
        openblas_make_complex_float(0.0, 0.0), // a01
        openblas_make_complex_float(1.0, 0.0)  // a11 - must be real
    };
    
    cblas_chpr2(CblasRowMajor, uplo, n, &alpha, x, 1, y, 1, a);
    
    // For Hermitian matrices, diagonal elements should be real
    int failed = 0;
    failed += assert_eq(openblas_complex_float_imag(a[0]), 0.0, "chpr2 a[0,0] imag should be 0");
    failed += assert_eq(openblas_complex_float_imag(a[2]), 0.0, "chpr2 a[1,1] imag should be 0");
    // Check that the diagonal elements changed
    if (openblas_complex_float_real(a[0]) == 1.0) {
        printf("cblas_chpr2 Test Failed: diagonal element not modified\n");
        failed++;
    }
    return failed;
}