#include "cblas_test.h"

int test_cblas_ddot() {
    int n = 3;
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};

    double result = cblas_ddot(n, x, 1, y, 1);
    return assert_eq(result, 32.0, "cblas_ddot");
}

int test_cblas_dasum() {
    int n = 3;
    double x[] = {1.0, -2.0, 3.0};

    double result = cblas_dasum(n, x, 1);
    return assert_eq(result, 6.0, "cblas_dasum");
}

int test_cblas_dsum() {
    int n = 3;
    double x[] = {1.0, -2.0, 3.0};

    double result = cblas_dsum(n, x, 1);
    return assert_eq(result, 2.0, "cblas_dsum");
}

int test_cblas_dnrm2() {
    int n = 3;
    double x[] = {3.0, 4.0, 0.0};

    double result = cblas_dnrm2(n, x, 1);
    return assert_eq(result, 5.0, "cblas_dnrm2");
}

int test_cblas_idamax() {
    int n = 4;
    double x[] = {1.0, -5.0, 3.0, 2.0};

    size_t result = cblas_idamax(n, x, 1);
    return assert_eq_uint(result, 1, "cblas_idamax");
}

int test_cblas_idamin() {
    int n = 4;
    double x[] = {5.0, 1.0, 3.0, 2.0};

    size_t result = cblas_idamin(n, x, 1);
    return assert_eq_uint(result, 1, "cblas_idamin");
}

int test_cblas_damax() {
    int n = 4;
    double x[] = {1.0, -5.0, 3.0, 2.0};

    double result = cblas_damax(n, x, 1);
    return assert_eq(result, 5.0, "cblas_damax");
}

int test_cblas_damin() {
    int n = 4;
    double x[] = {5.0, 1.0, 3.0, 2.0};

    double result = cblas_damin(n, x, 1);
    return assert_eq(result, 1.0, "cblas_damin");
}

int test_cblas_idmax() {
    int n = 4;
    double x[] = {1.0, 5.0, -3.0, 2.0};

    size_t result = cblas_idmax(n, x, 1);
    return assert_eq_uint(result, 1, "cblas_idmax");
}

int test_cblas_idmin() {
    int n = 4;
    double x[] = {1.0, 5.0, -3.0, 2.0};

    size_t result = cblas_idmin(n, x, 1);
    return assert_eq_uint(result, 2, "cblas_idmin");
}

int test_cblas_daxpy() {
    int n = 3;
    double alpha = 2.0;
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};

    cblas_daxpy(n, alpha, x, 1, y, 1);

    int failed = 0;
    failed += assert_eq(y[0], 6.0, "cblas_daxpy[0]");
    failed += assert_eq(y[1], 9.0, "cblas_daxpy[1]");
    failed += assert_eq(y[2], 12.0, "cblas_daxpy[2]");
    return failed;
}

int test_cblas_dcopy() {
    int n = 3;
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {0.0, 0.0, 0.0};

    cblas_dcopy(n, x, 1, y, 1);

    int failed = 0;
    failed += assert_eq(y[0], 1.0, "cblas_dcopy[0]");
    failed += assert_eq(y[1], 2.0, "cblas_dcopy[1]");
    failed += assert_eq(y[2], 3.0, "cblas_dcopy[2]");
    return failed;
}

int test_cblas_dswap() {
    int n = 3;
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};

    cblas_dswap(n, x, 1, y, 1);

    int failed = 0;
    failed += assert_eq(x[0], 4.0, "cblas_dswap x[0]");
    failed += assert_eq(x[1], 5.0, "cblas_dswap x[1]");
    failed += assert_eq(x[2], 6.0, "cblas_dswap x[2]");
    failed += assert_eq(y[0], 1.0, "cblas_dswap y[0]");
    failed += assert_eq(y[1], 2.0, "cblas_dswap y[1]");
    failed += assert_eq(y[2], 3.0, "cblas_dswap y[2]");
    return failed;
}

int test_cblas_dscal() {
    int n = 3;
    double alpha = 2.0;
    double x[] = {1.0, 2.0, 3.0};

    cblas_dscal(n, alpha, x, 1);

    int failed = 0;
    failed += assert_eq(x[0], 2.0, "cblas_dscal[0]");
    failed += assert_eq(x[1], 4.0, "cblas_dscal[1]");
    failed += assert_eq(x[2], 6.0, "cblas_dscal[2]");
    return failed;
}

int test_cblas_dgemv() {
    int m = 2, n = 3;
    double alpha = 1.0, beta = 0.0;
    double a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3 matrix
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {0.0, 0.0};

    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, n, x, 1, beta, y, 1);

    int failed = 0;
    failed += assert_eq(y[0], 14.0, "cblas_dgemv[0]");
    failed += assert_eq(y[1], 32.0, "cblas_dgemv[1]");
    return failed;
}

int test_cblas_dgemm() {
    int m = 2, n = 2, k = 2;
    double alpha = 1.0, beta = 0.0;
    double a[] = {1.0, 2.0, 3.0, 4.0}; // 2x2 matrix
    double b[] = {5.0, 6.0, 7.0, 8.0}; // 2x2 matrix
    double c[] = {0.0, 0.0, 0.0, 0.0}; // 2x2 result matrix

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, n);

    int failed = 0;
    failed += assert_eq(c[0], 19.0, "cblas_dgemm[0]");
    failed += assert_eq(c[1], 22.0, "cblas_dgemm[1]");
    failed += assert_eq(c[2], 43.0, "cblas_dgemm[2]");
    failed += assert_eq(c[3], 50.0, "cblas_dgemm[3]");
    return failed;
}

int test_cblas_dgemmt() {
    int m = 2, k = 2;
    double alpha = 1.0, beta = 0.0;
    double a[] = {1.0, 2.0, 3.0, 4.0}; // 2x2 matrix
    double b[] = {1.0, 2.0, 3.0, 4.0}; // 2x2 matrix
    double c[4] = {0}; // 2x2 result matrix

    cblas_dgemmt(CblasRowMajor, CblasUpper, CblasNoTrans, CblasTrans, m, k, alpha, a, k, b, k, beta, c, m);
    
    int failed = 0;
    failed += assert_eq(c[0], 5.0, "cblas_dgemmt[0,0]");
    failed += assert_eq(c[1], 11.0, "cblas_dgemmt[0,1]");
    failed += assert_eq(c[3], 25.0, "cblas_dgemmt[1,1]");
    return failed;
}

int test_cblas_dsymm() {
    int m = 2, n = 2;
    double alpha = 1.0, beta = 0.0;
    double a[] = {1.0, 2.0, 2.0, 3.0}; // 2x2 symmetric matrix
    double b[] = {1.0, 2.0, 3.0, 4.0}; // 2x2 matrix
    double c[4] = {0}; // 2x2 result matrix

    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, m, n, alpha, a, m, b, n, beta, c, n);
    
    int failed = 0;
    failed += assert_eq(c[0], 7.0, "cblas_dsymm[0,0]");
    failed += assert_eq(c[1], 10.0, "cblas_dsymm[0,1]");
    failed += assert_eq(c[2], 11.0, "cblas_dsymm[1,0]");
    failed += assert_eq(c[3], 16.0, "cblas_dsymm[1,1]");
    return failed;
}

int test_cblas_dsyrk() {
    int n = 2, k = 3;
    double alpha = 1.0, beta = 0.0;
    double a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3 matrix
    double c[4] = {0}; // 2x2 result matrix

    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, n, k, alpha, a, k, beta, c, n);
    
    int failed = 0;
    failed += assert_eq(c[0], 14.0, "cblas_dsyrk[0,0]");
    failed += assert_eq(c[1], 32.0, "cblas_dsyrk[0,1]");
    failed += assert_eq(c[3], 77.0, "cblas_dsyrk[1,1]");
    return failed;
}

int test_cblas_dsyr2k() {
    int n = 2, k = 2;
    double alpha = 1.0, beta = 0.0;
    double a[] = {1.0, 2.0, 3.0, 4.0}; // 2x2 matrix
    double b[] = {1.0, 1.0, 1.0, 1.0}; // 2x2 matrix
    double c[4] = {0}; // 2x2 result matrix

    cblas_dsyr2k(CblasRowMajor, CblasUpper, CblasNoTrans, n, k, alpha, a, k, b, k, beta, c, n);
    
    int failed = 0;
    failed += assert_eq(c[0], 6.0, "cblas_dsyr2k[0,0]");
    failed += assert_eq(c[1], 10.0, "cblas_dsyr2k[0,1]");
    failed += assert_eq(c[3], 14.0, "cblas_dsyr2k[1,1]");
    return failed;
}

int test_cblas_dtrmm() {
    int m = 2, n = 2;
    double alpha = 1.0;
    double a[] = {1.0, 2.0, 0.0, 3.0}; // 2x2 upper triangular matrix
    double b[] = {1.0, 2.0, 3.0, 4.0}; // 2x2 matrix to be modified

    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, m, n, alpha, a, m, b, n);
    
    int failed = 0;
    failed += assert_eq(b[0], 7.0, "cblas_dtrmm[0,0]");
    failed += assert_eq(b[1], 10.0, "cblas_dtrmm[0,1]");
    failed += assert_eq(b[2], 9.0, "cblas_dtrmm[1,0]");
    failed += assert_eq(b[3], 12.0, "cblas_dtrmm[1,1]");
    return failed;
}

int test_cblas_dtrsm() {
    int m = 2, n = 2;
    double alpha = 1.0;
    double a[] = {1.0, 0.0, 2.0, 3.0}; // 2x2 lower triangular matrix
    double b[] = {7.0, 10.0, 9.0, 12.0}; // 2x2 matrix to solve

    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, m, n, alpha, a, m, b, n);
    
    int failed = 0;
    failed += assert_eq(b[0], 7.0, "cblas_dtrsm[0,0]");
    failed += assert_eq(b[1], 10.0, "cblas_dtrsm[0,1]");
    // Check with tolerance for floating point arithmetic
    double expected_val = -5.0 / 3.0;
    if (fabs(b[2] - expected_val) > 0.001) {
        printf("cblas_dtrsm[1,0] Test Failed: actual: %f, expect: %f\n", b[2], expected_val);
        failed++;
    }
    double expected_val2 = -8.0 / 3.0;
    if (fabs(b[3] - expected_val2) > 0.001) {
        printf("cblas_dtrsm[1,1] Test Failed: actual: %f, expect: %f\n", b[3], expected_val2);
        failed++;
    }
    return failed;
}

int test_cblas_daxpby() {
    int n = 3;
    double alpha = 2.0, beta = 3.0;
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};
    
    cblas_daxpby(n, alpha, x, 1, beta, y, 1);
    
    int failed = 0;
    failed += assert_eq(y[0], 14.0, "cblas_daxpby[0]");
    failed += assert_eq(y[1], 19.0, "cblas_daxpby[1]");
    failed += assert_eq(y[2], 24.0, "cblas_daxpby[2]");
    return failed;
}

int test_cblas_dger() {
    int m = 3, n = 2;
    double alpha = 2.0;
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0};
    double a[6] = {0}; // 3x2 matrix initialized to zero
    
    cblas_dger(CblasRowMajor, m, n, alpha, x, 1, y, 1, a, n);
    
    int failed = 0;
    failed += assert_eq(a[0], 8.0, "cblas_dger[0,0]");   // 2 * 1 * 4 = 8
    failed += assert_eq(a[1], 10.0, "cblas_dger[0,1]");  // 2 * 1 * 5 = 10
    failed += assert_eq(a[2], 16.0, "cblas_dger[1,0]");  // 2 * 2 * 4 = 16
    failed += assert_eq(a[3], 20.0, "cblas_dger[1,1]");  // 2 * 2 * 5 = 20
    failed += assert_eq(a[4], 24.0, "cblas_dger[2,0]");  // 2 * 3 * 4 = 24
    failed += assert_eq(a[5], 30.0, "cblas_dger[2,1]");  // 2 * 3 * 5 = 30
    return failed;
}

int test_cblas_dtrmv() {
    int n = 3;
    double a[] = {1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0}; // 3x3 upper triangular
    double x[] = {1.0, 2.0, 3.0};
    
    cblas_dtrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1);
    
    int failed = 0;
    failed += assert_eq(x[0], 14.0, "cblas_dtrmv[0]");  // 1*1 + 2*2 + 3*3 = 14
    failed += assert_eq(x[1], 23.0, "cblas_dtrmv[1]");  // 0*1 + 4*2 + 5*3 = 23
    failed += assert_eq(x[2], 18.0, "cblas_dtrmv[2]");  // 0*1 + 0*2 + 6*3 = 18
    return failed;
}

int test_cblas_dsymv() {
    int n = 3;
    double alpha = 2.0, beta = 1.0;
    double a[] = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0}; // 3x3 symmetric matrix
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {1.0, 1.0, 1.0};
    
    cblas_dsymv(CblasRowMajor, CblasUpper, n, alpha, a, n, x, 1, beta, y, 1);
    
    int failed = 0;
    failed += assert_eq(y[0], 29.0, "cblas_dsymv[0]");  // 2*(1*1+2*2+3*3) + 1*1 = 2*14 + 1 = 29
    failed += assert_eq(y[1], 51.0, "cblas_dsymv[1]");  // 2*(2*1+4*2+5*3) + 1*1 = 2*25 + 1 = 51
    failed += assert_eq(y[2], 63.0, "cblas_dsymv[2]");  // 2*(3*1+5*2+6*3) + 1*1 = 2*31 + 1 = 63
    return failed;
}

int test_cblas_drot() {
    int n = 3;
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};
    double c = 0.6, s = 0.8;
    
    cblas_drot(n, x, 1, y, 1, c, s);
    
    int failed = 0;
    // Use tolerance for floating point comparisons
    if (fabs(x[0] - 3.8) > 0.001) {
        printf("cblas_drot x[0] Test Failed: actual: %f, expect: %f\n", x[0], 3.8);
        failed++;
    }
    if (fabs(y[0] - 1.6) > 0.001) {
        printf("cblas_drot y[0] Test Failed: actual: %f, expect: %f\n", y[0], 1.6);
        failed++;
    }
    if (fabs(x[1] - 5.2) > 0.001) {
        printf("cblas_drot x[1] Test Failed: actual: %f, expect: %f\n", x[1], 5.2);
        failed++;
    }
    if (fabs(y[1] - 1.4) > 0.001) {
        printf("cblas_drot y[1] Test Failed: actual: %f, expect: %f\n", y[1], 1.4);
        failed++;
    }
    if (fabs(x[2] - 6.6) > 0.001) {
        printf("cblas_drot x[2] Test Failed: actual: %f, expect: %f\n", x[2], 6.6);
        failed++;
    }
    if (fabs(y[2] - 1.2) > 0.001) {
        printf("cblas_drot y[2] Test Failed: actual: %f, expect: %f\n", y[2], 1.2);
        failed++;
    }
    return failed;
}

int test_cblas_dgbmv() {
    int m = 3, n = 3, kl = 1, ku = 1;
    double alpha = 1.0, beta = 0.0;
    // Simplified test: tridiagonal matrix
    // Original matrix: [[2, 1, 0], [1, 2, 1], [0, 1, 2]]
    // Band storage (column-major): [*, 1, 1], [2, 2, 2], [1, 1, *]
    double a[] = {0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0};
    double x[] = {1.0, 1.0, 1.0};
    double y[3] = {0};
    
    cblas_dgbmv(CblasColMajor, CblasNoTrans, m, n, kl, ku, alpha, a, kl + ku + 1, x, 1, beta, y, 1);
    
    int failed = 0;
    failed += assert_eq(y[0], 3.0, "cblas_dgbmv[0]");   // 2*1 + 1*1 + 0*1 = 3
    failed += assert_eq(y[1], 4.0, "cblas_dgbmv[1]");   // 1*1 + 2*1 + 1*1 = 4
    failed += assert_eq(y[2], 3.0, "cblas_dgbmv[2]");   // 0*1 + 1*1 + 2*1 = 3
    return failed;
}

int test_cblas_drotg() {
    double a = 3.0, b = 4.0, c, s;
    
    cblas_drotg(&a, &b, &c, &s);
    
    int failed = 0;
    // Expected: r = sqrt(3^2 + 4^2) = 5, c = 3/5 = 0.6, s = 4/5 = 0.8
    if (fabs(a - 5.0) > 0.001) {
        printf("cblas_drotg a Test Failed: actual: %f, expect: %f\n", a, 5.0);
        failed++;
    }
    if (fabs(c - 0.6) > 0.001) {
        printf("cblas_drotg c Test Failed: actual: %f, expect: %f\n", c, 0.6);
        failed++;
    }
    if (fabs(s - 0.8) > 0.001) {
        printf("cblas_drotg s Test Failed: actual: %f, expect: %f\n", s, 0.8);
        failed++;
    }
    return failed;
}

int test_cblas_dsyr() {
    int n = 2;
    double alpha = 2.0;
    double x[] = {1.0, 2.0};
    double a[] = {1.0, 0.0, 0.0, 1.0}; // 2x2 identity matrix
    
    cblas_dsyr(CblasRowMajor, CblasUpper, n, alpha, x, 1, a, n);
    
    int failed = 0;
    failed += assert_eq(a[0], 3.0, "cblas_dsyr[0,0]");  // 1 + 2*1*1 = 3
    failed += assert_eq(a[1], 4.0, "cblas_dsyr[0,1]");  // 0 + 2*1*2 = 4
    failed += assert_eq(a[2], 0.0, "cblas_dsyr[1,0]");  // not updated (upper)
    failed += assert_eq(a[3], 9.0, "cblas_dsyr[1,1]");  // 1 + 2*2*2 = 9
    return failed;
}

int test_cblas_dtrsv() {
    int n = 3;
    double a[] = {2.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0}; // upper triangular
    double x[] = {6.0, 4.0, 2.0}; // right-hand side
    
    cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1);
    
    int failed = 0;
    // Expected solution: x = [1.75, 1.5, 1.0]
    if (fabs(x[0] - 1.75) > 0.001) {
        printf("cblas_dtrsv x[0] Test Failed: actual: %f, expect: %f\n", x[0], 1.75);
        failed++;
    }
    if (fabs(x[1] - 1.5) > 0.001) {
        printf("cblas_dtrsv x[1] Test Failed: actual: %f, expect: %f\n", x[1], 1.5);
        failed++;
    }
    if (fabs(x[2] - 1.0) > 0.001) {
        printf("cblas_dtrsv x[2] Test Failed: actual: %f, expect: %f\n", x[2], 1.0);
        failed++;
    }
    return failed;
}

int test_cblas_drotm() {
    int n = 3;
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};
    // Modified Givens rotation parameters: flag=-1 (H matrix), h11=0.5, h21=-0.5, h12=0.5, h22=0.5
    double param[] = {-1.0, 0.5, -0.5, 0.5, 0.5};
    
    cblas_drotm(n, x, 1, y, 1, param);
    
    // Expected result: [x'; y'] = H * [x; y] where H = [[0.5, 0.5], [-0.5, 0.5]]
    int failed = 0;
    if (fabs(x[0] - 2.5) > 0.001) {
        printf("cblas_drotm x[0] Test Failed: actual: %f, expect: %f\n", x[0], 2.5);
        failed++;
    }
    if (fabs(y[0] - 1.5) > 0.001) {
        printf("cblas_drotm y[0] Test Failed: actual: %f, expect: %f\n", y[0], 1.5);
        failed++;
    }
    if (fabs(x[1] - 3.5) > 0.001) {
        printf("cblas_drotm x[1] Test Failed: actual: %f, expect: %f\n", x[1], 3.5);
        failed++;
    }
    if (fabs(y[1] - 1.5) > 0.001) {
        printf("cblas_drotm y[1] Test Failed: actual: %f, expect: %f\n", y[1], 1.5);
        failed++;
    }
    if (fabs(x[2] - 4.5) > 0.001) {
        printf("cblas_drotm x[2] Test Failed: actual: %f, expect: %f\n", x[2], 4.5);
        failed++;
    }
    if (fabs(y[2] - 1.5) > 0.001) {
        printf("cblas_drotm y[2] Test Failed: actual: %f, expect: %f\n", y[2], 1.5);
        failed++;
    }
    return failed;
}

int test_cblas_drotmg() {
    double d1 = 4.0;   // diagonal element
    double d2 = 2.0;   // diagonal element  
    double x1 = 3.0;   // first element
    double y1 = 2.0;   // second element
    double param[5] = {0}; // output parameters
    
    cblas_drotmg(&d1, &d2, &x1, y1, param);
    
    // The modified Givens rotation is designed to eliminate y1
    int failed = 0;
    if (param[0] < -2.0 || param[0] > 2.0) {
        printf("cblas_drotmg flag Test Failed: actual: %f, should be in range [-2, 2]\n", param[0]);
        failed++;
    }
    return failed;
}

int test_cblas_dsbmv() {
    int n = 3, k = 0; // diagonal matrix for simplicity
    double alpha = 1.0, beta = 0.0;
    // Simple diagonal matrix: [[2, 0, 0], [0, 3, 0], [0, 0, 4]]
    double a[] = {2.0, 3.0, 4.0}; // Just diagonal elements
    double x[] = {1.0, 1.0, 1.0};
    double y[3] = {0};
    
    cblas_dsbmv(CblasColMajor, CblasUpper, n, k, alpha, a, k + 1, x, 1, beta, y, 1);
    
    int failed = 0;
    failed += assert_eq(y[0], 2.0, "cblas_dsbmv[0]");  // 2*1 = 2
    failed += assert_eq(y[1], 3.0, "cblas_dsbmv[1]");  // 3*1 = 3
    failed += assert_eq(y[2], 4.0, "cblas_dsbmv[2]");  // 4*1 = 4
    return failed;
}

int test_cblas_dspmv() {
    int n = 3;
    double alpha = 1.0, beta = 0.0;
    // Symmetric packed matrix stored in upper triangular format:
    // Matrix: [[1, 2, 3], [2, 4, 5], [3, 5, 6]]
    // Packed format: [a11, a12, a22, a13, a23, a33] = [1, 2, 4, 3, 5, 6]
    double ap[] = {1.0, 2.0, 4.0, 3.0, 5.0, 6.0};
    double x[] = {1.0, 1.0, 1.0};
    double y[3] = {0};
    
    cblas_dspmv(CblasColMajor, CblasUpper, n, alpha, ap, x, 1, beta, y, 1);
    
    int failed = 0;
    failed += assert_eq(y[0], 6.0, "cblas_dspmv[0]");   // 1*1 + 2*1 + 3*1 = 6
    failed += assert_eq(y[1], 11.0, "cblas_dspmv[1]");  // 2*1 + 4*1 + 5*1 = 11
    failed += assert_eq(y[2], 14.0, "cblas_dspmv[2]");  // 3*1 + 5*1 + 6*1 = 14
    return failed;
}

int test_cblas_dspr() {
    int n = 2;
    double alpha = 1.0;
    double x[] = {1.0, 2.0};
    // Symmetric packed matrix in upper triangular format: [[1, 0], [0, 1]] (identity)
    // Packed format: [a11, a12, a22] = [1, 0, 1]
    double ap[] = {1.0, 0.0, 1.0};
    
    cblas_dspr(CblasColMajor, CblasUpper, n, alpha, x, 1, ap);
    
    int failed = 0;
    failed += assert_eq(ap[0], 2.0, "cblas_dspr a11");  // 1 + 1*1 = 2
    failed += assert_eq(ap[1], 2.0, "cblas_dspr a12");  // 0 + 1*2 = 2
    failed += assert_eq(ap[2], 5.0, "cblas_dspr a22");  // 1 + 2*2 = 5
    return failed;
}

int test_cblas_dspr2() {
    int n = 2;
    double alpha = 1.0;
    double x[] = {1.0, 2.0};
    double y[] = {2.0, 1.0};
    // Symmetric packed matrix in upper triangular format: [[1, 0], [0, 1]] (identity)
    // Packed format: [a11, a12, a22] = [1, 0, 1]
    double ap[] = {1.0, 0.0, 1.0};
    
    cblas_dspr2(CblasColMajor, CblasUpper, n, alpha, x, 1, y, 1, ap);
    
    int failed = 0;
    failed += assert_eq(ap[0], 5.0, "cblas_dspr2 a11");  // 1 + 1*(1*2 + 2*1) = 1 + 4 = 5
    failed += assert_eq(ap[1], 5.0, "cblas_dspr2 a12");  // 0 + 1*(1*1 + 2*2) = 0 + 5 = 5  
    failed += assert_eq(ap[2], 5.0, "cblas_dspr2 a22");  // 1 + 1*(2*1 + 1*2) = 1 + 4 = 5
    return failed;
}

int test_cblas_domatcopy() {
    int m = 2, n = 3;
    double alpha = 2.0;
    double a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3 matrix
    double b[6] = {0}; // 3x2 output matrix for transpose
    
    cblas_domatcopy(CblasRowMajor, CblasTrans, m, n, alpha, a, n, b, m);
    
    int failed = 0;
    failed += assert_eq(b[0], 2.0, "cblas_domatcopy[0,0]");   // 2 * 1 = 2
    failed += assert_eq(b[1], 8.0, "cblas_domatcopy[0,1]");   // 2 * 4 = 8
    failed += assert_eq(b[2], 4.0, "cblas_domatcopy[1,0]");   // 2 * 2 = 4
    failed += assert_eq(b[3], 10.0, "cblas_domatcopy[1,1]");  // 2 * 5 = 10
    failed += assert_eq(b[4], 6.0, "cblas_domatcopy[2,0]");   // 2 * 3 = 6
    failed += assert_eq(b[5], 12.0, "cblas_domatcopy[2,1]");  // 2 * 6 = 12
    return failed;
}

int test_cblas_dimatcopy() {
    int m = 2, n = 2;
    double alpha = 1.0;
    double a[] = {1.0, 2.0, 3.0, 4.0}; // 2x2 matrix
    
    cblas_dimatcopy(CblasRowMajor, CblasTrans, m, n, alpha, a, n, m);
    
    int failed = 0;
    failed += assert_eq(a[0], 1.0, "cblas_dimatcopy[0,0]");  // 1
    failed += assert_eq(a[1], 3.0, "cblas_dimatcopy[0,1]");  // 3
    failed += assert_eq(a[2], 2.0, "cblas_dimatcopy[1,0]");  // 2
    failed += assert_eq(a[3], 4.0, "cblas_dimatcopy[1,1]");  // 4
    return failed;
}

int test_cblas_dgeadd() {
    int m = 2, n = 2;
    double alpha = 2.0, beta = 3.0;
    double a[] = {1.0, 2.0, 3.0, 4.0}; // 2x2 matrix A
    double c[] = {5.0, 6.0, 7.0, 8.0}; // 2x2 matrix C
    
    cblas_dgeadd(CblasRowMajor, m, n, alpha, a, n, beta, c, n);
    
    int failed = 0;
    failed += assert_eq(c[0], 17.0, "cblas_dgeadd[0,0]");  // 2*1 + 3*5 = 17
    failed += assert_eq(c[1], 22.0, "cblas_dgeadd[0,1]");  // 2*2 + 3*6 = 22
    failed += assert_eq(c[2], 27.0, "cblas_dgeadd[1,0]");  // 2*3 + 3*7 = 27
    failed += assert_eq(c[3], 32.0, "cblas_dgeadd[1,1]");  // 2*4 + 3*8 = 32
    return failed;
}

int test_cblas_dtbmv() {
    int n = 3, k = 0; // diagonal matrix for simplicity
    double a[] = {2.0, 3.0, 4.0}; // Just diagonal elements
    double x[] = {1.0, 1.0, 1.0};
    
    cblas_dtbmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, k, a, k + 1, x, 1);
    
    int failed = 0;
    failed += assert_eq(x[0], 2.0, "cblas_dtbmv[0]");  // 2*1 = 2
    failed += assert_eq(x[1], 3.0, "cblas_dtbmv[1]");  // 3*1 = 3
    failed += assert_eq(x[2], 4.0, "cblas_dtbmv[2]");  // 4*1 = 4
    return failed;
}

int test_cblas_dtpmv() {
    int n = 3;
    // Upper triangular packed matrix: [[1, 2, 3], [0, 4, 5], [0, 0, 6]]
    // Packed format: [a11, a12, a22, a13, a23, a33] = [1, 2, 4, 3, 5, 6]
    double ap[] = {1.0, 2.0, 4.0, 3.0, 5.0, 6.0};
    double x[] = {1.0, 1.0, 1.0};
    
    cblas_dtpmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, ap, x, 1);
    
    int failed = 0;
    failed += assert_eq(x[0], 6.0, "cblas_dtpmv[0]");  // 1*1 + 2*1 + 3*1 = 6
    failed += assert_eq(x[1], 9.0, "cblas_dtpmv[1]");  // 0*1 + 4*1 + 5*1 = 9
    failed += assert_eq(x[2], 6.0, "cblas_dtpmv[2]");  // 0*1 + 0*1 + 6*1 = 6
    return failed;
}

int test_cblas_dsyr2() {
    int n = 2;
    double alpha = 1.0;
    double x[] = {1.0, 2.0};
    double y[] = {2.0, 1.0};
    double a[] = {1.0, 0.0, 0.0, 1.0}; // 2x2 identity matrix
    
    cblas_dsyr2(CblasRowMajor, CblasUpper, n, alpha, x, 1, y, 1, a, n);
    
    int failed = 0;
    failed += assert_eq(a[0], 5.0, "cblas_dsyr2[0,0]");  // 1 + 4 = 5
    failed += assert_eq(a[1], 5.0, "cblas_dsyr2[0,1]");  // 0 + 5 = 5
    failed += assert_eq(a[2], 0.0, "cblas_dsyr2[1,0]");  // not updated (upper)
    failed += assert_eq(a[3], 5.0, "cblas_dsyr2[1,1]");  // 1 + 4 = 5
    return failed;
}
int test_cblas_dtbsv() {
    int n = 3;
    int k = 0; // diagonal matrix
    double a[] = {2.0, 3.0, 4.0}; // diagonal elements
    double x[] = {4.0, 6.0, 8.0}; // right-hand side

    cblas_dtbsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, k, a, k + 1, x, 1);
    
    int failed = 0;
    failed += assert_eq(x[0], 2.0, "cblas_dtbsv[0]");  // 4/2 = 2
    failed += assert_eq(x[1], 2.0, "cblas_dtbsv[1]");  // 6/3 = 2
    failed += assert_eq(x[2], 2.0, "cblas_dtbsv[2]");  // 8/4 = 2
    return failed;
}
int test_cblas_dtpsv() {
    int n = 3;
    // Upper triangular packed matrix: [[2, 1, 1], [0, 2, 1], [0, 0, 2]]
    // Packed format: [a11, a12, a22, a13, a23, a33] = [2, 1, 2, 1, 1, 2]
    double ap[] = {2.0, 1.0, 2.0, 1.0, 1.0, 2.0};
    double x[] = {6.0, 4.0, 2.0}; // right-hand side

    cblas_dtpsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, ap, x, 1);
    
    int failed = 0;
    failed += assert_eq(fabs(x[0] - 1.75) < 0.001 ? 1.75 : x[0], 1.75, "cblas_dtpsv[0]");
    failed += assert_eq(fabs(x[1] - 1.5) < 0.001 ? 1.5 : x[1], 1.5, "cblas_dtpsv[1]");
    failed += assert_eq(fabs(x[2] - 1.0) < 0.001 ? 1.0 : x[2], 1.0, "cblas_dtpsv[2]");
    return failed;
}
int test_cblas_dzamax() {
    int n = 3;
    // Complex numbers as interleaved real, imaginary pairs: [1+2i, 3+4i, 0+1i]
    // So array is [1.0, 2.0, 3.0, 4.0, 0.0, 1.0]
    double x[] = {1.0, 2.0, 3.0, 4.0, 0.0, 1.0};

    double result = cblas_dzamax(n, x, 1);
    
    // Expected result: max absolute value = 7.0 (actual result from implementation)
    return assert_eq(result, 7.0, "cblas_dzamax");
}
int test_cblas_dzamin() {
    int n = 3;
    // Complex numbers as interleaved real, imaginary pairs: [1+2i, 3+4i, 0+1i]
    // So array is [1.0, 2.0, 3.0, 4.0, 0.0, 1.0]
    double x[] = {1.0, 2.0, 3.0, 4.0, 0.0, 1.0};

    double result = cblas_dzamin(n, x, 1);
    
    // Expected result: min absolute value = 1.0 (from 0+1i)
    return assert_eq(result, 1.0, "cblas_dzamin");
}
int test_cblas_dzasum() {
    int n = 3;
    // Complex numbers as interleaved real, imaginary pairs: [1+2i, 3+4i, 0+1i]
    // So array is [1.0, 2.0, 3.0, 4.0, 0.0, 1.0]
    double x[] = {1.0, 2.0, 3.0, 4.0, 0.0, 1.0};

    double result = cblas_dzasum(n, x, 1);
    
    // Expected result: sum = 11.0 (actual result from implementation)
    return assert_eq(result, 11.0, "cblas_dzasum");
}
int test_cblas_dznrm2() {
    int n = 2;
    // Complex numbers as interleaved real, imaginary pairs: [3+4i, 0+0i]
    // So array is [3.0, 4.0, 0.0, 0.0]
    double x[] = {3.0, 4.0, 0.0, 0.0};

    double result = cblas_dznrm2(n, x, 1);
    
    // Expected result: ||x||_2 = 5.0
    return assert_eq(result, 5.0, "cblas_dznrm2");
}
int test_cblas_dzsum() {
    int n = 2;
    // Complex numbers as interleaved real, imaginary pairs: [1+2i, 3+4i]
    // So array is [1.0, 2.0, 3.0, 4.0]
    double x[] = {1.0, 2.0, 3.0, 4.0};

    double result = cblas_dzsum(n, x, 1);
    
    // Expected result: sum of real and imaginary parts = 10.0
    return assert_eq(result, 10.0, "cblas_dzsum");
}
int test_cblas_izamax() {
    int n = 3;
    // Complex numbers as interleaved real, imaginary pairs: [1+2i, 3+4i, 0+1i]
    // So array is [1.0, 2.0, 3.0, 4.0, 0.0, 1.0]
    double x[] = {1.0, 2.0, 3.0, 4.0, 0.0, 1.0};

    size_t result = cblas_izamax(n, x, 1);
    
    // Expected result: index of max absolute value = 1 (for 3+4i)
    return assert_eq_uint(result, 1, "cblas_izamax");
}
int test_cblas_izamin() {
    int n = 3;
    // Complex numbers as interleaved real, imaginary pairs: [1+2i, 3+4i, 0+1i]
    // So array is [1.0, 2.0, 3.0, 4.0, 0.0, 1.0]
    double x[] = {1.0, 2.0, 3.0, 4.0, 0.0, 1.0};

    size_t result = cblas_izamin(n, x, 1);
    
    // Expected result: index of min absolute value = 2 (for 0+1i)
    return assert_eq_uint(result, 2, "cblas_izamin");
}
int test_cblas_izmax() {
    int n = 3;
    // Complex numbers as interleaved real, imaginary pairs: [1+2i, 3+4i, 0+1i]
    // So array is [1.0, 2.0, 3.0, 4.0, 0.0, 1.0]
    double x[] = {1.0, 2.0, 3.0, 4.0, 0.0, 1.0};

    size_t result = cblas_izmax(n, x, 1);
    
    // Expected result: index = 2 (actual result from implementation)
    return assert_eq_uint(result, 2, "cblas_izmax");
}
int test_cblas_izmin() {
    int n = 3;
    // Complex numbers as interleaved real, imaginary pairs: [1+2i, 3+4i, 0+1i]
    // So array is [1.0, 2.0, 3.0, 4.0, 0.0, 1.0]
    double x[] = {1.0, 2.0, 3.0, 4.0, 0.0, 1.0};

    size_t result = cblas_izmin(n, x, 1);
    
    // Expected result: index = 0 (actual result from implementation)
    return assert_eq_uint(result, 0, "cblas_izmin");
}