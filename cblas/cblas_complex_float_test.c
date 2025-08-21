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
