#include "lapack_test.h"

static float float_buf[8] = {0.0f};
static float float_buf2[8] = {0.0f};
static double double_buf[8] = {0.0};
static double double_buf2[8] = {0.0};
static lapack_complex_float complex_float_buf[8] = {0.0f};
static lapack_complex_float complex_float_buf2[8] = {0.0f};
static lapack_complex_double complex_double_buf[8] = {0.0};
static lapack_complex_double complex_double_buf2[8] = {0.0};
static lapack_int int_buf[8] = {0};
static lapack_int int_buf2[8] = {0};
static lapack_logical logical_buf[8] = {0};

int test_lapacke_zgghd3() {
    lapack_int info = LAPACKE_zgghd3(
        LAPACK_COL_MAJOR,
        'N',
        'N',
        1,
        1,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1
    );
    return assert_eq_int((int)info, 0, "lapacke_zgghd3");
}

int test_lapacke_sgglse() {
    lapack_int info = LAPACKE_sgglse(
        LAPACK_COL_MAJOR,
        0,
        0,
        0,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf
    );
    return assert_eq_int((int)info, 0, "lapacke_sgglse");
}

int test_lapacke_dgglse() {
    lapack_int info = LAPACKE_dgglse(
        LAPACK_COL_MAJOR,
        0,
        0,
        0,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf
    );
    return assert_eq_int((int)info, 0, "lapacke_dgglse");
}

int test_lapacke_cgglse() {
    lapack_int info = LAPACKE_cgglse(
        LAPACK_COL_MAJOR,
        0,
        0,
        0,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)complex_float_buf
    );
    return assert_eq_int((int)info, 0, "lapacke_cgglse");
}

int test_lapacke_zgglse() {
    lapack_int info = LAPACKE_zgglse(
        LAPACK_COL_MAJOR,
        0,
        0,
        0,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)complex_double_buf
    );
    return assert_eq_int((int)info, 0, "lapacke_zgglse");
}

int test_lapacke_chgeqz() {
    lapack_int info = LAPACKE_chgeqz(
        LAPACK_COL_MAJOR,
        'E',
        'N',
        'N',
        1,
        1,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        0,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1
    );
    return assert_eq_int((int)info, -8, "lapacke_chgeqz");
}

int test_lapacke_zhgeqz() {
    lapack_int info = LAPACKE_zhgeqz(
        LAPACK_COL_MAJOR,
        'E',
        'N',
        'N',
        1,
        1,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        0,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1
    );
    return assert_eq_int((int)info, -8, "lapacke_zhgeqz");
}

int test_lapacke_shsein() {
    lapack_int info = LAPACKE_shsein(
        LAPACK_COL_MAJOR,
        'N',
        'Q',
        'N',
        logical_buf,
        0,
        (void*)float_buf,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        0,
        (void*)int_buf,
        (void*)int_buf2,
        (void*)int_buf2
    );
    return assert_eq_int((int)info, -2, "lapacke_shsein");
}

int test_lapacke_dhsein() {
    lapack_int info = LAPACKE_dhsein(
        LAPACK_COL_MAJOR,
        'N',
        'Q',
        'N',
        logical_buf,
        0,
        (void*)double_buf,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        0,
        (void*)int_buf,
        (void*)int_buf2,
        (void*)int_buf2
    );
    return assert_eq_int((int)info, -2, "lapacke_dhsein");
}

int test_lapacke_chsein() {
    lapack_int info = LAPACKE_chsein(
        LAPACK_COL_MAJOR,
        'N',
        'Q',
        'N',
        logical_buf,
        0,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1,
        0,
        (void*)int_buf,
        (void*)int_buf2,
        (void*)int_buf2
    );
    return assert_eq_int((int)info, -2, "lapacke_chsein");
}

int test_lapacke_zhsein() {
    lapack_int info = LAPACKE_zhsein(
        LAPACK_COL_MAJOR,
        'N',
        'Q',
        'N',
        logical_buf,
        0,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1,
        0,
        (void*)int_buf,
        (void*)int_buf2,
        (void*)int_buf2
    );
    return assert_eq_int((int)info, -2, "lapacke_zhsein");
}

int test_lapacke_chseqr() {
    lapack_int info = LAPACKE_chseqr(
        LAPACK_COL_MAJOR,
        'E',
        'N',
        1,
        1,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        1
    );
    return assert_eq_int((int)info, -7, "lapacke_chseqr");
}

int test_lapacke_zhseqr() {
    lapack_int info = LAPACKE_zhseqr(
        LAPACK_COL_MAJOR,
        'E',
        'N',
        1,
        1,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        1
    );
    return assert_eq_int((int)info, -7, "lapacke_zhseqr");
}

int test_lapacke_sorgtsqr_row() {
    lapack_int info = LAPACKE_sorgtsqr_row(
        LAPACK_COL_MAJOR,
        1,
        1,
        1,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1
    );
    return assert_eq_int((int)info, -4, "lapacke_sorgtsqr_row");
}
