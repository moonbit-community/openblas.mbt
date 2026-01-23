#include "lapack_test.h"

static float float_buf[1] = {0.0f};
static float float_stat_buf[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static double double_buf[1] = {0.0};
static lapack_complex_float complex_float_buf[1] = {0.0f};
static lapack_complex_double complex_double_buf[1] = {0.0};
static lapack_int int_buf[1] = {0};
static lapack_int int_stat_buf[3] = {0, 0, 0};
static char byte_buf[1] = {0};

static lapack_logical select_sgees(const float *wr, const float *wi) {
    (void)wr;
    (void)wi;
    return 0;
}

int test_lapack_make_complex_float() {
    lapack_complex_float v = lapack_make_complex_float(1.0f, 2.0f);
    int failed = 0;
    failed += assert_close(crealf(v), 1.0, 1e-6, "lapack_make_complex_float re");
    failed += assert_close(cimagf(v), 2.0, 1e-6, "lapack_make_complex_float im");
    return failed;
}

int test_lapack_make_complex_double() {
    lapack_complex_double v = lapack_make_complex_double(1.0, 2.0);
    int failed = 0;
    failed += assert_close(creal(v), 1.0, 1e-6, "lapack_make_complex_double re");
    failed += assert_close(cimag(v), 2.0, 1e-6, "lapack_make_complex_double im");
    return failed;
}

int test_lapacke_sbdsdc() {
    lapack_int info = LAPACKE_sbdsdc(LAPACK_COL_MAJOR, 'U', 'N', 0, (void*)float_buf, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_sbdsdc");
}

int test_lapacke_dbdsdc() {
    lapack_int info = LAPACKE_dbdsdc(LAPACK_COL_MAJOR, 'U', 'N', 0, (void*)double_buf, (void*)double_buf, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_dbdsdc");
}

int test_lapacke_sbdsqr() {
    lapack_int info = LAPACKE_sbdsqr(LAPACK_COL_MAJOR, 'U', 0, 0, 0, 0, (void*)float_buf, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sbdsqr");
}

int test_lapacke_dbdsqr() {
    lapack_int info = LAPACKE_dbdsqr(LAPACK_COL_MAJOR, 'U', 0, 0, 0, 0, (void*)double_buf, (void*)double_buf, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_dbdsqr");
}

int test_lapacke_cbdsqr() {
    lapack_int info = LAPACKE_cbdsqr(LAPACK_COL_MAJOR, 'U', 0, 0, 0, 0, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_cbdsqr");
}

int test_lapacke_zbdsqr() {
    lapack_int info = LAPACKE_zbdsqr(LAPACK_COL_MAJOR, 'U', 0, 0, 0, 0, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_zbdsqr");
}

int test_lapacke_sbdsvdx() {
    lapack_int info = LAPACKE_sbdsvdx(LAPACK_COL_MAJOR, 'U', 'N', 'A', 0, (void*)float_buf, (void*)float_buf, 0.0f, 0.0f, 1, 1, (void*)int_buf, (void*)float_buf, (void*)float_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_sbdsvdx");
}

int test_lapacke_dbdsvdx() {
    lapack_int info = LAPACKE_dbdsvdx(LAPACK_COL_MAJOR, 'U', 'N', 'A', 0, (void*)double_buf, (void*)double_buf, 0.0, 0.0, 1, 1, (void*)int_buf, (void*)double_buf, (void*)double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_dbdsvdx");
}

int test_lapacke_sdisna() {
    lapack_int info = LAPACKE_sdisna('E', 0, 0, (void*)float_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sdisna");
}

int test_lapacke_ddisna() {
    lapack_int info = LAPACKE_ddisna('E', 0, 0, (void*)double_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_ddisna");
}

int test_lapacke_sgbbrd() {
    lapack_int info = LAPACKE_sgbbrd(LAPACK_COL_MAJOR, 'N', 0, 0, 0, 0, 0, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sgbbrd");
}

int test_lapacke_dgbbrd() {
    lapack_int info = LAPACKE_dgbbrd(LAPACK_COL_MAJOR, 'N', 0, 0, 0, 0, 0, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_dgbbrd");
}

int test_lapacke_cgbbrd() {
    lapack_int info = LAPACKE_cgbbrd(LAPACK_COL_MAJOR, 'N', 0, 0, 0, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_cgbbrd");
}

int test_lapacke_zgbbrd() {
    lapack_int info = LAPACKE_zgbbrd(LAPACK_COL_MAJOR, 'N', 0, 0, 0, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_zgbbrd");
}

int test_lapacke_sgbcon() {
    lapack_int info = LAPACKE_sgbcon(LAPACK_COL_MAJOR, '1', 0, 0, 0, (void*)float_buf, 1, (void*)int_buf, 0.0f, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgbcon");
}

int test_lapacke_dgbcon() {
    lapack_int info = LAPACKE_dgbcon(LAPACK_COL_MAJOR, '1', 0, 0, 0, (void*)double_buf, 1, (void*)int_buf, 0.0, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgbcon");
}

int test_lapacke_cgbcon() {
    lapack_int info = LAPACKE_cgbcon(LAPACK_COL_MAJOR, '1', 0, 0, 0, (void*)complex_float_buf, 1, (void*)int_buf, 0.0f, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgbcon");
}

int test_lapacke_zgbcon() {
    lapack_int info = LAPACKE_zgbcon(LAPACK_COL_MAJOR, '1', 0, 0, 0, (void*)complex_double_buf, 1, (void*)int_buf, 0.0, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgbcon");
}

int test_lapacke_sgbequ() {
    lapack_int info = LAPACKE_sgbequ(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgbequ");
}

int test_lapacke_dgbequ() {
    lapack_int info = LAPACKE_dgbequ(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgbequ");
}

int test_lapacke_cgbequ() {
    lapack_int info = LAPACKE_cgbequ(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgbequ");
}

int test_lapacke_zgbequ() {
    lapack_int info = LAPACKE_zgbequ(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgbequ");
}

int test_lapacke_sgbequb() {
    lapack_int info = LAPACKE_sgbequb(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgbequb");
}

int test_lapacke_dgbequb() {
    lapack_int info = LAPACKE_dgbequb(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgbequb");
}

int test_lapacke_cgbequb() {
    lapack_int info = LAPACKE_cgbequb(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgbequb");
}

int test_lapacke_zgbequb() {
    lapack_int info = LAPACKE_zgbequb(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgbequb");
}

int test_lapacke_sgbrfs() {
    lapack_int info = LAPACKE_sgbrfs(LAPACK_COL_MAJOR, 'N', 0, 0, 0, 0, (void*)float_buf, 1, (void*)float_buf, 1, (void*)int_buf, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgbrfs");
}

int test_lapacke_dgbrfs() {
    lapack_int info = LAPACKE_dgbrfs(LAPACK_COL_MAJOR, 'N', 0, 0, 0, 0, (void*)double_buf, 1, (void*)double_buf, 1, (void*)int_buf, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgbrfs");
}

int test_lapacke_cgbrfs() {
    lapack_int info = LAPACKE_cgbrfs(LAPACK_COL_MAJOR, 'N', 0, 0, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)int_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgbrfs");
}

int test_lapacke_zgbrfs() {
    lapack_int info = LAPACKE_zgbrfs(LAPACK_COL_MAJOR, 'N', 0, 0, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)int_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgbrfs");
}

int test_lapacke_sgbsv() {
    lapack_int info = LAPACKE_sgbsv(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)float_buf, 1, (void*)int_buf, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sgbsv");
}

int test_lapacke_dgbsv() {
    lapack_int info = LAPACKE_dgbsv(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)double_buf, 1, (void*)int_buf, (void*)double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_dgbsv");
}

int test_lapacke_cgbsv() {
    lapack_int info = LAPACKE_cgbsv(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)complex_float_buf, 1, (void*)int_buf, (void*)complex_float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_cgbsv");
}

int test_lapacke_zgbsv() {
    lapack_int info = LAPACKE_zgbsv(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)complex_double_buf, 1, (void*)int_buf, (void*)complex_double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_zgbsv");
}

int test_lapacke_sgbsvx() {
    lapack_int info = LAPACKE_sgbsvx(LAPACK_COL_MAJOR, 'N', 'N', 0, 0, 0, 0, (void*)float_buf, 1, (void*)float_buf, 1, (void*)int_buf, (void*)byte_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgbsvx");
}

int test_lapacke_dgbsvx() {
    lapack_int info = LAPACKE_dgbsvx(LAPACK_COL_MAJOR, 'N', 'N', 0, 0, 0, 0, (void*)double_buf, 1, (void*)double_buf, 1, (void*)int_buf, (void*)byte_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgbsvx");
}

int test_lapacke_cgbsvx() {
    lapack_int info = LAPACKE_cgbsvx(LAPACK_COL_MAJOR, 'N', 'N', 0, 0, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)int_buf, (void*)byte_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgbsvx");
}

int test_lapacke_zgbsvx() {
    lapack_int info = LAPACKE_zgbsvx(LAPACK_COL_MAJOR, 'N', 'N', 0, 0, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)int_buf, (void*)byte_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgbsvx");
}

int test_lapacke_sgbtrf() {
    lapack_int info = LAPACKE_sgbtrf(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)float_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgbtrf");
}

int test_lapacke_dgbtrf() {
    lapack_int info = LAPACKE_dgbtrf(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgbtrf");
}

int test_lapacke_cgbtrf() {
    lapack_int info = LAPACKE_cgbtrf(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)complex_float_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgbtrf");
}

int test_lapacke_zgbtrf() {
    lapack_int info = LAPACKE_zgbtrf(LAPACK_COL_MAJOR, 0, 0, 0, 0, (void*)complex_double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgbtrf");
}

int test_lapacke_sgbtrs() {
    lapack_int info = LAPACKE_sgbtrs(LAPACK_COL_MAJOR, 'N', 0, 0, 0, 0, (void*)float_buf, 1, (void*)int_buf, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sgbtrs");
}

int test_lapacke_dgbtrs() {
    lapack_int info = LAPACKE_dgbtrs(LAPACK_COL_MAJOR, 'N', 0, 0, 0, 0, (void*)double_buf, 1, (void*)int_buf, (void*)double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_dgbtrs");
}

int test_lapacke_cgbtrs() {
    lapack_int info = LAPACKE_cgbtrs(LAPACK_COL_MAJOR, 'N', 0, 0, 0, 0, (void*)complex_float_buf, 1, (void*)int_buf, (void*)complex_float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_cgbtrs");
}

int test_lapacke_zgbtrs() {
    lapack_int info = LAPACKE_zgbtrs(LAPACK_COL_MAJOR, 'N', 0, 0, 0, 0, (void*)complex_double_buf, 1, (void*)int_buf, (void*)complex_double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_zgbtrs");
}

int test_lapacke_sgebak() {
    lapack_int info = LAPACKE_sgebak(LAPACK_COL_MAJOR, 'N', 'R', 1, 1, 1, (void*)float_buf, 1, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sgebak");
}

int test_lapacke_dgebak() {
    lapack_int info = LAPACKE_dgebak(LAPACK_COL_MAJOR, 'N', 'R', 1, 1, 1, (void*)double_buf, 1, (void*)double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_dgebak");
}

int test_lapacke_cgebak() {
    lapack_int info = LAPACKE_cgebak(LAPACK_COL_MAJOR, 'N', 'R', 1, 1, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_cgebak");
}

int test_lapacke_zgebak() {
    lapack_int info = LAPACKE_zgebak(LAPACK_COL_MAJOR, 'N', 'R', 1, 1, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_zgebak");
}

int test_lapacke_sgebal() {
    lapack_int info = LAPACKE_sgebal(LAPACK_COL_MAJOR, 'N', 0, (void*)float_buf, 1, (void*)int_buf, (void*)int_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgebal");
}

int test_lapacke_dgebal() {
    lapack_int info = LAPACKE_dgebal(LAPACK_COL_MAJOR, 'N', 0, (void*)double_buf, 1, (void*)int_buf, (void*)int_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgebal");
}

int test_lapacke_cgebal() {
    lapack_int info = LAPACKE_cgebal(LAPACK_COL_MAJOR, 'N', 0, (void*)complex_float_buf, 1, (void*)int_buf, (void*)int_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgebal");
}

int test_lapacke_zgebal() {
    lapack_int info = LAPACKE_zgebal(LAPACK_COL_MAJOR, 'N', 0, (void*)complex_double_buf, 1, (void*)int_buf, (void*)int_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgebal");
}

int test_lapacke_sgebrd() {
    lapack_int info = LAPACKE_sgebrd(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgebrd");
}

int test_lapacke_dgebrd() {
    lapack_int info = LAPACKE_dgebrd(LAPACK_COL_MAJOR, 0, 0, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgebrd");
}

int test_lapacke_cgebrd() {
    lapack_int info = LAPACKE_cgebrd(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgebrd");
}

int test_lapacke_zgebrd() {
    lapack_int info = LAPACKE_zgebrd(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgebrd");
}

int test_lapacke_sgecon() {
    lapack_int info = LAPACKE_sgecon(LAPACK_COL_MAJOR, '1', 0, (void*)float_buf, 1, 0.0f, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgecon");
}

int test_lapacke_dgecon() {
    lapack_int info = LAPACKE_dgecon(LAPACK_COL_MAJOR, '1', 0, (void*)double_buf, 1, 0.0, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgecon");
}

int test_lapacke_cgecon() {
    lapack_int info = LAPACKE_cgecon(LAPACK_COL_MAJOR, '1', 0, (void*)complex_float_buf, 1, 0.0f, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgecon");
}

int test_lapacke_zgecon() {
    lapack_int info = LAPACKE_zgecon(LAPACK_COL_MAJOR, '1', 0, (void*)complex_double_buf, 1, 0.0, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgecon");
}

int test_lapacke_sgeequ() {
    lapack_int info = LAPACKE_sgeequ(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgeequ");
}

int test_lapacke_dgeequ() {
    lapack_int info = LAPACKE_dgeequ(LAPACK_COL_MAJOR, 0, 0, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgeequ");
}

int test_lapacke_cgeequ() {
    lapack_int info = LAPACKE_cgeequ(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgeequ");
}

int test_lapacke_zgeequ() {
    lapack_int info = LAPACKE_zgeequ(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgeequ");
}

int test_lapacke_sgeequb() {
    lapack_int info = LAPACKE_sgeequb(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgeequb");
}

int test_lapacke_dgeequb() {
    lapack_int info = LAPACKE_dgeequb(LAPACK_COL_MAJOR, 0, 0, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgeequb");
}

int test_lapacke_cgeequb() {
    lapack_int info = LAPACKE_cgeequb(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgeequb");
}

int test_lapacke_zgeequb() {
    lapack_int info = LAPACKE_zgeequb(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgeequb");
}

int test_lapacke_sgees() {
    lapack_int info = LAPACKE_sgees(LAPACK_COL_MAJOR, 'N', 'S', select_sgees, 1, (void*)float_buf, 1, (void*)int_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sgees");
}

int test_lapacke_sgeev() {
    lapack_int info = LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'N', 0, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sgeev");
}

int test_lapacke_dgeev() {
    lapack_int info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'N', 0, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, (void*)double_buf, 1, (void*)double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_dgeev");
}

int test_lapacke_cgeev() {
    lapack_int info = LAPACKE_cgeev(LAPACK_COL_MAJOR, 'N', 'N', 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_cgeev");
}

int test_lapacke_zgeev() {
    lapack_int info = LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', 'N', 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_zgeev");
}

int test_lapacke_sgeevx() {
    lapack_int info = LAPACKE_sgeevx(LAPACK_COL_MAJOR, 'N', 'N', 'N', 'N', 0, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1, (void*)int_buf, (void*)int_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgeevx");
}

int test_lapacke_dgeevx() {
    lapack_int info = LAPACKE_dgeevx(LAPACK_COL_MAJOR, 'N', 'N', 'N', 'N', 0, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, (void*)double_buf, 1, (void*)double_buf, 1, (void*)int_buf, (void*)int_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgeevx");
}

int test_lapacke_cgeevx() {
    lapack_int info = LAPACKE_cgeevx(LAPACK_COL_MAJOR, 'N', 'N', 'N', 'N', 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)int_buf, (void*)int_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgeevx");
}

int test_lapacke_zgeevx() {
    lapack_int info = LAPACKE_zgeevx(LAPACK_COL_MAJOR, 'N', 'N', 'N', 'N', 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)int_buf, (void*)int_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgeevx");
}

int test_lapacke_sgehrd() {
    lapack_int info = LAPACKE_sgehrd(LAPACK_COL_MAJOR, 1, 1, 1, (void*)float_buf, 1, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgehrd");
}

int test_lapacke_dgehrd() {
    lapack_int info = LAPACKE_dgehrd(LAPACK_COL_MAJOR, 1, 1, 1, (void*)double_buf, 1, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgehrd");
}

int test_lapacke_cgehrd() {
    lapack_int info = LAPACKE_cgehrd(LAPACK_COL_MAJOR, 1, 1, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgehrd");
}

int test_lapacke_zgehrd() {
    lapack_int info = LAPACKE_zgehrd(LAPACK_COL_MAJOR, 1, 1, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgehrd");
}

int test_lapacke_sgejsv() {
    lapack_int info = LAPACKE_sgejsv(LAPACK_COL_MAJOR, 'A', 'N', 'N', 'N', 'N', 'N', 0, 0, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_stat_buf, (void*)int_stat_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgejsv");
}

int test_lapacke_dgejsv() {
    lapack_int info = LAPACKE_dgejsv(LAPACK_COL_MAJOR, 'A', 'N', 'N', 'N', 'N', 'N', 0, 0, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, (void*)int_stat_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgejsv");
}

int test_lapacke_cgejsv() {
    lapack_int info = LAPACKE_cgejsv(LAPACK_COL_MAJOR, 'A', 'N', 'N', 'N', 'N', 'N', 1, 1, (void*)complex_float_buf, 1, (void*)float_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)float_stat_buf, (void*)int_stat_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgejsv");
}

int test_lapacke_zgejsv() {
    lapack_int info = LAPACKE_zgejsv(LAPACK_COL_MAJOR, 'A', 'N', 'N', 'N', 'N', 'N', 1, 1, (void*)complex_double_buf, 1, (void*)double_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)int_stat_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgejsv");
}

int test_lapacke_sgelq2() {
    lapack_int info = LAPACKE_sgelq2(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgelq2");
}

int test_lapacke_dgelq2() {
    lapack_int info = LAPACKE_dgelq2(LAPACK_COL_MAJOR, 0, 0, (void*)double_buf, 1, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgelq2");
}

int test_lapacke_cgelq2() {
    lapack_int info = LAPACKE_cgelq2(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgelq2");
}

int test_lapacke_zgelq2() {
    lapack_int info = LAPACKE_zgelq2(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgelq2");
}

int test_lapacke_sgelqf() {
    lapack_int info = LAPACKE_sgelqf(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgelqf");
}

int test_lapacke_dgelqf() {
    lapack_int info = LAPACKE_dgelqf(LAPACK_COL_MAJOR, 0, 0, (void*)double_buf, 1, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgelqf");
}

int test_lapacke_cgelqf() {
    lapack_int info = LAPACKE_cgelqf(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgelqf");
}

int test_lapacke_zgelqf() {
    lapack_int info = LAPACKE_zgelqf(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgelqf");
}

int test_lapacke_sgels() {
    lapack_int info = LAPACKE_sgels(LAPACK_COL_MAJOR, 'N', 0, 0, 0, (void*)float_buf, 1, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sgels");
}

int test_lapacke_dgels() {
    lapack_int info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', 0, 0, 0, (void*)double_buf, 1, (void*)double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_dgels");
}

int test_lapacke_cgels() {
    lapack_int info = LAPACKE_cgels(LAPACK_COL_MAJOR, 'N', 0, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_cgels");
}

int test_lapacke_zgels() {
    lapack_int info = LAPACKE_zgels(LAPACK_COL_MAJOR, 'N', 0, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_zgels");
}

int test_lapacke_sgelsd() {
    lapack_int info = LAPACKE_sgelsd(LAPACK_COL_MAJOR, 0, 0, 0, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, 0.0f, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgelsd");
}

int test_lapacke_dgelsd() {
    lapack_int info = LAPACKE_dgelsd(LAPACK_COL_MAJOR, 0, 0, 0, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, 0.0, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgelsd");
}

int test_lapacke_cgelsd() {
    lapack_int info = LAPACKE_cgelsd(LAPACK_COL_MAJOR, 0, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)float_buf, 0.0f, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgelsd");
}

int test_lapacke_zgelsd() {
    lapack_int info = LAPACKE_zgelsd(LAPACK_COL_MAJOR, 0, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)double_buf, 0.0, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgelsd");
}

int test_lapacke_sgelss() {
    lapack_int info = LAPACKE_sgelss(LAPACK_COL_MAJOR, 0, 0, 0, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, 0.0f, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgelss");
}

int test_lapacke_dgelss() {
    lapack_int info = LAPACKE_dgelss(LAPACK_COL_MAJOR, 0, 0, 0, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, 0.0, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgelss");
}

int test_lapacke_cgelss() {
    lapack_int info = LAPACKE_cgelss(LAPACK_COL_MAJOR, 0, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)float_buf, 0.0f, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgelss");
}

int test_lapacke_zgelss() {
    lapack_int info = LAPACKE_zgelss(LAPACK_COL_MAJOR, 0, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)double_buf, 0.0, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgelss");
}

int test_lapacke_sgelsy() {
    lapack_int info = LAPACKE_sgelsy(LAPACK_COL_MAJOR, 0, 0, 0, (void*)float_buf, 1, (void*)float_buf, 1, (void*)int_buf, 0.0f, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgelsy");
}

int test_lapacke_dgelsy() {
    lapack_int info = LAPACKE_dgelsy(LAPACK_COL_MAJOR, 0, 0, 0, (void*)double_buf, 1, (void*)double_buf, 1, (void*)int_buf, 0.0, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgelsy");
}

int test_lapacke_cgelsy() {
    lapack_int info = LAPACKE_cgelsy(LAPACK_COL_MAJOR, 0, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)int_buf, 0.0f, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgelsy");
}

int test_lapacke_zgelsy() {
    lapack_int info = LAPACKE_zgelsy(LAPACK_COL_MAJOR, 0, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)int_buf, 0.0, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgelsy");
}

int test_lapacke_sgeqlf() {
    lapack_int info = LAPACKE_sgeqlf(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgeqlf");
}

int test_lapacke_dgeqlf() {
    lapack_int info = LAPACKE_dgeqlf(LAPACK_COL_MAJOR, 0, 0, (void*)double_buf, 1, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgeqlf");
}

int test_lapacke_cgeqlf() {
    lapack_int info = LAPACKE_cgeqlf(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgeqlf");
}

int test_lapacke_zgeqlf() {
    lapack_int info = LAPACKE_zgeqlf(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgeqlf");
}

int test_lapacke_sgeqp3() {
    lapack_int info = LAPACKE_sgeqp3(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)int_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgeqp3");
}

int test_lapacke_dgeqp3() {
    lapack_int info = LAPACKE_dgeqp3(LAPACK_COL_MAJOR, 0, 0, (void*)double_buf, 1, (void*)int_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgeqp3");
}

int test_lapacke_cgeqp3() {
    lapack_int info = LAPACKE_cgeqp3(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)int_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgeqp3");
}

int test_lapacke_zgeqp3() {
    lapack_int info = LAPACKE_zgeqp3(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)int_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgeqp3");
}

int test_lapacke_sgeqpf() {
    lapack_int info = LAPACKE_sgeqpf(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)int_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgeqpf");
}

int test_lapacke_dgeqpf() {
    lapack_int info = LAPACKE_dgeqpf(LAPACK_COL_MAJOR, 0, 0, (void*)double_buf, 1, (void*)int_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgeqpf");
}

int test_lapacke_cgeqpf() {
    lapack_int info = LAPACKE_cgeqpf(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)int_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgeqpf");
}

int test_lapacke_zgeqpf() {
    lapack_int info = LAPACKE_zgeqpf(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)int_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgeqpf");
}

int test_lapacke_sgeqr2() {
    lapack_int info = LAPACKE_sgeqr2(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgeqr2");
}

int test_lapacke_dgeqr2() {
    lapack_int info = LAPACKE_dgeqr2(LAPACK_COL_MAJOR, 0, 0, (void*)double_buf, 1, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgeqr2");
}

int test_lapacke_cgeqr2() {
    lapack_int info = LAPACKE_cgeqr2(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgeqr2");
}

int test_lapacke_zgeqr2() {
    lapack_int info = LAPACKE_zgeqr2(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgeqr2");
}

int test_lapacke_sgeqrf() {
    lapack_int info = LAPACKE_sgeqrf(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgeqrf");
}

int test_lapacke_cgeqrf() {
    lapack_int info = LAPACKE_cgeqrf(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgeqrf");
}

int test_lapacke_zgeqrf() {
    lapack_int info = LAPACKE_zgeqrf(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgeqrf");
}

int test_lapacke_sgeqrfp() {
    lapack_int info = LAPACKE_sgeqrfp(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgeqrfp");
}

int test_lapacke_dgeqrfp() {
    lapack_int info = LAPACKE_dgeqrfp(LAPACK_COL_MAJOR, 0, 0, (void*)double_buf, 1, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgeqrfp");
}

int test_lapacke_cgeqrfp() {
    lapack_int info = LAPACKE_cgeqrfp(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgeqrfp");
}

int test_lapacke_zgeqrfp() {
    lapack_int info = LAPACKE_zgeqrfp(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgeqrfp");
}

int test_lapacke_sgerfs() {
    lapack_int info = LAPACKE_sgerfs(LAPACK_COL_MAJOR, 'N', 0, 0, (void*)float_buf, 1, (void*)float_buf, 1, (void*)int_buf, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgerfs");
}

int test_lapacke_dgerfs() {
    lapack_int info = LAPACKE_dgerfs(LAPACK_COL_MAJOR, 'N', 0, 0, (void*)double_buf, 1, (void*)double_buf, 1, (void*)int_buf, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgerfs");
}

int test_lapacke_cgerfs() {
    lapack_int info = LAPACKE_cgerfs(LAPACK_COL_MAJOR, 'N', 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)int_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgerfs");
}

int test_lapacke_zgerfs() {
    lapack_int info = LAPACKE_zgerfs(LAPACK_COL_MAJOR, 'N', 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)int_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgerfs");
}

int test_lapacke_sgerqf() {
    lapack_int info = LAPACKE_sgerqf(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgerqf");
}

int test_lapacke_dgerqf() {
    lapack_int info = LAPACKE_dgerqf(LAPACK_COL_MAJOR, 0, 0, (void*)double_buf, 1, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgerqf");
}

int test_lapacke_cgerqf() {
    lapack_int info = LAPACKE_cgerqf(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgerqf");
}

int test_lapacke_zgerqf() {
    lapack_int info = LAPACKE_zgerqf(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgerqf");
}

int test_lapacke_sgesdd() {
    lapack_int info = LAPACKE_sgesdd(LAPACK_COL_MAJOR, 'N', 0, 0, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sgesdd");
}

int test_lapacke_dgesdd() {
    lapack_int info = LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'N', 0, 0, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, 1, (void*)double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_dgesdd");
}

int test_lapacke_cgesdd() {
    lapack_int info = LAPACKE_cgesdd(LAPACK_COL_MAJOR, 'N', 0, 0, (void*)complex_float_buf, 1, (void*)float_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_cgesdd");
}

int test_lapacke_zgesdd() {
    lapack_int info = LAPACKE_zgesdd(LAPACK_COL_MAJOR, 'N', 0, 0, (void*)complex_double_buf, 1, (void*)double_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_zgesdd");
}

int test_lapacke_sgesv() {
    lapack_int info = LAPACKE_sgesv(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)int_buf, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sgesv");
}

int test_lapacke_cgesv() {
    lapack_int info = LAPACKE_cgesv(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)int_buf, (void*)complex_float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_cgesv");
}

int test_lapacke_zgesv() {
    lapack_int info = LAPACKE_zgesv(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)int_buf, (void*)complex_double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_zgesv");
}

int test_lapacke_dsgesv() {
    lapack_int info = LAPACKE_dsgesv(LAPACK_COL_MAJOR, 0, 0, (void*)double_buf, 1, (void*)int_buf, (void*)double_buf, 1, (void*)double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_dsgesv");
}

int test_lapacke_zcgesv() {
    lapack_int info = LAPACKE_zcgesv(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)int_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_zcgesv");
}

int test_lapacke_sgesvd() {
    lapack_int info = LAPACKE_sgesvd(LAPACK_COL_MAJOR, 'N', 'N', 0, 0, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgesvd");
}

int test_lapacke_cgesvd() {
    lapack_int info = LAPACKE_cgesvd(LAPACK_COL_MAJOR, 'N', 'N', 0, 0, (void*)complex_float_buf, 1, (void*)float_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgesvd");
}

int test_lapacke_zgesvd() {
    lapack_int info = LAPACKE_zgesvd(LAPACK_COL_MAJOR, 'N', 'N', 0, 0, (void*)complex_double_buf, 1, (void*)double_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgesvd");
}

int test_lapacke_sgesvdx() {
    lapack_int info = LAPACKE_sgesvdx(LAPACK_COL_MAJOR, 'N', 'N', 'A', 0, 0, (void*)float_buf, 1, 0.0f, 0.0f, 1, 1, (void*)int_buf, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgesvdx");
}

int test_lapacke_dgesvdx() {
    lapack_int info = LAPACKE_dgesvdx(LAPACK_COL_MAJOR, 'N', 'N', 'A', 0, 0, (void*)double_buf, 1, 0.0, 0.0, 1, 1, (void*)int_buf, (void*)double_buf, (void*)double_buf, 1, (void*)double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgesvdx");
}

int test_lapacke_cgesvdx() {
    lapack_int info = LAPACKE_cgesvdx(LAPACK_COL_MAJOR, 'N', 'N', 'A', 0, 0, (void*)complex_float_buf, 1, 0.0f, 0.0f, 1, 1, (void*)int_buf, (void*)float_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgesvdx");
}

int test_lapacke_zgesvdx() {
    lapack_int info = LAPACKE_zgesvdx(LAPACK_COL_MAJOR, 'N', 'N', 'A', 0, 0, (void*)complex_double_buf, 1, 0.0, 0.0, 1, 1, (void*)int_buf, (void*)double_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgesvdx");
}

int test_lapacke_sgesvdq() {
    lapack_int info = LAPACKE_sgesvdq(LAPACK_COL_MAJOR, 'A', 'N', 'N', 'N', 'N', 0, 0, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgesvdq");
}

int test_lapacke_dgesvdq() {
    lapack_int info = LAPACKE_dgesvdq(LAPACK_COL_MAJOR, 'A', 'N', 'N', 'N', 'N', 0, 0, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, 1, (void*)double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgesvdq");
}

int test_lapacke_cgesvdq() {
    lapack_int info = LAPACKE_cgesvdq(LAPACK_COL_MAJOR, 'A', 'N', 'N', 'N', 'N', 0, 0, (void*)complex_float_buf, 1, (void*)float_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgesvdq");
}

int test_lapacke_zgesvdq() {
    lapack_int info = LAPACKE_zgesvdq(LAPACK_COL_MAJOR, 'A', 'N', 'N', 'N', 'N', 0, 0, (void*)complex_double_buf, 1, (void*)double_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, -10, "lapacke_zgesvdq");
}

int test_lapacke_sgesvj() {
    lapack_int info = LAPACKE_sgesvj(LAPACK_COL_MAJOR, 'G', 'N', 'N', 0, 0, (void*)float_buf, 1, (void*)float_buf, 0, (void*)float_buf, 1, (void*)float_stat_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgesvj");
}

int test_lapacke_dgesvj() {
    lapack_int info = LAPACKE_dgesvj(LAPACK_COL_MAJOR, 'G', 'N', 'N', 0, 0, (void*)double_buf, 1, (void*)double_buf, 0, (void*)double_buf, 1, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgesvj");
}

int test_lapacke_cgesvj() {
    lapack_int info = LAPACKE_cgesvj(LAPACK_COL_MAJOR, 'G', 'N', 'N', 1, 1, (void*)complex_float_buf, 1, (void*)float_buf, 1, (void*)complex_float_buf, 1, (void*)float_stat_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgesvj");
}

int test_lapacke_zgesvj() {
    lapack_int info = LAPACKE_zgesvj(LAPACK_COL_MAJOR, 'G', 'N', 'N', 1, 1, (void*)complex_double_buf, 1, (void*)double_buf, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgesvj");
}

int test_lapacke_sgesvx() {
    lapack_int info = LAPACKE_sgesvx(LAPACK_COL_MAJOR, 'N', 'N', 0, 0, (void*)float_buf, 1, (void*)float_buf, 1, (void*)int_buf, (void*)byte_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgesvx");
}

int test_lapacke_dgesvx() {
    lapack_int info = LAPACKE_dgesvx(LAPACK_COL_MAJOR, 'N', 'N', 0, 0, (void*)double_buf, 1, (void*)double_buf, 1, (void*)int_buf, (void*)byte_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgesvx");
}

int test_lapacke_cgesvx() {
    lapack_int info = LAPACKE_cgesvx(LAPACK_COL_MAJOR, 'N', 'N', 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)int_buf, (void*)byte_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgesvx");
}

int test_lapacke_zgesvx() {
    lapack_int info = LAPACKE_zgesvx(LAPACK_COL_MAJOR, 'N', 'N', 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)int_buf, (void*)byte_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgesvx");
}

int test_lapacke_sgetf2() {
    lapack_int info = LAPACKE_sgetf2(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgetf2");
}

int test_lapacke_dgetf2() {
    lapack_int info = LAPACKE_dgetf2(LAPACK_COL_MAJOR, 0, 0, (void*)double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgetf2");
}

int test_lapacke_cgetf2() {
    lapack_int info = LAPACKE_cgetf2(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgetf2");
}

int test_lapacke_zgetf2() {
    lapack_int info = LAPACKE_zgetf2(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgetf2");
}

int test_lapacke_sgetrf() {
    lapack_int info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgetrf");
}

int test_lapacke_cgetrf() {
    lapack_int info = LAPACKE_cgetrf(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgetrf");
}

int test_lapacke_zgetrf() {
    lapack_int info = LAPACKE_zgetrf(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgetrf");
}

int test_lapacke_sgetrf2() {
    lapack_int info = LAPACKE_sgetrf2(LAPACK_COL_MAJOR, 0, 0, (void*)float_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgetrf2");
}

int test_lapacke_dgetrf2() {
    lapack_int info = LAPACKE_dgetrf2(LAPACK_COL_MAJOR, 0, 0, (void*)double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgetrf2");
}

int test_lapacke_cgetrf2() {
    lapack_int info = LAPACKE_cgetrf2(LAPACK_COL_MAJOR, 0, 0, (void*)complex_float_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgetrf2");
}

int test_lapacke_zgetrf2() {
    lapack_int info = LAPACKE_zgetrf2(LAPACK_COL_MAJOR, 0, 0, (void*)complex_double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgetrf2");
}

int test_lapacke_sgetri() {
    lapack_int info = LAPACKE_sgetri(LAPACK_COL_MAJOR, 0, (void*)float_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_sgetri");
}

int test_lapacke_dgetri() {
    lapack_int info = LAPACKE_dgetri(LAPACK_COL_MAJOR, 0, (void*)double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_dgetri");
}

int test_lapacke_cgetri() {
    lapack_int info = LAPACKE_cgetri(LAPACK_COL_MAJOR, 0, (void*)complex_float_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_cgetri");
}

int test_lapacke_zgetri() {
    lapack_int info = LAPACKE_zgetri(LAPACK_COL_MAJOR, 0, (void*)complex_double_buf, 1, (void*)int_buf);
    return assert_eq_int((int)info, 0, "lapacke_zgetri");
}

int test_lapacke_sgetrs() {
    lapack_int info = LAPACKE_sgetrs(LAPACK_COL_MAJOR, 'N', 0, 0, (void*)float_buf, 1, (void*)int_buf, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sgetrs");
}

int test_lapacke_dgetrs() {
    lapack_int info = LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', 0, 0, (void*)double_buf, 1, (void*)int_buf, (void*)double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_dgetrs");
}

int test_lapacke_cgetrs() {
    lapack_int info = LAPACKE_cgetrs(LAPACK_COL_MAJOR, 'N', 0, 0, (void*)complex_float_buf, 1, (void*)int_buf, (void*)complex_float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_cgetrs");
}

int test_lapacke_zgetrs() {
    lapack_int info = LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', 0, 0, (void*)complex_double_buf, 1, (void*)int_buf, (void*)complex_double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_zgetrs");
}

int test_lapacke_sggbak() {
    lapack_int info = LAPACKE_sggbak(LAPACK_COL_MAJOR, 'N', 'R', 1, 1, 1, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sggbak");
}

int test_lapacke_dggbak() {
    lapack_int info = LAPACKE_dggbak(LAPACK_COL_MAJOR, 'N', 'R', 1, 1, 1, (void*)double_buf, (void*)double_buf, 1, (void*)double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_dggbak");
}

int test_lapacke_cggbak() {
    lapack_int info = LAPACKE_cggbak(LAPACK_COL_MAJOR, 'N', 'R', 1, 1, 1, (void*)complex_float_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_cggbak");
}

int test_lapacke_zggbak() {
    lapack_int info = LAPACKE_zggbak(LAPACK_COL_MAJOR, 'N', 'R', 1, 1, 1, (void*)complex_double_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_zggbak");
}

int test_lapacke_sggbal() {
    lapack_int info = LAPACKE_sggbal(LAPACK_COL_MAJOR, 'N', 0, (void*)float_buf, 1, (void*)float_buf, 1, (void*)int_buf, (void*)int_buf, (void*)float_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sggbal");
}

int test_lapacke_dggbal() {
    lapack_int info = LAPACKE_dggbal(LAPACK_COL_MAJOR, 'N', 0, (void*)double_buf, 1, (void*)double_buf, 1, (void*)int_buf, (void*)int_buf, (void*)double_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dggbal");
}

int test_lapacke_cggbal() {
    lapack_int info = LAPACKE_cggbal(LAPACK_COL_MAJOR, 'N', 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)int_buf, (void*)int_buf, (void*)complex_float_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cggbal");
}

int test_lapacke_zggbal() {
    lapack_int info = LAPACKE_zggbal(LAPACK_COL_MAJOR, 'N', 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)int_buf, (void*)int_buf, (void*)complex_double_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zggbal");
}

int test_lapacke_sggev() {
    lapack_int info = LAPACKE_sggev(LAPACK_COL_MAJOR, 'N', 'N', 0, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sggev");
}

int test_lapacke_dggev() {
    lapack_int info = LAPACKE_dggev(LAPACK_COL_MAJOR, 'N', 'N', 0, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf, 1, (void*)double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_dggev");
}

int test_lapacke_cggev() {
    lapack_int info = LAPACKE_cggev(LAPACK_COL_MAJOR, 'N', 'N', 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_cggev");
}

int test_lapacke_zggev() {
    lapack_int info = LAPACKE_zggev(LAPACK_COL_MAJOR, 'N', 'N', 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_zggev");
}

int test_lapacke_sggev3() {
    lapack_int info = LAPACKE_sggev3(LAPACK_COL_MAJOR, 'N', 'N', 0, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sggev3");
}

int test_lapacke_dggev3() {
    lapack_int info = LAPACKE_dggev3(LAPACK_COL_MAJOR, 'N', 'N', 0, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf, 1, (void*)double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_dggev3");
}

int test_lapacke_cggev3() {
    lapack_int info = LAPACKE_cggev3(LAPACK_COL_MAJOR, 'N', 'N', 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_cggev3");
}

int test_lapacke_zggev3() {
    lapack_int info = LAPACKE_zggev3(LAPACK_COL_MAJOR, 'N', 'N', 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_zggev3");
}

int test_lapacke_sggevx() {
    lapack_int info = LAPACKE_sggevx(LAPACK_COL_MAJOR, 'N', 'N', 'N', 'N', 0, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf, 1, (void*)float_buf, 1, (void*)int_buf, (void*)int_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sggevx");
}

int test_lapacke_dggevx() {
    lapack_int info = LAPACKE_dggevx(LAPACK_COL_MAJOR, 'N', 'N', 'N', 'N', 0, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf, 1, (void*)double_buf, 1, (void*)int_buf, (void*)int_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dggevx");
}

int test_lapacke_cggevx() {
    lapack_int info = LAPACKE_cggevx(LAPACK_COL_MAJOR, 'N', 'N', 'N', 'N', 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)int_buf, (void*)int_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cggevx");
}

int test_lapacke_zggevx() {
    lapack_int info = LAPACKE_zggevx(LAPACK_COL_MAJOR, 'N', 'N', 'N', 'N', 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)int_buf, (void*)int_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zggevx");
}

int test_lapacke_sggglm() {
    lapack_int info = LAPACKE_sggglm(LAPACK_COL_MAJOR, 0, 0, 0, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, (void*)float_buf, (void*)float_buf);
    return assert_eq_int((int)info, 0, "lapacke_sggglm");
}

int test_lapacke_dggglm() {
    lapack_int info = LAPACKE_dggglm(LAPACK_COL_MAJOR, 0, 0, 0, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, (void*)double_buf, (void*)double_buf);
    return assert_eq_int((int)info, 0, "lapacke_dggglm");
}

int test_lapacke_cggglm() {
    lapack_int info = LAPACKE_cggglm(LAPACK_COL_MAJOR, 0, 0, 0, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, (void*)complex_float_buf, (void*)complex_float_buf);
    return assert_eq_int((int)info, 0, "lapacke_cggglm");
}

int test_lapacke_zggglm() {
    lapack_int info = LAPACKE_zggglm(LAPACK_COL_MAJOR, 0, 0, 0, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf, (void*)complex_double_buf, (void*)complex_double_buf);
    return assert_eq_int((int)info, 0, "lapacke_zggglm");
}

int test_lapacke_sgghrd() {
    lapack_int info = LAPACKE_sgghrd(LAPACK_COL_MAJOR, 'N', 'N', 1, 1, 1, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sgghrd");
}

int test_lapacke_dgghrd() {
    lapack_int info = LAPACKE_dgghrd(LAPACK_COL_MAJOR, 'N', 'N', 1, 1, 1, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_dgghrd");
}

int test_lapacke_cgghrd() {
    lapack_int info = LAPACKE_cgghrd(LAPACK_COL_MAJOR, 'N', 'N', 1, 1, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_cgghrd");
}

int test_lapacke_zgghrd() {
    lapack_int info = LAPACKE_zgghrd(LAPACK_COL_MAJOR, 'N', 'N', 1, 1, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1, (void*)complex_double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_zgghrd");
}

int test_lapacke_sgghd3() {
    lapack_int info = LAPACKE_sgghd3(LAPACK_COL_MAJOR, 'N', 'N', 1, 1, 1, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, 1, (void*)float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_sgghd3");
}

int test_lapacke_dgghd3() {
    lapack_int info = LAPACKE_dgghd3(LAPACK_COL_MAJOR, 'N', 'N', 1, 1, 1, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, 1, (void*)double_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_dgghd3");
}

int test_lapacke_cgghd3() {
    lapack_int info = LAPACKE_cgghd3(LAPACK_COL_MAJOR, 'N', 'N', 1, 1, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1, (void*)complex_float_buf, 1);
    return assert_eq_int((int)info, 0, "lapacke_cgghd3");
}
