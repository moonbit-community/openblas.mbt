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
static char byte_buf[8] = {0};

int test_lapacke3_smoke() {
    int failed = 0;
    lapack_int info_lapacke_dorgtsqr_row = LAPACKE_dorgtsqr_row(
        LAPACK_ROW_MAJOR,
        1,
        1,
        1,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dorgtsqr_row, 0, "lapacke_dorgtsqr_row");

    lapack_int info_lapacke_sormbr = LAPACKE_sormbr(
        LAPACK_COL_MAJOR,
        'Q',
        'L',
        'N',
        1,
        1,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sormbr, 0, "lapacke_sormbr");

    lapack_int info_lapacke_dormbr = LAPACKE_dormbr(
        LAPACK_COL_MAJOR,
        'Q',
        'L',
        'N',
        1,
        1,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dormbr, 0, "lapacke_dormbr");

    lapack_int info_lapacke_sormhr = LAPACKE_sormhr(
        LAPACK_COL_MAJOR,
        'L',
        'N',
        1,
        1,
        1,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sormhr, 0, "lapacke_sormhr");

    lapack_int info_lapacke_dormhr = LAPACKE_dormhr(
        LAPACK_COL_MAJOR,
        'L',
        'N',
        1,
        1,
        1,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dormhr, 0, "lapacke_dormhr");

    lapack_int info_lapacke_sormlq = LAPACKE_sormlq(
        LAPACK_COL_MAJOR,
        'L',
        'N',
        1,
        1,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sormlq, 0, "lapacke_sormlq");

    lapack_int info_lapacke_dormlq = LAPACKE_dormlq(
        LAPACK_COL_MAJOR,
        'L',
        'N',
        1,
        1,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dormlq, 0, "lapacke_dormlq");

    lapack_int info_lapacke_sormql = LAPACKE_sormql(
        LAPACK_COL_MAJOR,
        'L',
        'N',
        1,
        1,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sormql, 0, "lapacke_sormql");

    lapack_int info_lapacke_dormql = LAPACKE_dormql(
        LAPACK_COL_MAJOR,
        'L',
        'N',
        1,
        1,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dormql, 0, "lapacke_dormql");

    lapack_int info_lapacke_sormqr = LAPACKE_sormqr(
        LAPACK_COL_MAJOR,
        'L',
        'N',
        1,
        1,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sormqr, 0, "lapacke_sormqr");

    lapack_int info_lapacke_dormqr = LAPACKE_dormqr(
        LAPACK_COL_MAJOR,
        'L',
        'N',
        1,
        1,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dormqr, 0, "lapacke_dormqr");

    lapack_int info_lapacke_sormrq = LAPACKE_sormrq(
        LAPACK_COL_MAJOR,
        'L',
        'N',
        1,
        1,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sormrq, 0, "lapacke_sormrq");

    lapack_int info_lapacke_dormrq = LAPACKE_dormrq(
        LAPACK_COL_MAJOR,
        'L',
        'N',
        1,
        1,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dormrq, 0, "lapacke_dormrq");

    lapack_int info_lapacke_sormrz = LAPACKE_sormrz(
        LAPACK_COL_MAJOR,
        'L',
        'N',
        1,
        1,
        1,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sormrz, 0, "lapacke_sormrz");

    lapack_int info_lapacke_dormrz = LAPACKE_dormrz(
        LAPACK_COL_MAJOR,
        'L',
        'N',
        1,
        1,
        1,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dormrz, 0, "lapacke_dormrz");

    lapack_int info_lapacke_sormtr = LAPACKE_sormtr(
        LAPACK_COL_MAJOR,
        'L',
        'U',
        'N',
        1,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sormtr, 0, "lapacke_sormtr");

    lapack_int info_lapacke_dormtr = LAPACKE_dormtr(
        LAPACK_COL_MAJOR,
        'L',
        'U',
        'N',
        1,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dormtr, 0, "lapacke_dormtr");

    lapack_int info_lapacke_spbcon = LAPACKE_spbcon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)float_buf,
        1,
        1,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_spbcon, 0, "lapacke_spbcon");

    lapack_int info_lapacke_dpbcon = LAPACKE_dpbcon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)double_buf,
        1,
        1,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dpbcon, 0, "lapacke_dpbcon");

    lapack_int info_lapacke_cpbcon = LAPACKE_cpbcon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)complex_float_buf,
        1,
        1,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cpbcon, 0, "lapacke_cpbcon");

    lapack_int info_lapacke_zpbcon = LAPACKE_zpbcon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)complex_double_buf,
        1,
        1,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zpbcon, 0, "lapacke_zpbcon");

    lapack_int info_lapacke_spbequ = LAPACKE_spbequ(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_spbequ, 0, "lapacke_spbequ");

    lapack_int info_lapacke_dpbequ = LAPACKE_dpbequ(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dpbequ, 0, "lapacke_dpbequ");

    lapack_int info_lapacke_cpbequ = LAPACKE_cpbequ(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cpbequ, 0, "lapacke_cpbequ");

    lapack_int info_lapacke_zpbequ = LAPACKE_zpbequ(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zpbequ, 0, "lapacke_zpbequ");

    lapack_int info_lapacke_spbrfs = LAPACKE_spbrfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_spbrfs, 0, "lapacke_spbrfs");

    lapack_int info_lapacke_dpbrfs = LAPACKE_dpbrfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dpbrfs, 0, "lapacke_dpbrfs");

    lapack_int info_lapacke_cpbrfs = LAPACKE_cpbrfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cpbrfs, 0, "lapacke_cpbrfs");

    lapack_int info_lapacke_zpbrfs = LAPACKE_zpbrfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zpbrfs, 0, "lapacke_zpbrfs");

    lapack_int info_lapacke_spbstf = LAPACKE_spbstf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_spbstf, 0, "lapacke_spbstf");

    lapack_int info_lapacke_dpbstf = LAPACKE_dpbstf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dpbstf, 0, "lapacke_dpbstf");

    lapack_int info_lapacke_cpbstf = LAPACKE_cpbstf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)complex_float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cpbstf, 0, "lapacke_cpbstf");

    lapack_int info_lapacke_zpbstf = LAPACKE_zpbstf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)complex_double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zpbstf, 0, "lapacke_zpbstf");

    lapack_int info_lapacke_spbsv = LAPACKE_spbsv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_spbsv, 0, "lapacke_spbsv");

    lapack_int info_lapacke_dpbsv = LAPACKE_dpbsv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dpbsv, 0, "lapacke_dpbsv");

    lapack_int info_lapacke_cpbsv = LAPACKE_cpbsv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cpbsv, 0, "lapacke_cpbsv");

    lapack_int info_lapacke_zpbsv = LAPACKE_zpbsv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zpbsv, 0, "lapacke_zpbsv");

    lapack_int info_lapacke_spbsvx = LAPACKE_spbsvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        0,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)byte_buf,
        (void*)float_buf,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_spbsvx, 0, "lapacke_spbsvx");

    lapack_int info_lapacke_dpbsvx = LAPACKE_dpbsvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        0,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)byte_buf,
        (void*)double_buf,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dpbsvx, 0, "lapacke_dpbsvx");

    lapack_int info_lapacke_cpbsvx = LAPACKE_cpbsvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        0,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1,
        (void*)byte_buf,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cpbsvx, 0, "lapacke_cpbsvx");

    lapack_int info_lapacke_zpbsvx = LAPACKE_zpbsvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        0,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1,
        (void*)byte_buf,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zpbsvx, 0, "lapacke_zpbsvx");

    lapack_int info_lapacke_spbtrf = LAPACKE_spbtrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_spbtrf, 0, "lapacke_spbtrf");

    lapack_int info_lapacke_dpbtrf = LAPACKE_dpbtrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dpbtrf, 0, "lapacke_dpbtrf");

    lapack_int info_lapacke_cpbtrf = LAPACKE_cpbtrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)complex_float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cpbtrf, 0, "lapacke_cpbtrf");

    lapack_int info_lapacke_zpbtrf = LAPACKE_zpbtrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        (void*)complex_double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zpbtrf, 0, "lapacke_zpbtrf");

    lapack_int info_lapacke_spbtrs = LAPACKE_spbtrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_spbtrs, 0, "lapacke_spbtrs");

    lapack_int info_lapacke_dpbtrs = LAPACKE_dpbtrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dpbtrs, 0, "lapacke_dpbtrs");

    lapack_int info_lapacke_cpbtrs = LAPACKE_cpbtrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cpbtrs, 0, "lapacke_cpbtrs");

    lapack_int info_lapacke_zpbtrs = LAPACKE_zpbtrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        0,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zpbtrs, 0, "lapacke_zpbtrs");

    lapack_int info_lapacke_spftrf = LAPACKE_spftrf(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        (void*)float_buf
    );
    failed += assert_eq_int((int)info_lapacke_spftrf, 0, "lapacke_spftrf");

    lapack_int info_lapacke_dpftrf = LAPACKE_dpftrf(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        (void*)double_buf
    );
    failed += assert_eq_int((int)info_lapacke_dpftrf, 0, "lapacke_dpftrf");

    lapack_int info_lapacke_cpftrf = LAPACKE_cpftrf(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        (void*)complex_float_buf
    );
    failed += assert_eq_int((int)info_lapacke_cpftrf, 0, "lapacke_cpftrf");

    lapack_int info_lapacke_zpftrf = LAPACKE_zpftrf(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        (void*)complex_double_buf
    );
    failed += assert_eq_int((int)info_lapacke_zpftrf, 0, "lapacke_zpftrf");

    lapack_int info_lapacke_spftri = LAPACKE_spftri(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        (void*)float_buf
    );
    failed += assert_eq_int((int)info_lapacke_spftri, 0, "lapacke_spftri");

    lapack_int info_lapacke_dpftri = LAPACKE_dpftri(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        (void*)double_buf
    );
    failed += assert_eq_int((int)info_lapacke_dpftri, 0, "lapacke_dpftri");

    lapack_int info_lapacke_cpftri = LAPACKE_cpftri(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        (void*)complex_float_buf
    );
    failed += assert_eq_int((int)info_lapacke_cpftri, 0, "lapacke_cpftri");

    lapack_int info_lapacke_zpftri = LAPACKE_zpftri(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        (void*)complex_double_buf
    );
    failed += assert_eq_int((int)info_lapacke_zpftri, 0, "lapacke_zpftri");

    lapack_int info_lapacke_spftrs = LAPACKE_spftrs(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_spftrs, 0, "lapacke_spftrs");

    lapack_int info_lapacke_dpftrs = LAPACKE_dpftrs(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dpftrs, 0, "lapacke_dpftrs");

    lapack_int info_lapacke_cpftrs = LAPACKE_cpftrs(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cpftrs, 0, "lapacke_cpftrs");

    lapack_int info_lapacke_zpftrs = LAPACKE_zpftrs(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zpftrs, 0, "lapacke_zpftrs");

    lapack_int info_lapacke_spocon = LAPACKE_spocon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)float_buf,
        1,
        1,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_spocon, 0, "lapacke_spocon");

    lapack_int info_lapacke_dpocon = LAPACKE_dpocon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)double_buf,
        1,
        1,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dpocon, 0, "lapacke_dpocon");

    lapack_int info_lapacke_cpocon = LAPACKE_cpocon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_float_buf,
        1,
        1,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cpocon, 0, "lapacke_cpocon");

    lapack_int info_lapacke_zpocon = LAPACKE_zpocon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_double_buf,
        1,
        1,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zpocon, 0, "lapacke_zpocon");

    lapack_int info_lapacke_spoequ = LAPACKE_spoequ(
        LAPACK_COL_MAJOR,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_spoequ, 0, "lapacke_spoequ");

    lapack_int info_lapacke_dpoequ = LAPACKE_dpoequ(
        LAPACK_COL_MAJOR,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dpoequ, 0, "lapacke_dpoequ");

    lapack_int info_lapacke_cpoequ = LAPACKE_cpoequ(
        LAPACK_COL_MAJOR,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cpoequ, 0, "lapacke_cpoequ");

    lapack_int info_lapacke_zpoequ = LAPACKE_zpoequ(
        LAPACK_COL_MAJOR,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zpoequ, 0, "lapacke_zpoequ");

    lapack_int info_lapacke_spoequb = LAPACKE_spoequb(
        LAPACK_COL_MAJOR,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_spoequb, 0, "lapacke_spoequb");

    lapack_int info_lapacke_dpoequb = LAPACKE_dpoequb(
        LAPACK_COL_MAJOR,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dpoequb, 0, "lapacke_dpoequb");

    lapack_int info_lapacke_cpoequb = LAPACKE_cpoequb(
        LAPACK_COL_MAJOR,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cpoequb, 0, "lapacke_cpoequb");

    lapack_int info_lapacke_zpoequb = LAPACKE_zpoequb(
        LAPACK_COL_MAJOR,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zpoequb, 0, "lapacke_zpoequb");

    lapack_int info_lapacke_sporfs = LAPACKE_sporfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_sporfs, 0, "lapacke_sporfs");

    lapack_int info_lapacke_dporfs = LAPACKE_dporfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dporfs, 0, "lapacke_dporfs");

    lapack_int info_lapacke_cporfs = LAPACKE_cporfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cporfs, 0, "lapacke_cporfs");

    lapack_int info_lapacke_zporfs = LAPACKE_zporfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zporfs, 0, "lapacke_zporfs");

    lapack_int info_lapacke_sposv = LAPACKE_sposv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sposv, 0, "lapacke_sposv");

    lapack_int info_lapacke_dposv = LAPACKE_dposv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dposv, 0, "lapacke_dposv");

    lapack_int info_lapacke_cposv = LAPACKE_cposv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cposv, 0, "lapacke_cposv");

    lapack_int info_lapacke_zposv = LAPACKE_zposv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zposv, 0, "lapacke_zposv");

    lapack_int info_lapacke_dsposv = LAPACKE_dsposv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        1,
        (void*)int_buf
    );
    failed += assert_eq_int((int)info_lapacke_dsposv, 0, "lapacke_dsposv");

    lapack_int info_lapacke_zcposv = LAPACKE_zcposv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        1,
        (void*)int_buf
    );
    failed += assert_eq_int((int)info_lapacke_zcposv, 0, "lapacke_zcposv");

    lapack_int info_lapacke_sposvx = LAPACKE_sposvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)byte_buf,
        (void*)float_buf,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_sposvx, 0, "lapacke_sposvx");

    lapack_int info_lapacke_dposvx = LAPACKE_dposvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)byte_buf,
        (void*)double_buf,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dposvx, 0, "lapacke_dposvx");

    lapack_int info_lapacke_cposvx = LAPACKE_cposvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1,
        (void*)byte_buf,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cposvx, 0, "lapacke_cposvx");

    lapack_int info_lapacke_zposvx = LAPACKE_zposvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1,
        (void*)byte_buf,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zposvx, 0, "lapacke_zposvx");

    lapack_int info_lapacke_spotrf2 = LAPACKE_spotrf2(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_spotrf2, 0, "lapacke_spotrf2");

    lapack_int info_lapacke_dpotrf2 = LAPACKE_dpotrf2(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dpotrf2, 0, "lapacke_dpotrf2");

    lapack_int info_lapacke_cpotrf2 = LAPACKE_cpotrf2(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cpotrf2, 0, "lapacke_cpotrf2");

    lapack_int info_lapacke_zpotrf2 = LAPACKE_zpotrf2(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zpotrf2, 0, "lapacke_zpotrf2");

    lapack_int info_lapacke_spotrf = LAPACKE_spotrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_spotrf, 0, "lapacke_spotrf");

    lapack_int info_lapacke_dpotrf = LAPACKE_dpotrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dpotrf, 0, "lapacke_dpotrf");

    lapack_int info_lapacke_cpotrf = LAPACKE_cpotrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cpotrf, 0, "lapacke_cpotrf");

    lapack_int info_lapacke_zpotrf = LAPACKE_zpotrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zpotrf, 0, "lapacke_zpotrf");

    lapack_int info_lapacke_spotri = LAPACKE_spotri(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_spotri, 0, "lapacke_spotri");

    lapack_int info_lapacke_dpotri = LAPACKE_dpotri(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dpotri, 0, "lapacke_dpotri");

    lapack_int info_lapacke_cpotri = LAPACKE_cpotri(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cpotri, 0, "lapacke_cpotri");

    lapack_int info_lapacke_zpotri = LAPACKE_zpotri(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zpotri, 0, "lapacke_zpotri");

    lapack_int info_lapacke_spotrs = LAPACKE_spotrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_spotrs, 0, "lapacke_spotrs");

    lapack_int info_lapacke_dpotrs = LAPACKE_dpotrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dpotrs, 0, "lapacke_dpotrs");

    lapack_int info_lapacke_cpotrs = LAPACKE_cpotrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cpotrs, 0, "lapacke_cpotrs");

    lapack_int info_lapacke_zpotrs = LAPACKE_zpotrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zpotrs, 0, "lapacke_zpotrs");

    lapack_int info_lapacke_sppcon = LAPACKE_sppcon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_sppcon, 0, "lapacke_sppcon");

    lapack_int info_lapacke_dppcon = LAPACKE_dppcon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dppcon, 0, "lapacke_dppcon");

    lapack_int info_lapacke_cppcon = LAPACKE_cppcon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cppcon, 0, "lapacke_cppcon");

    lapack_int info_lapacke_zppcon = LAPACKE_zppcon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zppcon, 0, "lapacke_zppcon");

    lapack_int info_lapacke_sppequ = LAPACKE_sppequ(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_sppequ, 0, "lapacke_sppequ");

    lapack_int info_lapacke_dppequ = LAPACKE_dppequ(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dppequ, 0, "lapacke_dppequ");

    lapack_int info_lapacke_cppequ = LAPACKE_cppequ(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cppequ, 0, "lapacke_cppequ");

    lapack_int info_lapacke_zppequ = LAPACKE_zppequ(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zppequ, 0, "lapacke_zppequ");

    lapack_int info_lapacke_spprfs = LAPACKE_spprfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_spprfs, 0, "lapacke_spprfs");

    lapack_int info_lapacke_dpprfs = LAPACKE_dpprfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dpprfs, 0, "lapacke_dpprfs");

    lapack_int info_lapacke_cpprfs = LAPACKE_cpprfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cpprfs, 0, "lapacke_cpprfs");

    lapack_int info_lapacke_zpprfs = LAPACKE_zpprfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zpprfs, 0, "lapacke_zpprfs");

    lapack_int info_lapacke_sppsv = LAPACKE_sppsv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sppsv, 0, "lapacke_sppsv");

    lapack_int info_lapacke_dppsv = LAPACKE_dppsv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dppsv, 0, "lapacke_dppsv");

    lapack_int info_lapacke_cppsv = LAPACKE_cppsv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cppsv, 0, "lapacke_cppsv");

    lapack_int info_lapacke_zppsv = LAPACKE_zppsv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zppsv, 0, "lapacke_zppsv");

    lapack_int info_lapacke_sppsvx = LAPACKE_sppsvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)byte_buf,
        (void*)float_buf,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_sppsvx, 0, "lapacke_sppsvx");

    lapack_int info_lapacke_dppsvx = LAPACKE_dppsvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)byte_buf,
        (void*)double_buf,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dppsvx, 0, "lapacke_dppsvx");

    lapack_int info_lapacke_cppsvx = LAPACKE_cppsvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)byte_buf,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        1,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cppsvx, 0, "lapacke_cppsvx");

    lapack_int info_lapacke_zppsvx = LAPACKE_zppsvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)byte_buf,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zppsvx, 0, "lapacke_zppsvx");

    lapack_int info_lapacke_spptrf = LAPACKE_spptrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)float_buf
    );
    failed += assert_eq_int((int)info_lapacke_spptrf, 0, "lapacke_spptrf");

    lapack_int info_lapacke_dpptrf = LAPACKE_dpptrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)double_buf
    );
    failed += assert_eq_int((int)info_lapacke_dpptrf, 0, "lapacke_dpptrf");

    lapack_int info_lapacke_cpptrf = LAPACKE_cpptrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_float_buf
    );
    failed += assert_eq_int((int)info_lapacke_cpptrf, 0, "lapacke_cpptrf");

    lapack_int info_lapacke_zpptrf = LAPACKE_zpptrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_double_buf
    );
    failed += assert_eq_int((int)info_lapacke_zpptrf, 0, "lapacke_zpptrf");

    lapack_int info_lapacke_spptri = LAPACKE_spptri(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)float_buf
    );
    failed += assert_eq_int((int)info_lapacke_spptri, 0, "lapacke_spptri");

    lapack_int info_lapacke_dpptri = LAPACKE_dpptri(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)double_buf
    );
    failed += assert_eq_int((int)info_lapacke_dpptri, 0, "lapacke_dpptri");

    lapack_int info_lapacke_cpptri = LAPACKE_cpptri(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_float_buf
    );
    failed += assert_eq_int((int)info_lapacke_cpptri, 0, "lapacke_cpptri");

    lapack_int info_lapacke_zpptri = LAPACKE_zpptri(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_double_buf
    );
    failed += assert_eq_int((int)info_lapacke_zpptri, 0, "lapacke_zpptri");

    lapack_int info_lapacke_spptrs = LAPACKE_spptrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_spptrs, 0, "lapacke_spptrs");

    lapack_int info_lapacke_dpptrs = LAPACKE_dpptrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dpptrs, 0, "lapacke_dpptrs");

    lapack_int info_lapacke_cpptrs = LAPACKE_cpptrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cpptrs, 0, "lapacke_cpptrs");

    lapack_int info_lapacke_zpptrs = LAPACKE_zpptrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zpptrs, 0, "lapacke_zpptrs");

    lapack_int info_lapacke_spstrf = LAPACKE_spstrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)float_buf,
        1,
        (void*)int_buf,
        (void*)int_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_spstrf, 0, "lapacke_spstrf");

    lapack_int info_lapacke_dpstrf = LAPACKE_dpstrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)double_buf,
        1,
        (void*)int_buf,
        (void*)int_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dpstrf, 0, "lapacke_dpstrf");

    lapack_int info_lapacke_cpstrf = LAPACKE_cpstrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_float_buf,
        1,
        (void*)int_buf,
        (void*)int_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cpstrf, 0, "lapacke_cpstrf");

    lapack_int info_lapacke_zpstrf = LAPACKE_zpstrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_double_buf,
        1,
        (void*)int_buf,
        (void*)int_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zpstrf, 0, "lapacke_zpstrf");

    lapack_int info_lapacke_sptcon = LAPACKE_sptcon(
        1,
        (void*)float_buf,
        (void*)float_buf2,
        1,
        (void*)float_buf
    );
    failed += assert_eq_int((int)info_lapacke_sptcon, 0, "lapacke_sptcon");

    lapack_int info_lapacke_dptcon = LAPACKE_dptcon(
        1,
        (void*)double_buf,
        (void*)double_buf2,
        1,
        (void*)double_buf
    );
    failed += assert_eq_int((int)info_lapacke_dptcon, 0, "lapacke_dptcon");

    lapack_int info_lapacke_cptcon = LAPACKE_cptcon(
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        1,
        (void*)complex_float_buf
    );
    failed += assert_eq_int((int)info_lapacke_cptcon, 0, "lapacke_cptcon");

    lapack_int info_lapacke_zptcon = LAPACKE_zptcon(
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf
    );
    failed += assert_eq_int((int)info_lapacke_zptcon, 0, "lapacke_zptcon");

    lapack_int info_lapacke_spteqr = LAPACKE_spteqr(
        LAPACK_COL_MAJOR,
        'N',
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_spteqr, 0, "lapacke_spteqr");

    lapack_int info_lapacke_dpteqr = LAPACKE_dpteqr(
        LAPACK_COL_MAJOR,
        'N',
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dpteqr, 0, "lapacke_dpteqr");

    lapack_int info_lapacke_cpteqr = LAPACKE_cpteqr(
        LAPACK_COL_MAJOR,
        'N',
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cpteqr, 0, "lapacke_cpteqr");

    lapack_int info_lapacke_zpteqr = LAPACKE_zpteqr(
        LAPACK_COL_MAJOR,
        'N',
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zpteqr, 0, "lapacke_zpteqr");

    lapack_int info_lapacke_sptrfs = LAPACKE_sptrfs(
        LAPACK_COL_MAJOR,
        1,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_sptrfs, 0, "lapacke_sptrfs");

    lapack_int info_lapacke_dptrfs = LAPACKE_dptrfs(
        LAPACK_COL_MAJOR,
        1,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dptrfs, 0, "lapacke_dptrfs");

    lapack_int info_lapacke_cptrfs = LAPACKE_cptrfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cptrfs, 0, "lapacke_cptrfs");

    lapack_int info_lapacke_zptrfs = LAPACKE_zptrfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zptrfs, 0, "lapacke_zptrfs");

    lapack_int info_lapacke_sptsv = LAPACKE_sptsv(
        LAPACK_COL_MAJOR,
        1,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sptsv, 0, "lapacke_sptsv");

    lapack_int info_lapacke_dptsv = LAPACKE_dptsv(
        LAPACK_COL_MAJOR,
        1,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dptsv, 0, "lapacke_dptsv");

    lapack_int info_lapacke_cptsv = LAPACKE_cptsv(
        LAPACK_COL_MAJOR,
        1,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cptsv, 0, "lapacke_cptsv");

    lapack_int info_lapacke_zptsv = LAPACKE_zptsv(
        LAPACK_COL_MAJOR,
        1,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zptsv, 0, "lapacke_zptsv");

    lapack_int info_lapacke_sptsvx = LAPACKE_sptsvx(
        LAPACK_COL_MAJOR,
        'N',
        1,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf
    );
    failed += assert_eq_int((int)info_lapacke_sptsvx, 0, "lapacke_sptsvx");

    lapack_int info_lapacke_dptsvx = LAPACKE_dptsvx(
        LAPACK_COL_MAJOR,
        'N',
        1,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf
    );
    failed += assert_eq_int((int)info_lapacke_dptsvx, 0, "lapacke_dptsvx");

    lapack_int info_lapacke_cptsvx = LAPACKE_cptsvx(
        LAPACK_COL_MAJOR,
        'N',
        1,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)complex_float_buf
    );
    failed += assert_eq_int((int)info_lapacke_cptsvx, 0, "lapacke_cptsvx");

    lapack_int info_lapacke_zptsvx = LAPACKE_zptsvx(
        LAPACK_COL_MAJOR,
        'N',
        1,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)complex_double_buf
    );
    failed += assert_eq_int((int)info_lapacke_zptsvx, 0, "lapacke_zptsvx");

    lapack_int info_lapacke_spttrf = LAPACKE_spttrf(
        1,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_spttrf, 0, "lapacke_spttrf");

    lapack_int info_lapacke_dpttrf = LAPACKE_dpttrf(
        1,
        (void*)double_buf,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dpttrf, 0, "lapacke_dpttrf");

    lapack_int info_lapacke_cpttrf = LAPACKE_cpttrf(
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cpttrf, 0, "lapacke_cpttrf");

    lapack_int info_lapacke_zpttrf = LAPACKE_zpttrf(
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zpttrf, 0, "lapacke_zpttrf");

    lapack_int info_lapacke_spttrs = LAPACKE_spttrs(
        LAPACK_COL_MAJOR,
        1,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_spttrs, 0, "lapacke_spttrs");

    lapack_int info_lapacke_dpttrs = LAPACKE_dpttrs(
        LAPACK_COL_MAJOR,
        1,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dpttrs, 0, "lapacke_dpttrs");

    lapack_int info_lapacke_cpttrs = LAPACKE_cpttrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cpttrs, 0, "lapacke_cpttrs");

    lapack_int info_lapacke_zpttrs = LAPACKE_zpttrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zpttrs, 0, "lapacke_zpttrs");

    lapack_int info_lapacke_ssbev = LAPACKE_ssbev(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        0,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_ssbev, 0, "lapacke_ssbev");

    lapack_int info_lapacke_dsbev = LAPACKE_dsbev(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        0,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dsbev, 0, "lapacke_dsbev");

    lapack_int info_lapacke_ssbevd = LAPACKE_ssbevd(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        0,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_ssbevd, 0, "lapacke_ssbevd");

    lapack_int info_lapacke_dsbevd = LAPACKE_dsbevd(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        0,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dsbevd, 0, "lapacke_dsbevd");

    lapack_int info_lapacke_ssbevx = LAPACKE_ssbevx(
        LAPACK_COL_MAJOR,
        'N',
        'A',
        'U',
        1,
        0,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        1,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)float_buf,
        (void*)float_buf2,
        1,
        (void*)int_buf2
    );
    failed += assert_eq_int((int)info_lapacke_ssbevx, 0, "lapacke_ssbevx");

    lapack_int info_lapacke_dsbevx = LAPACKE_dsbevx(
        LAPACK_COL_MAJOR,
        'N',
        'A',
        'U',
        1,
        0,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        1,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)double_buf,
        (void*)double_buf2,
        1,
        (void*)int_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dsbevx, 0, "lapacke_dsbevx");

    lapack_int info_lapacke_ssbgst = LAPACKE_ssbgst(
        LAPACK_COL_MAJOR,
        'Q',
        'U',
        1,
        0,
        0,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_ssbgst, 0, "lapacke_ssbgst");

    lapack_int info_lapacke_dsbgst = LAPACKE_dsbgst(
        LAPACK_COL_MAJOR,
        'Q',
        'U',
        1,
        0,
        0,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dsbgst, 0, "lapacke_dsbgst");

    lapack_int info_lapacke_ssbgv = LAPACKE_ssbgv(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        0,
        0,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_ssbgv, 0, "lapacke_ssbgv");

    lapack_int info_lapacke_dsbgv = LAPACKE_dsbgv(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        0,
        0,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dsbgv, 0, "lapacke_dsbgv");

    lapack_int info_lapacke_ssbgvd = LAPACKE_ssbgvd(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        0,
        0,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_ssbgvd, 0, "lapacke_ssbgvd");

    lapack_int info_lapacke_dsbgvd = LAPACKE_dsbgvd(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        0,
        0,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dsbgvd, 0, "lapacke_dsbgvd");

    lapack_int info_lapacke_ssbgvx = LAPACKE_ssbgvx(
        LAPACK_COL_MAJOR,
        'N',
        'A',
        'U',
        1,
        0,
        0,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        1,
        1,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)float_buf2,
        (void*)float_buf,
        1,
        (void*)int_buf2
    );
    failed += assert_eq_int((int)info_lapacke_ssbgvx, 0, "lapacke_ssbgvx");

    lapack_int info_lapacke_dsbgvx = LAPACKE_dsbgvx(
        LAPACK_COL_MAJOR,
        'N',
        'A',
        'U',
        1,
        0,
        0,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        1,
        1,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)double_buf2,
        (void*)double_buf,
        1,
        (void*)int_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dsbgvx, 0, "lapacke_dsbgvx");

    lapack_int info_lapacke_ssbtrd = LAPACKE_ssbtrd(
        LAPACK_COL_MAJOR,
        'Q',
        'U',
        1,
        0,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        (void*)float_buf,
        (void*)float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_ssbtrd, 0, "lapacke_ssbtrd");

    lapack_int info_lapacke_dsbtrd = LAPACKE_dsbtrd(
        LAPACK_COL_MAJOR,
        'Q',
        'U',
        1,
        0,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        (void*)double_buf,
        (void*)double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dsbtrd, 0, "lapacke_dsbtrd");

    lapack_int info_lapacke_ssfrk = LAPACKE_ssfrk(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        'N',
        1,
        1,
        1,
        (void*)float_buf,
        1,
        1,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_ssfrk, 0, "lapacke_ssfrk");

    lapack_int info_lapacke_dsfrk = LAPACKE_dsfrk(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        'N',
        1,
        1,
        1,
        (void*)double_buf,
        1,
        1,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dsfrk, 0, "lapacke_dsfrk");

    lapack_int info_lapacke_sspcon = LAPACKE_sspcon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)float_buf,
        (void*)int_buf,
        1,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_sspcon, 0, "lapacke_sspcon");

    lapack_int info_lapacke_dspcon = LAPACKE_dspcon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)double_buf,
        (void*)int_buf,
        1,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dspcon, 0, "lapacke_dspcon");

    lapack_int info_lapacke_cspcon = LAPACKE_cspcon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_float_buf,
        (void*)int_buf,
        1,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cspcon, 0, "lapacke_cspcon");

    lapack_int info_lapacke_zspcon = LAPACKE_zspcon(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_double_buf,
        (void*)int_buf,
        1,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zspcon, 0, "lapacke_zspcon");

    lapack_int info_lapacke_sspev = LAPACKE_sspev(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sspev, 0, "lapacke_sspev");

    lapack_int info_lapacke_dspev = LAPACKE_dspev(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dspev, 0, "lapacke_dspev");

    lapack_int info_lapacke_sspevd = LAPACKE_sspevd(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sspevd, 0, "lapacke_sspevd");

    lapack_int info_lapacke_dspevd = LAPACKE_dspevd(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dspevd, 0, "lapacke_dspevd");

    lapack_int info_lapacke_sspevx = LAPACKE_sspevx(
        LAPACK_COL_MAJOR,
        'N',
        'A',
        'U',
        1,
        (void*)float_buf,
        1,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)float_buf2,
        (void*)float_buf,
        1,
        (void*)int_buf2
    );
    failed += assert_eq_int((int)info_lapacke_sspevx, 0, "lapacke_sspevx");

    lapack_int info_lapacke_dspevx = LAPACKE_dspevx(
        LAPACK_COL_MAJOR,
        'N',
        'A',
        'U',
        1,
        (void*)double_buf,
        1,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)double_buf2,
        (void*)double_buf,
        1,
        (void*)int_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dspevx, 0, "lapacke_dspevx");

    lapack_int info_lapacke_sspgst = LAPACKE_sspgst(
        LAPACK_COL_MAJOR,
        1,
        'U',
        1,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_sspgst, 0, "lapacke_sspgst");

    lapack_int info_lapacke_dspgst = LAPACKE_dspgst(
        LAPACK_COL_MAJOR,
        1,
        'U',
        1,
        (void*)double_buf,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dspgst, 0, "lapacke_dspgst");

    lapack_int info_lapacke_sspgv = LAPACKE_sspgv(
        LAPACK_COL_MAJOR,
        1,
        'N',
        'U',
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        (void*)float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sspgv, 0, "lapacke_sspgv");

    lapack_int info_lapacke_dspgv = LAPACKE_dspgv(
        LAPACK_COL_MAJOR,
        1,
        'N',
        'U',
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        (void*)double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dspgv, 0, "lapacke_dspgv");

    lapack_int info_lapacke_sspgvd = LAPACKE_sspgvd(
        LAPACK_COL_MAJOR,
        1,
        'N',
        'U',
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        (void*)float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sspgvd, 0, "lapacke_sspgvd");

    lapack_int info_lapacke_dspgvd = LAPACKE_dspgvd(
        LAPACK_COL_MAJOR,
        1,
        'N',
        'U',
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        (void*)double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dspgvd, 0, "lapacke_dspgvd");

    lapack_int info_lapacke_sspgvx = LAPACKE_sspgvx(
        LAPACK_COL_MAJOR,
        1,
        'N',
        'A',
        'U',
        1,
        (void*)float_buf,
        (void*)float_buf2,
        1,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)float_buf,
        (void*)float_buf2,
        1,
        (void*)int_buf2
    );
    failed += assert_eq_int((int)info_lapacke_sspgvx, 0, "lapacke_sspgvx");

    lapack_int info_lapacke_dspgvx = LAPACKE_dspgvx(
        LAPACK_COL_MAJOR,
        1,
        'N',
        'A',
        'U',
        1,
        (void*)double_buf,
        (void*)double_buf2,
        1,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)double_buf,
        (void*)double_buf2,
        1,
        (void*)int_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dspgvx, 0, "lapacke_dspgvx");

    lapack_int info_lapacke_ssprfs = LAPACKE_ssprfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)int_buf,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_ssprfs, 0, "lapacke_ssprfs");

    lapack_int info_lapacke_dsprfs = LAPACKE_dsprfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)int_buf,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dsprfs, 0, "lapacke_dsprfs");

    lapack_int info_lapacke_csprfs = LAPACKE_csprfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)int_buf,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_csprfs, 0, "lapacke_csprfs");

    lapack_int info_lapacke_zsprfs = LAPACKE_zsprfs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)int_buf,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zsprfs, 0, "lapacke_zsprfs");

    lapack_int info_lapacke_sspsv = LAPACKE_sspsv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)float_buf,
        (void*)int_buf,
        (void*)float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sspsv, 0, "lapacke_sspsv");

    lapack_int info_lapacke_dspsv = LAPACKE_dspsv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)double_buf,
        (void*)int_buf,
        (void*)double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dspsv, 0, "lapacke_dspsv");

    lapack_int info_lapacke_cspsv = LAPACKE_cspsv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_float_buf,
        (void*)int_buf,
        (void*)complex_float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cspsv, 0, "lapacke_cspsv");

    lapack_int info_lapacke_zspsv = LAPACKE_zspsv(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_double_buf,
        (void*)int_buf,
        (void*)complex_double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zspsv, 0, "lapacke_zspsv");

    lapack_int info_lapacke_sspsvx = LAPACKE_sspsvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)int_buf,
        (void*)float_buf,
        1,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf
    );
    failed += assert_eq_int((int)info_lapacke_sspsvx, 0, "lapacke_sspsvx");

    lapack_int info_lapacke_dspsvx = LAPACKE_dspsvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)int_buf,
        (void*)double_buf,
        1,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf
    );
    failed += assert_eq_int((int)info_lapacke_dspsvx, 0, "lapacke_dspsvx");

    lapack_int info_lapacke_cspsvx = LAPACKE_cspsvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)int_buf,
        (void*)complex_float_buf,
        1,
        (void*)complex_float_buf2,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)complex_float_buf
    );
    failed += assert_eq_int((int)info_lapacke_cspsvx, 0, "lapacke_cspsvx");

    lapack_int info_lapacke_zspsvx = LAPACKE_zspsvx(
        LAPACK_COL_MAJOR,
        'N',
        'U',
        1,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)int_buf,
        (void*)complex_double_buf,
        1,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)complex_double_buf
    );
    failed += assert_eq_int((int)info_lapacke_zspsvx, 0, "lapacke_zspsvx");

    lapack_int info_lapacke_ssptrd = LAPACKE_ssptrd(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_ssptrd, 0, "lapacke_ssptrd");

    lapack_int info_lapacke_dsptrd = LAPACKE_dsptrd(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        (void*)double_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dsptrd, 0, "lapacke_dsptrd");

    lapack_int info_lapacke_ssptrf = LAPACKE_ssptrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)float_buf,
        (void*)int_buf
    );
    failed += assert_eq_int((int)info_lapacke_ssptrf, 0, "lapacke_ssptrf");

    lapack_int info_lapacke_dsptrf = LAPACKE_dsptrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)double_buf,
        (void*)int_buf
    );
    failed += assert_eq_int((int)info_lapacke_dsptrf, 0, "lapacke_dsptrf");

    lapack_int info_lapacke_csptrf = LAPACKE_csptrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_float_buf,
        (void*)int_buf
    );
    failed += assert_eq_int((int)info_lapacke_csptrf, 0, "lapacke_csptrf");

    lapack_int info_lapacke_zsptrf = LAPACKE_zsptrf(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_double_buf,
        (void*)int_buf
    );
    failed += assert_eq_int((int)info_lapacke_zsptrf, 0, "lapacke_zsptrf");

    lapack_int info_lapacke_ssptri = LAPACKE_ssptri(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)float_buf,
        (void*)int_buf
    );
    failed += assert_eq_int((int)info_lapacke_ssptri, 0, "lapacke_ssptri");

    lapack_int info_lapacke_dsptri = LAPACKE_dsptri(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)double_buf,
        (void*)int_buf
    );
    failed += assert_eq_int((int)info_lapacke_dsptri, 0, "lapacke_dsptri");

    lapack_int info_lapacke_csptri = LAPACKE_csptri(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_float_buf,
        (void*)int_buf
    );
    failed += assert_eq_int((int)info_lapacke_csptri, 0, "lapacke_csptri");

    lapack_int info_lapacke_zsptri = LAPACKE_zsptri(
        LAPACK_COL_MAJOR,
        'U',
        1,
        (void*)complex_double_buf,
        (void*)int_buf
    );
    failed += assert_eq_int((int)info_lapacke_zsptri, 0, "lapacke_zsptri");

    lapack_int info_lapacke_ssptrs = LAPACKE_ssptrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)float_buf,
        (void*)int_buf,
        (void*)float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_ssptrs, 0, "lapacke_ssptrs");

    lapack_int info_lapacke_dsptrs = LAPACKE_dsptrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)double_buf,
        (void*)int_buf,
        (void*)double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dsptrs, 0, "lapacke_dsptrs");

    lapack_int info_lapacke_csptrs = LAPACKE_csptrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_float_buf,
        (void*)int_buf,
        (void*)complex_float_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_csptrs, 0, "lapacke_csptrs");

    lapack_int info_lapacke_zsptrs = LAPACKE_zsptrs(
        LAPACK_COL_MAJOR,
        'U',
        1,
        1,
        (void*)complex_double_buf,
        (void*)int_buf,
        (void*)complex_double_buf2,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zsptrs, 0, "lapacke_zsptrs");

    lapack_int info_lapacke_sstebz = LAPACKE_sstebz(
        'A',
        'R',
        1,
        1,
        1,
        1,
        1,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)int_buf,
        (void*)int_buf2,
        (void*)float_buf,
        (void*)int_buf,
        (void*)int_buf2
    );
    failed += assert_eq_int((int)info_lapacke_sstebz, 0, "lapacke_sstebz");

    lapack_int info_lapacke_dstebz = LAPACKE_dstebz(
        'A',
        'R',
        1,
        1,
        1,
        1,
        1,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)int_buf,
        (void*)int_buf2,
        (void*)double_buf,
        (void*)int_buf,
        (void*)int_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dstebz, 0, "lapacke_dstebz");

    lapack_int info_lapacke_sstedc = LAPACKE_sstedc(
        LAPACK_COL_MAJOR,
        'N',
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_sstedc, 0, "lapacke_sstedc");

    lapack_int info_lapacke_dstedc = LAPACKE_dstedc(
        LAPACK_COL_MAJOR,
        'N',
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dstedc, 0, "lapacke_dstedc");

    lapack_int info_lapacke_cstedc = LAPACKE_cstedc(
        LAPACK_COL_MAJOR,
        'N',
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_cstedc, 0, "lapacke_cstedc");

    lapack_int info_lapacke_zstedc = LAPACKE_zstedc(
        LAPACK_COL_MAJOR,
        'N',
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zstedc, 0, "lapacke_zstedc");

    lapack_int info_lapacke_sstegr = LAPACKE_sstegr(
        LAPACK_COL_MAJOR,
        'N',
        'A',
        1,
        (void*)float_buf,
        (void*)float_buf2,
        1,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)float_buf,
        (void*)float_buf2,
        1,
        (void*)int_buf2
    );
    failed += assert_eq_int((int)info_lapacke_sstegr, 0, "lapacke_sstegr");

    lapack_int info_lapacke_dstegr = LAPACKE_dstegr(
        LAPACK_COL_MAJOR,
        'N',
        'A',
        1,
        (void*)double_buf,
        (void*)double_buf2,
        1,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)double_buf,
        (void*)double_buf2,
        1,
        (void*)int_buf2
    );
    failed += assert_eq_int((int)info_lapacke_dstegr, 0, "lapacke_dstegr");

    lapack_int info_lapacke_cstegr = LAPACKE_cstegr(
        LAPACK_COL_MAJOR,
        'N',
        'A',
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        1,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        1,
        (void*)int_buf2
    );
    failed += assert_eq_int((int)info_lapacke_cstegr, 0, "lapacke_cstegr");

    lapack_int info_lapacke_zstegr = LAPACKE_zstegr(
        LAPACK_COL_MAJOR,
        'N',
        'A',
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        1,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        1,
        (void*)int_buf2
    );
    failed += assert_eq_int((int)info_lapacke_zstegr, 0, "lapacke_zstegr");

    lapack_int info_lapacke_sstein = LAPACKE_sstein(
        LAPACK_COL_MAJOR,
        1,
        (void*)float_buf,
        (void*)float_buf2,
        1,
        (void*)float_buf,
        (void*)int_buf,
        (void*)int_buf2,
        (void*)float_buf2,
        1,
        (void*)int_buf
    );
    failed += assert_eq_int((int)info_lapacke_sstein, 0, "lapacke_sstein");

    lapack_int info_lapacke_dstein = LAPACKE_dstein(
        LAPACK_COL_MAJOR,
        1,
        (void*)double_buf,
        (void*)double_buf2,
        1,
        (void*)double_buf,
        (void*)int_buf,
        (void*)int_buf2,
        (void*)double_buf2,
        1,
        (void*)int_buf
    );
    failed += assert_eq_int((int)info_lapacke_dstein, 0, "lapacke_dstein");

    lapack_int info_lapacke_cstein = LAPACKE_cstein(
        LAPACK_COL_MAJOR,
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        1,
        (void*)complex_float_buf,
        (void*)int_buf,
        (void*)int_buf2,
        (void*)complex_float_buf2,
        1,
        (void*)int_buf
    );
    failed += assert_eq_int((int)info_lapacke_cstein, 0, "lapacke_cstein");

    lapack_int info_lapacke_zstein = LAPACKE_zstein(
        LAPACK_COL_MAJOR,
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        1,
        (void*)complex_double_buf,
        (void*)int_buf,
        (void*)int_buf2,
        (void*)complex_double_buf2,
        1,
        (void*)int_buf
    );
    failed += assert_eq_int((int)info_lapacke_zstein, 0, "lapacke_zstein");

    lapack_int info_lapacke_sstemr = LAPACKE_sstemr(
        LAPACK_COL_MAJOR,
        'N',
        'A',
        1,
        (void*)float_buf,
        (void*)float_buf2,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)float_buf,
        (void*)float_buf2,
        1,
        1,
        (void*)int_buf2,
        0
    );
    failed += assert_eq_int((int)info_lapacke_sstemr, 0, "lapacke_sstemr");

    lapack_int info_lapacke_dstemr = LAPACKE_dstemr(
        LAPACK_COL_MAJOR,
        'N',
        'A',
        1,
        (void*)double_buf,
        (void*)double_buf2,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)double_buf,
        (void*)double_buf2,
        1,
        1,
        (void*)int_buf2,
        0
    );
    failed += assert_eq_int((int)info_lapacke_dstemr, 0, "lapacke_dstemr");

    lapack_int info_lapacke_cstemr = LAPACKE_cstemr(
        LAPACK_COL_MAJOR,
        'N',
        'A',
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        1,
        1,
        (void*)int_buf2,
        0
    );
    failed += assert_eq_int((int)info_lapacke_cstemr, 0, "lapacke_cstemr");

    lapack_int info_lapacke_zstemr = LAPACKE_zstemr(
        LAPACK_COL_MAJOR,
        'N',
        'A',
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        1,
        1,
        1,
        1,
        (void*)int_buf,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        1,
        1,
        (void*)int_buf2,
        0
    );
    failed += assert_eq_int((int)info_lapacke_zstemr, 0, "lapacke_zstemr");

    lapack_int info_lapacke_ssteqr = LAPACKE_ssteqr(
        LAPACK_COL_MAJOR,
        'N',
        1,
        (void*)float_buf,
        (void*)float_buf2,
        (void*)float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_ssteqr, 0, "lapacke_ssteqr");

    lapack_int info_lapacke_dsteqr = LAPACKE_dsteqr(
        LAPACK_COL_MAJOR,
        'N',
        1,
        (void*)double_buf,
        (void*)double_buf2,
        (void*)double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_dsteqr, 0, "lapacke_dsteqr");

    lapack_int info_lapacke_csteqr = LAPACKE_csteqr(
        LAPACK_COL_MAJOR,
        'N',
        1,
        (void*)complex_float_buf,
        (void*)complex_float_buf2,
        (void*)complex_float_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_csteqr, 0, "lapacke_csteqr");

    lapack_int info_lapacke_zsteqr = LAPACKE_zsteqr(
        LAPACK_COL_MAJOR,
        'N',
        1,
        (void*)complex_double_buf,
        (void*)complex_double_buf2,
        (void*)complex_double_buf,
        1
    );
    failed += assert_eq_int((int)info_lapacke_zsteqr, 0, "lapacke_zsteqr");

    lapack_int info_lapacke_ssterf = LAPACKE_ssterf(
        1,
        (void*)float_buf,
        (void*)float_buf2
    );
    failed += assert_eq_int((int)info_lapacke_ssterf, 0, "lapacke_ssterf");

    return failed;
}