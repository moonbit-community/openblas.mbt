#include "lapack_test.h"

int test_lapacke_dgesv() {
    lapack_int n = 2;
    lapack_int nrhs = 1;
    double a[] = {
        3.0, 1.0,
        1.0, 2.0
    };
    double b[] = {4.0, 3.0};
    lapack_int ipiv[2] = {0, 0};

    lapack_int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, a, n, ipiv, b, n);
    int failed = 0;
    failed += assert_eq_int((int)info, 0, "lapacke_dgesv info");
    failed += assert_close(b[0], 1.0, 1e-6, "lapacke_dgesv x");
    failed += assert_close(b[1], 1.0, 1e-6, "lapacke_dgesv y");
    return failed;
}

int test_lapacke_dgetrf() {
    lapack_int n = 2;
    double a[] = {
        1.0, 0.0,
        0.0, 1.0
    };
    lapack_int ipiv[2] = {0, 0};

    lapack_int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a, n, ipiv);
    int failed = 0;
    failed += assert_eq_int((int)info, 0, "lapacke_dgetrf info");
    failed += assert_eq_int((int)ipiv[0], 1, "lapacke_dgetrf ipiv[0]");
    failed += assert_eq_int((int)ipiv[1], 2, "lapacke_dgetrf ipiv[1]");
    failed += assert_close(a[0], 1.0, 1e-6, "lapacke_dgetrf a[0]");
    failed += assert_close(a[1], 0.0, 1e-6, "lapacke_dgetrf a[1]");
    failed += assert_close(a[2], 0.0, 1e-6, "lapacke_dgetrf a[2]");
    failed += assert_close(a[3], 1.0, 1e-6, "lapacke_dgetrf a[3]");
    return failed;
}

int test_lapacke_dgeqrf() {
    lapack_int n = 2;
    double a[] = {
        1.0, 0.0,
        0.0, 1.0
    };
    double tau[] = {0.0, 0.0};

    lapack_int info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, n, a, n, tau);
    int failed = 0;
    failed += assert_eq_int((int)info, 0, "lapacke_dgeqrf info");
    failed += assert_close(fabs(a[0]), 1.0, 1e-6, "lapacke_dgeqrf a[0]");
    failed += assert_close(fabs(a[3]), 1.0, 1e-6, "lapacke_dgeqrf a[3]");
    failed += assert_close(a[1], 0.0, 1e-6, "lapacke_dgeqrf a[1]");
    failed += assert_close(a[2], 0.0, 1e-6, "lapacke_dgeqrf a[2]");
    return failed;
}

int test_lapacke_dgesvd() {
    lapack_int n = 2;
    double a[] = {
        3.0, 0.0,
        0.0, 2.0
    };
    double s[] = {0.0, 0.0};
    double u[] = {0.0, 0.0, 0.0, 0.0};
    double vt[] = {0.0, 0.0, 0.0, 0.0};
    double superb[] = {0.0};

    lapack_int info = LAPACKE_dgesvd(
        LAPACK_COL_MAJOR,
        'S',
        'S',
        n,
        n,
        a,
        n,
        s,
        u,
        n,
        vt,
        n,
        superb
    );
    int failed = 0;
    failed += assert_eq_int((int)info, 0, "lapacke_dgesvd info");
    failed += assert_close(s[0], 3.0, 1e-6, "lapacke_dgesvd s[0]");
    failed += assert_close(s[1], 2.0, 1e-6, "lapacke_dgesvd s[1]");
    return failed;
}
