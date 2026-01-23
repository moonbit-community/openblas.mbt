#include "lapack_test.h"

int assert_close(double actual, double expect, double eps, const char* msg) {
    if (fabs(actual - expect) > eps) {
        printf("%s Test Failed: actual: %.6f, expect: %.6f\n", msg, actual, expect);
        return 1;
    }
    return 0;
}

int assert_eq_int(int actual, int expect, const char* msg) {
    if (actual != expect) {
        printf("%s Test Failed: actual: %d, expect: %d\n", msg, actual, expect);
        return 1;
    }
    return 0;
}

int main() {
    int failed = 0;

    failed += test_lapacke_dgesv();
    failed += test_lapacke_dgetrf();
    failed += test_lapacke_dgeqrf();
    failed += test_lapacke_dgesvd();

    if (failed > 0) {
        printf("%d tests failed.\n", failed);
        return 1;
    }
    printf("All tests passed.\n");
    return 0;
}
