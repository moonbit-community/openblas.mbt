#include <stdio.h>
#include <cblas.h>

int assert_eq(float actual, float expect, const char* msg) {
    int ia = *(int*)&actual;
    int ib = *(int*)&expect;
    if (ia != ib) {
        printf("%s Test Failed: actual: %f(0x%x), expect: %f(0x%x);\n",
                msg, actual, ia, expect, ib);
        return 1;
    }
    return 0;
}

int test_cblas_sdsdot() {
    int n = 3;
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {4.0f, 5.0f, 6.0f};

    float result = cblas_sdsdot(n, 1.0f, x, 1, y, 1);
    return assert_eq(result, 33.0f, "cblas_sdsdot");
}

int test_cblas_dsdot() {
    int n = 3;
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};

    double result = cblas_ddot(n, x, 1, y, 1);
    return assert_eq(result, 32.0, "cblas_ddot");
}

int main() {
    int failed = 0;
    failed += test_cblas_sdsdot();
    failed += test_cblas_dsdot();
    if (failed > 0) {
        printf("%d tests failed.\n", failed);
        return 1;
    }
    printf("All tests passed.\n");
    return 0;
}
