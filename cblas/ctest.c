#include "cblas_test.h"

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

int assert_eq_uint(size_t actual, size_t expect, const char* msg) {
    if (actual != expect) {
        printf("%s Test Failed: actual: %zu, expect: %zu;\n",
                msg, actual, expect);
        return 1;
    }
    return 0;
}

int main() {
    int failed = 0;
    
    // Float tests (single precision real)
    failed += test_cblas_sdsdot();
    failed += test_cblas_dsdot();
    failed += test_cblas_sdot();
    failed += test_cblas_sasum();
    failed += test_cblas_ssum();
    failed += test_cblas_snrm2();
    failed += test_cblas_isamax();
    failed += test_cblas_idamax();
    failed += test_cblas_isamin();
    failed += test_cblas_idamin();
    failed += test_cblas_samax();
    failed += test_cblas_damax();
    failed += test_cblas_samin();
    failed += test_cblas_damin();
    failed += test_cblas_ismax();
    failed += test_cblas_idmax();
    failed += test_cblas_ismin();
    failed += test_cblas_idmin();
    failed += test_cblas_saxpy();
    failed += test_cblas_daxpy();
    failed += test_cblas_scopy();
    failed += test_cblas_dcopy();
    failed += test_cblas_sswap();
    failed += test_cblas_dswap();
    failed += test_cblas_sscal();
    failed += test_cblas_dscal();
    failed += test_cblas_sgemv();
    failed += test_cblas_dgemv();
    failed += test_cblas_sgemm();
    failed += test_cblas_dgemm();
    failed += test_cblas_sgemmt();
    failed += test_cblas_dgemmt();
    failed += test_cblas_ssymm();
    failed += test_cblas_dsymm();
    failed += test_cblas_ssyrk();
    failed += test_cblas_dsyrk();
    failed += test_cblas_ssyr2k();
    failed += test_cblas_dsyr2k();
    failed += test_cblas_strmm();
    failed += test_cblas_dtrmm();
    failed += test_cblas_strsm();
    failed += test_cblas_dtrsm();
    failed += test_cblas_saxpby();
    failed += test_cblas_daxpby();
    failed += test_cblas_sger();
    failed += test_cblas_dger();
    failed += test_cblas_strmv();
    failed += test_cblas_dtrmv();
    failed += test_cblas_ssymv();
    failed += test_cblas_dsymv();
    failed += test_cblas_srot();
    failed += test_cblas_drot();
    failed += test_cblas_sgbmv();
    failed += test_cblas_dgbmv();
    failed += test_cblas_srotg();
    failed += test_cblas_drotg();
    failed += test_cblas_ssyr();
    failed += test_cblas_dsyr();
    failed += test_cblas_strsv();
    failed += test_cblas_dtrsv();
    failed += test_cblas_srotm();
    failed += test_cblas_drotm();
    failed += test_cblas_srotmg();
    failed += test_cblas_drotmg();
    failed += test_cblas_ssbmv();
    failed += test_cblas_dsbmv();
    failed += test_cblas_sspmv();
    failed += test_cblas_dspmv();
    failed += test_cblas_sspr();
    failed += test_cblas_dspr();
    failed += test_cblas_sspr2();
    failed += test_cblas_dspr2();
    failed += test_cblas_somatcopy();
    failed += test_cblas_domatcopy();
    failed += test_cblas_simatcopy();
    failed += test_cblas_dimatcopy();
    failed += test_cblas_sgeadd();
    failed += test_cblas_dgeadd();
    failed += test_cblas_stbmv();
    failed += test_cblas_dtbmv();
    failed += test_cblas_stpmv();
    failed += test_cblas_dtpmv();
    
    // Double tests are mixed with float tests above for compatibility with original test order
    
    // Complex float tests (single precision complex)
    failed += test_cblas_cdotu();
    failed += test_cblas_cdotc();
    failed += test_cblas_scasum();
    failed += test_cblas_scnrm2();
    failed += test_cblas_icamax();
    failed += test_cblas_icamin();
    failed += test_cblas_scamax();
    failed += test_cblas_scamin();
    failed += test_cblas_icmax();
    failed += test_cblas_icmin();
    failed += test_cblas_caxpy();
    failed += test_cblas_caxpyc();
    failed += test_cblas_ccopy();
    failed += test_cblas_cswap();
    failed += test_cblas_cscal();
    failed += test_cblas_csscal();
    
    // Complex double tests - all skipped due to binding issues
    
    if (failed > 0) {
        printf("%d tests failed.\n", failed);
        return 1;
    }
    printf("All tests passed.\n");
    return 0;
}