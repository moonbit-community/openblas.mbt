#ifndef CBLAS_TEST_H
#define CBLAS_TEST_H

#include <stdio.h>
#include <math.h>
#include <cblas.h>

// Utility functions
int assert_eq(float actual, float expect, const char* msg);
int assert_eq_uint(size_t actual, size_t expect, const char* msg);

// Float test functions (single precision real)
int test_cblas_sdsdot();
int test_cblas_dsdot();
int test_cblas_sdot();
int test_cblas_sasum();
int test_cblas_ssum();
int test_cblas_snrm2();
int test_cblas_isamax();
int test_cblas_isamin();
int test_cblas_samax();
int test_cblas_samin();
int test_cblas_ismax();
int test_cblas_ismin();
int test_cblas_saxpy();
int test_cblas_scopy();
int test_cblas_sswap();
int test_cblas_sscal();
int test_cblas_sgemv();
int test_cblas_sgemm();
int test_cblas_sgemmt();
int test_cblas_ssymm();
int test_cblas_ssyrk();
int test_cblas_ssyr2k();
int test_cblas_strmm();
int test_cblas_strsm();
int test_cblas_saxpby();
int test_cblas_sger();
int test_cblas_strmv();
int test_cblas_ssymv();
int test_cblas_srot();
int test_cblas_sgbmv();
int test_cblas_srotg();
int test_cblas_ssyr();
int test_cblas_strsv();
int test_cblas_srotm();
int test_cblas_srotmg();
int test_cblas_ssbmv();
int test_cblas_sspmv();
int test_cblas_sspr();
int test_cblas_sspr2();
int test_cblas_somatcopy();
int test_cblas_simatcopy();
int test_cblas_sgeadd();
int test_cblas_stbmv();
int test_cblas_stpmv();
int test_cblas_ssyr2();
int test_cblas_stbsv();
int test_cblas_stpsv();
int test_cblas_scsum();

// Double test functions (double precision real)
int test_cblas_ddot();
int test_cblas_dasum();
int test_cblas_dsum();
int test_cblas_dnrm2();
int test_cblas_idamax();
int test_cblas_idamin();
int test_cblas_damax();
int test_cblas_damin();
int test_cblas_idmax();
int test_cblas_idmin();
int test_cblas_daxpy();
int test_cblas_dcopy();
int test_cblas_dswap();
int test_cblas_dscal();
int test_cblas_dgemv();
int test_cblas_dgemm();
int test_cblas_dgemmt();
int test_cblas_dsymm();
int test_cblas_dsyrk();
int test_cblas_dsyr2k();
int test_cblas_dtrmm();
int test_cblas_dtrsm();
int test_cblas_daxpby();
int test_cblas_dger();
int test_cblas_dtrmv();
int test_cblas_dsymv();
int test_cblas_drot();
int test_cblas_dgbmv();
int test_cblas_drotg();
int test_cblas_dsyr();
int test_cblas_dtrsv();
int test_cblas_drotm();
int test_cblas_drotmg();
int test_cblas_dsbmv();
int test_cblas_dspmv();
int test_cblas_dspr();
int test_cblas_dspr2();
int test_cblas_domatcopy();
int test_cblas_dimatcopy();
int test_cblas_dgeadd();
int test_cblas_dtbmv();
int test_cblas_dtpmv();
int test_cblas_dsyr2();
int test_cblas_dtbsv();
int test_cblas_dtpsv();
int test_cblas_dzamax();
int test_cblas_dzamin();
int test_cblas_dzasum();
int test_cblas_dznrm2();
int test_cblas_dzsum();
int test_cblas_izamax();
int test_cblas_izamin();
int test_cblas_izmax();
int test_cblas_izmin();

// Complex float test functions (single precision complex)
int test_cblas_cdotu();
int test_cblas_cdotc();
int test_cblas_scasum();
int test_cblas_scnrm2();
int test_cblas_icamax();
int test_cblas_icamin();
int test_cblas_scamax();
int test_cblas_scamin();
int test_cblas_icmax();
int test_cblas_icmin();
int test_cblas_caxpy();
int test_cblas_caxpyc();
int test_cblas_ccopy();
int test_cblas_cswap();
int test_cblas_cscal();
int test_cblas_csscal();
int test_cblas_cdotu_sub();
int test_cblas_cdotc_sub();
int test_cblas_crotg();
int test_cblas_caxpby();
int test_cblas_cgemv();
int test_cblas_cgeru();
int test_cblas_cgerc();

// Complex double test functions (double precision complex)
int test_cblas_zdotc();
int test_cblas_zaxpy();
int test_cblas_zaxpyc();
int test_cblas_zcopy();
int test_cblas_zswap();
int test_cblas_zscal();
int test_cblas_zdscal();
int test_cblas_zrotg();
int test_cblas_zgemv();
int test_cblas_zgeru();
int test_cblas_zgerc();
// Note: Some complex double functions may have binding issues

#endif // CBLAS_TEST_H
