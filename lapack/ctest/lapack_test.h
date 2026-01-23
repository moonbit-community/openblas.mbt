#ifndef LAPACK_TEST_H
#define LAPACK_TEST_H

#include <stdio.h>
#include <math.h>
#include <lapacke.h>

int assert_close(double actual, double expect, double eps, const char* msg);
int assert_eq_int(int actual, int expect, const char* msg);

int test_lapacke_dgesv();
int test_lapacke_dgetrf();
int test_lapacke_dgeqrf();
int test_lapacke_dgesvd();

#endif
