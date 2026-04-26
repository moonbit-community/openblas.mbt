#include <moonbit.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

moonbit_string_t cstr_to_moonbit_string(void *ptr) {
  char *cptr = (char *)ptr;
  int32_t len = strlen(cptr);
  moonbit_string_t ms = moonbit_make_string(len, 0);
  for (int i = 0; i < len; i++) {
    ms[i] = (uint16_t)cptr[i];
  }
  return ms;
}

void free_cstr(char* p) {
  if (p) {
    free(p);
  }
}

void* get_null() {
  return (void*)0;
}

int voidptr_is_null(void* p) {
  return p == NULL;
}

static int32_t vector_start(int32_t n, int32_t inc) {
  return inc < 0 ? (1 - n) * inc : 0;
}

static int32_t matrix_index(int32_t order, int32_t row, int32_t col, int32_t ld) {
  return order == 101 ? row * ld + col : col * ld + row;
}

static float matrix_value_float(
  int32_t order,
  int32_t transpose,
  const float* matrix,
  int32_t ld,
  int32_t row,
  int32_t col
) {
  return transpose == 111
    ? matrix[matrix_index(order, row, col, ld)]
    : matrix[matrix_index(order, col, row, ld)];
}

static double matrix_value_double(
  int32_t order,
  int32_t transpose,
  const double* matrix,
  int32_t ld,
  int32_t row,
  int32_t col
) {
  return transpose == 111
    ? matrix[matrix_index(order, row, col, ld)]
    : matrix[matrix_index(order, col, row, ld)];
}

float mbt_cblas_scsum(int32_t n, const float* x, int32_t incx) {
  if (n <= 0 || incx == 0) {
    return 0.0f;
  }
  int32_t ix = 2 * vector_start(n, incx);
  int32_t step = 2 * incx;
  float result = 0.0f;
  for (int32_t i = 0; i < n; i++) {
    result += x[ix] + x[ix + 1];
    ix += step;
  }
  return result;
}

double mbt_cblas_dzsum(int32_t n, const double* x, int32_t incx) {
  if (n <= 0 || incx == 0) {
    return 0.0;
  }
  int32_t ix = 2 * vector_start(n, incx);
  int32_t step = 2 * incx;
  double result = 0.0;
  for (int32_t i = 0; i < n; i++) {
    result += x[ix] + x[ix + 1];
    ix += step;
  }
  return result;
}

float mbt_cblas_samax(int32_t n, const float* x, int32_t incx) {
  if (n <= 0 || incx == 0) {
    return 0.0f;
  }
  int32_t ix = vector_start(n, incx);
  float result = fabsf(x[ix]);
  for (int32_t i = 1; i < n; i++) {
    ix += incx;
    float value = fabsf(x[ix]);
    if (value > result) {
      result = value;
    }
  }
  return result;
}

double mbt_cblas_damax(int32_t n, const double* x, int32_t incx) {
  if (n <= 0 || incx == 0) {
    return 0.0;
  }
  int32_t ix = vector_start(n, incx);
  double result = fabs(x[ix]);
  for (int32_t i = 1; i < n; i++) {
    ix += incx;
    double value = fabs(x[ix]);
    if (value > result) {
      result = value;
    }
  }
  return result;
}

float mbt_cblas_samin(int32_t n, const float* x, int32_t incx) {
  if (n <= 0 || incx == 0) {
    return 0.0f;
  }
  int32_t ix = vector_start(n, incx);
  float result = fabsf(x[ix]);
  for (int32_t i = 1; i < n; i++) {
    ix += incx;
    float value = fabsf(x[ix]);
    if (value < result) {
      result = value;
    }
  }
  return result;
}

double mbt_cblas_damin(int32_t n, const double* x, int32_t incx) {
  if (n <= 0 || incx == 0) {
    return 0.0;
  }
  int32_t ix = vector_start(n, incx);
  double result = fabs(x[ix]);
  for (int32_t i = 1; i < n; i++) {
    ix += incx;
    double value = fabs(x[ix]);
    if (value < result) {
      result = value;
    }
  }
  return result;
}

float mbt_cblas_scamax(int32_t n, const float* x, int32_t incx) {
  if (n <= 0 || incx == 0) {
    return 0.0f;
  }
  int32_t ix = 2 * vector_start(n, incx);
  int32_t step = 2 * incx;
  float result = fabsf(x[ix]) + fabsf(x[ix + 1]);
  for (int32_t i = 1; i < n; i++) {
    ix += step;
    float value = fabsf(x[ix]) + fabsf(x[ix + 1]);
    if (value > result) {
      result = value;
    }
  }
  return result;
}

double mbt_cblas_dzamax(int32_t n, const double* x, int32_t incx) {
  if (n <= 0 || incx == 0) {
    return 0.0;
  }
  int32_t ix = 2 * vector_start(n, incx);
  int32_t step = 2 * incx;
  double result = fabs(x[ix]) + fabs(x[ix + 1]);
  for (int32_t i = 1; i < n; i++) {
    ix += step;
    double value = fabs(x[ix]) + fabs(x[ix + 1]);
    if (value > result) {
      result = value;
    }
  }
  return result;
}

float mbt_cblas_scamin(int32_t n, const float* x, int32_t incx) {
  if (n <= 0 || incx == 0) {
    return 0.0f;
  }
  int32_t ix = 2 * vector_start(n, incx);
  int32_t step = 2 * incx;
  float result = fabsf(x[ix]) + fabsf(x[ix + 1]);
  for (int32_t i = 1; i < n; i++) {
    ix += step;
    float value = fabsf(x[ix]) + fabsf(x[ix + 1]);
    if (value < result) {
      result = value;
    }
  }
  return result;
}

double mbt_cblas_dzamin(int32_t n, const double* x, int32_t incx) {
  if (n <= 0 || incx == 0) {
    return 0.0;
  }
  int32_t ix = 2 * vector_start(n, incx);
  int32_t step = 2 * incx;
  double result = fabs(x[ix]) + fabs(x[ix + 1]);
  for (int32_t i = 1; i < n; i++) {
    ix += step;
    double value = fabs(x[ix]) + fabs(x[ix + 1]);
    if (value < result) {
      result = value;
    }
  }
  return result;
}

void mbt_cblas_caxpyc(
  int32_t n,
  const float* alpha,
  const float* x,
  int32_t incx,
  float* y,
  int32_t incy
) {
  if (n <= 0 || incx == 0 || incy == 0) {
    return;
  }
  float ar = alpha[0];
  float ai = alpha[1];
  int32_t ix = 2 * vector_start(n, incx);
  int32_t iy = 2 * vector_start(n, incy);
  int32_t stepx = 2 * incx;
  int32_t stepy = 2 * incy;
  for (int32_t i = 0; i < n; i++) {
    float xr = x[ix];
    float xi = x[ix + 1];
    y[iy] += ar * xr + ai * xi;
    y[iy + 1] += ai * xr - ar * xi;
    ix += stepx;
    iy += stepy;
  }
}

void mbt_cblas_zaxpyc(
  int32_t n,
  const double* alpha,
  const double* x,
  int32_t incx,
  double* y,
  int32_t incy
) {
  if (n <= 0 || incx == 0 || incy == 0) {
    return;
  }
  double ar = alpha[0];
  double ai = alpha[1];
  int32_t ix = 2 * vector_start(n, incx);
  int32_t iy = 2 * vector_start(n, incy);
  int32_t stepx = 2 * incx;
  int32_t stepy = 2 * incy;
  for (int32_t i = 0; i < n; i++) {
    double xr = x[ix];
    double xi = x[ix + 1];
    y[iy] += ar * xr + ai * xi;
    y[iy + 1] += ai * xr - ar * xi;
    ix += stepx;
    iy += stepy;
  }
}

void mbt_cblas_drotg(double* a, double* b, double* c, double* s) {
  double aa = *a;
  double bb = *b;
  double roe = fabs(aa) > fabs(bb) ? aa : bb;
  double scale = fabs(aa) + fabs(bb);
  if (scale == 0.0) {
    *a = 0.0;
    *b = 0.0;
    *c = 1.0;
    *s = 0.0;
    return;
  }

  double sa = aa / scale;
  double sb = bb / scale;
  double r = scale * sqrt(sa * sa + sb * sb);
  if (roe < 0.0) {
    r = -r;
  }
  *c = aa / r;
  *s = bb / r;
  if (fabs(aa) > fabs(bb)) {
    *b = *s;
  } else if (*c != 0.0) {
    *b = 1.0 / *c;
  } else {
    *b = 1.0;
  }
  *a = r;
}

void mbt_cblas_sgemmt(
  int32_t order,
  int32_t uplo,
  int32_t trans_a,
  int32_t trans_b,
  int32_t m,
  int32_t k,
  float alpha,
  const float* a,
  int32_t lda,
  const float* b,
  int32_t ldb,
  float beta,
  float* c,
  int32_t ldc
) {
  for (int32_t row = 0; row < m; row++) {
    for (int32_t col = 0; col < m; col++) {
      if ((uplo == 121 && col < row) || (uplo == 122 && col > row)) {
        continue;
      }
      float sum = 0.0f;
      for (int32_t p = 0; p < k; p++) {
        sum += matrix_value_float(order, trans_a, a, lda, row, p) *
          matrix_value_float(order, trans_b, b, ldb, p, col);
      }
      int32_t ci = matrix_index(order, row, col, ldc);
      c[ci] = alpha * sum + beta * c[ci];
    }
  }
}

void mbt_cblas_dgemmt(
  int32_t order,
  int32_t uplo,
  int32_t trans_a,
  int32_t trans_b,
  int32_t m,
  int32_t k,
  double alpha,
  const double* a,
  int32_t lda,
  const double* b,
  int32_t ldb,
  double beta,
  double* c,
  int32_t ldc
) {
  for (int32_t row = 0; row < m; row++) {
    for (int32_t col = 0; col < m; col++) {
      if ((uplo == 121 && col < row) || (uplo == 122 && col > row)) {
        continue;
      }
      double sum = 0.0;
      for (int32_t p = 0; p < k; p++) {
        sum += matrix_value_double(order, trans_a, a, lda, row, p) *
          matrix_value_double(order, trans_b, b, ldb, p, col);
      }
      int32_t ci = matrix_index(order, row, col, ldc);
      c[ci] = alpha * sum + beta * c[ci];
    }
  }
}
