#include <complex.h>

typedef struct {
    float re;
    float im;
} mbt_complex_float;

typedef struct {
    double re;
    double im;
} mbt_complex_double;

typedef float _Complex lapack_complex_float;
typedef double _Complex lapack_complex_double;

extern lapack_complex_float lapack_make_complex_float(float re, float im);
extern lapack_complex_double lapack_make_complex_double(double re, double im);

mbt_complex_float mbt_lapack_make_complex_float(float re, float im) {
    lapack_complex_float v = lapack_make_complex_float(re, im);
    mbt_complex_float out = {crealf(v), cimagf(v)};
    return out;
}

mbt_complex_double mbt_lapack_make_complex_double(double re, double im) {
    lapack_complex_double v = lapack_make_complex_double(re, im);
    mbt_complex_double out = {creal(v), cimag(v)};
    return out;
}
