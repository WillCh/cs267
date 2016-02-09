#include <emmintrin.h> 

const char* dgemm_desc = "trans dgemm.";




double tmp[1024 * 1024];
void square_dgemm(int N, double* mul1, double* mul2, double* res) {
    int i, j, k;
    double *__restrict__ rres = res;
    double *__restrict__ rtmp = tmp;
    double *__restrict__ rmul1 = mul1;
    double *__restrict__ rmul2 = mul2;

    for (i = 0; i < N; ++i)
        for (j = 0; j < N; ++j)
            rtmp[i * N + j] = rmul1[j * N + i];

    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            double rres_ij = 0;
            for (k = 0; k < N; ++k) {
                rres_ij += rmul2[i * N + k] * rtmp[j * N + k];
            }
            rres[i * N + j] += rres_ij;
        }
    }


}
