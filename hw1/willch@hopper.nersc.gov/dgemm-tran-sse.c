


#include <emmintrin.h> /* header file for the SSE intrinsics */
#include <string.h>
double tmp2[10];
double tmpA[1024 * 1024];
const char* dgemm_desc = "SEE + trans dgemm.";

void square_dgemm(int N, double* A, double* B, double* C) {
    int i, j, k;
    double *__restrict__ rC = C;
    double *__restrict__ rtmpA = tmpA;
    double *__restrict__ rtmp2 = tmp2;
    double *__restrict__ rB = B;
    double *__restrict__ rA = A;
    register __m128d cTmp1, cTmp2, cTmp3, cTmp4, cTmp5,  aTmp1, aTmp2, aTmp3, aTmp4, aTmp5,  bTmp1, bTmp2, bTmp3, bTmp4, bTmp5;

    for (i = 0; i < N; ++i)
        for (j = 0; j < N; ++j)
            rtmpA[i * N + j] = rA[j * N + i];
    
    

    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {

            //memset(rtmp2, 0, 2*sizeof(double));
            cTmp = _mm_setzero_pd();
            double* adda = rtmpA + i*N;
            double* addb = rB + j*N;
            for (k = 0; k < N/8*8; k += 8) {
                double* adda2 = adda + k;
                double* addb2 = addb + k;
                aTmp1 = _mm_loadu_pd(adda2);
                bTmp1 = _mm_loadu_pd(addb2);
                cTmp1 = _mm_add_pd(cTmp1, _mm_mul_pd(aTmp1, bTmp1));

                aTmp2 = _mm_loadu_pd(adda2 + 2);
                bTmp2 = _mm_loadu_pd(addb2 + 2);
                cTmp2 = _mm_add_pd(cTmp2, _mm_mul_pd(aTmp2, bTmp2));   


                aTmp3 = _mm_loadu_pd(adda2 + 4);
                bTmp3 = _mm_loadu_pd(addb2 + 4);
                cTmp3 = _mm_add_pd(cTmp3, _mm_mul_pd(aTmp3, bTmp3));   

                aTmp4 = _mm_loadu_pd(adda2 + 6);
                bTmp4 = _mm_loadu_pd(addb2 + 6);
                cTmp4 = _mm_add_pd(cTmp4, _mm_mul_pd(aTmp4, bTmp4));               
            }

            for (k = N/8*8; k < N/2*2; k += 2) {
                aTmp5 = _mm_loadu_pd(adda + k);
                bTmp5 = _mm_loadu_pd(addb + k);
                cTmp5 = _mm_add_pd(cTmp5, _mm_mul_pd(aTmp5, bTmp5)); 
            }

            _mm_storeu_pd(rtmp2, cTmp1);
            _mm_storeu_pd(rtmp2 + 2, cTmp2);
            _mm_storeu_pd(rtmp2 + 4, cTmp3);
            _mm_storeu_pd(rtmp2 + 6, cTmp4);
            _mm_storeu_pd(rtmp2 + 8, cTmp5);

            rC[i + N * j] += rtmp2[0] + rtmp2[1] + rtmp2[2] + rtmp2[3] + rtmp2[4] + rtmp2[5]
                            + rtmp2[6] + rtmp2[7] + rtmp2[8] + rtmp2[9];

            for (k = N/2*2; k < N; k++) {
                rC[i + N * j] += rtmpA[i*N + k] * rB[j*N + k];
            }
            
        }
    }


}
