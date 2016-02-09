


#include <emmintrin.h> /* header file for the SSE intrinsics */
#include <string.h>
double tmp2[2];
double tmpA[1024 * 1024];
const char* dgemm_desc = "SEE + trans dgemm.";

void square_dgemm(int N, double* A, double* B, double* C) {
    int i, j, k;
    double *__restrict__ rC = C;
    double *__restrict__ rtmpA = tmpA;
    double *__restrict__ rtmp2 = tmp2;
    double *__restrict__ rB = B;
    double *__restrict__ rA = A;
    //register __m128d cTmp1, cTmp2, cTmp3, cTmp4, cTmp5,  aTmp1, aTmp2, aTmp3, aTmp4, aTmp5,  bTmp1, bTmp2, bTmp3, bTmp4, bTmp5;
    register __m128d cTmp, aTmp, bTmp;
    for (i = 0; i < N; ++i)
        for (j = 0; j < N; ++j)
            rtmpA[i * N + j] = rA[j * N + i];
    
    

    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {

            //memset(rtmp2, 0, 2*sizeof(double));
            cTmp = _mm_setzero_pd();
            //cTmp2 = _mm_setzero_pd();
            //cTmp3 = _mm_setzero_pd();
            //cTmp4 = _mm_setzero_pd();
            //cTmp5 = _mm_setzero_pd();

            double* __restrict__ adda = rtmpA + i*N;
            double* __restrict__ addb = rB + j*N;


            _mm_prefetch(adda, _MM_HINT_T0);
            _mm_prefetch(addb, _MM_HINT_T0);
            /**
            for (k = 0; k < N/4*8; k += 8) {
                double* __restrict__ adda2 = adda + k;
                double* __restrict__ addb2 = addb + k;
                //_mm_prefetch(adda2 + 4, _MM_HINT_T0);
                //_mm_prefetch(addb2 + 4, _MM_HINT_T0);
                //_mm_prefetch(adda2 + 6, _MM_HINT_T0);
                //_mm_prefetch(addb2 + 6, _MM_HINT_T0);

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
            **/

            for (k = 0; k < N/2*2; k += 2) {
                aTmp = _mm_loadu_pd(adda + k);
                bTmp = _mm_loadu_pd(addb + k);
                cTmp = _mm_add_pd(cTmp, _mm_mul_pd(aTmp, bTmp)); 
            }

            _mm_storeu_pd(rtmp2, cTmp);
            /**
            _mm_storeu_pd(rtmp2 + 2, cTmp2);
            _mm_storeu_pd(rtmp2 + 4, cTmp3);
            _mm_storeu_pd(rtmp2 + 6, cTmp4);
            _mm_storeu_pd(rtmp2 + 8, cTmp5);
            **/
            rC[i + N * j] += rtmp2[0] + rtmp2[1];

            for (k = N/2*2; k < N; k++) {
                rC[i + N * j] += rtmpA[i*N + k] * rB[j*N + k];
            }
            
        }
    }


}
