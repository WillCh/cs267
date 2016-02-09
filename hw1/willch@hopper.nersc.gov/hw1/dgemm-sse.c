#include <emmintrin.h> /* header file for the SSE intrinsics */

const char* dgemm_desc = "SEE dgemm.";
/** vector part **/



/** this fun use the SSE to implement the mul **/
void square_dgemm(int lda, double* A, double* B, double* C) {
    // define the variable here

    register __m128d cTmp, aTmp, bTmp; 

    for (int j = 0; j < lda; j++) {
        for (int k = 0; k < lda; k++) {
            // copy the B's val to fill the bTmp
            bTmp = _mm_load1_pd(B + k + j*lda);

            double* adda_mid = A + k*lda;
            double* addc_mid = C + j*lda;
            for (int i = 0; i < lda/8*8; i += 8) {
                double* adda = adda_mid + i;
                double* addc = addc_mid + i;
                
                aTmp = _mm_loadu_pd(adda);
                cTmp = _mm_loadu_pd(addc);
                cTmp = _mm_add_pd(cTmp, _mm_mul_pd(bTmp, aTmp));
                _mm_stream_pd(addc, cTmp);

                aTmp = _mm_loadu_pd(adda + 2);
                cTmp = _mm_loadu_pd(addc + 2);
                cTmp = _mm_add_pd(cTmp, _mm_mul_pd(bTmp, aTmp));
                _mm_stream_pd((addc + 2), cTmp);

                aTmp = _mm_loadu_pd(adda + 4);
                cTmp = _mm_loadu_pd(addc + 4);
                cTmp = _mm_add_pd(cTmp, _mm_mul_pd(bTmp, aTmp));
                _mm_stream_pd((addc + 4), cTmp);

                aTmp = _mm_loadu_pd(adda + 6);
                cTmp = _mm_loadu_pd(addc + 6);
                cTmp = _mm_add_pd(cTmp, _mm_mul_pd(bTmp, aTmp));
                _mm_stream_pd((addc + 6), cTmp);
            }

            for (int i = lda/8*8; i < lda/2*2; i += 2) {
                double* adda = adda_mid + i;
                double* addc = addc_mid + i;
                
                aTmp = _mm_loadu_pd(adda);
                cTmp = _mm_loadu_pd(addc);
                cTmp = _mm_add_pd(cTmp, _mm_mul_pd(bTmp, aTmp));
                _mm_stream_pd(addc, cTmp);
            }

            // the last case
            for (int i = lda/2*2; i < lda; i ++) {
                C[i + j*lda] += A[i + k*lda] * B[k+j*lda];

            }
        }
    }
}

/** then try the block method **/