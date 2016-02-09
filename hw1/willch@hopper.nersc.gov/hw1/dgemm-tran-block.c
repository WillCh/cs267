


#include <emmintrin.h> /* header file for the SSE intrinsics */
#include <string.h>
const char* dgemm_desc = " trans dgemm  + matrix block";
//double tmp2[2];
double tmpA[1024 * 1024];
#define min(a,b) (((a)<(b))?(a):(b))

void square_dgemm(int lda, double* A, double* B, double* C) {
    int i, j, k, ii, jj, kk;
    int block_size= 64;

    double *__restrict__ rC = C;
    double *__restrict__ rtmpA = tmpA;
    double *__restrict__ rB = B;
    double *__restrict__ rA = A;
    


    for (i = 0; i < lda; ++i)
        for (j = 0; j < lda; ++j)
            rtmpA[i * lda + j] = rA[j * lda + i];
    
    

    for (i = 0; i < lda; i += block_size) {
        for (j = 0; j < lda; j += block_size) {
            for (k = 0; k < lda; k += block_size) {
                // get the corret boundary
                int M = min(block_size, lda - i);
                int N = min(block_size, lda - j);
                int K = min(block_size, lda - k);
                
                // get the add of the sub-matrice
                //double* headA = rtmpA + i*lda + k;
                //double* headB = rB + j*lda + k;
                //double* headC = rC + i + j*lda;

                for (ii = 0; ii < M; ii++) {
                    for (jj = 0; jj < N; jj++) {
                        //memset(tmp2, 0, 2*sizeof(double));
                        //cTmp = _mm_setzero_pd();
                        //double* adda = headA + ii*lda;
                        //double* addb = headB + jj*lda;
                        double rres_ij = 0;
                        for (kk = 0; kk < K; kk ++){
                            rres_ij += rtmpA[ii * lda + kk] * rB[jj * lda + kk];
                        }

                        rC[ii + lda * jj] += rres_ij;
                    }
                } 

            }
            
            
        }
    }


}
