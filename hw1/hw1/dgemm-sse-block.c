#include <string.h>
#include <emmintrin.h> /* header file for the SSE intrinsics */

double new_A[50000] __attribute__((aligned(16)));
double new_B[200000] __attribute__((aligned(16)));
double new_C[40000] __attribute__((aligned(16)));
#define min(a,b) (((a)<(b))?(a):(b))
const char* dgemm_desc = "SEE dgemm.";
/** then try the block method **/

/**
int changeSize(int n) {
    switch(n % 4) {
        case 0: return n;
        case 1: return n + 3;
        case 2: return n + 2;
        case 3: return n + 1;
    }
    return 0;
}


// A = lda * lda; A_p = W * H
// here the W is the width, H is height, and we will padding
void padding_copy(int lda, int W, int H, double* A, double* A_p) {
    //int new_W = (W % 2) ? (W + 1): W;
    //int new_H = (H % 2) ? (H + 1): H;
    int new_W = changeSize(W);
    int new_H = changeSize(H);
    //__m128d tmp;
    memset(A_p, 0.0, sizeof(double) * new_H* new_W);
    for (int j = 0; j < W; j ++) {
        memcpy((A_p + j*new_H), (A + j*lda), H*sizeof(double));
    }
}


void square_dgemm(int lda, double* A, double* B, double* C) {

    // here write them to the same method to save the link time
    //int block_size = 64;

    // here define the block size, it should be even num
    register __m128d c0, c1, c2, c3, a0, a1, a2, a3, b00, b01, b02, b03, \
                     b10, b11, b12, b13, b20, b21, b22, b23, b30, b31, b32, b33, \
                    tmp0, tmp1, tmp2, tmp3; 

    int block_size_K = 64;
    int block_size_I = 64;
    int block_size_J = 64;


    for (int k = 0; k < lda; k += block_size_K) {
        int K = min(block_size_K, lda - k);
        // load the belt of the B 
        padding_copy(lda, lda, K, B + k, new_B); // B is K* N
        
        for (int i = 0; i < lda; i += block_size_I) {
            // for each block dgemms
            double* tmpA = A + i + k*lda;
            int M = min(block_size_I, lda - i);
            padding_copy(lda, K, M, tmpA, new_A); // A ~ M* K
            // load the region of the a

            for (int j = 0; j < lda; j += block_size_J) {
                
                int N = min(block_size_J, lda - j);
                
                // then do the mul for each block
                double* tmpC = C + i + j*lda;
                
                //double* tmpB = B + k + j*lda;
                //padding_copy(lda, N, K, tmpB, new_B); // B is K* N
                padding_copy(lda, N, M, tmpC, new_C); // C is M * N
                
                // the get the proper add of the B

                //int new_M = (M % 2) ? (M + 1): M;
                //int new_N = (N % 2) ? (N + 1): N;
                //int new_K = (K % 2) ? (K + 1): K;
                
                int new_M = changeSize(M);
                int new_N = changeSize(N);
                int new_K = changeSize(K);

                double* tmpB = new_B + j*new_K;


                for (int kk = 0; kk < new_K; kk += 4) {
                    for (int jj = 0; jj < new_N; jj += 4) {

                        double* addB = tmpB + kk + jj*new_K;

                        double* tmpAddB1 = addB + new_K;
                        double* tmpAddB2 = addB + new_K * 2;
                        double* tmpAddB3 = addB + new_K * 3;

                        b00 = _mm_load1_pd(addB);
                        b01 = _mm_load1_pd(tmpAddB1);
                        b02 = _mm_load1_pd(tmpAddB2);
                        b03 = _mm_load1_pd(tmpAddB3);

                        b10 = _mm_load1_pd(addB + 1);
                        b11 = _mm_load1_pd(tmpAddB1 + 1);
                        b12 = _mm_load1_pd(tmpAddB2 + 1);
                        b13 = _mm_load1_pd(tmpAddB3 + 1);

                        b20 = _mm_load1_pd(addB + 2);
                        b21 = _mm_load1_pd(tmpAddB1 + 2);
                        b22 = _mm_load1_pd(tmpAddB2 + 2);
                        b23 = _mm_load1_pd(tmpAddB3 + 2);

                        b30 = _mm_load1_pd(addB + 3);
                        b31 = _mm_load1_pd(tmpAddB1 + 3);
                        b32 = _mm_load1_pd(tmpAddB2 + 3);
                        b33 = _mm_load1_pd(tmpAddB3 + 3);

                        double* adda_mid = new_A + kk*new_M;
                        double* addc_mid = new_C + jj*new_M;

                        for (int ii = 0; ii < new_M; ii += 2) {
                            double* addA = adda_mid + ii;
                            double* addc = addc_mid + ii;
                            
                            a0 = _mm_load_pd(addA);
                            a1 = _mm_load_pd(addA + new_M);
                            a2 = _mm_load_pd(addA + new_M * 2);
                            a3 = _mm_load_pd(addA + new_M * 3);

                            c0 = _mm_load_pd(addc);
                            c1 = _mm_load_pd(addc + new_M);
                            c2 = _mm_load_pd(addc + new_M * 2);
                            c3 = _mm_load_pd(addc + new_M * 3);

                            tmp0 = _mm_add_pd(c0, _mm_mul_pd(a0, b00));
                            tmp1 = _mm_add_pd(c1, _mm_mul_pd(a0, b01));
                            tmp2 = _mm_add_pd(c2, _mm_mul_pd(a0, b02));
                            tmp3 = _mm_add_pd(c3, _mm_mul_pd(a0, b03));

                            tmp0 = _mm_add_pd(tmp0, _mm_mul_pd(a1, b10));
                            tmp1 = _mm_add_pd(tmp1, _mm_mul_pd(a1, b11));
                            tmp2 = _mm_add_pd(tmp2, _mm_mul_pd(a1, b12));
                            tmp3 = _mm_add_pd(tmp3, _mm_mul_pd(a1, b13));

                            tmp0 = _mm_add_pd(tmp0, _mm_mul_pd(a2, b20));
                            tmp1 = _mm_add_pd(tmp1, _mm_mul_pd(a2, b21));
                            tmp2 = _mm_add_pd(tmp2, _mm_mul_pd(a2, b22));
                            tmp3 = _mm_add_pd(tmp3, _mm_mul_pd(a2, b23));

                            c0 = _mm_add_pd(tmp0, _mm_mul_pd(a3, b30));
                            c1 = _mm_add_pd(tmp1, _mm_mul_pd(a3, b31));
                            c2 = _mm_add_pd(tmp2, _mm_mul_pd(a3, b32));
                            c3 = _mm_add_pd(tmp3, _mm_mul_pd(a3, b33));

                            _mm_store_pd(addc, c0);
                            _mm_store_pd(addc + new_M, c1);
                            _mm_store_pd(addc + new_M * 2, c2);
                            _mm_store_pd(addc + new_M * 3, c3);
                            //tmpC[ii + jj*lda] += tmpA[ii + kk*lda] * tmpB[kk + jj*lda];
                        }
                        
                    }
                }
                // then need to copy back
                for (int jj = 0; jj < N; jj++) {
                    memcpy(tmpC + jj*lda, new_C + jj*new_M, M*sizeof(double) );
                }

            }
        }
    }
}
**/
 
/** 41 edition **/
// A = lda * lda; A_p = W * H
// here the W is the width, H is height, and we will padding
void padding_copy(int lda, int W, int H, double* A, double* A_p) {
    int new_W = (W % 2) ? (W + 1): W;
    int new_H = (H % 2) ? (H + 1): H;
    //__m128d tmp;
    memset(A_p, 0.0, sizeof(double) * new_H* new_W);
    for (int j = 0; j < W; j ++) {
        memcpy((A_p + j*new_H), (A + j*lda), H*sizeof(double));
    }
}

/**
void square_dgemm(int lda, double* A, double* B, double* C) {

    // here write them to the same method to save the link time
    //int block_size = 64;

    // here define the block size, it should be even num
    register __m128d c0, c1, a0, a1, b00, b01, b10, b11, tmp0, tmp1; 

    int block_size_K = 64;
    int block_size_I = 64;
    int block_size_J = 64;


    for (int k = 0; k < lda; k += block_size_K) {
        int K = min(block_size_K, lda - k);
        // load the belt of the B 
        padding_copy(lda, lda, K, B + k, new_B); // B is K* N
        
        for (int i = 0; i < lda; i += block_size_I) {
            // for each block dgemms
            double* tmpA = A + i + k*lda;
            int M = min(block_size_I, lda - i);
            padding_copy(lda, K, M, tmpA, new_A); // A ~ M* K
            // load the region of the a

            for (int j = 0; j < lda; j += block_size_J) {
                
                int N = min(block_size_J, lda - j);
                
                // then do the mul for each block
                double* tmpC = C + i + j*lda;
                
                //double* tmpB = B + k + j*lda;
                //padding_copy(lda, N, K, tmpB, new_B); // B is K* N
                padding_copy(lda, N, M, tmpC, new_C); // C is M * N
                
                // the get the proper add of the B

                int new_M = (M % 2) ? (M + 1): M;
                int new_N = (N % 2) ? (N + 1): N;
                int new_K = (K % 2) ? (K + 1): K;
                double* tmpB = new_B + j*new_K;


                for (int kk = 0; kk < new_K; kk += 2) {
                    for (int jj = 0; jj < new_N; jj += 2) {

                        double* addB = tmpB + kk + jj*new_K;

                        b00 = _mm_load1_pd(addB);
                        b01 = _mm_load1_pd(addB + new_K);
                        b10 = _mm_load1_pd(addB + 1);
                        b11 = _mm_load1_pd(addB + 1 + new_K);

                        double* adda_mid = new_A + kk*new_M;
                        double* addc_mid = new_C + jj*new_M;
                        for (int ii = 0; ii < new_M/2*2; ii += 2) {
                            double* addA = adda_mid + ii;
                            double* addc = addc_mid + ii;
                            
                            a0 = _mm_load_pd(addA);
                            a1 = _mm_load_pd(addA + new_M);

                            c0 = _mm_load_pd(addc);
                            c1 = _mm_load_pd(addc + new_M);

                            tmp0 = _mm_add_pd(c0, _mm_mul_pd(a0, b00));
                            tmp1 = _mm_add_pd(c1, _mm_mul_pd(a0, b01));

                            c0 = _mm_add_pd(tmp0, _mm_mul_pd(a1, b10));
                            c1 = _mm_add_pd(tmp1, _mm_mul_pd(a1, b11));

                            _mm_store_pd(addc, c0);
                            _mm_store_pd(addc + new_M, c1);
                            //tmpC[ii + jj*lda] += tmpA[ii + kk*lda] * tmpB[kk + jj*lda];
                        }
                        
                    }
                }
                // then need to copy back
                for (int jj = 0; jj < N; jj++) {
                    memcpy(tmpC + jj*lda, new_C + jj*new_M, M*sizeof(double) );
                }

            }
        }
    }
}
**/

/**
35 edition **/
void square_dgemm(int lda, double* A, double* B, double* C) {

    // here write them to the same method to save the link time
    //int block_size = 64;

    // here define the block size, it should be even num
    register __m128d c0, c1, a0, a1, b00, b01, b10, b11, tmp0, tmp1; 

    int block_size_K = 64;
    int block_size_I = 64;
    int block_size_J = 64;


    for (int k = 0; k < lda; k += block_size_K)

        for (int i = 0; i < lda; i += block_size_I) {
            // for each block dgemms
            for (int j = 0; j < lda; j += block_size_J) {
                int M = min(block_size_I, lda - i);
                int N = min(block_size_J, lda - j);
                int K = min(block_size_K, lda - k);
                // then do the mul for each block
                double* tmpC = C + i + j*lda;
                double* tmpA = A + i + k*lda;
                double* tmpB = B + k + j*lda;
                padding_copy(lda, N, M, tmpC, new_C); // C is M * N
                padding_copy(lda, N, K, tmpB, new_B); // B is K* N
                padding_copy(lda, K, M, tmpA, new_A); // A ~ M* K
                int new_M = (M % 2) ? (M + 1): M;
                int new_N = (N % 2) ? (N + 1): N;
                int new_K = (K % 2) ? (K + 1): K;

                for (int kk = 0; kk < new_K; kk += 2) {
                    for (int jj = 0; jj < new_N; jj += 2) {
                        double* addB = new_B + kk + jj*new_K;
                        b00 = _mm_load1_pd(addB);
                        b01 = _mm_load1_pd(addB + new_K);
                        b10 = _mm_load1_pd(addB + 1);
                        b11 = _mm_load1_pd(addB + 1 + new_K);

                        double* adda_mid = new_A + kk*new_M;
                        double* addc_mid = new_C + jj*new_M;
                        for (int ii = 0; ii < new_M/2*2; ii += 2) {
                            double* addA = adda_mid + ii;
                            double* addc = addc_mid + ii;
                            
                            a0 = _mm_load_pd(addA);
                            a1 = _mm_load_pd(addA + new_M);

                            c0 = _mm_load_pd(addc);
                            c1 = _mm_load_pd(addc + new_M);

                            tmp0 = _mm_add_pd(c0, _mm_mul_pd(a0, b00));
                            tmp1 = _mm_add_pd(c1, _mm_mul_pd(a0, b01));

                            c0 = _mm_add_pd(tmp0, _mm_mul_pd(a1, b10));
                            c1 = _mm_add_pd(tmp1, _mm_mul_pd(a1, b11));

                            _mm_store_pd(addc, c0);
                            _mm_store_pd(addc + new_M, c1);
                            //tmpC[ii + jj*lda] += tmpA[ii + kk*lda] * tmpB[kk + jj*lda];
                        }
                        
                    }
                }
                // then need to copy back
                for (int jj = 0; jj < N; jj++) {
                    memcpy(tmpC + jj*lda, new_C + jj*new_M, M*sizeof(double) );
                }

            }
        }
}
/**/