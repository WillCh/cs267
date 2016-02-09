#include <string.h>
#include <math.h>

#include <omp.h>
#include <emmintrin.h>
#include "benchmark.h"
#include "pmmintrin.h"


#include <stdio.h>

// try padding
void eig(float *v, float *A, float *u, size_t n, unsigned iters) {

    
    //omp_set_dynamic(0);     
    //omp_set_num_threads(16);
    register __m128 aTmp, vTmp, uTmp, muTmp;
    //dynamic padding
    int N;
    
    int index = n%64;

    if (n <= 4) {
        N = 4;
    } else if (n <= 16){
        N = 16;

    } else if (n <= 32) {
        N = 32;
    } else if (n < 64) {
        N = 64;
    }else if (index == 0 || index == 4 || index == 16 || index == 48) {
        N = n;
    } else if (index < 4) {
        N = (n/4 + 1)*4;
    } else if (index < 16) {
        N = (n/16 + 1)*16;
    } else if (index < 32) {
        N = (n/32 + 1)*32;
    } else if (index < 48) {
        N = (n/48 + 1)*48;
    } else {
        N = (n/64 + 1)*64;
    }
    

    // padding the data
    float *v_p = (float*) malloc(N*N*sizeof(float));
    float *A_p = (float*) malloc(N*N*sizeof(float));
    //float *u_p = (float*) malloc(N*N*sizeof(float));

    memset(v_p, 0, sizeof(float) * N * N);
    memset(A_p, 0, sizeof(float) * N * N);
    //memset(u_p, 0, sizeof(float) * N * N);

    #pragma omp parallel
    {
        #pragma omp for shared(v_p, v, A, A_p)

        for (int i = 0; i < n; i++)
        {

            memcpy((v_p+i*N), (v+i*n), n*sizeof(float));
            memcpy((A_p+i*N), (A+i*n), n*sizeof(float));
            
        }
    }

    for (size_t k = 0; k < iters; k += 1) {
        
        memset(v_p, 0, N * N * sizeof(float));
        #pragma omp parallel for shared(A_p, v_p, u) private(aTmp, vTmp, uTmp)
        
        for (size_t l = 0; l < n; l += 1) {

            for (size_t j = 0; j < n; j += 1) {
                int Nj = N*j;
                //int ln = l*n;
                int lN = l*N;
                uTmp = _mm_load1_ps(u + j+l*n);
                
                for (size_t i = 0; i < N/64*64; i += 64) {
                    float *add1 = A_p + Nj + i;
                    float *add2 = v_p + i + lN;
                    aTmp = _mm_loadu_ps(add1);
                    vTmp = _mm_loadu_ps(add2);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2),vTmp);

                    aTmp = _mm_loadu_ps(add1 +4);
                    vTmp = _mm_loadu_ps(add2 +4);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +4),vTmp);

                    aTmp = _mm_loadu_ps(add1 +8);
                    vTmp = _mm_loadu_ps(add2+8);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +8),vTmp);

                    aTmp = _mm_loadu_ps(add1 +12);
                    vTmp = _mm_loadu_ps(add2 +12);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +12),vTmp);

                    aTmp = _mm_loadu_ps(add1 +16);
                    vTmp = _mm_loadu_ps(add2 +16);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +16),vTmp);

                    aTmp = _mm_loadu_ps(add1 +20);
                    vTmp = _mm_loadu_ps(add2 +20);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +20),vTmp);

                    aTmp = _mm_loadu_ps(add1 +24);
                    vTmp = _mm_loadu_ps(add2 +24);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +24),vTmp);

                    aTmp = _mm_loadu_ps(add1 +28);
                    vTmp = _mm_loadu_ps(add2 +28);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +28),vTmp);

                    aTmp = _mm_loadu_ps(add1 +32);
                    vTmp = _mm_loadu_ps(add2 +32);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +32),vTmp);

                    aTmp = _mm_loadu_ps(add1 +36);
                    vTmp = _mm_loadu_ps(add2 +36);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +36),vTmp);

                    aTmp = _mm_loadu_ps(add1 +40);
                    vTmp = _mm_loadu_ps(add2 +40);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +40),vTmp);

                    aTmp = _mm_loadu_ps(add1 +44);
                    vTmp = _mm_loadu_ps(add2 +44);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +44),vTmp);

                    aTmp = _mm_loadu_ps(add1 +48);
                    vTmp = _mm_loadu_ps(add2 +48);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +48),vTmp);

                    aTmp = _mm_loadu_ps(add1 +52);
                    vTmp = _mm_loadu_ps(add2 +52);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +52),vTmp);

                    aTmp = _mm_loadu_ps(add1 +56);
                    vTmp = _mm_loadu_ps(add2 +56);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +56),vTmp);

                    aTmp = _mm_loadu_ps(add1 +60);
                    vTmp = _mm_loadu_ps(add2 +60);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +60),vTmp);
                    
                }
                // edge case
                for (size_t i = N/64*64; i < N/32*32; i += 32) {
                    float *add1 = A_p + Nj+i;
                    float *add2 = v_p + i + lN;
                    aTmp = _mm_loadu_ps(add1);
                    vTmp = _mm_loadu_ps(add2);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2),vTmp);

                    aTmp = _mm_loadu_ps(add1 +4);
                    vTmp = _mm_loadu_ps(add2 +4);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +4),vTmp);

                    aTmp = _mm_loadu_ps(add1 +8);
                    vTmp = _mm_loadu_ps(add2+8);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +8),vTmp);

                    aTmp = _mm_loadu_ps(add1 +12);
                    vTmp = _mm_loadu_ps(add2 +12);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +12),vTmp);

                    aTmp = _mm_loadu_ps(add1 +16);
                    vTmp = _mm_loadu_ps(add2 +16);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +16),vTmp);

                    aTmp = _mm_loadu_ps(add1 +20);
                    vTmp = _mm_loadu_ps(add2 +20);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +20),vTmp);

                    aTmp = _mm_loadu_ps(add1 +24);
                    vTmp = _mm_loadu_ps(add2 +24);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +24),vTmp);

                    aTmp = _mm_loadu_ps(add1 +28);
                    vTmp = _mm_loadu_ps(add2 +28);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +28),vTmp);
                }

                for (size_t i = N/32*32; i < N/16*16; i += 16) {
                    float *add1 = A_p + Nj+i;
                    float *add2 = v_p + i + lN;
                    aTmp = _mm_loadu_ps(add1);
                    vTmp = _mm_loadu_ps(add2);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2),vTmp);

                    aTmp = _mm_loadu_ps(add1 +4);
                    vTmp = _mm_loadu_ps(add2 +4);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +4),vTmp);

                    aTmp = _mm_loadu_ps(add1 +8);
                    vTmp = _mm_loadu_ps(add2+8);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +8),vTmp);

                    aTmp = _mm_loadu_ps(add1 +12);
                    vTmp = _mm_loadu_ps(add2 +12);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +12),vTmp);
                }

                for (size_t i = N/16*16; i < N; i += 4) {
                    float *add2 = v_p + i + lN;
                    aTmp = _mm_loadu_ps(A_p + Nj+i);
                    vTmp = _mm_loadu_ps(add2);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2),vTmp);
                }
                
            }
        }
        
        float mu[n];
        float tmp[4];
        memset(mu, 0, n * sizeof(float));
        
        #pragma omp parallel for shared(v_p, mu) private(vTmp, uTmp,tmp)
        
        for (size_t l = 0; l < n; l += 1) {
            //__m128  vTmp, uTmp;
            
            memset(tmp, 0, 4 * sizeof(float));
            uTmp = _mm_setzero_ps();
            int lN = l*N;
            for (size_t i = 0; i < N/64*64; i += 64) {
                float *add1 = v_p + i + lN;
                vTmp = _mm_loadu_ps(add1);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 8);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 16);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 20);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 24);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 28);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 32);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 36);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 40);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 44);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 48);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 52);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 56);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 60);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));
            }

            for (size_t i = N/64*64; i < N/32*32; i += 32) {
                float *add1 = v_p + i +lN;
                vTmp = _mm_loadu_ps(add1);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 8);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 16);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 20);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 24);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 28);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));
            }

            for (size_t i = N/32*32; i < N/16*16; i += 16) {
                float *add1 = v_p + i +lN;
                vTmp = _mm_loadu_ps(add1);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 8);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));
            }

            for (size_t i = N/16*16; i < N; i += 4) {
                
                vTmp = _mm_loadu_ps(v_p + i +lN);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));
            }
            
            _mm_storeu_ps(tmp, uTmp);
            mu[l] = tmp[0] +tmp[1] + tmp[2] +tmp[3];
            
            //mu[l] = sqrt(mu[l]);
        }

        #pragma omp parallel for shared(mu)
        
        for (size_t l = 0; l < n/4*4; l += 4) {
            muTmp = _mm_loadu_ps(mu + l);
            _mm_storeu_ps(mu + l, _mm_sqrt_ps(muTmp));
        }
        for (size_t l = n/4*4; l < n; l ++) {
            mu[l] = sqrt(mu[l]);
        }

        #pragma omp parallel for shared(v_p, u) private(vTmp, uTmp, muTmp)
        for (size_t l = 0; l < n; l += 1) {
            muTmp = _mm_set_ps(mu[l] , mu[l], mu[l], mu[l]);
            int ln = l*n;
            int lN = l*N;
            for (size_t i = 0; i < n/64*64; i += 64) {
                float *add1 = v_p +i +lN;
                float *add2 = u + i +ln;
                vTmp = _mm_loadu_ps(add1);
                _mm_storeu_ps(add2, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                _mm_storeu_ps(add2 +4, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 +8);
                _mm_storeu_ps(add2 + 8, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                _mm_storeu_ps(add2 +12, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 16);
                _mm_storeu_ps(add2 +16, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 20);
                _mm_storeu_ps(add2 +20, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 24);
                _mm_storeu_ps(add2 +24, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 28);
                _mm_storeu_ps(add2 +28, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 32);
                _mm_storeu_ps(add2 +32, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 36);
                _mm_storeu_ps(add2 +36, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 40);
                _mm_storeu_ps(add2 +40, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 44);
                _mm_storeu_ps(add2 +44, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 48);
                _mm_storeu_ps(add2 +48, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 52);
                _mm_storeu_ps(add2 +52, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 56);
                _mm_storeu_ps(add2 +56, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 60);
                _mm_storeu_ps(add2 +60, _mm_div_ps(vTmp, muTmp));

            }

            for (size_t i = n/64*64; i < n/32*32; i += 32) {
                float *add1 = v_p +i +lN;
                float *add2 = u + i +ln;
                vTmp = _mm_loadu_ps(add1);
                _mm_storeu_ps(add2, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                _mm_storeu_ps(add2 +4, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 +8);
                _mm_storeu_ps(add2 + 8, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                _mm_storeu_ps(add2 +12, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 16);
                _mm_storeu_ps(add2 +16, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 20);
                _mm_storeu_ps(add2 +20, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 24);
                _mm_storeu_ps(add2 +24, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 28);
                _mm_storeu_ps(add2 +28, _mm_div_ps(vTmp, muTmp));
            }

            for (size_t i = n/32*32; i < n/16*16; i +=16){
                float *add1 = v_p +i +lN;
                float *add2 = u + i +ln;
                vTmp = _mm_loadu_ps(add1);
                _mm_storeu_ps(add2, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                _mm_storeu_ps(add2 +4, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 +8);
                _mm_storeu_ps(add2 + 8, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                _mm_storeu_ps(add2 +12, _mm_div_ps(vTmp, muTmp));
            }

            for (size_t i = n/16*16; i < n/4*4; i +=4) {
                vTmp = _mm_loadu_ps( v_p + i + lN);
                _mm_storeu_ps( u + i + ln, _mm_div_ps(vTmp, muTmp));
            }
            for (size_t i = n/4*4; i < n; i++) {
                u[i + ln] = v_p[i + lN] / mu[l];
            }
        }
    }
    // convert back to the original one
    #pragma omp parallel
    {
        #pragma omp for shared(v, v_p)

        for (int i = 0; i < n; i++)
        {
            memcpy((v+i*n), (v_p+i*N), n*sizeof(float));
            
        }
    }
    free(v_p);

    free(A_p);
}

