#include "gemm_bench.h"

static void naive_gemm(const float* A,
                           const float* B,
                           float* C,
                           int M,
                           int N,
                           int K,
                           int lda,
                           int ldb,
                           int ldc) {
    int i, j, k;

    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            float a = A[i * lda + k];
            const float* b_row = &B[k * ldb];
            float* c_row = &C[i * ldc];
            for (j = 0; j < N; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

GEMM_REGISTER_F32("naive_gemm", naive_gemm);

