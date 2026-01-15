#include "gemm_bench.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>

static void* bench_aligned_alloc(size_t alignment, size_t size) {
    void* p = 0;
#if defined(_MSC_VER)
    p = _aligned_malloc(size, alignment);
    return p;
#else
    if (posix_memalign(&p, alignment, size) != 0) return 0;
    return p;
#endif
}

static void bench_aligned_free(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

// 简单可复现实验的伪随机数（避免依赖更复杂的 RNG）
static unsigned int lcg_next(unsigned int* state) {
    // Numerical Recipes
    *state = (*state) * 1664525u + 1013904223u;
    return *state;
}

static void fill_rand_f32(float* x, int n, unsigned int seed) {
    int i;
    unsigned int s = seed ? seed : 1u;
    for (i = 0; i < n; ++i) {
        unsigned int r = lcg_next(&s);
        // [-0.5, 0.5)
        x[i] = ((int)(r >> 9) / (float)(1u << 23)) - 0.5f;
    }
}

static void zero_f32(float* x, int n) {
    int i;
    for (i = 0; i < n; ++i) x[i] = 0.0f;
}

static void gemm_ref_f32(const float* A,
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
        for (j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (k = 0; k < K; ++k) {
                acc += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = acc;
        }
    }
}

static int check_close_f32(const float* got,
                           const float* ref,
                           int M,
                           int N,
                           int ldc,
                           float* out_max_abs,
                           float* out_max_rel) {
    int i, j;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (i = 0; i < M; ++i) {
        const float* g = got + i * ldc;
        const float* r = ref + i * ldc;
        for (j = 0; j < N; ++j) {
            float a = g[j];
            float b = r[j];
        float diff = a - b;
        if (diff < 0.0f) diff = -diff;
        if (diff > max_abs) max_abs = diff;
        float denom = b;
        if (denom < 0.0f) denom = -denom;
        if (denom < 1e-6f) denom = 1e-6f;
        float rel = diff / denom;
        if (rel > max_rel) max_rel = rel;
        }
    }
    if (out_max_abs) *out_max_abs = max_abs;
    if (out_max_rel) *out_max_rel = max_rel;

    // 经验阈值：教学起步阶段够用；后续可按你的课题调严/调宽
    if (max_abs > 1e-3f && max_rel > 1e-3f) return 0;
    return 1;
}

int bench_run_one(const GemmKernelDesc* k, const BenchArgs* args) {
    if (!k || !k->fn || !args) return -1;

    int M = args->M;
    int N = args->N;
    int K = args->K;
    int lda = args->lda ? args->lda : K;
    int ldb = args->ldb ? args->ldb : N;
    int ldc = args->ldc ? args->ldc : N;
    int warmup = args->warmup > 0 ? args->warmup : 5;
    int iters = args->iters > 0 ? args->iters : 30;

    if (M <= 0 || N <= 0 || K <= 0) return -2;
    if (lda < K || ldb < N || ldc < N) return -3;

    size_t bytesA = (size_t)M * (size_t)lda * sizeof(float);
    size_t bytesB = (size_t)K * (size_t)ldb * sizeof(float);
    size_t bytesC = (size_t)M * (size_t)ldc * sizeof(float);

    float* A = (float*)bench_aligned_alloc(64, bytesA);
    float* B = (float*)bench_aligned_alloc(64, bytesB);
    float* C = (float*)bench_aligned_alloc(64, bytesC);
    float* Cref = 0;
    if (args->check) Cref = (float*)bench_aligned_alloc(64, bytesC);

    if (!A || !B || !C || (args->check && !Cref)) {
        bench_aligned_free(A);
        bench_aligned_free(B);
        bench_aligned_free(C);
        bench_aligned_free(Cref);
        return -4;
    }

    fill_rand_f32(A, M * lda, (unsigned int)args->seed + 1u);
    fill_rand_f32(B, K * ldb, (unsigned int)args->seed + 2u);
    zero_f32(C, M * ldc);
    if (Cref) zero_f32(Cref, M * ldc);

    // correctness check（先跑一次）
    if (args->check) {
        gemm_ref_f32(A, B, Cref, M, N, K, lda, ldb, ldc);
        k->fn(A, B, C, M, N, K, lda, ldb, ldc);
        float max_abs = 0.0f, max_rel = 0.0f;
        int ok = check_close_f32(C, Cref, M, N, ldc, &max_abs, &max_rel);
        if (!ok) {
            printf("[FAIL] %-24s  max_abs=%g  max_rel=%g\n", k->name, max_abs, max_rel);
            bench_aligned_free(A);
            bench_aligned_free(B);
            bench_aligned_free(C);
            bench_aligned_free(Cref);
            return -5;
        }
    }

    // warmup
    for (int t = 0; t < warmup; ++t) {
        k->fn(A, B, C, M, N, K, lda, ldb, ldc);
    }

    // timing
    using clock = std::chrono::steady_clock;
    clock::time_point t0 = clock::now();
    for (int t = 0; t < iters; ++t) {
        k->fn(A, B, C, M, N, K, lda, ldb, ldc);
    }
    clock::time_point t1 = clock::now();

    double seconds = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();
    double sec_per = seconds / (double)iters;
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (flops / sec_per) / 1e9;

    printf("[OK]   %-24s  %dx%dx%d  %.3f ms/iter  %.2f GFLOP/s\n",
           k->name,
           M,
           N,
           K,
           sec_per * 1e3,
           gflops);

    bench_aligned_free(A);
    bench_aligned_free(B);
    bench_aligned_free(C);
    bench_aligned_free(Cref);
    return 0;
}

int bench_run_all(const BenchArgs* args, const char* only_name) {
    int n = gemm_get_count();
    const GemmKernelDesc* list = gemm_get_list();
    int ran = 0;

    for (int i = 0; i < n; ++i) {
        const GemmKernelDesc* k = &list[i];
        if (!k->name || !k->fn) continue;
        if (only_name && only_name[0]) {
            if (strcmp(k->name, only_name) != 0) continue;
        }
        int rc = bench_run_one(k, args);
        if (rc != 0) return rc;
        ++ran;
    }

    if (ran == 0) {
        if (only_name && only_name[0]) {
            printf("No kernel matched: %s\n", only_name);
        } else {
            printf("No kernels registered.\n");
        }
        return -10;
    }
    return 0;
}

// 参考实现也注册进来，便于作为 baseline 对比
static void gemm_ref_kernel(const float* A,
                            const float* B,
                            float* C,
                            int M,
                            int N,
                            int K,
                            int lda,
                            int ldb,
                            int ldc) {
    gemm_ref_f32(A, B, C, M, N, K, lda, ldb, ldc);
}

GEMM_REGISTER_F32("ref", gemm_ref_kernel);

