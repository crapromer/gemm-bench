#ifndef GEMM_BENCH_H
#define GEMM_BENCH_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*gemm_f32_fn)(const float* A,
                           const float* B,
                           float* C,
                           int M,
                           int N,
                           int K,
                           int lda,
                           int ldb,
                           int ldc);

typedef struct GemmKernelDesc {
    const char* name;
    gemm_f32_fn fn;
} GemmKernelDesc;

// 注册/查询
int gemm_register_f32(const char* name, gemm_f32_fn fn);
int gemm_get_count(void);
const GemmKernelDesc* gemm_get_list(void);
const GemmKernelDesc* gemm_find(const char* name);

typedef struct BenchArgs {
    int M;
    int N;
    int K;
    int lda;
    int ldb;
    int ldc;
    int warmup;   // 预热次数
    int iters;    // 计时迭代次数
    int check;    // 1=校验正确性(默认), 0=不校验
    int seed;     // 随机种子
    int verbose;  // 1=打印更多信息
} BenchArgs;

// 运行单个 kernel；返回 0 表示成功
int bench_run_one(const GemmKernelDesc* k, const BenchArgs* args);

// 运行所有 kernel（可用 only_name 过滤；传 NULL 表示不过滤）
int bench_run_all(const BenchArgs* args, const char* only_name);

// 宏注册：在实现文件中写 GEMM_REGISTER_F32("my_gemm", my_gemm_fn);
#define GEMM_REGISTER_F32(name_str, fn_sym) \
    static int gemm_reg_dummy_##fn_sym = gemm_register_f32(name_str, fn_sym)

#ifdef __cplusplus
} // extern "C"
#endif

#endif // GEMM_BENCH_H

