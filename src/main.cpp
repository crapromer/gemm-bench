#include "gemm_bench.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char* prog) {
    printf("Usage:\n");
    printf("  %s [options]\n", prog);
    printf("\n");
    printf("Options:\n");
    printf("  --list                 列出所有已注册的 GEMM kernel\n");
    printf("  --kernel <name>        只运行指定 kernel（默认运行全部）\n");
    printf("  -M <int>               矩阵维度 M（默认 512）\n");
    printf("  -N <int>               矩阵维度 N（默认 512）\n");
    printf("  -K <int>               矩阵维度 K（默认 512）\n");
    printf("  --iters <int>          计时迭代次数（默认 30）\n");
    printf("  --warmup <int>         预热次数（默认 5）\n");
    printf("  --no-check             不做正确性校验（更快）\n");
    printf("  --seed <int>           随机种子（默认 1）\n");
    printf("  -v                     打印更多信息\n");
    printf("  -h, --help             帮助\n");
    printf("\n");
    printf("Kernel contract:\n");
    printf("  kernel(A, B, C, M, N, K, lda, ldb, ldc) 需计算 C = A * B（覆盖 C）。\n");
    printf("\n");
    printf("How to add your GEMM:\n");
    printf("  1) 新增一个 .cpp，例如 src/kernels/gemm_myopt.cpp\n");
    printf("  2) 实现函数签名 gemm_f32_fn\n");
    printf("  3) 在文件末尾写：GEMM_REGISTER_F32(\"myopt\", my_fn);\n");
}

static void list_kernels(void) {
    int n = gemm_get_count();
    const GemmKernelDesc* list = gemm_get_list();
    printf("Registered kernels (%d):\n", n);
    for (int i = 0; i < n; ++i) {
        if (list[i].name && list[i].fn) {
            printf("  %s\n", list[i].name);
        }
    }
}

int main(int argc, char** argv) {
    BenchArgs args;
    args.M = 512;
    args.N = 512;
    args.K = 512;
    args.lda = 0; // 0 表示默认使用 K
    args.ldb = 0; // 0 表示默认使用 N
    args.ldc = 0; // 0 表示默认使用 N
    args.warmup = 5;
    args.iters = 30;
    args.check = 1;
    args.seed = 1;
    args.verbose = 0;

    const char* only_kernel = 0;
    int do_list = 0;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (strcmp(a, "--list") == 0) {
            do_list = 1;
        } else if (strcmp(a, "--kernel") == 0) {
            if (i + 1 >= argc) {
                printf("Missing value for --kernel\n");
                return 1;
            }
            only_kernel = argv[++i];
        } else if (strcmp(a, "-M") == 0) {
            if (i + 1 >= argc) return 1;
            args.M = atoi(argv[++i]);
        } else if (strcmp(a, "-N") == 0) {
            if (i + 1 >= argc) return 1;
            args.N = atoi(argv[++i]);
        } else if (strcmp(a, "-K") == 0) {
            if (i + 1 >= argc) return 1;
            args.K = atoi(argv[++i]);
        } else if (strcmp(a, "--iters") == 0) {
            if (i + 1 >= argc) return 1;
            args.iters = atoi(argv[++i]);
        } else if (strcmp(a, "--warmup") == 0) {
            if (i + 1 >= argc) return 1;
            args.warmup = atoi(argv[++i]);
        } else if (strcmp(a, "--seed") == 0) {
            if (i + 1 >= argc) return 1;
            args.seed = atoi(argv[++i]);
        } else if (strcmp(a, "--no-check") == 0) {
            args.check = 0;
        } else if (strcmp(a, "-v") == 0) {
            args.verbose = 1;
        } else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            printf("Unknown arg: %s\n", a);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (do_list) {
        list_kernels();
        return 0;
    }

    if (args.verbose) {
        printf("M=%d N=%d K=%d, warmup=%d iters=%d, check=%d\n",
               args.M,
               args.N,
               args.K,
               args.warmup,
               args.iters,
               args.check);
    }

    return bench_run_all(&args, only_kernel);
}
