#include "gemm_bench.h"

#include <string.h> // strcmp

// 简单静态注册表（避免复杂依赖/特性）
#ifndef GEMM_REGISTRY_MAX
#define GEMM_REGISTRY_MAX 256
#endif

static GemmKernelDesc g_kernels[GEMM_REGISTRY_MAX];
static int g_kernel_count = 0;

int gemm_register_f32(const char* name, gemm_f32_fn fn) {
    int i;
    if (!name || !fn) return -1;

    // 去重：同名直接忽略（避免重复链接/包含）
    for (i = 0; i < g_kernel_count; ++i) {
        if (g_kernels[i].name && strcmp(g_kernels[i].name, name) == 0) {
            return i;
        }
    }

    if (g_kernel_count >= GEMM_REGISTRY_MAX) return -2;

    g_kernels[g_kernel_count].name = name;
    g_kernels[g_kernel_count].fn = fn;
    ++g_kernel_count;
    return g_kernel_count - 1;
}

int gemm_get_count(void) { return g_kernel_count; }

const GemmKernelDesc* gemm_get_list(void) { return g_kernels; }

const GemmKernelDesc* gemm_find(const char* name) {
    int i;
    if (!name) return 0;
    for (i = 0; i < g_kernel_count; ++i) {
        if (g_kernels[i].name && strcmp(g_kernels[i].name, name) == 0) {
            return &g_kernels[i];
        }
    }
    return 0;
}

