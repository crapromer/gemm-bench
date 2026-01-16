# GEMM 性能测试框架

一个简单的 GEMM（矩阵乘法）CPU 性能测试框架。

## 快速开始

## 如何添加自己的 GEMM Kernel

只需三步：

### 1. 创建源文件

在 `src/kernels/` 目录下创建新的 `.cpp` 文件，例如 `src/kernels/gemm_myopt.cpp`：

```cpp
#include "gemm_bench.h"

static void my_gemm(const float* A,
                    const float* B,
                    float* C,
                    int M, int N, int K,
                    int lda, int ldb, int ldc) {
    // 实现你的 GEMM：计算 C = A * B（覆盖式写 C）
    // A: M×K 行主序，B: K×N 行主序，C: M×N 行主序
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

// 注册 kernel
GEMM_REGISTER_F32("myopt", my_gemm);
```

### 2. 编译

```bash
xmake
```

### 3. 运行测试

```bash
xmake run
```

框架会自动：
- 生成随机测试数据
- 进行正确性校验（对比参考实现）
- 预热并计时
- 输出性能报告（GFLOP/s）

## Kernel 函数签名

```cpp
void your_gemm(const float* A,  // 输入矩阵 A，M×K 行主序
               const float* B,  // 输入矩阵 B，K×N 行主序
               float* C,        // 输出矩阵 C，M×N 行主序
               int M,           // A 的行数，C 的行数
               int N,           // B 的列数，C 的列数
               int K,           // A 的列数，B 的行数
               int lda,         // A 的 leading dimension（通常等于 K）
               int ldb,         // B 的 leading dimension（通常等于 N）
               int ldc);        // C 的 leading dimension（通常等于 N）
```

**重要约定**：
- 计算 `C = A * B`（覆盖式写入 C，不是累加）
- 矩阵按**行主序**存储
- 框架会在计时前处理转置（`--transA/--transB`），你的 kernel 只需要实现 NN 情况

## 命令行选项

```
--list                 列出所有已注册的 kernel
--kernel <name>        只运行指定 kernel（默认运行全部）
-M <int>               矩阵维度 M（默认 512）
-N <int>               矩阵维度 N（默认 512）
-K <int>               矩阵维度 K（默认 512）
--iters <int>          计时迭代次数（默认 30）
--warmup <int>         预热次数（默认 5）
--no-check             不做正确性校验（更快）
--seed <int>           随机种子（默认 1）
-v                     打印更多信息
-h, --help             帮助
```

## 项目结构

```
.
├── include/
│   └── gemm_bench.h          # 框架头文件
├── src/
│   ├── main.cpp              # 主程序（CLI）
│   ├── bench_gemm.cpp        # 性能测试逻辑
│   ├── gemm_registry.cpp     # Kernel 注册表
│   └── kernels/              # 你的 GEMM 实现放这里
│       ├── gemm_naive.cpp
│       ├── simd_gemm.cpp
│       └── ...
├── xmake.lua                 # 构建配置
└── README.md                 # 本文件
```

## 注意事项

- 使用简单的 C++11 特性，避免高级 C++ 特性
- 框架会自动处理内存对齐和分配
- 正确性校验阈值：相对误差和绝对误差均 < 1e-3
- 支持 AVX2/FMA 等 SIMD 指令（需要硬件支持）
