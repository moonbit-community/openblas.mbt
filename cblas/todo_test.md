# CBLAS 未测试函数汇总

基于 `moon test --verbose` 的结果和 `cblas.mbti` 接口文件的对比分析，以下是尚未被测试的 CBLAS 函数汇总。

## 测试覆盖情况概述

- **总CBLAS函数数量**: 202 个函数
- **已测试函数数量**: 137 个函数 (更新时间: 2024年12月)  
- **未测试函数数量**: 65 个函数
- **测试覆盖率**: 67.8%

## 已测试的函数列表 (137个)

### BLAS Level 1 (向量操作)
- `cblas_sdsdot`, `cblas_dsdot`, `cblas_sdot`, `cblas_ddot`
- `cblas_cdotu`, `cblas_cdotc`
- `cblas_sasum`, `cblas_dasum`, `cblas_scasum`, `cblas_ssum`, `cblas_dsum`
- `cblas_snrm2`, `cblas_dnrm2`, `cblas_scnrm2`
- `cblas_isamax`, `cblas_idamax`, `cblas_icamax`
- `cblas_isamin`, `cblas_idamin`, `cblas_icamin`
- `cblas_samax`, `cblas_damax`, `cblas_scamax`
- `cblas_samin`, `cblas_damin`, `cblas_scamin`
- `cblas_ismax`, `cblas_idmax`, `cblas_icmax`
- `cblas_ismin`, `cblas_idmin`, `cblas_icmin`
- `cblas_saxpy`, `cblas_daxpy`, `cblas_caxpy`, `cblas_caxpyc`
- `cblas_scopy`, `cblas_dcopy`, `cblas_ccopy`
- `cblas_sswap`, `cblas_dswap`, `cblas_cswap`
- `cblas_sscal`, `cblas_dscal`, `cblas_cscal`, `cblas_csscal`
- `cblas_saxpby`, `cblas_daxpby`
- `cblas_srot`, `cblas_drot`
- `cblas_srotg`, `cblas_drotg`
- `cblas_srotm`, `cblas_drotm` (新增)
- `cblas_srotmg`, `cblas_drotmg` (新增)
- `cblas_ssyr2`, `cblas_dsyr2` (最新增加)

### BLAS Level 2 (矩阵-向量操作)
- `cblas_sgemv`, `cblas_dgemv`
- `cblas_sger`, `cblas_dger`
- `cblas_strmv`, `cblas_dtrmv`
- `cblas_ssymv`, `cblas_dsymv`
- `cblas_sgbmv`, `cblas_dgbmv`
- `cblas_ssyr`, `cblas_dsyr`
- `cblas_strsv`, `cblas_dtrsv`
- `cblas_ssbmv`, `cblas_dsbmv` (新增)
- `cblas_sspmv`, `cblas_dspmv` (新增)
- `cblas_sspr`, `cblas_dspr` (新增)
- `cblas_sspr2`, `cblas_dspr2` (新增)
- `cblas_stbmv`, `cblas_dtbmv` (新增)
- `cblas_stpmv`, `cblas_dtpmv` (新增)
- `cblas_stbsv`, `cblas_dtbsv` (最新增加)
- `cblas_stpsv`, `cblas_dtpsv` (最新增加)

### BLAS Level 3 (矩阵-矩阵操作)
- `cblas_sgemm`, `cblas_dgemm`
- `cblas_sgemmt`, `cblas_dgemmt`
- `cblas_ssymm`, `cblas_dsymm`
- `cblas_ssyrk`, `cblas_dsyrk`
- `cblas_ssyr2k`, `cblas_dsyr2k`
- `cblas_strmm`, `cblas_dtrmm`
- `cblas_strsm`, `cblas_dtrsm`

### BLAS 扩展函数 (新增)
- `cblas_somatcopy`, `cblas_domatcopy` (异地矩阵转置拷贝)
- `cblas_simatcopy`, `cblas_dimatcopy` (原地矩阵转置)
- `cblas_sgeadd`, `cblas_dgeadd` (矩阵加法)
- `cblas_scsum` (单精度复数求和) (最新增加)
- `cblas_dzamax`, `cblas_dzamin`, `cblas_dzasum`, `cblas_dznrm2`, `cblas_dzsum` (复数双精度实数操作) (最新增加)
- `cblas_izamax`, `cblas_izamin`, `cblas_izmax`, `cblas_izmin` (复数双精度索引) (最新增加)

### 复数函数 (新增18个)

#### 复数双精度 Level 1 (8个) ✅
- `cblas_zdotc` (复数共轭点积)
- `cblas_zaxpy`, `cblas_zaxpyc` (复数向量加法)
- `cblas_zcopy`, `cblas_zswap` (复数向量拷贝和交换)
- `cblas_zscal`, `cblas_zdscal` (复数向量缩放)
- `cblas_zrotg` (复数 Givens 旋转生成)

#### 复数单精度 Level 1 (4个) ✅
- `cblas_cdotu_sub`, `cblas_cdotc_sub` (复数点积子程序版本)
- `cblas_crotg` (复数 Givens 旋转生成)
- `cblas_caxpby` (复数向量线性组合)

#### 复数双精度 Level 2 (3个) ✅
- `cblas_zgemv` (复数通用矩阵向量乘法)
- `cblas_zgeru`, `cblas_zgerc` (复数通用矩阵rank-1更新)

#### 复数单精度 Level 2 (3个) ✅
- `cblas_cgemv` (复数通用矩阵向量乘法)
- `cblas_cgeru`, `cblas_cgerc` (复数通用矩阵rank-1更新)

## 未测试的函数列表 (65个)

### 1. Complex Double Precision (Z-前缀函数, 37个)

#### Level 1 函数 (2个) - 大部分已完成 ✅
- `cblas_zdotu`, `cblas_zdotu_sub` (无共轭点积)
- `cblas_zdrot`, `cblas_zaxpby` (旋转和向量线性组合)

#### Level 2 函数 (12个) - 基础函数已完成 ✅
- `cblas_zgbmv` (通用带状矩阵向量乘法)
- `cblas_zhemv`, `cblas_zhbmv`, `cblas_zhpmv` (Hermitian 矩阵向量乘法)
- `cblas_zher`, `cblas_zher2`, `cblas_zhpr`, `cblas_zhpr2` (Hermitian 矩阵更新)
- `cblas_ztrmv`, `cblas_ztbmv`, `cblas_ztpmv` (三角矩阵向量乘法)
- `cblas_ztrsv`, `cblas_ztbsv`, `cblas_ztpsv` (三角系统求解)

#### Level 3 函数 (12个)
- `cblas_zgemm`, `cblas_zgemm3m`, `cblas_zgemm_batch`
- `cblas_zgemmt`, `cblas_zhemm`
- `cblas_zherk`, `cblas_zher2k`
- `cblas_zsymm`, `cblas_zsyrk`, `cblas_zsyr2k`
- `cblas_ztrmm`, `cblas_ztrsm`

#### 扩展函数 (8个)
- `cblas_zgeadd`, `cblas_zimatcopy`, `cblas_zomatcopy`
- 其他复数矩阵操作函数

### 2. Complex Single Precision (C-前缀函数, 28个)

#### Level 1 函数 (1个) - 大部分已完成 ✅
- `cblas_csrot` (复数-实数 Givens 旋转)

#### Level 2 函数 (12个) - 基础函数已完成 ✅
- `cblas_cgbmv` (通用带状矩阵向量乘法)
- `cblas_chemv`, `cblas_chbmv`, `cblas_chpmv` (Hermitian 矩阵向量乘法)
- `cblas_cher`, `cblas_cher2`, `cblas_chpr`, `cblas_chpr2` (Hermitian 矩阵更新)
- `cblas_ctrmv`, `cblas_ctbmv`, `cblas_ctpmv` (三角矩阵向量乘法)
- `cblas_ctrsv`, `cblas_ctbsv`, `cblas_ctpsv` (三角系统求解)

#### Level 3 函数 (12个)
- `cblas_cgemm`, `cblas_cgemm3m`, `cblas_cgemm_batch`
- `cblas_cgemmt`, `cblas_chemm`
- `cblas_cherk`, `cblas_cher2k`
- `cblas_csymm`, `cblas_csyrk`, `cblas_csyr2k`
- `cblas_ctrmm`, `cblas_ctrsm`

#### 扩展函数 (6个)
- `cblas_cgeadd`, `cblas_cimatcopy`, `cblas_comatcopy`
- 其他复数矩阵操作函数

### 3. Double Precision Real (D-前缀函数, 1个)

D-前缀函数的测试覆盖率极高，仅剩下批量操作函数：

#### Level 3 函数 (1个)
- `cblas_dgemm_batch` (批量矩阵乘法)

### 4. Single Precision Real (S-前缀函数, 1个)

S-前缀函数的测试覆盖率极高，仅剩下批量操作函数：

#### Level 3 函数 (1个)
- `cblas_sgemm_batch` (批量矩阵乘法)

### 5. Index Functions (I-前缀函数, 0个)

所有复数双精度数组索引函数已完成测试 ✅

### 6. OpenBLAS 特有函数 (配置函数) - 未在CBLAS统计中

#### 配置函数 (9个)
- `openblas_get_config`, `openblas_get_corename`
- `openblas_get_num_procs`, `openblas_get_num_threads`
- `openblas_get_parallel`, `openblas_set_num_threads`
- `openblas_set_num_threads_local`
- `openblas_set_threads_callback_function`
- `goto_set_num_threads`

#### 常量 (3个)
- `OPENBLAS_OPENMP`, `OPENBLAS_SEQUENTIAL`, `OPENBLAS_THREAD`

## 建议的测试优先级

### 高优先级 (核心 BLAS 函数) - 已完成 ✅
基础的单精度和双精度实数BLAS函数已全面覆盖，包括：
1. **Level 1**: 向量运算 (点积、范数、索引、缩放、拷贝等) ✅
2. **Level 2**: 矩阵-向量运算 (gemv, ger, trmv, symv, gbmv, syr, trsv等) ✅  
3. **Level 3**: 矩阵-矩阵运算 (gemm, gemmt, symm, syrk, syr2k, trmm, trsm等) ✅
4. **扩展函数**: 矩阵操作 (omatcopy, imatcopy, geadd) ✅

### 中优先级 (剩余实数函数) - 建议下一步
1. **对称矩阵rank-2更新**: `cblas_ssyr2`, `cblas_dsyr2` (2个函数)
2. **三角求解扩展**: `cblas_stbsv`, `cblas_dtbsv`, `cblas_stpsv`, `cblas_dtpsv` (4个函数)
3. **批量操作**: `cblas_sgemm_batch`, `cblas_dgemm_batch` (2个函数)
4. **复数求和**: `cblas_scsum` (1个函数)
5. **复数双精度实数操作**: `cblas_dzamax`, `cblas_dzamin`, `cblas_dzasum`, `cblas_dznrm2`, `cblas_dzsum` (5个函数)
6. **复数双精度索引**: `cblas_izamax`, `cblas_izamin`, `cblas_izmax`, `cblas_izmin` (4个函数)

**小计**: 18个剩余实数和实数相关函数

### 低优先级 (复数函数)
1. **复数单精度**: 37个C-前缀函数 (主要是完整的复数数学运算)
2. **复数双精度**: 45个Z-前缀函数 (主要是完整的复数数学运算)

**小计**: 82个复数函数

### 最低优先级 (配置函数)
1. **OpenBLAS 配置**: 9个配置和线程管理函数
2. **OpenBLAS 常量**: 3个常量定义

## 测试实施分析

### 当前测试模式分析
1. **MoonBit测试**: 使用`test "function_name"`块，重点验证数值计算正确性
2. **C测试对照**: ctest.c提供相同的测试用例，用于验证绑定正确性
3. **测试覆盖**: 当前主要覆盖单精度和双精度实数运算，复数运算测试有限

### 未测试函数的技术障碍分析
1. **复数函数**: 主要困难在于`ComplexFloat`和`ComplexDouble`类型的参数传递
2. **批量操作**: `VoidPtr`参数类型需要特殊处理
3. **三角求解**: 需要构造适当的测试矩阵以避免数值不稳定

### 下一步测试建议
1. **立即实施**: 剩余18个实数相关函数 (中优先级)
2. **分类测试**: 按数据类型和操作级别组织
3. **参数验证**: 重点测试边界条件和错误处理
4. **数值精度**: 验证计算结果的数值准确性，特别是浮点精度问题

## 统筹分析总结

### 当前测试状态 (2024年12月)
- **测试覆盖率**: 67.8% (137/202个CBLAS函数)
- **实数函数覆盖**: 极高 (~98%) - 单精度和双精度实数运算基本完成
- **复数函数覆盖**: 低 (~15%) - 已测试复数相关基础函数
- **扩展函数覆盖**: 优秀 - 矩阵操作扩展函数全面覆盖

### 测试质量评估
1. **已测试函数**: 测试设计合理，既有MoonBit测试又有C对照测试
2. **数值验证**: 测试用例覆盖了基本的数学正确性验证
3. **边界测试**: 部分函数包含了浮点精度和特殊值测试

### 主要测试Gap
1. **技术Gap**: 复数类型绑定和参数传递复杂性
2. **功能Gap**: 82个复数函数和18个剩余实数函数
3. **优先级Gap**: 剩余实数函数优先级更高，更易实现

### 下一阶段建议
**立即实施**: 优先完成剩余18个实数相关函数，可将整体实数函数覆盖率提升到接近100%
**中期目标**: 解决复数绑定技术问题，开始复数函数测试
**长期目标**: 实现完整的CBLAS测试覆盖

## 下一步实施计划

### 第一阶段：完善实数函数测试 (优先级最高)
**目标**: 将实数函数覆盖率提升至接近100%

**已完成实施** (16个函数) ✅:
1. **对称矩阵扩展**: `cblas_ssyr2`, `cblas_dsyr2` (2个) ✅
2. **三角求解**: `cblas_stbsv`, `cblas_dtbsv`, `cblas_stpsv`, `cblas_dtpsv` (4个) ✅  
3. **复数求和**: `cblas_scsum` (1个) ✅
4. **复数实部操作**: `cblas_dzamax`, `cblas_dzamin`, `cblas_dzasum`, `cblas_dznrm2`, `cblas_dzsum` (5个) ✅
5. **复数索引**: `cblas_izamax`, `cblas_izamin`, `cblas_izmax`, `cblas_izmin` (4个) ✅

**实际结果**: 测试覆盖率已提升至 67.8% (137/202)

**下一步实施** (2个函数):
1. **批量操作**: `cblas_sgemm_batch`, `cblas_dgemm_batch` (2个)

### 第二阶段：解决复数绑定问题 (中期目标)
**技术重点**: 
1. 研究`ComplexFloat`和`ComplexDouble`在MoonBit中的正确绑定方式
2. 解决复数参数传递的技术问题
3. 实现基础复数函数测试框架

### 第三阶段：复数函数全面测试 (长期目标)
**目标**: 实现82个复数函数的测试覆盖
**预期结果**: 测试覆盖率达到90%以上

## 更新记录

### 2024年12月 - 复数函数测试增强完成 ✅
- **新增测试**: 成功增加18个CBLAS复数函数的测试覆盖
- **测试覆盖率**: 从58.9%提升至67.8%
- **复数函数突破**: 完成18个复数函数测试，包括Level 1和Level 2核心功能
- **测试质量**: 所有137个测试通过，C对照测试验证binding正确性
- **数学验证**: 修正了复数矩阵运算的数学计算，确保测试的准确性

#### 本轮完成的具体函数
**复数双精度 (8个Level 1 + 3个Level 2)**:
- Level 1: `zdotc`, `zaxpy`, `zaxpyc`, `zcopy`, `zswap`, `zscal`, `zdscal`, `zrotg`
- Level 2: `zgemv`, `zgeru`, `zgerc`

**复数单精度 (4个Level 1 + 3个Level 2)**:
- Level 1: `cdotu_sub`, `cdotc_sub`, `crotg`, `caxpby`
- Level 2: `cgemv`, `cgeru`, `cgerc`

### 2024年12月初 - 实数函数测试增强完成
- **新增测试**: 成功增加16个CBLAS函数的测试覆盖
- **测试覆盖率**: 从50.5%提升至58.9%
- **高优先级函数**: 完成18个高优先级函数中的16个
- **实数函数**: 几乎完成所有实数函数测试(覆盖率98%)
- **复数基础**: 完成复数相关基础函数测试框架

### 之前的更新
- 完成高优先级和大部分中优先级函数的测试
- 新增 22 个中优先级 BLAS 函数的测试覆盖
- Level 1-3 基础实数运算全面覆盖
- 扩展函数(矩阵操作)基本完成
