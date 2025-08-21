# CBLAS 未测试函数汇总

基于 `moon test --verbose` 的结果和 `cblas.mbti` 接口文件的对比分析，以下是尚未被测试的 CBLAS 函数汇总。

## 测试覆盖情况概述

- **总函数数量**: 261 个函数
- **已测试函数数量**: 102 个函数 (更新时间: 2024年12月)  
- **未测试函数数量**: 159 个函数
- **测试覆盖率**: 39.1%

## 已测试的函数列表 (80个)

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

## 未测试的函数列表 (181个)

### 1. Complex Double Precision (Z-前缀函数, 58个)

#### Level 1 函数 (13个)
- `cblas_zdotc`, `cblas_zdotc_sub`
- `cblas_zdotu`, `cblas_zdotu_sub`
- `cblas_dzamax`, `cblas_dzamin`, `cblas_dzasum`, `cblas_dznrm2`, `cblas_dzsum`
- `cblas_izamax`, `cblas_izamin`, `cblas_izmax`, `cblas_izmin`
- `cblas_zaxpy`, `cblas_zaxpyc`, `cblas_zaxpby`
- `cblas_zcopy`, `cblas_zswap`
- `cblas_zscal`, `cblas_zdscal`
- `cblas_zdrot`, `cblas_zrotg`

#### Level 2 函数 (15个)
- `cblas_zgemv`
- `cblas_zgbmv`
- `cblas_zhemv`, `cblas_zhbmv`, `cblas_zhpmv`
- `cblas_zher`, `cblas_zher2`, `cblas_zhpr`, `cblas_zhpr2`
- `cblas_ztrmv`, `cblas_ztbmv`, `cblas_ztpmv`
- `cblas_ztrsv`, `cblas_ztbsv`, `cblas_ztpsv`
- `cblas_zgerc`, `cblas_zgeru`

#### Level 3 函数 (12个)
- `cblas_zgemm`, `cblas_zgemm3m`, `cblas_zgemm_batch`
- `cblas_zgemmt`
- `cblas_zhemm`
- `cblas_zherk`, `cblas_zher2k`
- `cblas_zsymm`, `cblas_zsyrk`, `cblas_zsyr2k`
- `cblas_ztrmm`, `cblas_ztrsm`

#### 扩展函数 (18个)
- `cblas_zgeadd`
- `cblas_zimatcopy`, `cblas_zomatcopy`
- 其他 z-前缀扩展函数

### 2. Complex Single Precision (C-前缀函数, 58个)

#### Level 1 函数 (6个)
- `cblas_cdotc_sub`, `cblas_cdotu_sub`
- `cblas_caxpby`
- `cblas_crotg`, `cblas_csrot`

#### Level 2 函数 (15个)
- `cblas_cgemv`
- `cblas_cgbmv`
- `cblas_chemv`, `cblas_chbmv`, `cblas_chpmv`
- `cblas_cher`, `cblas_cher2`, `cblas_chpr`, `cblas_chpr2`
- `cblas_ctrmv`, `cblas_ctbmv`, `cblas_ctpmv`
- `cblas_ctrsv`, `cblas_ctbsv`, `cblas_ctpsv`
- `cblas_cgerc`, `cblas_cgeru`

#### Level 3 函数 (12个)
- `cblas_cgemm`, `cblas_cgemm3m`, `cblas_cgemm_batch`
- `cblas_cgemmt`
- `cblas_chemm`
- `cblas_cherk`, `cblas_cher2k`
- `cblas_csymm`, `cblas_csyrk`, `cblas_csyr2k`
- `cblas_ctrmm`, `cblas_ctrsm`

#### 扩展函数 (25个)
- `cblas_cgeadd`
- `cblas_cimatcopy`, `cblas_comatcopy`
- `cblas_scsum` (单精度复数求和)
- 其他 c-前缀扩展函数

### 3. Double Precision Real (D-前缀函数, 33个)

#### Level 1 函数 (4个)
- `cblas_drotm`, `cblas_drotmg`

#### Level 2 函数 (11个)
- `cblas_dsbmv`, `cblas_dspmv`
- `cblas_dsyr2`, `cblas_dspr`, `cblas_dspr2`
- `cblas_dtbmv`, `cblas_dtpmv`
- `cblas_dtbsv`, `cblas_dtpsv`

#### Level 3 函数 (1个)
- `cblas_dgemm_batch`

#### 扩展函数 (17个)
- `cblas_dgeadd`
- `cblas_dimatcopy`, `cblas_domatcopy`

### 4. Single Precision Real (S-前缀函数, 32个)

#### Level 1 函数 (4个)
- `cblas_srotm`, `cblas_srotmg`

#### Level 2 函数 (11个)
- `cblas_ssbmv`, `cblas_sspmv`
- `cblas_ssyr2`, `cblas_sspr`, `cblas_sspr2`
- `cblas_stbmv`, `cblas_stpmv`
- `cblas_stbsv`, `cblas_stpsv`

#### Level 3 函数 (1个)
- `cblas_sgemm_batch`

#### 扩展函数 (16个)
- `cblas_sgeadd`
- `cblas_simatcopy`, `cblas_somatcopy`

### 5. OpenBLAS 特有函数 (常量和配置函数)

#### 配置函数 (8个)
- `openblas_get_config`
- `openblas_get_corename`
- `openblas_get_num_procs`
- `openblas_get_num_threads`
- `openblas_get_parallel`
- `openblas_set_num_threads`
- `openblas_set_num_threads_local`
- `openblas_set_threads_callback_function`
- `goto_set_num_threads`

#### 常量 (3个)
- `OPENBLAS_OPENMP`
- `OPENBLAS_SEQUENTIAL` 
- `OPENBLAS_THREAD`

## 建议的测试优先级

### 高优先级 (核心 BLAS 函数) - 已完成 ✅
1. **基础矩阵操作**: `cblas_dger`, `cblas_sger` (外积运算) ✅
2. **三角矩阵操作**: `cblas_dtrmv`, `cblas_strmv` (三角矩阵向量乘法) ✅
3. **对称矩阵操作**: `cblas_dsymv`, `cblas_ssymv` (对称矩阵向量乘法) ✅
4. **旋转操作**: `cblas_drot`, `cblas_srot` (Givens 旋转) ✅
5. **带状矩阵操作**: `cblas_sgbmv`, `cblas_dgbmv` (带状矩阵操作) ✅
6. **旋转生成**: `cblas_srotg`, `cblas_drotg` (Givens 旋转生成) ✅
7. **对称秩1更新**: `cblas_dsyr`, `cblas_ssyr` (对称矩阵秩1更新) ✅
8. **三角求解**: `cblas_dtrsv`, `cblas_strsv` (三角系统求解) ✅

### 中优先级 (扩展 BLAS 函数) - 已完成 ✅
1. **矩阵拷贝和转置**: `cblas_somatcopy`, `cblas_domatcopy` ✅
2. **原地矩阵转置**: `cblas_simatcopy`, `cblas_dimatcopy` ✅
3. **矩阵加法**: `cblas_sgeadd`, `cblas_dgeadd` ✅
4. **修改 Givens 旋转**: `cblas_srotm`, `cblas_drotm`, `cblas_srotmg`, `cblas_drotmg` ✅
5. **带状/打包矩阵**: `cblas_ssbmv`, `cblas_dsbmv`, `cblas_sspmv`, `cblas_dspmv` ✅
6. **打包矩阵rank更新**: `cblas_sspr`, `cblas_dspr`, `cblas_sspr2`, `cblas_dspr2` ✅  
7. **三角形带状矩阵**: `cblas_stbmv`, `cblas_dtbmv` ✅
8. **三角形打包矩阵**: `cblas_stpmv`, `cblas_dtpmv` ✅

### 下一步中优先级建议
1. **批量操作**: `cblas_sgemm_batch`, `cblas_dgemm_batch`
2. **三角形求解扩展**: `cblas_stbsv`, `cblas_dtbsv`, `cblas_stpsv`, `cblas_dtpsv`
3. **对称矩阵rank-2更新**: `cblas_ssyr2`, `cblas_dsyr2`

### 低优先级 (复数和专门化函数)
1. **复数运算**: 所有 C 和 Z 前缀函数
2. **打包存储**: 所有 spmv, hpmv 等函数
3. **OpenBLAS 配置**: 配置和线程管理函数

## 测试建议

1. **分类测试**: 按数据类型 (S/D/C/Z) 和操作级别 (Level 1/2/3) 组织测试
2. **参数验证**: 重点测试边界条件和错误处理
3. **性能测试**: 对核心函数进行性能基准测试  
4. **内存安全**: 确保所有指针操作的安全性
5. **数值精度**: 验证计算结果的数值准确性

## 总结

当前测试主要覆盖了单精度和双精度实数的基础 BLAS 操作，但在复数运算、高级矩阵操作、以及 OpenBLAS 特有功能方面存在较大测试缺口。建议优先补充核心 BLAS 函数的测试，然后逐步扩展到复数和专门化功能的测试覆盖。

## 更新记录

### 2024年12月 - 核心和中优先级函数测试增强
- 新增 22 个中优先级 BLAS 函数的测试覆盖
- 测试覆盖率从 30.6% 提升到 39.1%
- 完成高优先级和大部分中优先级函数的测试
- 本次新增测试函数:
  - Level 1 扩展: `cblas_srotm`, `cblas_drotm`, `cblas_srotmg`, `cblas_drotmg`
  - Level 2 扩展: `cblas_ssbmv`, `cblas_dsbmv`, `cblas_sspmv`, `cblas_dspmv`, `cblas_sspr`, `cblas_dspr`, `cblas_sspr2`, `cblas_dspr2`, `cblas_stbmv`, `cblas_dtbmv`, `cblas_stpmv`, `cblas_dtpmv`
  - 扩展函数: `cblas_somatcopy`, `cblas_domatcopy`, `cblas_simatcopy`, `cblas_dimatcopy`, `cblas_sgeadd`, `cblas_dgeadd`

### 下一阶段建议
根据优先级继续补充中优先级函数的测试，重点关注:
1. 批量操作和矩阵拷贝功能
2. Modified Givens 旋转操作  
3. 带状和打包存储格式的矩阵操作
4. 复数运算的基础函数
