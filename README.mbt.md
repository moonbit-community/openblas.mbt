# OpenBLAS.mbt

这是一个为 [MoonBit 编程语言](https://www.moonbitlang.com/) 提供的 OpenBLAS 绑定库。OpenBLAS 是一个高性能的 BLAS (Basic Linear Algebra Subprograms) 和部分 LAPACK (Linear Algebra PACKage) 实现，广泛用于科学计算和机器学习领域。

## ⚠️ 开发中警告

**重要提示**: 本库的 OpenBLAS 绑定已经完成，但许多 API 尚未经过充分测试。在生产环境中使用前，请仔细测试相关功能。我们欢迎社区贡献测试用例和 bug 报告。

## 系统要求与安装

### macOS

使用 Homebrew 安装 OpenBLAS：

```bash
brew install openblas
```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install libopenblas-dev
```

### Linux (CentOS/RHEL/Fedora)

```bash
# CentOS/RHEL
sudo yum install openblas-devel

# Fedora
sudo dnf install openblas-devel
```

### Windows

**注意**: Windows 理论上可用，但尚未经过测试。你可以尝试以下方法：

1. 使用 [vcpkg](https://vcpkg.io/en/) 安装 OpenBLAS
2. 或者从 [OpenBLAS 官网](https://www.openblas.net/) 下载预编译版本
3. 相应地调整环境配置文件

## 包管理与安装

### 更新包索引

```bash
moon update
```

### 安装包

```bash
moon add Kaida-Amethyst/openblas
```

## 项目配置

### 1. 包配置

在需要使用openblas的包内，配置 `moon.pkg.json`：

```json
{
  "import": [
    "Kaida-Amethyst/openblas/cblas",
    "Kaida-Amethyst/openblas/lapack"
  ],
  "link": {
    "native": {
      "cc": "$CC",
      "cc-flags": "$CC_FLAGS -w",
      "cc-link-flags": "$CC_LINK_FLAGS"
    }
  }
}
```

### 2. 环境配置

上面的`$CC`，`$CC_FLAGS`，`$CC_LINK_FLAGS`均为环境变量，建议配置`env.sh`。

```bash
# env.sh
OPENBLAS_PATH=$(brew --prefix openblas)  # macOS 示例

export CC=gcc
export CC_FLAGS="-I$OPENBLAS_PATH/include"
export CC_LINK_FLAGS="-L$OPENBLAS_PATH/lib -lopenblas"
export C_INCLUDE_PATH="$C_INCLUDE_PATH:/opt/homebrew/include:$OPENBLAS_PATH/include"
```

**Linux 用户请相应调整路径**:
```bash
# Linux 示例 (路径可能因发行版而异)
export CC=gcc
export CC_FLAGS="-I/usr/include/openblas"
export CC_LINK_FLAGS="-lopenblas"
```

在使用前记得加载环境变量：
```bash
source env.sh
```


## 使用示例

### CBLAS 示例 - 向量点积和矩阵乘法

以下是一个使用 CBLAS 进行基础线性代数运算的示例：

```moonbit
fn cblas_example() -> Unit {
  // 示例 1: 向量点积
  println("=== CBLAS 向量点积示例 ===")
  let n = 4
  let x : FixedArray[Float] = [1.0, 2.0, 3.0, 4.0]
  let y : FixedArray[Float] = [2.0, 1.0, 4.0, 3.0]
  
  // 计算 x · y = 1*2 + 2*1 + 3*4 + 4*3 = 28
  let dot_result = @cblas.cblas_sdot(n, x, 1, y, 1)
  println("向量点积结果: \{dot_result}")
  
  // 示例 2: 向量归一化
  println("\n=== 向量归一化示例 ===")
  let vec : FixedArray[Float] = [3.0, 4.0, 0.0]
  let norm = @cblas.cblas_snrm2(3, vec, 1)
  println("向量 [3, 4, 0] 的欧几里得范数: \{norm}")
  
  // 归一化向量 (除以范数)
  @cblas.cblas_sscal(3, 1.0 / norm, vec, 1)
  println("归一化后的向量: [\{vec[0]}, \{vec[1]}, \{vec[2]}]")
  
  // 示例 3: 矩阵向量乘法 (GEMV)
  println("\n=== 矩阵向量乘法示例 ===")
  let m = 3  // 矩阵行数
  let n = 3  // 矩阵列数
  let alpha : Float = 1.0
  let beta : Float = 0.0
  
  // 3x3 矩阵 A (行主序存储)
  let a : FixedArray[Float] = [
    1.0, 2.0, 3.0,  // 第一行
    4.0, 5.0, 6.0,  // 第二行
    7.0, 8.0, 9.0   // 第三行
  ]
  let input_vec : FixedArray[Float] = [1.0, 1.0, 1.0]
  let result_vec : FixedArray[Float] = [0.0, 0.0, 0.0]
  
  // 计算 result_vec = A * input_vec
  @cblas.cblas_sgemv(
    @cblas.CblasRowMajor, @cblas.CblasNoTrans,
    m, n, alpha, a, n, input_vec, 1, beta, result_vec, 1
  )
  
  println("矩阵 A * 向量 [1,1,1] = [\{result_vec[0]}, \{result_vec[1]}, \{result_vec[2]}]")
  
  // 示例 4: 向量加法 (AXPY: y = a*x + y)
  println("\n=== 向量加法示例 (AXPY) ===")
  let alpha2 : Float = 2.5
  let x2 : FixedArray[Float] = [1.0, 2.0, 3.0]
  let y2 : FixedArray[Float] = [4.0, 5.0, 6.0]
  
  println("操作前: x = [\{x2[0]}, \{x2[1]}, \{x2[2]}], y = [\{y2[0]}, \{y2[1]}, \{y2[2]}]")
  
  // 计算 y = 2.5 * x + y
  @cblas.cblas_saxpy(3, alpha2, x2, 1, y2, 1)
  
  println("y = 2.5 * x + y = [\{y2[0]}, \{y2[1]}, \{y2[2]}]")
}

fn main {
  cblas_example()
}
```

### LAPACK 示例 - 线性方程组求解和特征值分解

以下是一个使用 LAPACK 进行高级线性代数运算的示例：

```moonbit
fn lapack_example() -> Unit {
  // 示例 1: 求解线性方程组 Ax = b (使用LU分解)
  println("=== LAPACK 线性方程组求解示例 ===")
  
  let n = 3  // 矩阵维数
  let nrhs = 1  // 右端向量个数
  
  // 3x3 系数矩阵 A (列主序存储，LAPACK 要求)
  let a : FixedArray[Double] = [
    2.0, 1.0, 1.0,  // 第一列
    1.0, 3.0, 2.0,  // 第二列  
    1.0, 2.0, 4.0   // 第三列
  ]
  
  // 右端向量 b
  let b : FixedArray[Double] = [8.0, 13.0, 18.0]
  
  // LU 分解的行置换数组
  let ipiv : FixedArray[Int] = [0, 0, 0]
  
  println("求解方程组:")
  println("2x + y + z = 8")
  println("x + 3y + 2z = 13")  
  println("x + 2y + 4z = 18")
  
  // 调用 LAPACK 的 DGESV 函数求解
  let info = @lapack.lapacke_dgesv(
    @lapack.LAPACK_COL_MAJOR, n, nrhs, a, n, ipiv, b, n
  )
  
  if info == 0 {
    println("解为: x = \{b[0]}, y = \{b[1]}, z = \{b[2]}")
  } else {
    println("求解失败, 错误代码: \{info}")
  }
  
  // 示例 2: 矩阵的QR分解
  println("\n=== QR 分解示例 ===")
  
  let m2 = 3
  let n2 = 3
  
  // 待分解的矩阵 (列主序)
  let a2 : FixedArray[Double] = [
    1.0, 1.0, 1.0,  // 第一列
    2.0, 1.0, 0.0,  // 第二列
    3.0, 2.0, 1.0   // 第三列
  ]
  
  // 存储 Householder 反射向量的数组
  let tau : FixedArray[Double] = [0.0, 0.0, 0.0]
  
  println("对矩阵进行 QR 分解:")
  println("A = [[1, 2, 3], [1, 1, 2], [1, 0, 1]]")
  
  // 调用 DGEQRF 进行 QR 分解
  let info2 = @lapack.lapacke_dgeqrf(
    @lapack.LAPACK_COL_MAJOR, m2, n2, a2, m2, tau
  )
  
  if info2 == 0 {
    println("QR 分解成功完成")
    println("R 矩阵的对角元素: [\{a2[0]}, \{a2[4]}, \{a2[8]}]")
  } else {
    println("QR 分解失败, 错误代码: \{info2}")
  }
  
  // 示例 3: 计算矩阵的奇异值分解 (SVD)
  println("\n=== 奇异值分解示例 ===")
  
  let m3 = 2
  let n3 = 3
  
  // 2x3 矩阵
  let a3 : FixedArray[Double] = [
    1.0, 3.0,     // 第一列
    2.0, 4.0,     // 第二列  
    3.0, 5.0      // 第三列
  ]
  
  // 奇异值数组
  let s : FixedArray[Double] = [0.0, 0.0]
  
  // U 和 VT 矩阵 (这里简化处理，不提取)
  let u : FixedArray[Double] = []
  let vt : FixedArray[Double] = []
  
  println("计算矩阵 [[1, 2, 3], [3, 4, 5]] 的奇异值")
  
  // 调用 DGESVD 计算奇异值
  let info3 = @lapack.lapacke_dgesvd(
    @lapack.LAPACK_COL_MAJOR, 'N'.to_int().to_byte(), 'N'.to_int().to_byte(),
    m3, n3, a3, m3, s, u, m3, vt, n3
  )
  
  if info3 == 0 {
    println("奇异值: [\{s[0]}, \{s[1]}]")
  } else {
    println("SVD 计算失败, 错误代码: \{info3}")
  }
}

fn main {
  lapack_example()
}
```

## API 参考

### CBLAS 模块

本库提供了完整的 CBLAS Level 1、Level 2 和 Level 3 函数，包括：

- **Level 1**: 向量运算 (点积、范数、旋转等)
- **Level 2**: 矩阵-向量运算 (GEMV、SYR 等)  
- **Level 3**: 矩阵-矩阵运算 (GEMM、SYRK 等)

支持单精度 (`s`前缀) 和双精度 (`d`前缀) 浮点运算，以及复数运算 (`c`前缀单精度复数，`z`前缀双精度复数)。

### LAPACK 模块

本库提供了 LAPACK 的核心功能，包括：

- **线性系统求解**: GESV、POSV、SYSV 等
- **矩阵分解**: LU、QR、Cholesky、奇异值分解等
- **特征值问题**: SYEV、GEEV 等
- **最小二乘问题**: GELS 等

## 许可证

本项目采用 Apache-2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 相关链接

- [OpenBLAS 官网](https://www.openblas.net/)
- [MoonBit 语言](https://www.moonbitlang.com/)
- [项目仓库](https://github.com/moonbit-community/openblas.mbt)
- [问题反馈](https://github.com/moonbit-community/openblas.mbt/issues)
