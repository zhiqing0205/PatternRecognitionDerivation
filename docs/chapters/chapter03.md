# 3. 最大似然估计和贝叶斯参数估计

## PCA（主成分分析）数学推导

### 1. 问题定义

PCA的目标是找到一个投影方向 $\mathbf{w}$，使得数据在该方向上投影后的方差最大。

设有 $n$ 个 $d$ 维数据点：$\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$

### 2. 数据中心化

首先对数据进行中心化处理：
$$\tilde{\mathbf{x}}_i = \mathbf{x}_i - \boldsymbol{\mu}$$

其中样本均值为：
$$\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i$$

### 3. 投影方差最大化

数据在单位向量 $\mathbf{w}$（$\|\mathbf{w}\|=1$）方向上的投影为：
$$y_i = \mathbf{w}^T\tilde{\mathbf{x}}_i$$

投影后数据的方差为：
$$\text{Var}(y) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$$

由于数据已中心化，$\bar{y} = 0$，因此：
$$\text{Var}(y) = \frac{1}{n}\sum_{i=1}^{n}y_i^2 = \frac{1}{n}\sum_{i=1}^{n}(\mathbf{w}^T\tilde{\mathbf{x}}_i)^2$$

### 4. 目标函数

$$\text{Var}(y) = \frac{1}{n}\sum_{i=1}^{n}\mathbf{w}^T\tilde{\mathbf{x}}_i\tilde{\mathbf{x}}_i^T\mathbf{w} = \mathbf{w}^T\left(\frac{1}{n}\sum_{i=1}^{n}\tilde{\mathbf{x}}_i\tilde{\mathbf{x}}_i^T\right)\mathbf{w}$$

定义协方差矩阵：
$$\mathbf{S} = \frac{1}{n}\sum_{i=1}^{n}\tilde{\mathbf{x}}_i\tilde{\mathbf{x}}_i^T$$

因此目标函数为：
$$\max_{\mathbf{w}} \mathbf{w}^T\mathbf{S}\mathbf{w} \quad \text{subject to } \|\mathbf{w}\|^2 = 1$$

### 5. 拉格朗日乘子法求解

构建拉格朗日函数：
$$L(\mathbf{w}, \lambda) = \mathbf{w}^T\mathbf{S}\mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)$$

对 $\mathbf{w}$ 求偏导并令其为零：
$$\frac{\partial L}{\partial \mathbf{w}} = 2\mathbf{S}\mathbf{w} - 2\lambda\mathbf{w} = 0$$

得到特征值方程：
$$\mathbf{S}\mathbf{w} = \lambda\mathbf{w}$$

### 6. 解的性质

- $\mathbf{w}$ 是协方差矩阵 $\mathbf{S}$ 的特征向量
- $\lambda$ 是对应的特征值，等于投影方差：$\mathbf{w}^T\mathbf{S}\mathbf{w} = \lambda$
- 为最大化方差，选择最大特征值对应的特征向量
- 前 $k$ 个主成分对应前 $k$ 个最大特征值的特征向量

## LDA（线性判别分析）数学推导

### 1. 问题定义

LDA的目标是找到一个投影方向 $\mathbf{w}$，使得投影后类间距离最大、类内距离最小。

设有 $C$ 个类别，第 $i$ 类有 $n_i$ 个样本，总样本数 $n = \sum_{i=1}^{C} n_i$

### 2. 类内散布矩阵

第 $i$ 类的均值：
$$\boldsymbol{\mu}_i = \frac{1}{n_i}\sum_{\mathbf{x} \in \omega_i}\mathbf{x}$$

第 $i$ 类的散布矩阵：
$$\mathbf{S}_i = \sum_{\mathbf{x} \in \omega_i}(\mathbf{x} - \boldsymbol{\mu}_i)(\mathbf{x} - \boldsymbol{\mu}_i)^T$$

类内散布矩阵：
$$\mathbf{S}_W = \sum_{i=1}^{C}\mathbf{S}_i = \sum_{i=1}^{C}\sum_{\mathbf{x} \in \omega_i}(\mathbf{x} - \boldsymbol{\mu}_i)(\mathbf{x} - \boldsymbol{\mu}_i)^T$$

### 3. 类间散布矩阵

总体均值：
$$\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i = \frac{1}{n}\sum_{i=1}^{C}n_i\boldsymbol{\mu}_i$$

类间散布矩阵：
$$\mathbf{S}_B = \sum_{i=1}^{C}n_i(\boldsymbol{\mu}_i - \boldsymbol{\mu})(\boldsymbol{\mu}_i - \boldsymbol{\mu})^T$$

### 4. Fisher判别准则

在投影方向 $\mathbf{w}$ 上：
- 投影后第 $i$ 类均值：$\tilde{\mu}_i = \mathbf{w}^T\boldsymbol{\mu}_i$
- 投影后类内散布：$\tilde{s}_W^2 = \mathbf{w}^T\mathbf{S}_W\mathbf{w}$
- 投影后类间散布：$\tilde{s}_B^2 = \mathbf{w}^T\mathbf{S}_B\mathbf{w}$

Fisher判别准则（以二分类为例）：
$$J(\mathbf{w}) = \frac{(\tilde{\mu}_1 - \tilde{\mu}_2)^2}{\tilde{s}_1^2 + \tilde{s}_2^2} = \frac{\mathbf{w}^T\mathbf{S}_B\mathbf{w}}{\mathbf{w}^T\mathbf{S}_W\mathbf{w}}$$

### 5. 广义特征值问题

目标是最大化：
$$J(\mathbf{w}) = \frac{\mathbf{w}^T\mathbf{S}_B\mathbf{w}}{\mathbf{w}^T\mathbf{S}_W\mathbf{w}}$$

对 $\mathbf{w}$ 求导并令其为零：
$$\frac{\partial J}{\partial \mathbf{w}} = \frac{2\mathbf{S}_B\mathbf{w}(\mathbf{w}^T\mathbf{S}_W\mathbf{w}) - 2\mathbf{S}_W\mathbf{w}(\mathbf{w}^T\mathbf{S}_B\mathbf{w})}{(\mathbf{w}^T\mathbf{S}_W\mathbf{w})^2} = 0$$

简化得到广义特征值方程：
$$\mathbf{S}_B\mathbf{w} = \lambda\mathbf{S}_W\mathbf{w}$$

等价于：
$$\mathbf{S}_W^{-1}\mathbf{S}_B\mathbf{w} = \lambda\mathbf{w}$$

### 6. 解的性质

- 最优投影方向 $\mathbf{w}$ 是 $\mathbf{S}_W^{-1}\mathbf{S}_B$ 的特征向量
- 对应的特征值 $\lambda$ 等于Fisher判别准则的值
- 对于 $C$ 类问题，最多有 $C-1$ 个有意义的判别方向
- 选择最大的几个特征值对应的特征向量作为投影方向

## 多变量高斯分布的最大似然估计推导

多变量高斯分布在模式识别中应用广泛，其参数的最大似然估计是基础且重要的内容。我们分两种情况进行推导。

### 情况一：均值已知，协方差矩阵未知

设 $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$ 是来自 $d$ 维多变量高斯分布 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 的独立同分布样本，其中均值 $\boldsymbol{\mu}$ 已知，需要估计协方差矩阵 $\boldsymbol{\Sigma}$。

#### 1. 似然函数

多变量高斯分布的概率密度函数为：
$$p(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$

似然函数为：
$$L(\boldsymbol{\Sigma}) = \prod_{i=1}^{n} p(\mathbf{x}_i|\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

#### 2. 对数似然函数

$$\ln L(\boldsymbol{\Sigma}) = \sum_{i=1}^{n} \ln p(\mathbf{x}_i|\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

$$= -\frac{nd}{2}\ln(2\pi) - \frac{n}{2}\ln|\boldsymbol{\Sigma}| - \frac{1}{2}\sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}_i - \boldsymbol{\mu})$$

#### 3. 求导与最大化

利用矩阵求导公式：
- $\frac{\partial \ln|\boldsymbol{\Sigma}|}{\partial \boldsymbol{\Sigma}} = (\boldsymbol{\Sigma}^{-1})^T = \boldsymbol{\Sigma}^{-1}$（因为 $\boldsymbol{\Sigma}$ 对称）
- $\frac{\partial \text{tr}(\mathbf{A}\boldsymbol{\Sigma}^{-1})}{\partial \boldsymbol{\Sigma}} = -\boldsymbol{\Sigma}^{-1}\mathbf{A}\boldsymbol{\Sigma}^{-1}$

注意到：
$$\sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}_i - \boldsymbol{\mu}) = \text{tr}\left(\boldsymbol{\Sigma}^{-1}\sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T\right)$$

设 $\mathbf{S} = \sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T$，则：

$$\frac{\partial \ln L(\boldsymbol{\Sigma})}{\partial \boldsymbol{\Sigma}} = -\frac{n}{2}\boldsymbol{\Sigma}^{-1} + \frac{1}{2}\boldsymbol{\Sigma}^{-1}\mathbf{S}\boldsymbol{\Sigma}^{-1} = 0$$

#### 4. 最大似然估计解

从上式得到：
$$n\mathbf{I} = \mathbf{S}\boldsymbol{\Sigma}^{-1}$$

因此：
$$\hat{\boldsymbol{\Sigma}}_{ML} = \frac{1}{n}\sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T$$

### 情况二：均值和协方差矩阵均未知

当均值 $\boldsymbol{\mu}$ 和协方差矩阵 $\boldsymbol{\Sigma}$ 都未知时，需要同时估计这两个参数。

#### 1. 对数似然函数

$$\ln L(\boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\frac{nd}{2}\ln(2\pi) - \frac{n}{2}\ln|\boldsymbol{\Sigma}| - \frac{1}{2}\sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}_i - \boldsymbol{\mu})$$

#### 2. 对均值求导

$$\frac{\partial \ln L}{\partial \boldsymbol{\mu}} = \sum_{i=1}^{n}\boldsymbol{\Sigma}^{-1}(\mathbf{x}_i - \boldsymbol{\mu}) = 0$$

得到：
$$\hat{\boldsymbol{\mu}}_{ML} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i$$

#### 3. 对协方差矩阵求导

将估计的均值代入，设 $\mathbf{S} = \sum_{i=1}^{n}(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})^T$，类似于情况一的推导：

$$\frac{\partial \ln L}{\partial \boldsymbol{\Sigma}} = -\frac{n}{2}\boldsymbol{\Sigma}^{-1} + \frac{1}{2}\boldsymbol{\Sigma}^{-1}\mathbf{S}\boldsymbol{\Sigma}^{-1} = 0$$

#### 4. 最大似然估计解

$$\hat{\boldsymbol{\mu}}_{ML} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i$$

$$\hat{\boldsymbol{\Sigma}}_{ML} = \frac{1}{n}\sum_{i=1}^{n}(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})^T$$

## 问题：最大似然估计得到的结果是有偏的还是无偏的，二者差距多少，哪一个是更好的结果？

### 1. 均值估计的无偏性

**结论**：均值的最大似然估计是**无偏的**。

**证明**：
$$E[\hat{\boldsymbol{\mu}}_{ML}] = E\left[\frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i\right] = \frac{1}{n}\sum_{i=1}^{n}E[\mathbf{x}_i] = \frac{1}{n}\sum_{i=1}^{n}\boldsymbol{\mu} = \boldsymbol{\mu}$$

因此 $\hat{\boldsymbol{\mu}}_{ML}$ 是 $\boldsymbol{\mu}$ 的无偏估计。

### 2. 协方差矩阵估计的有偏性

**结论**：协方差矩阵的最大似然估计是**有偏的**。

**分析**：考虑情况二中的协方差矩阵估计：
$$\hat{\boldsymbol{\Sigma}}_{ML} = \frac{1}{n}\sum_{i=1}^{n}(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})^T$$

#### 有偏性证明

对于标量情况，我们知道：
$$E\left[\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2\right] = \frac{n-1}{n}\sigma^2$$

推广到多变量情况：
$$E[\hat{\boldsymbol{\Sigma}}_{ML}] = \frac{n-1}{n}\boldsymbol{\Sigma}$$

**证明思路**：
- 当使用样本均值 $\hat{\boldsymbol{\mu}}_{ML}$ 而非真实均值 $\boldsymbol{\mu}$ 时，样本偏差的平方和会系统性地偏小
- 这是因为样本均值是使平方和最小的点，导致低估了真实的方差

#### 偏差量化

- **偏差**：$\text{Bias}[\hat{\boldsymbol{\Sigma}}_{ML}] = E[\hat{\boldsymbol{\Sigma}}_{ML}] - \boldsymbol{\Sigma} = -\frac{1}{n}\boldsymbol{\Sigma}$
- **相对偏差**：$\frac{1}{n}$，随样本量增加而减小

### 3. 无偏估计

为得到协方差矩阵的无偏估计，使用**贝塞尔修正**：

$$\hat{\boldsymbol{\Sigma}}_{无偏} = \frac{1}{n-1}\sum_{i=1}^{n}(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})^T$$

**验证无偏性**：
$$E[\hat{\boldsymbol{\Sigma}}_{无偏}] = \frac{1}{n-1} \cdot \frac{n-1}{n} \cdot n \cdot \boldsymbol{\Sigma} = \boldsymbol{\Sigma}$$

### 4. 比较与选择

**有偏估计 vs 无偏估计**：

| 特性 | 最大似然估计（有偏） | 无偏估计 |
|------|---------------------|----------|
| **偏差** | $-\frac{1}{n}\boldsymbol{\Sigma}$ | $0$ |
| **方差** | 较小 | 较大 |
| **均方误差** | 小样本时可能更小 | 大样本时更优 |
| **实际应用** | 机器学习中常用 | 统计推断中常用 |

**哪个更好？**

- **大样本情况**（$n$ 很大）：两者差别很小，$\frac{1}{n} \approx 0$，都是渐近无偏的
- **小样本情况**：
  - 如果关注**无偏性**（如统计推断），选择无偏估计
  - 如果关注**预测精度**（如机器学习），最大似然估计的较小方差可能更有价值
- **实践建议**：
  - 统计分析：使用无偏估计（除以 $n-1$）
  - 机器学习：使用最大似然估计（除以 $n$），因为偏差在大数据下可忽略，且计算简单

### 5. 渐近性质

当 $n \to \infty$ 时：
- 两种估计都收敛到真实值：$\hat{\boldsymbol{\Sigma}}_{ML} \to \boldsymbol{\Sigma}$，$\hat{\boldsymbol{\Sigma}}_{无偏} \to \boldsymbol{\Sigma}$
- 最大似然估计具有渐近正态性和渐近效率性
- 在大样本下，最大似然估计是"最优"的