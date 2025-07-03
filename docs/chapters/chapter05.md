# 5. 线性判别函数

## 支持向量机(SVM)的数学推导

支持向量机是一种基于统计学习理论的机器学习方法，其核心思想是寻找一个最优超平面，使得不同类别之间的间隔最大化。

### 1. 硬间隔SVM的数学表述

#### 1.1 问题设定

设训练样本集为 $\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$，其中：
- $\mathbf{x}_i \in \mathbb{R}^d$ 是 $d$ 维特征向量
- $y_i \in \{-1, +1\}$ 是类别标签

假设数据线性可分，我们要找到一个超平面 $\mathbf{w}^T\mathbf{x} + b = 0$ 将两类数据分开。

#### 1.2 几何间隔与函数间隔

**函数间隔**：点 $(\mathbf{x}_i, y_i)$ 到超平面的函数间隔定义为：
$$\hat{\gamma}_i = y_i(\mathbf{w}^T\mathbf{x}_i + b)$$

**几何间隔**：点 $(\mathbf{x}_i, y_i)$ 到超平面的几何间隔定义为：
$$\gamma_i = \frac{y_i(\mathbf{w}^T\mathbf{x}_i + b)}{\|\mathbf{w}\|}$$

数据集关于超平面的几何间隔为：
$$\gamma = \min_{i=1,\ldots,n} \gamma_i$$

#### 1.3 最大间隔问题的原始形式

SVM的目标是最大化几何间隔，即：

$$\begin{align}
\max_{\mathbf{w}, b} &\quad \gamma \\
\text{s.t.} &\quad \frac{y_i(\mathbf{w}^T\mathbf{x}_i + b)}{\|\mathbf{w}\|} \geq \gamma, \quad i = 1, 2, \ldots, n
\end{align}$$

为了简化计算，我们可以固定函数间隔 $\hat{\gamma} = 1$（通过重新缩放 $\mathbf{w}$ 和 $b$ 实现），原问题等价于：

$$\begin{align}
\min_{\mathbf{w}, b} &\quad \frac{1}{2}\|\mathbf{w}\|^2 \\
\text{s.t.} &\quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad i = 1, 2, \ldots, n
\end{align}$$

### 2. 拉格朗日对偶方法求解

#### 2.1 拉格朗日函数

引入拉格朗日乘子 $\alpha_i \geq 0$，构造拉格朗日函数：

$$L(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^{n} \alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1]$$

其中 $\boldsymbol{\alpha} = (\alpha_1, \alpha_2, \ldots, \alpha_n)^T$。

#### 2.2 KKT条件

根据KKT(Karush-Kuhn-Tucker)条件，最优解 $(\mathbf{w}^*, b^*, \boldsymbol{\alpha}^*)$ 必须满足：

1. **平稳性条件**：
   $$\nabla_{\mathbf{w}} L = \mathbf{w} - \sum_{i=1}^{n} \alpha_i y_i \mathbf{x}_i = 0$$
   $$\frac{\partial L}{\partial b} = -\sum_{i=1}^{n} \alpha_i y_i = 0$$

2. **原始可行性**：
   $$y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 \geq 0, \quad i = 1, 2, \ldots, n$$

3. **对偶可行性**：
   $$\alpha_i \geq 0, \quad i = 1, 2, \ldots, n$$

4. **互补松弛条件**：
   $$\alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1] = 0, \quad i = 1, 2, \ldots, n$$

#### 2.3 对偶问题推导

从平稳性条件得到：
$$\mathbf{w}^* = \sum_{i=1}^{n} \alpha_i y_i \mathbf{x}_i$$
$$\sum_{i=1}^{n} \alpha_i y_i = 0$$

将这些条件代入拉格朗日函数，消除 $\mathbf{w}$ 和 $b$，得到对偶问题：

$$\begin{align}
\max_{\boldsymbol{\alpha}} &\quad W(\boldsymbol{\alpha}) = \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j \\
\text{s.t.} &\quad \sum_{i=1}^{n} \alpha_i y_i = 0 \\
&\quad \alpha_i \geq 0, \quad i = 1, 2, \ldots, n
\end{align}$$

#### 2.4 支持向量的确定

从互补松弛条件可知：
- 如果 $\alpha_i > 0$，则 $y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$，样本 $\mathbf{x}_i$ 是**支持向量**
- 如果 $\alpha_i = 0$，则 $y_i(\mathbf{w}^T\mathbf{x}_i + b) > 1$，样本 $\mathbf{x}_i$ 不是支持向量

### 3. 软间隔SVM

#### 3.1 引入松弛变量

当数据不完全线性可分时，引入松弛变量 $\xi_i \geq 0$：

$$\begin{align}
\min_{\mathbf{w}, b, \boldsymbol{\xi}} &\quad \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n} \xi_i \\
\text{s.t.} &\quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \quad i = 1, 2, \ldots, n \\
&\quad \xi_i \geq 0, \quad i = 1, 2, \ldots, n
\end{align}$$

其中 $C > 0$ 是正则化参数，控制对错误分类的惩罚程度。

#### 3.2 软间隔SVM的对偶问题

构造拉格朗日函数：
$$L(\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu}) = \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n} \xi_i - \sum_{i=1}^{n} \alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 + \xi_i] - \sum_{i=1}^{n} \mu_i \xi_i$$

通过KKT条件求解，得到软间隔SVM的对偶问题：

$$\begin{align}
\max_{\boldsymbol{\alpha}} &\quad W(\boldsymbol{\alpha}) = \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j \\
\text{s.t.} &\quad \sum_{i=1}^{n} \alpha_i y_i = 0 \\
&\quad 0 \leq \alpha_i \leq C, \quad i = 1, 2, \ldots, n
\end{align}$$

与硬间隔SVM相比，唯一的区别是约束条件从 $\alpha_i \geq 0$ 变为 $0 \leq \alpha_i \leq C$。

### 4. KKT求解的具体例子

#### 4.1 简单二维例子

考虑一个简单的二维线性可分问题：

**训练数据**：
- 正类：$\mathbf{x}_1 = (3, 3)^T$，$y_1 = +1$
- 正类：$\mathbf{x}_2 = (4, 3)^T$，$y_2 = +1$  
- 负类：$\mathbf{x}_3 = (1, 1)^T$，$y_3 = -1$

#### 4.2 对偶问题求解

对偶问题为：
$$\begin{align}
\max_{\boldsymbol{\alpha}} &\quad W(\boldsymbol{\alpha}) = \alpha_1 + \alpha_2 + \alpha_3 - \frac{1}{2}[\alpha_1^2(\mathbf{x}_1^T\mathbf{x}_1) + \alpha_2^2(\mathbf{x}_2^T\mathbf{x}_2) + \alpha_3^2(\mathbf{x}_3^T\mathbf{x}_3) \\
&\quad + 2\alpha_1\alpha_2 y_1 y_2(\mathbf{x}_1^T\mathbf{x}_2) + 2\alpha_1\alpha_3 y_1 y_3(\mathbf{x}_1^T\mathbf{x}_3) + 2\alpha_2\alpha_3 y_2 y_3(\mathbf{x}_2^T\mathbf{x}_3)] \\
\text{s.t.} &\quad \alpha_1 + \alpha_2 - \alpha_3 = 0 \\
&\quad \alpha_i \geq 0, \quad i = 1, 2, 3
\end{align}$$

**计算内积**：
- $\mathbf{x}_1^T\mathbf{x}_1 = 3^2 + 3^2 = 18$
- $\mathbf{x}_2^T\mathbf{x}_2 = 4^2 + 3^2 = 25$
- $\mathbf{x}_3^T\mathbf{x}_3 = 1^2 + 1^2 = 2$
- $\mathbf{x}_1^T\mathbf{x}_2 = 3 \times 4 + 3 \times 3 = 21$
- $\mathbf{x}_1^T\mathbf{x}_3 = 3 \times 1 + 3 \times 1 = 6$
- $\mathbf{x}_2^T\mathbf{x}_3 = 4 \times 1 + 3 \times 1 = 7$

**目标函数**：
$$W(\boldsymbol{\alpha}) = \alpha_1 + \alpha_2 + \alpha_3 - \frac{1}{2}[18\alpha_1^2 + 25\alpha_2^2 + 2\alpha_3^2 + 42\alpha_1\alpha_2 - 12\alpha_1\alpha_3 - 14\alpha_2\alpha_3]$$

#### 4.3 利用约束条件简化

由约束条件 $\alpha_1 + \alpha_2 - \alpha_3 = 0$，得 $\alpha_3 = \alpha_1 + \alpha_2$。

代入目标函数并化简：
$$W(\alpha_1, \alpha_2) = 2\alpha_1 + 2\alpha_2 - \frac{1}{2}[37\alpha_1^2 + 54\alpha_2^2 + 70\alpha_1\alpha_2]$$

#### 4.4 求解最优解

对 $\alpha_1$ 和 $\alpha_2$ 求偏导并令其为零：
$$\frac{\partial W}{\partial \alpha_1} = 2 - 37\alpha_1 - 35\alpha_2 = 0$$
$$\frac{\partial W}{\partial \alpha_2} = 2 - 54\alpha_2 - 35\alpha_1 = 0$$

解得：
$$\alpha_1 = \frac{2}{37}, \quad \alpha_2 = \frac{2}{54} = \frac{1}{27}, \quad \alpha_3 = \frac{2}{37} + \frac{1}{27} = \frac{91}{999}$$

#### 4.5 计算最优参数

**权重向量**：
$$\mathbf{w}^* = \sum_{i=1}^{3} \alpha_i y_i \mathbf{x}_i = \frac{2}{37}(3, 3)^T + \frac{1}{27}(4, 3)^T - \frac{91}{999}(1, 1)^T$$

**偏置项**：利用支持向量上的约束条件 $y_i(\mathbf{w}^{*T}\mathbf{x}_i + b^*) = 1$。

#### 4.6 KKT条件验证

最优解必须满足所有KKT条件：

1. **平稳性**：$\mathbf{w}^* = \sum_{i=1}^{3} \alpha_i^* y_i \mathbf{x}_i$ ✓
2. **原始可行性**：$y_i(\mathbf{w}^{*T}\mathbf{x}_i + b^*) \geq 1$ 对所有 $i$ 成立 ✓
3. **对偶可行性**：$\alpha_i^* \geq 0$ 对所有 $i$ 成立 ✓  
4. **互补松弛**：由于所有 $\alpha_i^* > 0$，所有样本都是支持向量，满足 $y_i(\mathbf{w}^{*T}\mathbf{x}_i + b^*) = 1$ ✓

### 5. 决策函数

求解完成后，SVM的决策函数为：
$$f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^{n} \alpha_i^* y_i \mathbf{x}_i^T\mathbf{x} + b^*\right)$$

只有支持向量（$\alpha_i > 0$ 的样本）对决策函数有贡献，这体现了SVM的稀疏性特点。

### 6. 核化SVM

对于非线性问题，通过核函数 $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)$ 将原始特征空间映射到高维特征空间，对偶问题变为：

$$\begin{align}
\max_{\boldsymbol{\alpha}} &\quad W(\boldsymbol{\alpha}) = \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \\
\text{s.t.} &\quad \sum_{i=1}^{n} \alpha_i y_i = 0 \\
&\quad 0 \leq \alpha_i \leq C, \quad i = 1, 2, \ldots, n
\end{align}$$

决策函数变为：
$$f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^{n} \alpha_i^* y_i K(\mathbf{x}_i, \mathbf{x}) + b^*\right)$$

KKT条件和求解方法保持不变，只是将内积 $\mathbf{x}_i^T\mathbf{x}_j$ 替换为核函数 $K(\mathbf{x}_i, \mathbf{x}_j)$。