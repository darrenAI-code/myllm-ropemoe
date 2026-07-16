在 RoPE（旋转位置编码）中，一个维度为 $d$ 的特征向量（通常是 Query 或 Key）并不是在单一的高维空间中整体旋转，而是两两分组，拆分到 $\frac{d}{2}$ 个独立的二维平面中分别进行旋转。
对于位置 $m$ 上的 Token，其每个维度上的旋转矩阵有“分块二维矩阵”和“整体稀疏矩阵”两种等价表达方式方式。以下是详细的公式和结构：
------------------------------
## 1. 核心参数：每个平面的旋转步长 $\theta_i$
在开始构造矩阵前，需要先计算出每个二维平面独立拥有的“基础旋转步长”（也叫频率）。
假设向量的总维度为 $d$，第 $i$ 个平面（$i = 0, 1, \dots, \frac{d}{2}-1$）的频率公式为：
$$\theta_i = 10000^{-\frac{2i}{d}}$$ 
随着维度 $i$ 的增大，$\theta_i$ 变得越来越小，意味着高维平面的旋转速度会变慢。
------------------------------
## 2. 独立二维平面上的旋转矩阵 $R_i(m)$
在位置 $m$ 处，第 $i$ 个二维平面所对应的旋转角度为 $m\theta_i$。该平面对应的标准 $2 \times 2$ 旋转矩阵为：
$$R_i(m) = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$ 
这个 $2 \times 2$ 矩阵会直接作用于该 Token 向量的第 $2i$ 和第 $2i+1$ 两个维度上。
------------------------------
## 3. 整个向量的整体旋转矩阵 $\boldsymbol{R}_{\Theta, m}^d$
为了能一举对整个 $d$ 维向量进行变换，我们需要把这 $\frac{d}{2}$ 个二维旋转矩阵组合成一个 $d \times d$ 的大矩阵。
## 表达方式 A：分块对角矩阵 (Block Diagonal Matrix)
将所有的 $R_i(m)$ 沿着主对角线依次排列，其余位置全部补 $0$。其标准的矩阵表达结构如下：
$$\boldsymbol{R}_{\Theta, m}^d = \begin{pmatrix} \begin{pmatrix} \cos(m\theta_0) & -\sin(m\theta_0) \\ \sin(m\theta_0) & \cos(m\theta_0) \end{pmatrix} & \boldsymbol{0} & \cdots & \boldsymbol{0} \\ \boldsymbol{0} & \begin{pmatrix} \cos(m\theta_1) & -\sin(m\theta_1) \\ \sin(m\theta_1) & \cos(m\theta_1) \end{pmatrix} & \cdots & \boldsymbol{0} \\ \vdots & \vdots & \ddots & \vdots \\ \boldsymbol{0} & \boldsymbol{0} & \cdots & \begin{pmatrix} \cos(m\theta_{\frac{d}{2}-1}) & -\sin(m\theta_{\frac{d}{2}-1}) \\ \sin(m\theta_{\frac{d}{2}-1}) & \cos(m\theta_{\frac{d}{2}-1}) \end{pmatrix} \end{pmatrix}$$ 
当这个 $d \times d$ 的矩阵与 Token 向量 $\boldsymbol{x} = [x_0, x_1, x_2, x_3, \dots, x_{d-1}]^T$ 相乘时，它会完美地将每一对相邻特征在各自的平面内旋转 $m\theta_i$ 度。
------------------------------
## 4. 工程实现中的等价变换公式
在实际的 Transformers 库（如 Hugging Face、FlashAttention）代码实现中，由于 $d \times d$ 的稀疏矩阵乘法极为消耗算力，工程师们通常不会真的去创建这个大矩阵。
他们通常会采用逐元素相乘（Element-wise Product, $\odot$）的等价公式来代替矩阵乘法，以极大地提升计算效率。对于向量 $\boldsymbol{x}$：
$$\boldsymbol{R}_{\Theta, m}^d \boldsymbol{x} = \begin{pmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \\ \vdots \\ x_{d-2} \\ x_{d-1} \end{pmatrix} \odot \begin{pmatrix} \cos(m\theta_0) \\ \cos(m\theta_0) \\ \cos(m\theta_1) \\ \cos(m\theta_1) \\ \vdots \\ \cos(m\theta_{\frac{d}{2}-1}) \\ \cos(m\theta_{\frac{d}{2}-1}) \end{pmatrix} + \begin{pmatrix} -x_1 \\ x_0 \\ -x_3 \\ x_2 \\ \vdots \\ -x_{d-1} \\ x_{d-2} \end{pmatrix} \odot \begin{pmatrix} \sin(m\theta_0) \\ \sin(m\theta_0) \\ \sin(m\theta_1) \\ \sin(m\theta_1) \\ \vdots \\ \sin(m\theta_{\frac{d}{2}-1}) \\ \sin(m\theta_{\frac{d}{2}-1}) \end{pmatrix}$$ 
注：右侧第二项中，向量 $\boldsymbol{x}$ 的相邻两维被两两交换了位置，且奇数维取了相反数（即 $[-x_1, x_0, -x_3, x_2, \dots]$），这在代码中通常被命名为 rotate_half 操作。
------------------------------
## ✅ 结论
在位置 $m$ 处，RoPE 作用于特征维度的矩阵是一个由多个不同旋转角度的 $2 \times 2$ 矩阵组合而成的 $d \times d$ 分块对角正交矩阵 [1]。每一个平面的具体旋转角度由 $m \cdot 10000^{-\frac{2i}{d}}$ 精确决定。
如果对代码实现感兴趣，我们可以进一步看看 PyTorch 中 rotate_half 函数的几行核心源码 是如何高效还原这个矩阵计算的；或者聊聊为什么这种两两分组的旋转能神奇地在点积中保留相对位置信息？想要了解哪部分可以随时告诉我。

