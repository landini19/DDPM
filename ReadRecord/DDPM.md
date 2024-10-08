# DDPM文章阅读

首先明确工作的目标：利用杂乱的噪声图像生成一张高质量图像。作者使用了一种全新的模型--Diffusion Model来完成该目标。Difussion Model: 有限时间内，使用变分推断方法训练参数化的马尔科夫链，来生成符合原数据的样本。

## Background

对网络训练与预测过程进行了数学建模。噪声图像($x_T$)与高质量图像($x_0$)之间的变化关系，可以想象成如图1所示。对$x_T$而言，可以视作是不断向$x_0$中添加Gauss noise得到的。整个添加过程视作一个Markov chain，因为添加噪声是在上一张图像的基础上进行，只与上一个过程有关。那么，其逆过程就可以视作不断添加相反方向的Gauss noise，最终还原为清晰图像。

因此训练过程就可以分为两部分，Noise和Denoise。从Data中随机采样得到$x_0$，对其添加Gauss noise得到$x_T$，把$x_T$投入Denoise Network中得到$D(x_T)$，将其与$x_0$比较并不断优化。这是一个很直观的思路。为了落实这个思路，我们需要搞清3个大问题：

    (1) 如何进行Noise过程？
    (2) 如何进行Denoise过程？
    (3) 如何评价结果并进行优化？

### Forward Process

Background中首先给出了第一个问题的回答。我们把Noise过程记作$q(x_t|x_{t-1})$，由于假定是添加Gauss noise，因此可以写出解析解。

$$
q(x_t|x_{t-1})=\mathcal{N}(x_{t};\sqrt{1-\beta_t}x_{t-1},\beta_t I)
$$

需要说明的是这里的均值和方差是文章的假设，$\beta_t$属于超参数。在此基础上，我们可以直接通过$x_0$求得t时刻的噪声图像$x_t$。这需要进行一些数学推导，中间涉及重参数化技巧(引入$\epsilon\thicksim\mathcal{N}(0,I)$)以及正态分布的性质。直接放结果：

$$
q(x_t|x_0)=\mathcal{N}(x_t;\sqrt{\overline{\alpha}_t}x_0,(1-\overline{\alpha}_t)I) \\

\overline{\alpha}_t=\prod_{i=1}^t \alpha_i \\

\alpha_t=1-\beta_t
$$

公式告诉我们，第一步Noise过程可以极大化简，一步到位。

### Reverse Process I

紧接着，文章说明当前向过程(Noise)中的超参数$\beta_t$很小时，noise与denoise过程具有相同的函数形式，即高斯分布(!!!这个结论文章并未给出证明)。因此，我们知道了Denoise过程实际上也是一个Gauss noise过程，不过相比于超参数$\beta_t$，Denoise是需要Network拟合参数的，即Gauss Distribution的mean and variance。我们把Denoise过程记作$p_{\theta}(x_{t-1}|x_t)$。

$$
p_{\theta}(x_{t-1}|x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))
$$

到目前为止，还是比较迷惑的，train的过程是怎样的，Network output出来的是什么东西，我们只是理清楚了forward process过程。逆向过程暂时搁置一旁。

### Loss Function

回顾最开始的操作，我们从高质量图像集合Data中Sample出一张$x_0$，可以记作$p_{data}(x_0)$，而denoise过程，可以记作$p_{\theta}(x_0)$，我们的希望是让两个分布尽可能相似。使用KL散度来衡量。

$$\begin{aligned}
D_{KL}(P_{data}||P_\theta) &= \int p_{data}(x)log\frac{p_{data}(x)}{p_\theta(x)}dx \\
&=\int p_{data}(x)log[p_{data}(x)]dx - \int p_{data}(x)log[p_\theta(x)]dx
\end{aligned}$$

事实上，KL散度的值与Data无关，因此尽可能相似实际上变为优化问题：

$$\begin{aligned}
   argmin_{\theta}D_{KL}(P_{data}||P_\theta) &= argmax_{\theta}\int p_{data}(x)log[p_\theta(x)]dx \\
   &\approx argmax_{\theta} \sum p_{data}(x_i)log[p_\theta(x_i)] \\
   &\approx argmax_{\theta} \sum log[p_\theta(x_i)]
\end{aligned}
$$

对于每一次输入$x_0$来说，我们希望输出结果能够满足$argmax_{\theta}log[p_\theta(x_0)]$。现在来看，train的目标直观了一些，让概率对数值取最大，上一节中我们推断概率分布同样满足高斯分布，这样一个简单的概率分布模型的优化，似乎能推导出一个更简便的解析表达式。



## Related Question

1. 什么是后验概率，这与贝叶斯定理有什么关系？
   
   通过观测值反推事件发生的概率，叫做后验概率。我们通过观测，推断出可能的事件分布，也就是后验分布，这是通过贝叶斯定理计算得到的。因此，后验分布也可叫做贝叶斯分布。

2. 变分推断(Variational Inference, VI)是什么？
   
   变分推断(后续称之为VI)是贝叶斯近似推断中的一大类方法。由于贝叶斯公式

   $$
   p(z|x)=\frac{p(z)p(x|z)}{p(x)}
   $$

   往往得不到一个解析表达式(例如高维demision下p(x)往往是一个不可解析的多重积分)，因此需要将后验问题转化为其他问题求解。变分推断将后验问题转化为优化问题求解。

3. 潜变量模型(Talent Variable Model)是什么？
   
   一种基于简单分布构建复杂概率模型的框架。简单框架，例如高斯分布这一类的指数型概率，我们都可使用极大似然估计方法转化为优化问题并得到解析解。但事实上模型可能有着复杂的函数形式，仅通过极大似然法无法求出解析解，这个时候需要使用其他方法。潜变量模型通过混叠简单模型去近似复杂模型。

   $$
   p(x;\theta)=\int p(x,z;\theta)dz=\int p(x|z;\theta)p(z;\theta)dz
   $$

   潜变量模型是通过对某个联合分布进行边缘化得到的一个简单概率模型。被边缘化的变量就是潜变量。

4. 为什么在Markov chain中，forward process进行的diffusion足够微小时，reverse process和forward process具有相同的函数形式？
   
   原文中并未找到答案。

5. 什么是KL散度(Kullback-Leibler divergence)？
   
   度量两个概率分布差异程度的工具。散度越大，差异越大；反之差异越小。

   $$
   D_{KL}(P||Q)=\int p(x)log\frac{p(x)}{q(x)}dx
   $$

   KL divergence是非负的，当且仅当$p=q$时取0。通常来说用p表示真实的分布，q为预测分布。