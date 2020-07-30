# What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

> Yarin Gal  University of Cambridge  yg279@cam.ac.uk
>
> Alex Kendall University of Cambridge agk34@cam.ac.uk

## 简介

![image-20200730191542498](https://tva1.sinaimg.cn/large/007S8ZIlly1gh97vw2yndj31520kadm2.jpg)

我们把uncertainty分为两种：aleatoric uncertainty(i.e. data uncertainty) 和 epidemic uncertainty(i.e. model uncertainty)。从上图可以看出

- data unceratinty主要出现在物体边缘和远处，该uncertainty源于标注员对物体边缘标注的精度误差和远处物体较差的成像质量

- epidemic uncertainty主要出现在model预测不好的地方，比如右下角的图，模型对人行道的分割结果较差，所以uncertainty比较高

  

## 本文贡献

![image-20200730105539691](https://tva1.sinaimg.cn/large/007S8ZIlly1gh97vwr8tjj31bs0ek78n.jpg)





## Epistemic Uncertainty

- Dropout variational inference
  - classification

    $$p(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{e^x+1}$$

    

    $$\mathcal{L}(\theta, p) = -\frac{1}{N}\sum_{i=1}^{N}{\log{p(y_i|f^{\widehat{W_i}}(x_i))}} + \frac{1-p}{2N}||\theta||^2$$

    其中，N是数据点数，p是dropout probability， $\widehat{W}_i \sim q_{\theta}^*(W)$，其中，$\theta$是simple distribution's parameters to be optim

    

    Epidemic Uncertainty approximation：

    $$p(y=c|x,X,Y)\approx\frac{1}{T}\sum_{t=1}^{T}Softmax(f^{\widehat{W}_i}(x))$$

    The uncertainty of probability vector **p** : $H(p) = -\sum_{c=1}^{C}p_c\log(p_c)$

  - regression

    $$-\log{p(y_i|f^{\widehat{W_i}}(x_i))} \propto \frac{1}{2\sigma^2}||y_i-f^{\widehat{W_i}}(x_i)||^2 + \frac{1}{2}\log{\sigma^2}$$

    Epidemic Uncertainty:

    $$Var(y)\approx\sigma^2+\frac{1}{T}\sum_{t=1}^{T}f^{\widehat{W_t}}(x)^Tf^{\widehat{W_t}}(x_t) - {E(y)^T}{E(y)}$$ 

    approximationn predictive mean: $E(y) = \frac{1}{T}\sum_{t=1}^{T}f^{\widehat{W}_t}(x)$

    $\sigma^2$ : the amount of noise inherent in the data 

  

##  Aleatoric Uncertainty

- Homoscedastic Aleatoric Uncertainty

  > Homoscedastic regression assumes constant observation noise σ for every input point x.

  对任何的输入，都具有的随机不确定性，与input data无关

- Heteroscedastic Aleatoric Uncertainty

  与输入数据有关的随机不确定性。比如，在深度估计任务中，一张图像是颜色一样的一面墙，另一张是包含vanishing lines的图像，前者的uncertainty应该比后者高



##  TODO

- 结合fata和model uncertainty的方法
- 实验



## Reference

[机器之心：如何创造可信任的机器学习模型？先要理解不确定性](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650755237&idx=3&sn=55beb3edcef0bb4ded4b56e1379efbda&scene=0#wechat_redirect) 推荐指数\****

[AI科技评论：学界 | 模型可解释性差？你考虑了各种不确定性了吗？](https://mp.weixin.qq.com/s?__biz=MzA5ODEzMjIyMA==&mid=2247496311&idx=3&sn=3e7f1df007926e6fba1124630046be76&source=41#wechat_redirect)

[CSDN:What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? 计算机视觉用于贝叶斯深度学习的不确定性]([https://blog.csdn.net/weixin_39779106/article/details/78968982#1%E5%B0%86%E5%BC%82%E6%96%B9%E5%B7%AE%E5%81%B6%E7%84%B6%E4%B8%8D%E7%A1%AE%E5%AE%9A%E6%80%A7%E5%92%8C%E8%AE%A4%E7%9F%A5%E4%B8%8D%E7%A1%AE%E5%AE%9A%E6%80%A7%E7%9B%B8%E7%BB%93%E5%90%88](https://blog.csdn.net/weixin_39779106/article/details/78968982#1将异方差偶然不确定性和认知不确定性相结合))

[Uncertainties in Bayesian Deep Learning - kid丶的文章 - 知乎]( https://zhuanlan.zhihu.com/p/100998668)

[Homoscedastic regression贝叶斯神经网络建模两类不确定性——NIPS2017 - Dannis的文章 - 知乎](https://zhuanlan.zhihu.com/p/88654038)

