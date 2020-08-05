# Note_Gaussian YOLOv3

## why ？

YOLOv3的输出包含3部分：objectness info, category info, bbox。objectness和category都是通过softmax/sigmoid classification得到的，有对应score可以用来建模uncertainty(值得指出的是，classification score不能直接用来作为uncertainty，需要calibration或者用其他方法建模)，但是bbox是regression得到的，没有对应的score来反应reliability of bbox。

YOLOv3 bbox输出回顾：

- bbox输出为$b_x, b_y, b_w, b_h$,  蓝色框是预测结果

- $(c_x, c_y )$是当前grid左上角角点的坐标，$\sigma(t_x), \sigma(t_y)$是prior box中心点相对于左上角角点的坐标

- $p_w, p_h$ 是prior box的宽和高

  > 有个问题，grid是啥？是现有grid再有prior box的吧，

  

<img src="/Users/lizhiwei/Documents/paper_notes/active_learning/image-20200805193456177.png" alt="image-20200805193456177" style="zoom:40%;" />

## what？

### Gaussian modeling

<img src="/Users/lizhiwei/Documents/paper_notes/active_learning/image-20200805153518552.png" alt="image-20200805153518552" style="zoom:50%;" />

在bbox regression中，总共需要回归4个变量($t_x, t_y, t_w, t_h$)。我们可以这4个变量的gaussian model来计算每个bbox的uncertainty。具体来说，输入x，对每个变量y，我们有
$$
p(y|x) = N(y;\mu(x), \sum(x))
$$
其中$\mu(x)$和$\sum(x)$是均值和方差。

为了输出bbox的uncertainty，我们对每个bbox变量进行gaussian modeling，即模型输出bbox时不直接输出x, y, w, h，而是输出

$\hat{\mu}_{t_x}, \hat{\sum}_{t_x}, \hat{\mu}_{t_y}, \hat{\sum}_{t_y}, \hat{\mu}_{t_w}, \hat{\sum}_{t_w},  \hat{\mu}_{t_h}, \hat{\sum}_{t_h}$。 考虑YOLOv3的输出层（是为了归一化？），我们用sigmoid处理$t_x, t_y, t_w, t_h$
$$
\mu_{t_x} = \sigma(\hat{\mu}_{t_x}), \mu_{t_y} = \sigma(\hat{\mu}_{t_y}), \mu_{t_w} = \hat{\mu}_{t_w}, \mu_{t_h} = \hat{\mu}_{t_h}
$$

$$
{\sum}_{t_x}= \sigma(\hat{\sum}_{t_x}), 
{\sum}_{t_y}= \sigma(\hat{\sum}_{t_y}) \\
{\sum}_{t_w}= \sigma(\hat{\sum}_{t_w}),
{\sum}_{t_h}= \sigma(\hat{\sum}_{t_h}),
$$

$$
\sigma(x) = \frac{1}{1+exp^{(-x)}}
$$

其中均值就是最后bbox的坐标，方差就是bbox的uncertainty。需要注意的是，$t_x, t_y$必须是bbox的中心，所以我们用sigmoid function来归一化，对$t_x, t_y, t_w, t_h$的variance我们也用sigmoid处理（归一化到0-1之间方便建模uncertainty，如何计算最后的uncertainty？？）。但是由于$t_w, t_h$是通过prior box + offset得到的，我们不用sigmoid处理他们（因为他们可正可负）。

<img src="/Users/lizhiwei/Documents/paper_notes/active_learning/image-20200805160928497.png" alt="image-20200805160928497" style="zoom:50%;" />

计算量

yolov3：99x10^9 Flops, gaussian yolov3 : 99.04x10^9 Flops 增加了4%的计算量



### Reconstruction of loss function

对bbox，yolov3使用sum square loss，对objectness和category，使用cross_entroy loss。

因为bbox的坐标已经用gaussian modeling，loss也应该改为negtive log likelihood（NLL） loss：
$$
L_x = -\sum_{i=1}^W\sum_{j=1}^H\sum_{k=1}^K\gamma_{ijk}\log(
N(x_{ijk}^G|\mu_{t_x}(x_{ijk}), {\sum}_{t_x}(x_{ijk}))
+\epsilon)
$$
$L_x$是x的NLL loss，对$y,w,h$同样可以计算$L_y,L_w,L_h$。其中，W,H是feature map的宽和高，K是anchors数量。更进一步地，$\mu_{t_x}(x_{ijk})$就是最后bbox中心点横坐标$t_x$，是在$(i,j)$这个位置的第k个ancho被offset修正后得到的。${\sum}_{t_x}(x_{ijk})$

也是网络输出值，代表$t_x$这个坐标的uncertainty。