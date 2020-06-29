# FCOS

## Inference

<img src="/Users/lizhiwei/Documents/paper_notes/obj_detection_anchor_free/FCOS/image-20200407113423085.png" alt="image-20200407113423085" style="zoom:40%;" />

## 

输入图像，经过网络forward计算，输出代表bbox的4D vector(l,t,r,b)和C个binary classifier。

## Train

![image-20200407113458460](/Users/lizhiwei/Documents/paper_notes/obj_detection_anchor_free/FCOS/image-20200407113458460.png)



<img src="/Users/lizhiwei/Documents/paper_notes/obj_detection_anchor_free/FCOS/image-20200407113819149.png" alt="image-20200407113819149" style="zoom:33%;" />

网络forward计算，在原有Feature Map上增加两个FeatureMap，通过FPN结构输出multi-level feature map, 最后分出两个branch 分别预测cls和回归bbox。预测cls用focal loss，回归bbox用IOU loss。





## 重点

- 在feature map上的每个点回归bbox，resulting high recall
- 通过FPN结构，设置不同level feature map的回归阈值（类似anchor bbox size)，过滤
- 设置center loss，过滤远离物体中心的bbox
- 提到了CenterNet，DenseBox的一个问题：如果同类物体的中心overlap，只能回归其中一个，会有漏检问题出现

