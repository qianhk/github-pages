---
title: 客户端码农学习ML —— 逻辑回归分类算法
date: 2018-06-19 16:18:36
tags: [AI, Logistic]
articleID: 客户端码农学习ML-逻辑回归分类算法
---

## 分类问题

在线性回归中，预测的是连续值，而在分类问题中，预测的是离散值，预测的结果是特征属于哪个类别以及概率，比如是否垃圾邮件、肿瘤是良性还是恶性、根据花瓣大小判断哪种花，一般从最简单的二元分类开始，通常将一种类别表示为1，另一种类别表示为0。

如下图，分别是几种不同的分类样式：

![ai_logistic_sample_3_kind](/images/ai_logistic_sample_3_kind.jpg)

<!--more-->

## 分类方法

如果我们用线性回归算法来解决一个分类问题，那么假设函数的输出值可能远大于1，或者远小于0，会导致吴恩达机器学习教程中提到的分类不准或者阈值难以选择问题。

![ai_logistic_linear_regression_problem](/images/ai_logistic_linear_regression_problem.jpg)

数学家们脑袋一转，想出来著名的sigmoid function，又名S形函数，是机器学习中诸多算法常用的一种激活函数，其图形如：

![ai_sigmoid_function](/images/ai_sigmoid_function.png)

输出始终在0-1之间，完美应用在二分类问题，同时还能给出预测的概率。

将线性回归的输出应用sigmoid函数后，即得到逻辑回归的模型函数，又名假设函数Hypothesis:

** 【如果看到公式是乱七八糟的字符，请刷新下网页】 **

$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e ^ {-\theta^T x}}$$

<!--![ai_logistic_hypothesis](/images/ai_logistic_hypothesis.png)-->

## 损失函数

在线性回归中，一般采用样本误差的平方和来计算损失，批量训练中求均方误差MSE或者均方根误差RMSE，此方法的损失函数是凸函数，我们通过遍历待定系数w画图形或者数学计算可证明。凸函数便于通过梯度下降法求出最优解。在一个特征下损失函数形如抛物线, 如下图中左边的子图:

![ai_loss_linear_rmse_logistic_rmse_log](/images/ai_loss_linear_rmse_logistic_rmse_log.jpg)

在加了sigmoid函数后，继续采用平方和来计算损失，得到的损失函数将不是个凸函数，难以求出全局最优解，图形绘制如上图中间的子图。

数学家们脑袋又一转，又想到了对数损失函数。

$$Cost(h_\theta(x),y)= \begin{cases}
-log(h_\theta(x)) & \text{if $y$ = 1} \\\\
-log(1-h_\theta(x)) & \text{if $y$ = 0} \\\\
\end{cases}$$

<!--![ai_logistic_loss_log_function](/images/ai_logistic_loss_log_function.png)-->

当y=1时，图形形如下图中的左图，如果预测正确，损失为0，如果预测错误，损失无穷大。当y=0时，同样如果预测正确，损失为0，如果预测错误，损失无穷大。

![ai_logistic_loss_log_shape](/images/ai_logistic_loss_log_shape.jpg)

写成一个完整的函数后变成：

$$Cost(h_\theta(x),y) = -y_ilog(h_\theta(x))-(1-y_i)log(1-h_\theta(x))$$


<!--![ai_logistic_loss_log_function_full](/images/ai_logistic_loss_log_function_full.png)-->

这个复杂的函数对于自变量θ是一个凸函数，画出的图形可看上面绿色图形中右边的子图，数学证明可求证二阶导数非负，可参考：[http://sofasofa.io/forum_main_post.php?postid=1000921](http://sofasofa.io/forum_main_post.php?postid=1000921)

## 代码实现分类

### 绘制损失函数图形

就是上面的绿色线条的图，首先准备数据和工具方法

```
import numpy as np
import matplotlib.pyplot as plt

data = [(1, 0), (2, 0), (10, 1)]

var_x = np.array([x for x, y in data])
var_y = np.array([y for x, y in data])

def sigmoid(z):
    exp = np_exp(-z)
    return 1.0 / (1 + exp)

def np_exp(array):
    return np.exp(np.minimum(array, 700))

def np_log(array):
    return np.log(np.maximum(array, 1e-250))
```

定义计算代码，实现多种损失函数计算，为便于绘制图形，假设b是固定值，只考虑w1系数变化，代码中的w参数等同于上述原理公式中的θ

```
loss_type = 3

def calc_loss(_b=0, _w1=0):
    result_mul = np.multiply(var_x, _w1)
    result_add = result_mul + _b
    result_sigmoid = sigmoid(result_add)

    if loss_type == 1 or loss_type == 2:
        if loss_type == 1:
            loss_array = np.square(result_add - var_y)  # linear regression square loss
        else:
            loss_array = np.square(result_sigmoid - var_y)  # logistic regression square loss
        return np.sqrt(np.mean(loss_array))
    elif loss_type == 3:
        first = np.multiply(-var_y, np_log(result_sigmoid))  # logistic regression log loss
        second = (1 - var_y) * np_log(1 - result_sigmoid)
        loss_array = first - second
        return np.average(loss_array)
    else:
        loss_array = np.maximum(result_add, 0) - result_add * var_y + np.log(1 + np.exp(-np.abs(result_add)))
        return np.average(loss_array)
```

绘制出图形

```
if __name__ == '__main__':
    loss_vec = []

    max_n = 1
    b = 0
    test_range = np.arange(-max_n, max_n, 0.01)
    for step in test_range:
        loss = calc_loss(b, step)
        loss_vec.append(loss)

    plt.figure()
    plt.title('loss type: ' + str(loss_type))
    plt.plot(test_range, loss_vec, 'g-')
    plt.show()
```

### 实现线性可分样本分类

即第一张图左边样本的分类，先给出完成效果看看：

![ai_logistic_train_result_blobs](/images/ai_logistic_train_result_blobs.png)


导入库，生成mock数据，并定义h(θ)

```
import numpy as np
import tensorflow as tf
from sklearn import datasets

random_state = np.random.RandomState(2)
data, target = datasets.make_blobs(n_samples=123, n_features=2, centers=2, cluster_std=1.5, random_state=random_state)
target = np.array(target, dtype=np.float32)
# print('data=%s' % data)
# print('target=%s' % target)

b = tf.Variable(0, dtype=tf.float32, name='b')
w1 = tf.Variable([[0]], dtype=tf.float32, name='w1')
w2 = tf.Variable([[0]], dtype=tf.float32, name='w2')

x_data1 = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x_data2 = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

result_matmul1 = tf.matmul(x_data1, w1)
result_matmul2 = tf.matmul(x_data2, w2)
result_add = result_matmul1 + result_matmul2 + b
```

定义优化器、训练并打出最优超参数看看。

```
use_base_method = 3

if use_base_method == 1:
    # result_sigmoid = 1.0 / (1 + tf.exp(tf.clip_by_value(-result_add, -1e250, 500)))
    # first = tf.multiply(-y_target, tf.log(tf.clip_by_value(result_sigmoid, 1e-250, 1.0)))
    # second = tf.multiply(1 - y_target, tf.log(tf.clip_by_value(1 - result_sigmoid, 1e-250, 1.0)))
    # 自己写的方法貌似由于计算有溢出，通常得不到正确的解，有高手知道如何写法的请指导指导
    result_sigmoid = tf.sigmoid(result_add)
    first = -y_target * tf.log_sigmoid(result_add)
    second = (1 - y_target) * tf.log(1 - result_sigmoid)
    loss = first - second
elif use_base_method == 2:
    x = result_add
    z = y_target
    loss = tf.maximum(x, 0) - x * z + tf.log(1 + tf.exp(-tf.abs(x)))
else:
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=result_add, labels=y_target)

loss = tf.reduce_mean(loss)

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

var_x1 = np.array([x[0] for i, x in enumerate(data)])
var_x2 = np.array([x[1] for i, x in enumerate(data)])

data_amount = len(var_x1)
batch_size = 20

loss_vec = []

for step in range(2001):
    rand_index = np.random.choice(data_amount, size=batch_size)
    tmp1 = var_x1[rand_index]
    tmp2 = [tmp1]
    x1 = np.transpose(tmp2)
    x2 = np.transpose([var_x2[rand_index]])
    y = np.transpose([target[rand_index]])
    sess.run(train, feed_dict={x_data1: x1, x_data2: x2, y_target: y})
    if step % 200 == 0:
        loss_value = sess.run(loss, feed_dict={x_data1: x1, x_data2: x2, y_target: y})
        loss_vec.append(loss_value)
        print('step=%d w1=%s w2=%s b=%s loss=%s' % (
            step, sess.run(w1)[0, 0], sess.run(w2)[0, 0], sess.run(b), loss_value))

[[_w1]] = sess.run(w1)
[[_w2]] = sess.run(w2)
_b = sess.run(b)
print('last W1=%f W2=%f B=%f' % (_w1, _w2, _b))
```

数据分类并模仿google的[A Neural Network playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.06451&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)绘制分类背景：

```
class1_x = [x[0] for i, x in enumerate(data) if target[i] == 1]
class1_y = [x[1] for i, x in enumerate(data) if target[i] == 1]
class2_x = [x[0] for i, x in enumerate(data) if target[i] != 1]
class2_y = [x[1] for i, x in enumerate(data) if target[i] != 1]

visualization_frame, _ = kai.make_visualization_frame(class1_x, class1_y, class2_x, class2_y)
series_x1 = visualization_frame['x1']
series_x2 = visualization_frame['x2']
x1 = np.transpose([series_x1])
x2 = np.transpose([series_x2])
visual_probabilities = sess.run(result_sigmoid, feed_dict={x_data1: x1, x_data2: x2}).T[0]
visualization_frame['probabilities'] = visual_probabilities

sess.close()

kai.show_visualization_data(class1_x, class1_y, class2_x, class2_y
                            , loss_vec
                            , target, probabilities
                            , 'blobs kai linear classifier'
                            , visualization_frame)

```

工具方法如下：除绘制图形外，顺便输出了预测精度。

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

def make_visualization_frame(class1_x, class1_y, class2_x, class2_y):
    min_x = min(min(class1_x), min(class2_x))
    min_y = min(min(class1_y), min(class2_y))
    max_x = max(max(class1_x), max(class2_x))
    max_y = max(max(class1_y), max(class2_y))
    n = int(max(max_x - min_x, max_y - min_y) * 3)
    xs = np.linspace(min_x, max_x, n)
    ys = np.linspace(min_y, max_y, n)
    X1, X2 = np.meshgrid(xs, ys)
    frame = pd.DataFrame()
    frame['x1'] = np.reshape(X1, n * n)
    frame['x2'] = np.reshape(X2, n * n)
    return frame, np.zeros(n * n)


def show_visualization_data(class1_x, class1_y, class2_x, class2_y
                            , log_losses
                            , target_series, probabilities,
                            title=None, visualization_frame=None):
    plt.figure(figsize=(10, 8))

    if title is not None:
        plt.title(title)

    ax = plt.subplot(121)
    if visualization_frame is not None:
        show_predict_probability(ax, visualization_frame)

    ax.scatter(class1_x, class1_y, c='r', marker='o')
    ax.scatter(class2_x, class2_y, c='b', marker='x')

    if log_losses is not None:
        ax = plt.subplot(222)
        ax.plot(log_losses, color='m', linewidth=1)

    if target_series is not None and probabilities is not None:
        ax = plt.subplot(224)
        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
            target_series, probabilities)
        ax.plot(false_positive_rate, true_positive_rate, c='c', label="our model")
        ax.plot([0, 1], [0, 1], 'y:', label="random classifier")
        accuracy = np.equal(target_series, np.round(probabilities)).astype(np.float32).mean()
        print('\n accuracy=%.4f%%' % (accuracy * 100))

    plt.show()


def show_predict_probability(ax, frame):
    x1 = frame['x1']
    x2 = frame['x2']
    probability = frame['probabilities']
    class1_x = [x1[i] for i, x in enumerate(probability) if x >= 0.5]
    class1_y = [x2[i] for i, x in enumerate(probability) if x >= 0.5]
    class2_x = [x1[i] for i, x in enumerate(probability) if x < 0.5]
    class2_y = [x2[i] for i, x in enumerate(probability) if x < 0.5]
    ax.scatter(class1_x, class1_y, c='r', alpha=0.2, marker='s')
    ax.scatter(class2_x, class2_y, c='b', alpha=0.2, marker='s')

```

### 实现圆形分类的样本分类

相对于上述线性可分样本，主要有两点不同，一是样本生成，二是h(θ):

```
# 样本生成：
data, target = datasets.make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=random_state)
```

```
# h(θ) result_matmul1 2的值改为平方后再乘以w系数
result_matmul1 = tf.matmul(x_data1 ** 2, w1)
result_matmul2 = tf.matmul(x_data2 ** 2, w2)
result_add = result_matmul1 + result_matmul2 + b
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=result_add, labels=y_target)

```

效果预览如下图：

![ai_logistic_train_result_circle](/images/ai_logistic_train_result_circle.png)

### 实现不规则分类的样本的分类

对于下图，一般不能立即看出什么假设函数能对应（数学高手或者有经验的除外），此时先用大杀器多项式多次试验，多项式写的够全、试验次数够多总能找到相对合适的模型。

```
# 同样样本生成：
data, target = datasets.make_moons(200, noise=0.10, random_state=random_state)
```

```
# 大量超参数定义
b = tf.Variable(0, dtype=tf.float32)
w1 = tf.Variable([[0]], dtype=tf.float32)
w2 = tf.Variable([[0]], dtype=tf.float32)
w3 = tf.Variable([[0]], dtype=tf.float32)
w4 = tf.Variable([[0]], dtype=tf.float32)
w5 = tf.Variable([[0]], dtype=tf.float32)
w6 = tf.Variable([[0]], dtype=tf.float32)

# h(θ) 加了3次方
result_1 = tf.matmul(x_data1, w1)
result_2 = tf.matmul(x_data2, w2)
result_3 = tf.matmul(x_data1 ** 2, w3)
result_4 = tf.matmul(x_data2 ** 2, w4)
result_5 = x_data1 ** 3 * w5
result_6 = x_data2 ** 3 * w6
result_add = b + result_1 + result_2 + result_3 + result_4 + result_5 + result_6
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=result_add, labels=y_target)
```

效果预览如下图：

![ai_logistic_train_result_moons](/images/ai_logistic_train_result_moons.png)

## 参考：

[http://sofasofa.io/forum_main_post.php?postid=1000921](http://sofasofa.io/forum_main_post.php?postid=1000921)

## 　

本文首发于[钱凯凯的博客](http://qianhk.com) : http://qianhk.com/2018/06/客户端码农学习ML-逻辑回归分类算法/


