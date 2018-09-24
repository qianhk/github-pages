---
title: 客户端码农学习ML —— 分类评估(准确率_精确率和召回率_ROC和曲线下面积)
date: 2018-08-14 10:45:21
tags: [AI, 评估]
articleID: 客户端码农学习ML-分类评估(准确率_精确率和召回率_ROC和曲线下面积)
---

判断各种机器学习方法训练而成的模型输出结果的好坏，需要一些评估方法，本文简单介绍二分类算法的几种评估方法，做个总结。

首先需要有一些基本概念：对于样本来说，通常有两种类型，一种是正类别、一种是负类别，正负类别本身没有褒义贬义的含义，纯粹为了区分二分类两种不同情况。

如“有没有狼的问题”，可以认为狼来了是正类别，没有狼是负类别；再比如“肿瘤是恶性还是良性”，恶性可以作为正类型，良性作为负类别。

对于模型的预测，可以用2x2混淆矩阵来总结，该矩阵描述了所有可能出现的结果，共4种，以肿瘤问题为例：

<!--more-->

![ai_score_tp_tn_fp_fn](/images/ai_score_tp_tn_fp_fn.png)

（图表来自google官方机器学习教程，简单修改了下数值）

从图表中可以看出共有100个肿瘤，其中绿色背景的预测正确，共94个（含1个恶性、93个良性），红色背景的预测错误，共6个（实际含5个恶性、1个良性）。

下面给出4个符号的定义：

真正例TP是指模型将正类别样本正确地预测为正类别。同样，真负例TN是指模型将负类别样本正确地预测为负类别。

假正例FP是指模型将负类别样本错误地预测为正类别，而假负例FN是指模型将正类别样本错误地预测为负类别。

简单的说： TP = 真正例，TN = 真负例，FP = 假正例，FN = 假负例。

## 准确率 Accuracy

准确率通常是我们最容易想到的一个评估分类模型的指标。通俗来说，准确率是指我们的模型预测正确的结果所占的比例。其值等于预测正确的样本数除以总样本数，取值范围[0,1]。

套用上面4个符号的定义，可以形成如下公式：

** 【如果看到公式是乱七八糟的字符，请刷新下网页】 **

$$\text{准确率} = \frac{预测准确数}{样本总数} = \frac{TP+TN}{TP+TN+FP+FN}$$

因此对于上面的图表：

$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN} = \frac{1+93}{1+93+1+5} = 0.94$

94%的准确性一般来说还不错，但是对于本身分类不平衡的样本来说会有很严重的问题，对于本例来说就是恶性肿瘤数本身占比肿瘤整体很少，只有6%。如果我们写个模型总是预测肿瘤为良性，那么准确率是94个良性除以样本100，结果还是94%的预测准确率，由此可以推断准确率不是个靠谱的指标。

因此科学家们定义了两个能够更好地评估分类不平衡问题的指标：精确率和召回率。



$\text{精确率} = \frac{TP}{TP+FP} = \frac{1}{1+1} = 0.5$

$\text{召回率} = \frac{TP}{TP+FN} = \frac{1}{1+8} = 0.11$

$\text{Precision} = \frac{TP}{TP + FP} = \frac{8}{8+2} = 0.8$

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{8}{8 + 3} = 0.73$$


![ai_score_Accuracy](../images/ai_score_Accuracy.png)

## 精确率 Precision 和召回率 Recall

精确率是指：在被识别为正类别的样本中，确实为正类别的比例是多少？

$$\text{Precision} = \frac{TP}{TP+FP}$$

召回率是指：在所有正类别样本中，被正确识别为正类别的比例是多少？

![ai_score_Precision](../images/ai_score_Precision.png)

![ai_score_Recall](../images/ai_score_Recall.png)

![ai_score_F1_score](../images/ai_score_F1_score.png)

精确率指标尝试回答以下问题：

在被识别为正类别的样本中，确实为正类别的比例是多少？

召回率
召回率尝试回答以下问题：

在所有正类别样本中，被正确识别为正类别的比例是多少？

## ROC、曲线下面积

TPR      true positive rate   等同召回率Recall
FPR      false positive rate
ROC     Receiver Operating Characteristic Curve 接收者操作特征曲线
AUC     Area Under Curve score 曲线下面积值

ROC 曲线（接收者操作特征曲线）是一种显示分类模型在所有分类阈值下的效果的图表。该曲线绘制了真正例率、假正例率两个参数。

![ai_score_FPR](../images/ai_score_FPR.png)

![ai_score_TPR](../images/ai_score_FPR.png)

![ai_score_roc_origin_data](../images/ai_score_roc_origin_data.png)

![ai_score_roc1](../images/ai_score_roc1.png)

![ai_score_roc2](../images/ai_score_roc2.png)

## 参考：

[准确率](https://developers.google.cn/machine-learning/crash-course/classification/accuracy)

[精确率和召回率](https://developers.google.cn/machine-learning/crash-course/classification/precision-and-recall)

[ROC 和曲线下面积](https://developers.google.cn/machine-learning/crash-course/classification/roc-and-auc)

## 　

本文首发于[钱凯凯的博客](http://qianhk.com) : http://qianhk.com/2018/08/分类评估(准确率_精确率和召回率_ROC和曲线下面积)/


