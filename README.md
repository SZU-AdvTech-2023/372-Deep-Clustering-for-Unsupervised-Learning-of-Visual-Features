# 基于不确定性感知深度聚类的图像特征自监督学
现过程中引用参考了caron2018发布的基于聚类的自监督方法代码，该方法是基础版本的深度聚类模型，（没有我们改进的特征融合以及不确定性度量模块），在Baseline中，关于Backbone的搭建，以及通过聚类进行高级特征的
提取，聚类，并最终使用高级特征的误差反传都进行了定义。为了实现特征融合，我们首先对主函数中的特征抽取\(compute_features(dataloader, model, N) \)函数进行
了重新定义。可以看到，与原代码相比，整个特征抽取模块都进行了改写，改写之后的函数能够对网络王的低级特征和高级特征都进行捕获，除此之外，我们还将低级特征融合与低级特征进行和融合拼接。原始代码使用的是高级特征进行聚类，但是在我们提出的新方法中，我们使用了融合特征进行聚类。

此外，我们借助了模糊聚类的思想定义了伪类标的不确定性，并将不确定性引入到模型的分类训练中。为了实现这一功能，我们对整个 \(clustering.py\) 文件都进行了重写，重写后的文件命名为  \(clustering.py\)，而原
来的文件被保留为  \(clustering0.py\)。这样做的目的是为了尽可能简化代码量，避免一些不必要的代码工作量。我
们对原始  \(clustering.py\) 的函数都进行了详细分析和学习，发现了一个非常重要的细节。特别是在  \(faiss\) 提供
的聚类库函数中，聚类的结果不仅包括了样本对应的簇标签，还包括了样本与簇心的距离。这一发现对我们的研究至关重要，因为
它意味着我们无需重新计算样本与簇心的距离，从而大大简化了计算过程。
	
我们进一步利用了这个距离信息，将其直接用于伪类标不确定性的计算中。具体而言，我们采用了基于距离的不确定性评
估方法。这种方法的优点在于它直接利用了 \(faiss\)聚类结果中已有的距离数据，从而避免了额外的计算负担。
	
为了更好地融合传统聚类方法和不确定性评估，我们在  \(clustering.py\) 中实现了一个新的功能，它能够在聚
类过程中同时计算每个样本的簇标签和与簇心的距离，并据此计算不确定性。这种方法的关键在于能够为每个样本赋予
一个不确定性权重，这个权重随后用于模型的分类训练过程。我们认为，通过这种方式，模型能够更有效地从数据中学习，
特别是在处理具有高度不确定性的样本时。
	
我们的方法的核心思想是，通过对不确定性的量化评估和利用，我们能够更全面地利用无标签数据集的信息，从而提升自
监督学习方法的性能。这种方法的应用不仅限于图像聚类任务，还可以扩展到其他类型的无监督或自监督学习任务中，为深度
学习模型的训练提供了一个新的视角。

## 实验环境
实验环境搭建以及模型参数设置参考了caron2018deep。
- Python 安装版本 2.7
- SciPy 和 scikit-learn 包
- PyTorch 安装版本 0.1.8 (pytorch.org)
- CUDA 8.0
- Faiss 安装
- ImageNet 数据集

## 使用说明
首先我们要进行自监督学习的前置任务训练，即根据大量不带标签的数据进行低级特征以及高级特征的抽取、聚类、不确定性度量和误差反传，这里使用代码的话
就直接对\(main_promoted2.1.sh\)进行修改，把数据集组织成需要的形式，然后运行脚本就可以。训练好的模型被保存到参数指定的文件夹中。

然后我们可以将训练好的模型用到下游任务里面，因为时间以及数据集规模的原因，这里还使用了ImageNet图像分类作为下游任务来进行
下一步训练（ImageNet图像分类作为下游任务是已有工作验证自监督效果常用的方式之一）。具体来说，我们将前置任务训练的模型的特征提取部分作为的
下游任务的预训练模型。这里使用的话直接修改\(eval_linear.sh\)的参数就可以。

## 创新点
本文方法的创新点如下：
- 结合低级和高级特征的融合机制：UAD方法通过融合来自卷积神经网络不同层次的低级和高级特征，有效地结合了低级特征中的细节信息和高级特征中的抽象信息。这种融合策略不仅增强了模型对图像内容和细节的理解，还提高了整体特征表达的能力
- 利用模糊聚类的思想定义伪类标不确定性：UAD方法借鉴了模糊聚类的概念，创新性地定义了基于样本与其簇心距离的伪类标不确定性。这种方法允许模型在聚类过程中考虑每个样本的不确定性，从而更有效地处理数据。
- 将不确定性引入模型的分类训练：通过将伪类标不确定性引入到分类模型的训练过程中，UAD方法能够为不同样本赋予不同的权重。这样做可以提高模型在处理高度不确定性样


## 自监督监督学习的训练
无监督训练可以通过运行以下命令来启动:
```
$ ./main.sh
```
数据文件夹的路径设置：
```
DIR=/datasets01/imagenet_full_size/061417/train
```
要训练一个AlexNet网络，请指定`ARCH=alexnet`；而要训练一个VGG-16卷积网络，请使用`ARCH=vgg16`。

指定保存聚类日志和checkpoint的位置，使用：
```
EXP=exp
```
在训练过程中，模型每隔n次迭代会被保存一次（通过`--checkpoints`标志设置），可以在`${EXP}/checkpoints/checkpoint_0.pth.tar`等位置找到。
每个时代中群集的分配日志可以在pickle文件`${EXP}/clusters`中找到。

无监督训练代码`main.py`、`main_promoted2.0.py`、`mainpromoted2.1.py`的完整文档：
```
usage: main.py [-h] [--arch ARCH] [--sobel] [--clustering {Kmeans,PIC}]
               [--nmb_cluster NMB_CLUSTER] [--lr LR] [--wd WD]
               [--reassign REASSIGN] [--workers WORKERS] [--epochs EPOCHS]
               [--start_epoch START_EPOCH] [--batch BATCH]
               [--momentum MOMENTUM] [--resume PATH]
               [--checkpoints CHECKPOINTS] [--seed SEED] [--exp EXP]
               [--verbose]
               DIR

PyTorch Implementation of DeepCluster

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  CNN architecture (default: alexnet)
  --sobel               Sobel filtering
  --clustering {Kmeans,PIC}
                        clustering algorithm (default: Kmeans)
  --nmb_cluster NMB_CLUSTER, --k NMB_CLUSTER
                        number of cluster for k-means (default: 10000)
  --lr LR               learning rate (default: 0.05)
  --wd WD               weight decay pow (default: -5)
  --reassign REASSIGN   how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)
  --workers WORKERS     number of data loading workers (default: 4)
  --epochs EPOCHS       number of total epochs to run (default: 200)
  --start_epoch START_EPOCH
                        manual epoch number (useful on restarts) (default: 0)
  --batch BATCH         mini-batch size (default: 256)
  --momentum MOMENTUM   momentum (default: 0.9)
  --resume PATH         path to checkpoint (default: None)
  --checkpoints CHECKPOINTS
                        how many iterations between two checkpoints (default:
                        25000)
  --seed SEED           random seed (default: 31)
  --exp EXP             path to exp folder
  --verbose             chatty
```

## 设置下游分类任务进一步训练
可以使用以下命令进行下游任务训练：
```
$ ./eval_linear.sh
```
需要指定监督数据（ImageNet或Places）的路径:
```
DATA=/datasets01/imagenet_full_size/061417/
```
指定模型路径:
```
MODEL=/private/home/mathilde/deepcluster/checkpoint.pth.tar
```
指定迁移的最高层：
```
CONV=3
```
指定checkpoint的位置：
```
EXP=exp
```

下游任务模型训练完整阐述:
```
usage: eval_linear.py [-h] [--data DATA] [--model MODEL] [--conv {1,2,3,4,5}]
                      [--tencrops] [--exp EXP] [--workers WORKERS]
                      [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
                      [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY]
                      [--seed SEED] [--verbose]

Train linear classifier on top of frozen convolutional layers of an AlexNet.

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           path to dataset
  --model MODEL         path to model
  --conv {1,2,3,4,5}    on top of which convolutional layer train logistic
                        regression
  --tencrops            validation accuracy averaged over 10 crops
  --exp EXP             exp folder
  --workers WORKERS     number of data loading workers (default: 4)
  --epochs EPOCHS       number of total epochs to run (default: 90)
  --batch_size BATCH_SIZE
                        mini-batch size (default: 256)
  --lr LR               learning rate
  --momentum MOMENTUM   momentum (default: 0.9)
  --weight_decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        weight decay pow (default: -4)
  --seed SEED           random seed
  --verbose             chatty





