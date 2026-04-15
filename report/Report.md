## Place365 Images Classification Project Report



### Dataset

This project implements and compares multiple neuro network architectures for scene classification. The applying is a subset of `Places365` - `Places365 Mini Hard`.

The Places365 dataset is designed following principles of human visual cognition. It is good for training artificaial system of high-level visual understanding tasks. The dataset contains more than 10 million images comprising 400+ unique scene categories. The dataset features 5000 to 30,000 training images per class, consistent with real-world frequencies of occurrence.

`Places365 Mini Hard` is a challenging subset of the Places365 scene recognition dataset, containing 10 scene categories.

| Label | Class Name       |
|-------|-----------------|
| 0     | Boxing Stage     |
| 1     | Inside a Car     |
| 2     | Airplane Cockpit |
| 3     | Forest           |
| 4     | Outside a Car    |
| 5     | Conference Room  |
| 6     | Rehearsal Room   |
| 7     | Stage            |
| 8     | Performance      |
| 9     | Conference Hall  |

### Using Instructions

+ Step 1: Open `DownloadetRemote.ipynb`, running the code of first cell, download the dataset from the remote server and save them to local folder `./data/*`

+ Step 2: Run the following file to execute the detailed code for the four models.

| Model      | Notebook                   |
|------------|----------------------------|
| SimpleCNN  | CNNClassfication.ipynb     |
| LeNet      | LeNetClassfication.ipynb   |
| ResNet50   | ResNetClassfication.ipynb  |
| ViT-B/16   | ViTClassfication.ipynb     |

The functions are explained as follows:

1. Loading files from the local folder and previewing them, I noticed that the image sizes are different.

2. Resize the images to 256 X 256; for the ViT model, resize them to 224 X 224 (as required by the pre-trained model). Then save them as `train_loader.pt` and `test_loader.pt`.

For time savings, the data will be loaded at the adjusted size the next time you use it.

3. Define the model structure separately.

4. Training, validation, and evaluation of the model. Then save the model parameters to a file.

5. Plot training and validation loss and accuracy. Confusion matrix on test data. F1 score on test data.


### Models

#### LeNet

LeNet is a series of convolutional neural network architectures created by a research group in AT&T Bell Laboratories during the 1988 to 1998 period, centered around Yann LeCun. I used the version is public in 1998, it is the most well-known version. It is also sometimes called LeNet-5.

The structure of LeNet is fixed, it applied big kernel size 5, 2 convolutional layers and 3 full connections layers.

![alt text](le_net_image.png)

After 20 epoch iterations, The training loss has tended to stabilize, And it approaches zero, showing a tendency towards overfitting. But the validation loss is not very stable, and there has been no significant improvement. 

![alt text](le_net_loss.png)
![alt text](le_net_accuracy.png)

Finally, the F1 Score is 0.5408.

![alt text](le_net_f1.png)

#### Simple CNN

Simple CNN 是现代典型基础网络结构，倾向于较小的卷积核 (例如 3x3)，结构比较灵活，2-3层卷积层，1-2层全连接层，网络结构如下图。

![alt text](cnn_structure.png)

15次迭代之后，由 Accuracy line chart 可以看出来，模型已经几乎学不到什么东西了，因为训练样本体量比较小。

![alt text](cnn_loss.png)
![alt text](cnn_accuracy.png)

Finally, the F1 Score is 0.7293. Comparing the F1 Score of LeNet, the model is available, but it is not good.

![alt text](cnn_f1.png)

于是，我们尝试使用迁移学习的方式。

#### Transfer Learning - ResNet50

ResNet50（Residual Network 50）是微软研究院于2015年提出的一种深度卷积神经网络结构，是ResNet（残差网络）系列的一员。它引入了 Residual Block 的概念，残差块的核心是跨层直连）shortcut connection, 当前卷积层的变换后，与变化前的特征一起传入下一层，避免了因为神经网络太深引起的梯度消失问题。

我选择了只训练它的后10层，包括 Stage 4 的一部分参数和最后的池化，全连接层。结构如下。

![alt text](res_net_structure.png)

它的训练效率很高，10次迭代之后就达到了如下的 F1 Score, 0.8534

![alt text](res_net_f1.png)

#### Transfer Learning - Vit-B/16

最后，尝试使用的最新的 ViT 预模型模型，ViT Basic version Patch Size 16. 它的结构比 ResNet 简单，是12层 Encoder Block 结构，我只训练了它的 11 层和 12层，以及最后的全连接分类层。结构如下。 

![alt text](vit_structure.png)

10次迭代之后，F1 Score 达到了 0.934

![alt text](vit_f1.png)

### Conclusion

以下是评估结果的统计，按从低到高排列

| Model      | Notebook                   | Batch Size | Epochs | Val Loss | Val Accuracy | Weighted F1 |
|------------|----------------------------|-----------|--------|----------|--------------|-------------|
| LeNet      | LeNetClassfication.ipynb   | 32        | 20     | 5.0790   | 44.92%       | 0.5194      |
| SimpleCNN  | CNNClassfication.ipynb     | 16        | 20     | 1.9833   | 58.63%       | 0.6482      |
| SimpleCNN  | CNNClassfication.ipynb     | 32        | 20     | 1.5336   | 63.82%       | 0.6648      |
| ResNet50   | ResNetClassfication.ipynb  | 32        | 10     | 0.3853   | 85.16%       | 0.8617      |
| ViT-B/16   | ViTClassfication.ipynb     | 32        | 10     | 0.5985   | 86.99%       | 0.9034      |

在解决图形识别问题上，迁移学习是个宝藏。海量的预训练模型记录着海量的特征信息。所以利用预训练模型，然后解决实际的业务上的问题就可以了。