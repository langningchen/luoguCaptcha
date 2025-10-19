当前版本使用了 ResNet（残差连接层）以及 SelfAttention BiLSTM

由于 ResNet 层数过多，在其内部就已经学习到的大量的全局特征（以至于直接使用 GlobalPooling 仍可获得 0.9 accuracy），可供 BiLSTM 学习的特征有限。

后续需要尝试减少残差连接层数，在第一轮训练中给 CNN 添加高 Dropout，将 GlobalPooling 替换为 Dense，减小第一轮训练迭代数等方式，在前期训练中限制 CNN 只学习局部特征，从而强迫 识别全局特征使用自注意力机制。