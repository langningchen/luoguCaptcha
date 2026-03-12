`lr_schedule = WarmupCosineDecay(d_model=D_MODEL, warmup_steps=2000, max_lr=5e-4)`

loss稳定降低，大概第6~7个epoch开始加速，后续位置，若loss降低太慢，考虑增加学习率到1e-3

若之后震荡或者不稳定，则考虑把学习率改回来

---

```
704/704 ━━━━━━━━━━━━━━━━━━━━ 74s 105ms/step - accuracy: 0.5499 - loss: 1.3834 - val_accuracy: 0.6370 - val_loss: 1.0730 - learning_rate: 4.4127e-04
Epoch 19/150
```

感觉下降不动了，先试一下提高学习率，3e-3

---

似乎学习率超过1.1e-3就突然慢了，还是上限改为1e-3吧

---

不是很稳定，先6e-4试一下，挂一晚上

704/704 ━━━━━━━━━━━━━━━━━━━━ 70s 99ms/step - accuracy: 0.6107 - loss: 1.1644 - val_accuracy: 0.6782 - val_loss: 0.9441 - learning_rate: 5.2952e-04
Epoch 19/150

训练暂停，明天继续，看起来这个参数还可以

附全阶段：

```
==================================================
Building Vision Transformer (Encoder-Only) Model
  Patch size: 5x5
  Num patches: 126 (7x18)
  D_model: 128, Heads: 4, Layers: 4, DFF: 256
==================================================

Model: "LuoguCaptcha_ViT"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input (InputLayer)                   │ (None, 35, 90, 1)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ patch_embedding (PatchEmbedding)     │ (None, None, 128)           │           3,328 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cls_tokens (CLSTokens)               │ (None, None, 128)           │             512 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ positional_encoding                  │ (None, 130, 128)            │          16,640 │
│ (LearnedPositionalEncoding)          │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 130, 128)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ transformer_block_0                  │ (None, 130, 128)            │         132,480 │
│ (TransformerEncoderBlock)            │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ transformer_block_1                  │ (None, 130, 128)            │         132,480 │
│ (TransformerEncoderBlock)            │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ transformer_block_2                  │ (None, 130, 128)            │         132,480 │
│ (TransformerEncoderBlock)            │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ transformer_block_3                  │ (None, 130, 128)            │         132,480 │
│ (TransformerEncoderBlock)            │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ final_norm (LayerNormalization)      │ (None, 130, 128)            │             256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ extract_cls (Lambda)                 │ (None, 4, 128)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cls_head_dense (Dense)               │ (None, 4, 256)              │          33,024 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_17 (Dropout)                 │ (None, 4, 256)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cls_head_output (Dense)              │ (None, 4, 256)              │          65,792 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 649,472 (2.48 MB)
 Trainable params: 649,472 (2.48 MB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/150
2026-03-12 22:59:22.311676: I tensorflow/core/kernels/data/tf_record_dataset_op.cc:390] TFRecordDataset `buffer_size` is unspecified, default to 262144
2026-03-12 22:59:22.785866: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:473] Loaded cuDNN version 90600
    704/Unknown 76s 97ms/step - accuracy: 0.0141 - loss: 4.89472026-03-12 23:00:32.412671: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-03-12 23:00:32.412815: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:00:32.412898: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:00:32.412976: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 5090305549196775289
2026-03-12 23:00:32.413081: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_4]]
/home/chanmao/luoguCaptcha/.conda/lib/python3.12/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
2026-03-12 23:00:34.850673: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[StatefulPartitionedCall/Shape/_6]]
2026-03-12 23:00:34.850752: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:00:34.850761: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:00:34.850765: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 79s 101ms/step - accuracy: 0.0161 - loss: 4.4631 - val_accuracy: 0.0172 - val_loss: 4.1104 - learning_rate: 2.1150e-04
Epoch 2/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 102ms/step - accuracy: 0.0169 - loss: 4.14052026-03-12 23:01:47.088089: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 1932261949288386439
2026-03-12 23:01:47.088151: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 17670532271325385841
2026-03-12 23:01:47.088159: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:01:47.088209: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 5090305549196775289
2026-03-12 23:01:47.088281: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:01:47.088324: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 166200773535152028
2026-03-12 23:01:49.017199: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[StatefulPartitionedCall/Shape/_6]]
2026-03-12 23:01:49.017371: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:01:49.017460: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:01:49.017523: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 74s 105ms/step - accuracy: 0.0168 - loss: 4.1319 - val_accuracy: 0.0166 - val_loss: 4.1033 - learning_rate: 4.2300e-04
Epoch 3/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 102ms/step - accuracy: 0.0161 - loss: 4.11452026-03-12 23:03:01.337374: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:03:01.337473: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:03:01.337511: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 5090305549196775289
2026-03-12 23:03:03.284403: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:03:03.284470: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:03:03.284479: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 74s 105ms/step - accuracy: 0.0162 - loss: 4.1111 - val_accuracy: 0.0177 - val_loss: 4.1004 - learning_rate: 5.9999e-04
Epoch 4/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 104ms/step - accuracy: 0.0203 - loss: 4.09102026-03-12 23:04:16.599780: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:04:16.599999: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 1932261949288386439
2026-03-12 23:04:16.600083: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 17670532271325385841
2026-03-12 23:04:16.600115: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:04:16.600165: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 5090305549196775289
2026-03-12 23:04:16.600238: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 166200773535152028
2026-03-12 23:04:18.525240: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[StatefulPartitionedCall/Shape/_6]]
2026-03-12 23:04:18.525309: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:04:18.525345: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:04:18.525402: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 75s 106ms/step - accuracy: 0.0226 - loss: 4.0769 - val_accuracy: 0.0275 - val_loss: 4.0431 - learning_rate: 5.9957e-04
Epoch 5/150
703/704 ━━━━━━━━━━━━━━━━━━━━ 0s 92ms/step - accuracy: 0.0349 - loss: 3.99142026-03-12 23:05:23.838723: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:05:23.838814: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:05:23.838853: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 5090305549196775289
2026-03-12 23:05:25.386234: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 7097523251311342287
2026-03-12 23:05:25.386296: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:05:25.386327: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:05:25.386380: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 67s 94ms/step - accuracy: 0.0489 - loss: 3.8868 - val_accuracy: 0.0989 - val_loss: 3.5408 - learning_rate: 5.9851e-04
Epoch 6/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 92ms/step - accuracy: 0.1064 - loss: 3.48292026-03-12 23:06:30.235610: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:06:30.235835: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 1932261949288386439
2026-03-12 23:06:30.235914: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 17670532271325385841
2026-03-12 23:06:30.235994: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:06:30.236009: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 166200773535152028
2026-03-12 23:06:31.930363: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:06:31.930435: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:06:31.930492: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 67s 94ms/step - accuracy: 0.1230 - loss: 3.3772 - val_accuracy: 0.1786 - val_loss: 3.0376 - learning_rate: 5.9681e-04
Epoch 7/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 87ms/step - accuracy: 0.1815 - loss: 3.02532026-03-12 23:07:33.877879: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:07:33.878102: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 1932261949288386439
2026-03-12 23:07:33.878181: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 17670532271325385841
2026-03-12 23:07:33.878192: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:07:33.878195: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 166200773535152028
2026-03-12 23:07:35.544512: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:07:35.544578: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:07:35.544594: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 64s 90ms/step - accuracy: 0.2025 - loss: 2.9106 - val_accuracy: 0.2698 - val_loss: 2.5597 - learning_rate: 5.9448e-04
Epoch 8/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 89ms/step - accuracy: 0.2543 - loss: 2.63002026-03-12 23:08:38.660512: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:08:38.660663: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:08:38.660716: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 5090305549196775289
2026-03-12 23:08:40.319256: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[StatefulPartitionedCall/Shape/_6]]
2026-03-12 23:08:40.319325: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:08:40.319363: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:08:40.319430: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 65s 91ms/step - accuracy: 0.2708 - loss: 2.5472 - val_accuracy: 0.3512 - val_loss: 2.1811 - learning_rate: 5.9153e-04
Epoch 9/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 84ms/step - accuracy: 0.3211 - loss: 2.31972026-03-12 23:09:39.825436: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:09:39.825604: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:09:39.825657: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 5090305549196775289
2026-03-12 23:09:41.497743: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:09:41.497807: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:09:41.497865: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 61s 86ms/step - accuracy: 0.3375 - loss: 2.2443 - val_accuracy: 0.4344 - val_loss: 1.8345 - learning_rate: 5.8795e-04
Epoch 10/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 92ms/step - accuracy: 0.3786 - loss: 2.05162026-03-12 23:10:46.776036: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:10:46.776185: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:10:46.776239: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 5090305549196775289
2026-03-12 23:10:48.440857: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:10:48.440921: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:10:48.440966: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 67s 95ms/step - accuracy: 0.3947 - loss: 1.9877 - val_accuracy: 0.4885 - val_loss: 1.6097 - learning_rate: 5.8376e-04
Epoch 11/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 88ms/step - accuracy: 0.4330 - loss: 1.83382026-03-12 23:11:50.984125: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:11:50.984189: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:11:50.984228: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 5090305549196775289
2026-03-12 23:11:50.481739: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 7097523251311342287
2026-03-12 23:11:50.481808: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:11:50.481817: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:11:50.481874: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 62s 88ms/step - accuracy: 0.4452 - loss: 1.7833 - val_accuracy: 0.5303 - val_loss: 1.4647 - learning_rate: 5.7897e-04
Epoch 12/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 89ms/step - accuracy: 0.4742 - loss: 1.67032026-03-12 23:12:53.211661: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 1932261949288386439
2026-03-12 23:12:53.211823: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 17670532271325385841
2026-03-12 23:12:53.211949: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 14090016303837445340
2026-03-12 23:12:53.211999: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 166200773535152028
2026-03-12 23:12:54.888490: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:12:54.888552: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:12:54.888612: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 64s 91ms/step - accuracy: 0.4849 - loss: 1.6331 - val_accuracy: 0.5607 - val_loss: 1.3554 - learning_rate: 5.7358e-04
Epoch 13/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 88ms/step - accuracy: 0.5057 - loss: 1.55332026-03-12 23:13:57.505455: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:13:57.505539: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:13:57.505574: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 5090305549196775289
2026-03-12 23:13:59.177119: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:13:59.177184: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:13:59.177196: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 64s 91ms/step - accuracy: 0.5137 - loss: 1.5218 - val_accuracy: 0.5938 - val_loss: 1.2276 - learning_rate: 5.6761e-04
Epoch 14/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 88ms/step - accuracy: 0.5318 - loss: 1.45112026-03-12 23:15:01.609990: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 1932261949288386439
2026-03-12 23:15:01.610071: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 17670532271325385841
2026-03-12 23:15:01.610122: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:15:01.610184: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:15:01.610228: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 166200773535152028
2026-03-12 23:15:03.271204: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:15:03.271265: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:15:03.271335: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 64s 91ms/step - accuracy: 0.5393 - loss: 1.4267 - val_accuracy: 0.6126 - val_loss: 1.1620 - learning_rate: 5.6107e-04
Epoch 15/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 89ms/step - accuracy: 0.5546 - loss: 1.36732026-03-12 23:16:06.242510: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:16:06.242719: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 1932261949288386439
2026-03-12 23:16:06.242799: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 17670532271325385841
2026-03-12 23:16:06.242813: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 166200773535152028
2026-03-12 23:16:07.917705: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:16:07.917769: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:16:07.917785: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 65s 91ms/step - accuracy: 0.5606 - loss: 1.3446 - val_accuracy: 0.6350 - val_loss: 1.0765 - learning_rate: 5.5398e-04
Epoch 16/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 90ms/step - accuracy: 0.5744 - loss: 1.29752026-03-12 23:17:11.404688: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 1932261949288386439
2026-03-12 23:17:11.404744: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:17:11.404886: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 166200773535152028
2026-03-12 23:17:11.404949: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 17670532271325385841
2026-03-12 23:17:13.087580: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[StatefulPartitionedCall/Shape/_6]]
2026-03-12 23:17:13.087662: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:17:13.087699: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:17:13.087759: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 65s 92ms/step - accuracy: 0.5792 - loss: 1.2785 - val_accuracy: 0.6538 - val_loss: 1.0259 - learning_rate: 5.4634e-04
Epoch 17/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 87ms/step - accuracy: 0.5897 - loss: 1.23972026-03-12 23:18:15.059068: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:18:15.059165: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 17670532271325385841
2026-03-12 23:18:15.059203: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:18:15.059210: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 5090305549196775289
2026-03-12 23:18:16.788077: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:18:16.788171: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:18:16.788296: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 64s 90ms/step - accuracy: 0.5968 - loss: 1.2165 - val_accuracy: 0.6670 - val_loss: 0.9785 - learning_rate: 5.3819e-04
Epoch 18/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 96ms/step - accuracy: 0.6057 - loss: 1.17872026-03-12 23:19:24.527726: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:19:24.527814: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:19:24.527852: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 5090305549196775289
2026-03-12 23:19:26.517242: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:19:26.517317: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:19:26.517326: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 70s 99ms/step - accuracy: 0.6107 - loss: 1.1644 - val_accuracy: 0.6782 - val_loss: 0.9441 - learning_rate: 5.2952e-04
Epoch 19/150
704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 106ms/step - accuracy: 0.6197 - loss: 1.13342026-03-12 23:20:41.385768: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8783352679716343436
2026-03-12 23:20:41.385848: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 8462593573958641343
2026-03-12 23:20:41.385861: I tensorflow/core/framework/local_rendezvous.cc:430] Local rendezvous send item cancelled. Key hash: 5090305549196775289
2026-03-12 23:20:43.371625: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 7437358193244969420
2026-03-12 23:20:43.371691: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 16832947115945297058
2026-03-12 23:20:43.371700: I tensorflow/core/framework/local_rendezvous.cc:426] Local rendezvous recv item cancelled. Key hash: 4016174104209971134
704/704 ━━━━━━━━━━━━━━━━━━━━ 77s 109ms/step - accuracy: 0.6241 - loss: 1.1201 - val_accuracy: 0.6868 - val_loss: 0.9033 - learning_rate: 5.2036e-04
Epoch 20/150
230/704 ━━━━━━━━━━━━━━━━━━━━ 38s 81ms/step - accuracy: 0.6225 - loss: 1.1076^CTraceback (most recent call last):
```