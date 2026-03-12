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