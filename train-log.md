`lr_schedule = WarmupCosineDecay(d_model=D_MODEL, warmup_steps=2000, max_lr=5e-4)`

loss稳定降低，大概第6~7个epoch开始加速，后续位置，若loss降低太慢，考虑增加学习率到1e-3

若之后震荡或者不稳定，则考虑把学习率改回来

---