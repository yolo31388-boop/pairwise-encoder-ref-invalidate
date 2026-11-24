# pairwise-encoder-ref-invalidate

## 领域

游戏、图形与多媒体

## 主题

编码参考帧失效

## 说明

IDR 清参考、在途帧作废与重排序复位。

初始状态下 `media/encoder_ref_invalidate.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
