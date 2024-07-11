# 混合提示模板

以 [SCKD](https://github.com/nju-websoft/SCKD) 模型为基础，探究了不同提示对模型性能的影响。

## 预训练模型

从 hugging 中下载预训练模型到 ./bert-base-uncased 目录中。

## 代码运行

**FewRel**
```
# 5shot
python trainner2.py --config ./config/fewrel/5shot.yml --pattern hybridprompt  --use_unused True

# 10shot
python trainner2.py --config ./config/fewrel/10shot.yml --pattern hybridprompt  --use_unused True

```


**TACRED**
```
# 5shot
python trainner2.py --config ./config/tacred/5shot.yml --pattern hybridprompt  --use_unused True

# 10shot
python trainner2.py --config ./config/tacred/10shot.yml --pattern hybridprompt  --use_unused True

```


