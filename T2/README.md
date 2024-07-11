# 有监督对比学习方法

在使用了混合提示的基础上使用了有监督对比学习结合 NCM 分类器的方法

## 代码运行

**FewRel**

```bash
# 5shot
python run_continual.py --dataname FewRel --num_k 5 --feat_dim 768 --use_unused 1 --pattern hybridprompt

# 10shot
python run_continual.py --dataname FewRel --num_k 10 --feat_dim 768 --use_unused 1 --pattern hybridprompt
```

**TACRED**

```bash
# 5shot
python run_continual.py --dataname TACRED --num_k 5 --feat_dim 768 --use_unused 1 --pattern hybridprompt

# 10shot
python run_continual.py --dataname TACRED --num_k 10 --feat_dim 768 --use_unused 1 --pattern hybridprompt
```





