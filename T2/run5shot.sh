
export CUDA_VISIBLE_DEVICES=2  
python run_continual.py --dataname TACRED --num_k 5 --feat_dim 768 --use_unused 1 --pattern hybridprompt > ./log/tacred5shot_hybrid.log
python run_continual.py --dataname FewRel --num_k 5 --feat_dim 768 --use_unused 1 --pattern hybridprompt > ./log/fewrel5shot_hybrid.log

