export CUDA_VISIBLE_DEVICES=0  
python run_continual.py --dataname TACRED --num_k 10 --feat_dim 768 --use_unused 1 --pattern hybridprompt > ./log/tacred10shot_hybrid.log
python run_continual.py --dataname FewRel --num_k 10 --feat_dim 768 --use_unused 1 --pattern hybridprompt > ./log/fewrel10shot_hybrid.log
