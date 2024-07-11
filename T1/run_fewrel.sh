#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=0

# python trainner.py --config ./config/fewrel/entity_marker_10shot.yml > log/fewrel/entity_marker_10shot.log
# python trainner.py --config ./config/fewrel/entity_marker_10shot.yml > log/fewrel/entity_marker_10shot_10_10_10.log
# python trainner.py --config ./config/fewrel/entity_marker_10shot.yml > log/fewrel/tmp2.log
# python trainner.py --config ./config/fewrel/prompt_10shot.yml > log/fewrel/prompt_10shot_10_10_10.log
# python trainner.py --config ./config/fewrel/prompt_10shot.yml > log/fewrel/prompt_10shot.log
# python trainner.py --config ./config/fewrel/entity_marker_5shot.yml > log/fewrel/entity_marker_5shot.log
# python trainner.py --config ./config/fewrel/entity_marker_5shot.yml > log/fewrel/entity_marker_5shot_10_10_10.log
# python trainner.py --config ./config/fewrel/entity_marker_5shot.yml > log/fewrel/tmp.log
# python trainner.py --config ./config/fewrel/prompt_5shot.yml > log/fewrel/prompt_5shot.log
# python trainner.py --config ./config/fewrel/prompt_5shot.yml > log/fewrel/prompt_5shot10_10_10.log


# python trainner.py --config ./config/fewrel/entity_marker_5shot.yml > log/fewrel/entity_marker_5shot_wo_distil.log
# python trainner.py --config ./config/fewrel/entity_marker_5shot.yml > log/fewrel/entity_marker_5shot_wo_hidden_distil.log


# python trainner.py --config ./config/fewrel/prompt_5shot.yml --pattern hard_prompt > log/fewrel/hard_prompt_5shot.log


# Replace <PID> with the actual process ID
# pid=193480

# while ps -p $pid > /dev/null; do
#     sleep 60  # You can adjust the sleep duration as needed
# done

# echo "Process has finished"

# python trainner.py --config ./config/fewrel/prompt_5shot.yml --pattern soft_prompt --retain_prev false > log/fewrel/5shot/soft_prompt.log
# python trainner.py --config ./config/fewrel/prompt_5shot.yml --pattern soft_prompt --retain_prev true > log/fewrel/5shot/soft_prompt_retain.log
# python trainner.py --config ./config/fewrel/prompt_5shot.yml --pattern hard_prompt --retain_prev true > log/fewrel/5shot/hard_prompt_retain1.log

# python trainner.py --config ./config/fewrel/prompt_5shot.yml --pattern hybrid_prompt --retain_prev true > log/fewrel/5shot/hybrid_prompt_retain1.log
# python trainner.py --config ./config/fewrel/prompt_5shot.yml --pattern hybrid_prompt --retain_prev false > log/fewrel/5shot/hybrid_prompt1.log

# python trainner.py --config ./config/fewrel/prompt_5shot.yml --pattern hard_prompt --retain_prev false > log/fewrel/5shot/hard_prompt1.log


# 硬提示
# python trainner2.py --config ./config/fewrel/5shot.yml --pattern hardprompt > log/fewrel/5shot/hardprompt.log

# 软提示
# python trainner2.py --config ./config/fewrel/5shot.yml --pattern softprompt > log/fewrel/5shot/softprompt.log

# 混合提示 提示长度为 1
# python trainner2.py --config ./config/fewrel/5shot.yml --pattern hybridprompt > log/fewrel/5shot/hybridprompt.log

# 实体标记
# python trainner2.py --config ./config/fewrel/5shot.yml --pattern entity_marker > log/fewrel/5shot/entity.log

# # unused
# # 硬提示
# python trainner2.py --config ./config/fewrel/5shot.yml --pattern hardprompt --use_unused True > log/fewrel/5shot/hardprompt_unused.log

# # 软提示
# python trainner2.py --config ./config/fewrel/5shot.yml --pattern softprompt  --use_unused True > log/fewrel/5shot/softprompt_unused.log

# # 混合提示 提示长度为 1
# python trainner2.py --config ./config/fewrel/5shot.yml --pattern hybridprompt  --use_unused True > log/fewrel/5shot/hybridprompt_unused.log

# # 实体标记
# python trainner2.py --config ./config/fewrel/5shot.yml --pattern entity_marker  --use_unused True > log/fewrel/5shot/entity_unused.log


# unused
# 硬提示
# python trainner2.py --config ./config/fewrel/10shot.yml --pattern hardprompt --use_unused True > log/fewrel/10shot/hardprompt_unused.log

# # 软提示
# python trainner2.py --config ./config/fewrel/10shot.yml --pattern softprompt  --use_unused True > log/fewrel/10shot/softprompt_unused.log

# # 混合提示 提示长度为 1
# python trainner2.py --config ./config/fewrel/10shot.yml --pattern hybridprompt  --use_unused True > log/fewrel/10shot/hybridprompt_unused.log

# # 实体标记
# python trainner2.py --config ./config/fewrel/10shot.yml --pattern entity_marker  --use_unused True > log/fewrel/10shot/entity_unused.log


# python trainner2.py --config ./config/fewrel/10shot.yml --pattern hybridprompt  --use_unused True > log/fewrel/10shot/hybridprompt3_unused.log
# python trainner2.py --config ./config/fewrel/10shot.yml --pattern entity_marker --seed 400 --total_round 3 > log/fewrel/10shot/entity.log

python trainner2.py --config ./config/fewrel/5shot.yml --pattern hybridprompt  --use_unused True --num_protos 2 > log/fewrel/5shot/hybridprompt_unused_m2.log
python trainner2.py --config ./config/fewrel/5shot.yml --pattern hybridprompt  --use_unused True --num_protos 3 > log/fewrel/5shot/hybridprompt_unused_m3.log

python trainner2.py --config ./config/fewrel/10shot.yml --pattern hybridprompt  --use_unused True --num_protos 2 > log/fewrel/10shot/hybridprompt_unused_m2.log
python trainner2.py --config ./config/fewrel/10shot.yml --pattern hybridprompt  --use_unused True --num_protos 3 > log/fewrel/10shot/hybridprompt_unused_m3.log



python trainner2.py --config ./config/fewrel/5shot.yml --pattern entity_marker --use_unused True > log/fewrel/5shot/entity_unused.log
python trainner2.py --config ./config/tacred/5shot.yml --pattern entity_marker --use_unused True > log/tacred/5shot/entity_unused.log
python trainner2.py --config ./config/fewrel/10shot.yml --pattern entity_marker --use_unused True > log/fewrel/10shot/entity_unused.log
python trainner2.py --config ./config/tacred/10shot.yml --pattern entity_marker --use_unused True > log/tacred/10shot/entity_unused.log