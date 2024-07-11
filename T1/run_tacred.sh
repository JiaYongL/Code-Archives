#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=1

# python trainner.py --config ./config/tacred/entity_marker_10shot.yml > log/tacred/entity_marker_10shot_10_10_10.log
# python trainner.py --config ./config/tacred/entity_marker_10shot.yml > log/tacred/xx.log
# python trainner.py --config ./config/tacred/entity_marker_10shot.yml > log/tacred/entity_marker_10shot.log
# python trainner.py --config ./config/tacred/prompt_10shot.yml > log/tacred/prompt_10shot.log
# python trainner.py --config ./config/tacred/prompt_10shot.yml > log/tacred/prompt_10shot_10_10_10.log
# python trainner.py --config ./config/tacred/entity_marker_5shot.yml > log/tacred/entity_marker_5shot.log
# python trainner.py --config ./config/tacred/entity_marker_5shot.yml > log/tacred/entity_marker_5shot_10_10_10.log
# python trainner.py --config ./config/tacred/entity_marker_5shot.yml > log/tacred/xxx.log
# python trainner.py --config ./config/tacred/prompt_5shot.yml > log/tacred/prompt_5shot.log
# python trainner.py --config ./config/tacred/prompt_5shot.yml > log/tacred/prompt_5shot_10_10_10.log


# python trainner.py --config ./config/tacred/entity_marker_5shot.yml > log/tacred/wo_distill.log

# python trainner.py --config ./config/tacred/entity_marker_5shot.yml > log/tacred/wo_hidden_distill.log
# python trainner.py --config ./config/tacred/prompt_5shot.yml > log/tacred/hybrid_prompt5shot.log
# python trainner.py --config ./config/tacred/prompt_5shot.yml > log/tacred/hybrid_prompt5shot_wo_distill.log

# python trainner.py --config ./config/tacred/prompt_5shot.yml --pattern hard_prompt > log/tacred/5shot/hard_prompt.log
# python trainner.py --config ./config/tacred/prompt_5shot.yml --pattern hard_prompt --retain_prev true > log/tacred/5shot/hard_prompt_retain.log

# python trainner.py --config ./config/tacred/prompt_5shot.yml --pattern soft_prompt > log/tacred/5shot/soft_prompt.log
# python trainner.py --config ./config/tacred/prompt_5shot.yml --pattern soft_prompt --retain_prev true > log/tacred/5shot/soft_prompt_retain.log

# python trainner.py --config ./config/tacred/prompt_5shot.yml --pattern hybrid_prompt > log/tacred/5shot/hybrid_prompt.log
# python trainner.py --config ./config/tacred/prompt_5shot.yml --pattern hybrid_prompt --retain_prev true > log/tacred/5shot/hybrid_prompt_retain.log

# python trainner.py --config ./config/fewrel/prompt_5shot.yml --pattern hard_prompt --retain_prev false > log/fewrel/5shot/hard_prompt.log
# python trainner.py --config ./config/fewrel/prompt_5shot.yml --pattern hard_prompt --retain_prev true > log/fewrel/5shot/hard_prompt_retain.log


# python trainner.py --config ./config/tacred/prompt_5shot.yml --pattern hybrid_prompt --retain_prev false > log/tacred/5shot/hybrid_prompt11.log


# 硬提示
# python trainner2.py --config ./config/tacred/5shot.yml --pattern hardprompt > log/tacred/5shot/hardprompt.log

# 软提示
# python trainner2.py --config ./config/tacred/5shot.yml --pattern softprompt > log/tacred/5shot/softprompt.log

# 混合提示 提示长度为 1
# python trainner2.py --config ./config/tacred/5shot.yml --pattern hybridprompt > log/tacred/5shot/hybridprompt.log

# 实体标记
# python trainner2.py --config ./config/tacred/5shot.yml --pattern entity_marker > log/tacred/5shot/entity.log

# python trainner2.py --config ./config/tacred/5shot.yml --pattern hybridprompt --prompt_length 3 > log/tacred/5shot/hybridprompt3.log


# ### unused
# # 硬提示
# python trainner2.py --config ./config/tacred/5shot.yml --pattern hardprompt --use_unused True > log/tacred/5shot/hardprompt_unused.log

# # 软提示
# python trainner2.py --config ./config/tacred/5shot.yml --pattern softprompt --use_unused True > log/tacred/5shot/softprompt_unused.log

# # 混合提示 提示长度为 1
# python trainner2.py --config ./config/tacred/5shot.yml --pattern hybridprompt --use_unused True > log/tacred/5shot/hybridprompt_unused.log

# # 实体标记
# python trainner2.py --config ./config/tacred/5shot.yml --pattern entity_marker --use_unused True > log/tacred/5shot/entity_unused.log

# python trainner2.py --config ./config/tacred/5shot.yml --pattern hybridprompt --prompt_length 3 --use_unused True > log/tacred/5shot/hybridprompt3_unused.log



# ### unused
# # 硬提示
# python trainner2.py --config ./config/tacred/10shot.yml --pattern hardprompt --use_unused True > log/tacred/10shot/hardprompt_unused.log

# # # 软提示
# python trainner2.py --config ./config/tacred/10shot.yml --pattern softprompt --use_unused True > log/tacred/10shot/softprompt_unused.log

# # # 混合提示 提示长度为 1
# python trainner2.py --config ./config/tacred/10shot.yml --pattern hybridprompt --use_unused True > log/tacred/10shot/hybridprompt_unused.log

# # # 实体标记
# python trainner2.py --config ./config/tacred/10shot.yml --pattern entity_marker --use_unused True > log/tacred/10shot/entity_unused.log

# python trainner2.py --config ./config/tacred/10shot.yml --pattern hybridprompt --prompt_length 3 --use_unused True > log/tacred/10shot/hybridprompt3_unused.log
python trainner2.py --config ./config/tacred/5shot.yml --pattern hybridprompt  --use_unused True --num_protos 2 > log/tacred/5shot/hybridprompt_unused_m2.log
python trainner2.py --config ./config/tacred/5shot.yml --pattern hybridprompt  --use_unused True --num_protos 3 > log/tacred/5shot/hybridprompt_unused_m3.log

python trainner2.py --config ./config/tacred/10shot.yml --pattern hybridprompt  --use_unused True --num_protos 2 > log/tacred/10shot/hybridprompt_unused_m2.log
python trainner2.py --config ./config/tacred/10shot.yml --pattern hybridprompt  --use_unused True --num_protos 3 > log/tacred/10shot/hybridprompt_unused_m3.log


