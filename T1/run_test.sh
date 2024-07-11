#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=1


# python trainner.py --config ./config/fewrel/prompt_5shot.yml > log/fewrel/hybrid_prompt_5shot.log
# debugpy-run trainner.py --config ./config/fewrel/prompt_5shot.yml

# python trainner.py --config ./config/fewrel/prompt_5shot.yml --pattern soft_prompt > log/fewrel/soft_prompt_5shot.log


# python trainner.py --config ./config/fewrel/prompt_5shot.yml --pattern soft_prompt --retain_prev true > log/fewrel/soft_prompt_retain_5shot.log

# python trainner.py --config ./config/tacred/prompt_5shot.yml --pattern soft_prompt --retain_prev true > log/tacred/soft_prompt_retain_5shot.log
# python trainner.py --config ./config/tacred/prompt_5shot.yml --pattern hybrid_prompt --retain_prev true > log/tacred/hybrid_prompt_retain_5shot.log
# python trainner.py --config ./config/tacred/prompt_5shot.yml --pattern hybrid_prompt --retain_prev false > log/tacred/hybrid_prompt_5shot.log
# python trainner.py --config ./config/tacred/prompt_5shot.yml --pattern hard_prompt --retain_prev false > log/tacred/hard_prompt_5shot.log
# python trainner.py --config ./config/tacred/prompt_5shot.yml --pattern hard_prompt --retain_prev true > log/tacred/hard_prompt_retain_5shot.log

python trainner2.py --config ./config/fewrel/test.yml --pattern hybridprompt > temp.log
