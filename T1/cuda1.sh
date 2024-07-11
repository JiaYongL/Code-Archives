wanted_gpu=$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader | sed -n "2p")

wait_pid=$(nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader | grep $wanted_gpu | sort -rn -k2 | head -n1 | awk '{if ($2 > 1000) print $0}' | awk -F ',' '{print $1}')

echo "waiting for user, cmd :" $(ps -p $wait_pid -o user,cmd | tail -1)

if [ ! -z "$wait_pid" ]; then
    while ps -p $wait_pid > /dev/null; do
        sleep 5  # You can adjust the sleep duration as needed
    done
fi

export CUDA_VISIBLE_DEVICES=1


# 混合提示 提示长度为 3
# python trainner2.py --config ./config/fewrel/5shot.yml --pattern hybridprompt --prompt_length 3 > log/fewrel/5shot/hybridprompt3.log

# 硬提示
# python trainner2.py --config ./config/fewrel/5shot.yml --pattern hardprompt > log/fewrel/5shot/hardprompt.log