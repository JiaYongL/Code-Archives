wanted_gpu=$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader | sed -n "1p")

wait_pid=$(nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader | grep $wanted_gpu | sort -rn -k2 | head -n1 | awk '{if ($2 > 1000) print $0}' | awk -F ',' '{print $1}')

echo $wait_pid

echo "waiting for user, cmd :" $(ps -p $wait_pid -o user,cmd | tail -1)

if [ ! -z "$wait_pid" ]; then
    while ps -p $wait_pid > /dev/null; do
        sleep 5  # You can adjust the sleep duration as needed
    done
fi

export CUDA_VISIBLE_DEVICES=0


# python trainner2.py --config ./config/tacred/5shot.yml --pattern hybridprompt --prompt_length 3 > log/tacred/5shot/hybridprompt3.log

