#!/bin/bash
###############################################################################
#
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################

## Usage: 
#./vllm_benchmark_report.sh -s $mode -m $hf_model -g $n_gpu -d $datatype -v $vllm_mode
## example:
## latency + throughput
#./vllm_benchmark_report.sh -s all -m NousResearch/Meta-Llama-3-8B -g 1 -d float16 -v $vllm_mode
## latency 
#./vllm_benchmark_report.sh -s latency -m NousResearch/Meta-Llama-3-8B -g 1 -d float16 -v $vllm_mode
## throughput
#./vllm_benchmark_report.sh -s throughput -m NousResearch/Meta-Llama-3-8B -g 1 -d float16 -v $vllm_mode

while getopts s:m:g:d:v: flag
do
    case "${flag}" in
        s) scenario=${OPTARG};;
        m) model=${OPTARG};;
        g) numgpu=${OPTARG};;
        d) datatype=${OPTARG};;
        v) vllmmode=${OPTARG};;
    esac
done

pip install pandas datasets 
pip install lm-eval[api]

# args
model_org_name=(${model//// })
model_name=${model_org_name[-1]}
tp=$numgpu

tag="vllm_ll4"
CON="16 32 64 128"
ISL_OSL=("1000:1000" "5000:1000" "10000:1000" "3200:800" "2000:150")

CON="16"
ISL_OSL=("1000:1000")

if [[ "$MAD_SYSTEM_GPU_ARCHITECTURE" == *"gfx94"* ]] ; then 
    CONFIG="online_config_rocm.csv"
    export VLLM_ROCM_FP8_PADDING=0 
    export VLLM_ROCM_USE_AITER=1 
    export VLLM_ROCM_USE_AITER_MOE=1 
    export VLLM_ROCM_USE_AITER_FP8_CHANNEL_SCALED_MOE=0
    export VLLM_ROCM_USE_AITER_RMSNORM=0 
    export VLLM_ROCM_USE_AITER_LINEAR=0 
else
    CONFIG="online_config_cuda.csv"
fi

report_dir="reports_${datatype}_${tag}"
report_summary_dir="${report_dir}/summary"
mkdir -p $report_dir
mkdir -p $report_summary_dir

echo $vllmmode

if [[ $vllmmode == "v1" ]]; then
    export VLLM_USE_V1=1 
    export SAFETENSORS_FAST_GPU=1 
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
else
    export VLLM_USE_V1=0
fi

if [[ $datatype == "float16" ]]; then
    DTYPE=" --dtype float16 "	
elif [[ $datatype == "float8" ]]; then
    if [[ $vllmmode == "v1" ]]; then
	DTYPE=" --dtype float16 --quantization fp8 " 
    else
	DTYPE=" --dtype float16 --quantization fp8 --kv-cache-dtype fp8 " 
    fi
fi

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 12000 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

if [ "$scenario" == "online_perf" ]; then

    echo "[INFO] ONLINE PERFORMANCE"
    echo "[INFO]" $MODEL_DIR

    date=$(date +"%Y-%m-%d")
    LOG="temp"
    backend="vllm"
    LOG_sum="benchmark_${backend}_${vllmmode}_${date}"

    while IFS="," read -r vllm_arg
    do
        echo $vllm_arg
        printf "%-15s" "model: " $MODEL_DIR     2>&1 | tee -a ${LOG_sum}.log
        printf "\n"                   2>&1 | tee -a ${LOG_sum}.log
        printf "%-15s" "option: " $vllm_arg     2>&1 | tee -a ${LOG_sum}.log
        printf "\n"                   2>&1 | tee -a ${LOG_sum}.log
        printf "%-15s" "==========="  2>&1 | tee -a ${LOG_sum}.log
        printf "\n"                   2>&1 | tee -a ${LOG_sum}.log

        vllm serve $MODEL_DIR $vllm_arg &
        wait_for_server 8080

        printf "%-15s" prompts                 2>&1 | tee -a ${LOG_sum}.log
        printf "%-15s" isl                     2>&1 | tee -a ${LOG_sum}.log
        printf "%-15s" osl                     2>&1 | tee -a ${LOG_sum}.log
        printf "%-15s" con                     2>&1 | tee -a ${LOG_sum}.log
        printf "%-15s" req_throughput          2>&1 | tee -a ${LOG_sum}.log
        printf "%-15s" median_e2e              2>&1 | tee -a ${LOG_sum}.log
        printf "%-15s" median_ttft             2>&1 | tee -a ${LOG_sum}.log
        printf "%-15s" median_tpot             2>&1 | tee -a ${LOG_sum}.log
        printf "%-15s" median_itl              2>&1 | tee -a ${LOG_sum}.log
        printf "%-15s" output_tps              2>&1 | tee -a ${LOG_sum}.log
        printf "%-15s" total_tps               2>&1 | tee -a ${LOG_sum}.log
        printf "\n"                            2>&1 | tee -a ${LOG_sum}.log

        for in_out in ${ISL_OSL[@]}
        do
            isl=$(echo $in_out | awk -F':' '{ print $1 }')
            osl=$(echo $in_out | awk -F':' '{ print $2 }')
            for con in $CON; do
            prompts=640

            echo "[RUNNING] prompts $prompts isl $isl osl $osl con $con"
            python3 /app/vllm/benchmarks/benchmark_serving.py \
                --model $MODEL_DIR \
                --dataset-name random \
                --random-input-len $isl \
                --random-output-len $osl \
                --num-prompts $prompts \
                --max-concurrency $con \
                --port 8080 \
                --ignore-eos \
                --percentile-metrics ttft,tpot,itl,e2el \
                2>&1 | tee ${LOG}.log

            rTh=$(grep -E "Request throughput" ${LOG}.log)
            e2eLat=$(grep -E "Median E2EL" ${LOG}.log)
            ttftLat=$(grep -E "Median TTFT" ${LOG}.log)
            tpotLat=$(grep -E "Median TPOT" ${LOG}.log)
            itlLat=$(grep -E "Median ITL" ${LOG}.log)
            outTh=$(grep -E "Output token throughput" ${LOG}.log)
            totTh=$(grep -E "Total Token throughput" ${LOG}.log)

            rTh_sp=(${rTh//:/ })
            e2eLat_sp=(${e2eLat//:/ })
            ttftLat_sp=(${ttftLat//:/ })
            tpotLat_sp=(${tpotLat//:/ })
            itlLat_sp=(${itlLat//:/ })
            outTh_sp=(${outTh//:/ })
            totTh_sp=(${totTh//:/ })

            rTh_val=${rTh_sp[3]}
            e2eLat_val=${e2eLat_sp[3]}
            ttftLat_val=${ttftLat_sp[3]}
            tpotLat_val=${tpotLat_sp[3]}
            itlLat_val=${itlLat_sp[3]}
            outTh_val=${outTh_sp[4]}
            totTh_val=${totTh_sp[4]}

            printf "%-15s" $prompts        2>&1 | tee -a ${LOG_sum}.log
            printf "%-15s" $isl            2>&1 | tee -a ${LOG_sum}.log
            printf "%-15s" $osl            2>&1 | tee -a ${LOG_sum}.log
            printf "%-15s" $con            2>&1 | tee -a ${LOG_sum}.log
            printf "%-15s" $rTh_val        2>&1 | tee -a ${LOG_sum}.log
            printf "%-15s" $e2eLat_val     2>&1 | tee -a ${LOG_sum}.log
            printf "%-15s" $ttftLat_val    2>&1 | tee -a ${LOG_sum}.log
            printf "%-15s" $tpotLat_val    2>&1 | tee -a ${LOG_sum}.log
            printf "%-15s" $itlLat_val     2>&1 | tee -a ${LOG_sum}.log
            printf "%-15s" $outTh_val      2>&1 | tee -a ${LOG_sum}.log
            printf "%-15s" $totTh_val      2>&1 | tee -a ${LOG_sum}.log
            printf "\n"                    2>&1 | tee -a ${LOG_sum}.log
            done
        done
    done < <(tail -n +2 $CONFIG)

elif [ "$scenario" == "online_accuracy" ]; then

    echo "[INFO] ONLINE ACCURACY"
    echo "[INFO]" $MODEL_DIR

    # pre-process
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git
    cd lm-evaluation-harness
    pip install -e .
    cd ..

    date=$(date +"%Y-%m-%d")
    LOG="temp"
    backend="vllm"
    LOG_sum="benchmark_${backend}_${vllmmode}_accuracy_${date}"

    while IFS="," read -r vllm_arg
    do
        echo $vllm_arg
        printf "%-15s" "model: " $MODEL_DIR     2>&1 | tee -a ${LOG_sum}.log
        printf "\n"                   2>&1 | tee -a ${LOG_sum}.log
        printf "%-15s" "option: " $vllm_arg     2>&1 | tee -a ${LOG_sum}.log
        printf "\n"                   2>&1 | tee -a ${LOG_sum}.log
        printf "%-15s" "==========="  2>&1 | tee -a ${LOG_sum}.log
        printf "\n"                   2>&1 | tee -a ${LOG_sum}.log

        vllm serve $MODEL_DIR $vllm_arg $DTYPE &
        wait_for_server 8080

        lm_eval --model local-completions --model_args model=$MODEL_DIR,base_url=http://0.0.0.0:8080/v1/completions,num_concurrent=10,max_retries=3 --tasks mmlu_pro --limit 100  2>&1 | tee -a ${LOG_sum}.log
        lm_eval --model local-completions --model_args model=$MODEL_DIR,base_url=http://0.0.0.0:8080/v1/completions,num_concurrent=10,max_retries=3 --tasks gpqa_diamond_cot_zeroshot --apply_chat_template  2>&1 | tee -a ${LOG_sum}.log
        lm_eval --model local-completions --model_args model=$MODEL_DIR,base_url=http://0.0.0.0:8080/v1/completions,num_concurrent=10,max_retries=3 --tasks gsm8k  2>&1 | tee -a ${LOG_sum}.log
    done < <(tail -n +2 $CONFIG)
fi

cp $LOG_sum.log $report_summary_dir/.
