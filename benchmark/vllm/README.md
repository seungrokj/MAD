# LLM inference performance validation with vLLM on the AMD Instinct MI300X accelerator

## Overview 🎉
--------

vLLM is a toolkit and library for large language model (LLM) inference and serving. It
deploys the PagedAttention algorithm, which reduces memory consumption
and increases throughput by leveraging dynamic key and value allocation
in GPU memory. vLLM also incorporates many recent LLM acceleration and
quantization algorithms. In addition, AMD implements high-performance custom
kernels and modules in vLLM to enhance performance further.

This Docker image packages vLLM with PyTorch for an AMD Instinct™ MI300X
accelerator. It includes:

-   ✅ ROCm™ 6.2.1
-   ✅ vLLM 0.6.4
-   ✅ PyTorch 2.5.0
-   ✅ Tuning files (.csv format)

With this Docker image, users can quickly validate the expected inference performance numbers on the MI300X accelerator. 
This guide also provides tips and techniques so that users can get optimal performance with popular AI models.


## Reproducing benchmark results 🚀
-----------------------------

Use the following instructions to reproduce the benchmark results on an
MI300X accelerator with a prebuilt vLLM Docker image.

Users have two choices to reproduce the benchmark results.

-   [MAD-integrated benchmarking](#mad-integrated-benchmarking)
-   [Standalone benchmarking](#standalone-benchmarking)

### NUMA balancing setting

To optimize performance, disable automatic NUMA balancing. Otherwise, the GPU
might hang until the periodic balancing is finalized. For further
details, refer to the [AMD Instinct MI300X system optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#disable-numa-auto-balancing) guide.

```sh
# disable automatic NUMA balancing
sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
# check if NUMA balancing is disabled (returns 0 if disabled)
cat /proc/sys/kernel/numa_balancing
0
```

### Download the Docker image 🐳

The following command pulls the Docker image from Docker Hub.

```sh
docker pull rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
```

### MAD-integrated benchmarking

Clone the ROCm Model Automation and Dashboarding (MAD) repository to a local directory and install the required packages on the host machine.

```sh
git clone https://github.com/ROCm/MAD
cd MAD
pip install -r requirements.txt
```

Use this command to run a performance benchmark test of the Llama 3.1 8B model on one GPU with float16 data type in the host machine. 

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
python3 tools/run_models.py --tags pyt_vllm_llama-3.1-8b --keep-model-dir --live-output --timeout 28800
```

ROCm MAD launches a Docker container with the name `container_ci-pyt_vllm_llama-3.1-8b`. The latency and throughput reports of the model are collected in the following path:

```sh
~/MAD/reports_float16/
```

Although the following models are pre-configured to collect latency and throughput performance data,
users can also change the benchmarking parameters. Refer to the [Standalone benchmarking](#standalone-benchmarking) section.

#### Available models

| model_name                  |
| --------------------------- |
| pyt_vllm_llama-3.1-8b       |
| pyt_vllm_llama-3.1-70b      |
| pyt_vllm_llama-3.1-405b     |
| pyt_vllm_llama-2-7b         |
| pyt_vllm_llama-2-70b        |
| pyt_vllm_mixtral-8x7b       |
| pyt_vllm_mixtral-8x22b      |
| pyt_vllm_mistral-7b         |
| pyt_vllm_qwen2-7b           |
| pyt_vllm_qwen2-72b          |
| pyt_vllm_jais-13b           |
| pyt_vllm_jais-30b           |
| pyt_vllm_llama-3.1-8b_fp8   |
| pyt_vllm_llama-3.1-70b_fp8  |
| pyt_vllm_llama-3.1-405b_fp8 |
| pyt_vllm_mixtral-8x7b_fp8   |
| pyt_vllm_mixtral-8x22b_fp8  |

### Standalone benchmarking
-----------------------------

Users also can run the benchmark tool after they launch a Docker container.

```sh
docker pull rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 128G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --cap-add=SYS_PTRACE -v $(pwd):/workspace --env HUGGINGFACE_HUB_CACHE=/workspace --name vllm_v0.6.4 rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
```

Now clone the ROCm MAD repository inside the Docker image and move to the benchmark scripts directory at *~/MAD/scripts/vllm*. 

```sh
git clone https://github.com/ROCm/MAD
cd MAD/scripts/vllm
```

#### Command

```sh
./vllm_benchmark_report.sh -s $test_option -m $model_repo -g $num_gpu -d $datatype
```

>[!NOTE]
>The input sequence length, output sequence length, and tensor parallel (TP) are already configured. You don't need to specify them with this script.

>[!NOTE]
>If you encounter this error, pass your access-authorized Hugging Face token to the gated models.
>```sh
>OSError: You are trying to access a gated repo.
>
># pass your HF_TOKEN
>export HF_TOKEN=$your_personal_hf_token
>```

#### Variables

| Name         | Options                                 | Description                                      |
| ------------ | --------------------------------------- | ------------------------------------------------ |
| $test_option | latency                                 | Measure decoding token latency                   |
|              | throughput                              | Measure token generation throughput              |
|              | all                                     | Measure both throughput and latency              |
| $model_repo  | meta-llama/Meta-Llama-3.1-8B-Instruct   | Llama 3.1 8B                                     |
| (float16)    | meta-llama/Meta-Llama-3.1-70B-Instruct  | Llama 3.1 70B                                    |
|              | meta-llama/Meta-Llama-3.1-405B-Instruct | Llama 3.1 405B                                   |
|              | meta-llama/Llama-2-7b-chat-hf           | Llama 2 7B                                       |
|              | meta-llama/Llama-2-70b-chat-hf          | Llama 2 70B                                      |
|              | mistralai/Mixtral-8x7B-Instruct-v0.1    | Mixtral 8x7B                                     |
|              | mistralai/Mixtral-8x22B-Instruct-v0.1   | Mixtral 8x22B                                    |
|              | mistralai/Mistral-7B-Instruct-v0.3      | Mistral 7B                                       |
|              | Qwen/Qwen2-7B-Instruct                  | Qwen2 7B                                         |
|              | Qwen/Qwen2-72B-Instruct                 | Qwen2 72B                                        |
|              | core42/jais-13b-chat                    | JAIS 13B                                         |
|              | core42/jais-30b-chat-v3                 | JAIS 30B                                         |
| $model_repo  | amd/Meta-Llama-3.1-8B-Instruct-FP8-KV   | Llama 3.1 8B                                     |
| (float8)     | amd/Meta-Llama-3.1-70B-Instruct-FP8-KV  | Llama 3.1 70B                                    |
|              | amd/Meta-Llama-3.1-405B-Instruct-FP8-KV | Llama 3.1 405B                                   |
|              | amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV   | Mixtral 8x7B                                     |
|              | amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV  | Mixtral 8x22B                                    |
| $num_gpu     | 1 or 8                                  | Number of GPUs                                   |
| $datatype    | float16, float8                         | Data type                                        |

#### Run the benchmark tests on the MI300X accelerator 🏃

Here are some examples and the test results:

- Benchmark example - latency

  Use this command to benchmark the latency of the Llama 3.1 8B model on one GPU with the float16 and float8 data type.

  ```sh
  ./vllm_benchmark_report.sh -s latency -m meta-llama/Meta-Llama-3.1-8B-Instruct -g 1 -d float16
  ./vllm_benchmark_report.sh -s latency -m amd/Meta-Llama-3.1-8B-Instruct-FP8-KV -g 1 -d float8
  ```

  The latency reports are available at:

  - `./reports_float16/summary/Meta-Llama-3.1-8B-Instruct_latency_report.csv`
  - `./reports_float8/summary/Meta-Llama-3.1-8B-Instruct-FP8-KV_latency_report.csv`

- Benchmark example - throughput

  Use this command to benchmark the throughput of the Llama 3.1 8B model on one GPU with the float16 and float8 data type.

  ```sh
  ./vllm_benchmark_report.sh -s throughput -m meta-llama/Meta-Llama-3.1-8B-Instruct -g 1 -d float16
  ./vllm_benchmark_report.sh -s throughput -m amd/Meta-Llama-3.1-8B-Instruct-FP8-KV -g 1 -d float8
  ```

  The throughput reports are available at:

  - `./reports_float16/summary/Meta-Llama-3.1-8B-Instruct_throughput_report.csv`
  - `./reports_float8/summary/Meta-Llama-3.1-8B-Instruct-FP8-KV_throughput_report.csv`

>[!NOTE]
>Throughput is calculated as:
>-   `throughput_tot = requests * (input lengths + output lengths) / elapsed_time`
>-   `throughput_gen = requests * output lengths / elapsed_time`

## References 🔎
----------

For an overview of the optional performance features of vLLM with
ROCm software, see [ROCm performance](https://github.com/ROCm/vllm/blob/main/ROCm_performance.md).

To learn more about the options for latency and throughput
benchmark scripts, see
<https://github.com/ROCm/vllm/tree/main/benchmarks>.

To learn how to run LLM models from Hugging Face or your own model, see the
[Using ROCm for AI](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/index.html) section of the ROCm documentation.

To learn how to optimize inference on LLMs, see the
[Fine-tuning LLMs and inference optimization](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/index.html) section of the ROCm documentation.

For a list of other ready-made Docker images for ROCm, see the 
[ROCm Docker image support matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/docker-image-support-matrix.html).

## Licensing information ⚠️
---------------------

Your use of this application is subject to the terms of the applicable
component-level license identified below. To the extent any subcomponent
in this container requires an offer for corresponding source code, AMD
hereby makes such an offer for corresponding source code form, which
will be made available upon request. By accessing and using this
application, you are agreeing to fully comply with the terms of this
license. If you do not agree to the terms of this license, do not access
or use this application.

The application is provided in a container image format that includes
the following separate and independent components:

| Package | License                                          | URL                  |
| ------- | ------------------------------------------------ | -------------------- |
| Ubuntu  | Creative Commons CC-BY-SA Version 3.0 UK License | [Ubuntu Legal](https://ubuntu.com/legal) |
| ROCm    | Custom/MIT/Apache V2.0/UIUC OSL                  | [ROCm Licensing Terms](https://rocm.docs.amd.com/en/latest/about/license.html) |
| PyTorch | Modified BSD                                     | [PyTorch License](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| vLLM    | Apache License 2.0                               | [vLLM License](https://github.com/vllm-project/vllm/blob/main/LICENSE)  |

### Disclaimer

The information contained herein is for informational purposes only and
is subject to change without notice. In addition, any stated support is
planned and is also subject to change. While every precaution has been
taken in the preparation of this document, it may contain technical
inaccuracies, omissions and typographical errors, and AMD is under no
obligation to update or otherwise correct this information. Advanced
Micro Devices, Inc. makes no representations or warranties with respect
to the accuracy or completeness of the contents of this document, and
assumes no liability of any kind, including the implied warranties of
noninfringement, merchantability or fitness for purposes, with respect
to the operation or use of AMD hardware, software or other products
described herein. No license, including implied or arising by estoppel,
to any intellectual property rights is granted by this document. Terms
and limitations applicable to the purchase or use of AMD's products are
as set forth in a signed agreement between the parties or in AMD\'s
Standard Terms and Conditions of Sale.

### Notices and attribution

© 2024 Advanced Micro Devices, Inc. All rights reserved. AMD, the AMD
Arrow logo, Instinct, Radeon Instinct, ROCm, and combinations thereof
are trademarks of Advanced Micro Devices, Inc.

Docker and the Docker logo are trademarks or registered trademarks of
Docker, Inc. in the United States and/or other countries. Docker, Inc.
and other parties may also have trademark rights in other terms used
herein. Linux® is the registered trademark of Linus Torvalds in the U.S.
and other countries.    

All other trademarks and copyrights are property of their respective
owners and are only mentioned for informative purposes.   


## Changelog
----------
This release note summarizes notable changes since the previous docker release (September 4, 2024).

-   The ROCm software version number was incremented from 6.2.0 to 6.2.1.

-   The vLLM version number was incremented from 0.4.3 to 0.6.4.

-   The PyTorch version number was incremented from 2.4.0 to 2.5.0.

-   The float16 data type benchmark test was updated to include the following models: Llama 2 70B, Mixtral 8x7B, Mixtral 8X22B, and Qwen2 72B.

-   float8 date type is available.

-   The float8 data type benchmark test was added to include the following models: Llama 3.1 8B, Llama 3.1 70B, Llama 3.1 405B, Mixtral 8x7B, and Mixtral 8X22B.



## Support 
----------
You can report bugs through our GitHub [issue tracker](https://github.com/ROCm/MAD/issues).
