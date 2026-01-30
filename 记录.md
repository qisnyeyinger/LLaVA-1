/data1/cwk/mllm/models/llava-v1.5-7b


"/data1/cwk/mllm/project/LLaVA/test.png"


CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli \
    --model-path /data1/cwk/mllm/models/llava-v1.5-7b \
    --image-file "/data1/cwk/mllm/project/LLaVA/test.png" \
    --load-4bit