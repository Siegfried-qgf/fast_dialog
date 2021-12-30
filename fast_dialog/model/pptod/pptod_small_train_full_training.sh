# 3090
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    learn.py \
    --dataset_name MultiWOZ_2.0\
    --model_name t5-small\
    --pretrained_path checkpoints/small\
    --epoch_num 60\
    --gradient_accumulation_steps 4\
    --number_of_gpu 8\
    --batch_size_per_gpu 4\
    --batch_size_per_gpu_eval 128\
    --train_data_ratio 0.1
# 2080
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#     --master_port 88887\
#     --nproc_per_node=4 \
#     --nnodes=1 \
#     --node_rank=0 \
#     learn.py \
#     --dataset_name MultiWOZ_2.0\
#     --model_name t5-small\
#     --epoch_num 60\
#     --gradient_accumulation_steps 16\
#     --number_of_gpu 4\
#     --batch_size_per_gpu 2\
#     --batch_size_per_gpu_eval 64\
#     --train_data_ratio 0.01