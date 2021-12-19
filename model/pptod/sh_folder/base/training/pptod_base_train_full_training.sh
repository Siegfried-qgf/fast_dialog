CUDA_VISIBLE_DEVICES=0 python ../../../learn.py\
    --data_path_prefix ../../../../../data/dataset/multiwoz/MultiWOZ_2.1\
    --model_name t5-base\
    --pretrained_path ../../../checkpoints/base/\
    --ckpt_save_path ../../../ckpt2.1/t5_base/full_training/\
    --epoch_num 60\
    --gradient_accumulation_steps 16\
    --number_of_gpu 1\
    --batch_size_per_gpu 4\
    --only_use_PLM