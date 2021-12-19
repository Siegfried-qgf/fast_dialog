CUDA_VISIBLE_DEVICES=2 python ../../../learn.py\
    --data_path_prefix ../../../../../data/dataset/multiwoz/MultiWOZ_2.2\
    --model_name t5-small\
    --pretrained_path ../../../checkpoints/small/\
    --ckpt_save_path ../../../ckpt2.2/small/full_training/\
    --epoch_num 60\
    --gradient_accumulation_steps 2\
    --number_of_gpu 1\
    --batch_size_per_gpu 16