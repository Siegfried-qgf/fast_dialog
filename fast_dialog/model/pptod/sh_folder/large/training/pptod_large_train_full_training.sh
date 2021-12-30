CUDA_VISIBLE_DEVICES=2,3 python ../../../learn.py\
    --data_path_prefix ../../../../../data/dataset/multiwoz/MultiWOZ_2.1\
    --model_name t5-large\
    --pretrained_path ../../../checkpoints/large/\
    --ckpt_save_path ../../../ckpt2.1/large/full_training/\
    --epoch_num 60\
    --gradient_accumulation_steps 16\
    --number_of_gpu 2\
    --batch_size_per_gpu 2