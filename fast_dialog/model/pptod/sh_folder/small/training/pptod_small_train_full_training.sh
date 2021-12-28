CUDA_VISIBLE_DEVICES=4,5,6,7 python ../../../learn.py\
    --dataset_name MultiWOZ_2.0\
    --model_name t5-small\
    --epoch_num 60\
    --gradient_accumulation_steps 16\
    --number_of_gpu 4\
    --batch_size_per_gpu 2