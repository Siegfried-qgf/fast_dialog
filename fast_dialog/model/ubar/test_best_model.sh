path='experiments/all_0729_sd11_lr0.0001_bs2_ga16/epoch43_trloss0.56_gpt2'
python train.py --mode test --datasets_name MultiWOZ_2.0 --cfg eval_load_path=$path use_true_prev_bspn=False use_true_prev_aspn=False use_true_db_pointer=False use_true_prev_resp=False use_true_curr_bspn=False use_true_curr_aspn=False use_all_previous_context=True cuda_device=7