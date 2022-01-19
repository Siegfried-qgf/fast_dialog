#3090
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch\
    --master_port 88888\
    --nproc_per_node=4\
    --nnodes=1\
    --node_rank=0\
    main.py\
    -version 2.0\
    -num_gpus 4\
    -run_type train\
    -batch_size_per_gpu 8\
    -batch_size_per_gpu_eval 64\
    -model_dir test_42_with_aux_task_lr_2e-3\
    -epochs 20\
    -save_best_model\
    -seed 42\
    -add_auxiliary_task\
    -learning_rate 2e-3\