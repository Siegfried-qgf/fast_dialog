CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch\
    --nproc_per_node=2\
    --nnodes=1\
    --node_rank=0\
    main.py\
    -run_type predict\
    -num_gpus 2\
    -ckpt test_/ckpt-epoch18\
    -output inference\
    -batch_size_per_gpu_eval 64\