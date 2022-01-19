"""
   MTTOD: main.py

   Command-line Interface for MTTOD.

   Copyright 2021 ETRI LIRS, Yohan Lee

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import random

import torch
import numpy as np

from config import get_config
from runner import MultiWOZRunner

from utils.io_utils import get_or_create_logger




def main():
    """ main function """
    cuda_available = torch.cuda.is_available()
    cfg = get_config()
    logger = get_or_create_logger(__name__, cfg.log_dir)
    if cuda_available:
        if cfg.num_gpus > 1:
            logger.info('Using Multi-GPU training, number of GPU is {}'.format(cfg.num_gpus))
            torch.cuda.set_device(cfg.local_rank)
            device = torch.device('cuda', cfg.local_rank)
            torch.distributed.init_process_group(backend='nccl')
        else:
            logger.info('Using single GPU training.')
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    setattr(cfg, "device", device)

    logger.info("Device: %s (the number of GPUs: %d)", str(device), cfg.num_gpus)

    if cfg.seed > 0:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        logger.info("Set random seed to %d", cfg.seed)

    runner = MultiWOZRunner(cfg)

    if cfg.run_type == "train":
        runner.train()
    else:
        runner.predict()


if __name__ == "__main__":
    main()
