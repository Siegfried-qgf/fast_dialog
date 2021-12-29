# PTM-based-E2E

## Datasets
- MultiWOZ 2.0\2.1\2.2

## Models
- UBAR: [UBAR: Towards Fully End-to-End Task-Oriented Dialog System with GPT-2](https://arxiv.org/pdf/2012.03539.pdf)
- PPTOD: [Multi-Task Pre-Training for Plug-and-Play Task-Oriented
Dialogue System](https://arxiv.org/pdf/2109.14739.pdf)
- DAMD: [Task-Oriented Dialog Systems that Consider Multiple Appropriate Responses
under the Same Context](https://arxiv.org/pdf/1911.10484.pdf)

## Data Preprocess
### MultiWOZ
**2.0**: fast_dialog/data/dataset/: ./MultiWOZ2.0_preparation.sh  
**2.1**: fast_dialog/data/dataset/: ./MultiWOZ2.1_preparation.sh  
**2.2**: fast_dialog/data/dataset/: ./MultiWOZ2.2_preparation.sh  

## Training
**PPTOD_small**: fast_dialog/model/pptod/: ./pptod_small_train_full_training.sh
**PPTOD_base**: fast_dialog/model/pptod/: ./pptod_base_train_full_training.sh
**PPTOD_large**: fast_dialog/model/pptod/: ./pptod_large_train_full_training.sh
