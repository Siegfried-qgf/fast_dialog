wget https://github.com/xiami2019/MultiWOZ_Datasets/raw/main/MultiWOZ_2.2.zip
unzip MultiWOZ_2.2.zip
cd ../utils
python data_analysis.py --version 2.2
python preprocess.py --version 2.2
python postprocessing_dataset.py --version 2.2
cd ..
cp special_token_list.txt ./MultiWOZ_2.0/multi-woz-fine-processed/special_token_list.txt