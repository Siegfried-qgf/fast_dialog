wget https://github.com/xiami2019/MultiWOZ_Datasets/raw/main/MultiWOZ_2.0.zip
unzip MultiWOZ_2.0.zip
cd ./utils
python data_analysis.py --version 2.0
python preprocess.py --version 2.0
python postprocessing_dataset.py --version 2.0
cd ..
cp special_token_list.txt ./MultiWOZ_2.0/multi-woz-fine-processed/special_token_list.txt
rm MultiWOZ_2.0.zip