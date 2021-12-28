wget https://github.com/xiami2019/MultiWOZ_Datasets/raw/main/MultiWOZ_2.1.zip
unzip MultiWOZ_2.1.zip
cd ./utils
python data_analysis.py --version 2.1
python preprocess.py --version 2.1
python postprocessing_dataset.py --version 2.1
cd ..
cp special_token_list.txt ./MultiWOZ_2.1/multi-woz-fine-processed/special_token_list.txt
rm MultiWOZ_2.1.zip