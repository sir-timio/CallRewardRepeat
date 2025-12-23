python -m spacy download en_core_web_sm
cd ./ubar-preprocessing/data
wget https://github.com/smartyfh/MultiWOZ2.4/blob/main/data/MULTIWOZ2.4.zip?raw=true -O MultiWOZ_2.4.zip
# unzip MultiWOZ_2.4.zip
# mv MULTIWOZ2.4 MultiWOZ_2.4

cd ..
cp ../../../src/multiwoz/fix_ubar_preprocess21.py preprocess21.py
python data_analysis21.py
python preprocess21.py
cd ..
# rm -rf ./ubar-preprocessing/data/MultiWOZ_2.4
rm -rf ./ubar-preprocessing/data/MultiWOZ_2.4.zip
# rm -rf ./ubar-preprocessing/data/__MACOSX
cp -r ./ubar-preprocessing/data ./
cp -r ./ubar-preprocessing/db ./data/
cd ./utlis
python postprocessing_dataset21.py
python convert_to_rawdata.py --data_version '2.1' --save_data_path_prefix '../data/multi-woz-2.4-rawdata'
cd ..
cp special_token_list.txt ./data/multi-woz-2.4-fine-processed/special_token_list.txt
cp schema.json ./data/multi-woz-2.4-fine-processed/schema.json
cp possible_slot_values.json ./data/multi-woz-2.4-fine-processed/possible_slot_values.json

find ./data -depth -name '*2.4*' | while read fname; do
    newfname=$(echo "$fname" | sed 's/2\.1/2.4/g')
    mv "$fname" "$newfname"
done