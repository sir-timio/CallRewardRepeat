echo "CamRest676"
mkdir raw_data
cd raw_data

git clone https://github.com/WING-NUS/sequicity.git
mv sequicity/data/CamRest676 CamRest676
rm -r sequicity

echo "SGD"
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git

echo "MS-E2E"
git clone https://github.com/xiul-msr/e2e_dialog_challenge.git

echo "WOZ"
git clone https://github.com/nmrksic/neural-belief-tracker.git

echo "TaskMaster"
git clone https://github.com/google-research-datasets/Taskmaster.git
