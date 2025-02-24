This repository contains a basic version of our code related to the model presented in the ISMIR2022 paper:
[A deep learning method for melody extraction from a polyphonic symbolic music representation](https://archives.ismir.net/ismir2022/paper/000091.pdf).
The data corpus that it is based on can be found [here](https://github.com/music-x-lab/POP909-Dataset/tree/master/POP909).

```
python -m venv .venv
git submodule init
git submodule update --remote --recursive
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python LStoM/process_data.py -i POP909-Dataset/POP909 -o ./processed_data
python LStoM/create_stats_dict.py -i ./processed_data -o ./yaml_files
python LStoM/train.py -sd ./yaml_files/stats_config_train_valid.yaml -ds ./yaml_files/data_split.yaml -o ./model_results -of test
```
