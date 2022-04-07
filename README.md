# An end-to-end binaural sound localization model based on the equalization and cancellation theory

code for the paper "An end-to-end binaural sound localization model based on the equalization and cancellation theory", which has been accepted by the 152 AES conference.

***if there is any problem, please contact me***

## Requirements
### python environment
- python 3.7
- tensorflow-gpu 2.4
- [gammatone](https://github.com/detly/gammatone): implementation of Gammatone filters
- [BasicTools](https://github.com/bingo-todd/BasicTools): a collection of scripts for some basic function, such as plotting and signal processing. 
- [LocTools](https://github.com/bingo-todd/LocTools): scripts for processing log files of localization.
- [RoomSimulator](https://github.com/bingo-todd/RoomSimulator): implementation of Image model for BRIR simulation.

### Dataset
- TIMIT[^TIMIT]
- [RealRoomBRIRs](https://github.com/IoSR-Surrey/RealRoomBRIRs)


## Run Steps
### Preparing training and validating dataset
#### Synthsizing BRIRs
```shell
cd syn_BRIRs
python syn_BRIRs.py
```
#### Generating training and validating dataset
```shell
cd gen_dataset
python gen_train_valid_dataset.py
```
### Preparing testing dataset
```shell
cd gen_dataset
python gen_test_1src_dataset.py
python gen_test_2src_dataset.py
python gen_test_3src_dataset.py
```
### trainig model
Three model scripts are provided,
- WaveLoc_EC: the basic model
```shell
python train-var_feat_filter_len-specify_cnn_f.py \ 
	--model_script_path WaveLoc_EC.py \  
	--file_reader_script_path file_read.py \ 
	--n_feat_filter 32 \  # channel number of CNN_feat kernel  
	--feat_filter_len 32 \  # length of CNN_feat kernel
	--cnn_coef 1 \  # parameter for activation function of CNN_feat 
	--model_dir exp/WaveLoc_EC \  
	--train_file_list Data/train/reverb/record_paths.txt \ 
	--valid_file_list Data/train/reverb/record_paths.txt \ 
	--gpu_id 0
```
- WaveLoc-EC-wav_pool: add single max-pooling layer after normalization layer  
```shell
python train-var_feat_filter_len-specify_cnn_f.py \ 
	--model_script_path WaveLoc_EC.py \  
	--file_reader_script_path file_read.py \ 
	--n_feat_filter 32 \  # channel number of CNN_feat kernel  
	--feat_filter_len 32 \  # length of CNN_feat kernel
	--cnn_coef 1 \  # parameter for activation function of CNN_feat
	--pool_len 2 \  # pooling size 
	--model_dir exp/WaveLoc_EC-wav_pool_2 \  
	--train_file_list Data/train/reverb/record_paths.txt \ 
	--valid_file_list Data/train/reverb/record_paths.txt \ 
	--gpu_id 0
```

- WaveLoc-EC-feat_pool: add signle max-pooling layer after the CNN layer which extract features from binaural signals
```shell
python train-var_feat_filter_len-specify_cnn_f.py \ 
	--model_script_path WaveLoc_EC.py \  
	--file_reader_script_path file_read.py \ 
	--n_feat_filter 32 \  # channel number of CNN_feat kernel  
	--feat_filter_len 32 \  # length of CNN_feat kernel
	--cnn_coef 1 \  # parameter for activation function of CNN_feat
	--pool_len 2 \  # pooling size 
	--model_dir exp/WaveLoc_EC-feat_pool_2 \  
	--train_file_list Data/train/reverb/record_paths.txt \ 
	--valid_file_list Data/train/reverb/record_paths.txt \ 
	--gpu_id 0
```

### Evaluating
```shell
bash run-evaluate-realroom.sh --model_script WaveLoc_EC.py \ 
	--model_dir exp/WaveLoc_EC \ 
	--n_src 1  # number of sound source, 1~3 
	--chunksize 1  # number of successive frames of model output to be averaged 
```

[^TIMIT]:need to convert \*.WAV  to \*.wav