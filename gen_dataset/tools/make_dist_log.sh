#!/usr/bin/env bash
set -e 

for set_type in train valid test
do
	if [ $set_type = 'valid' ];then
	       	brir_dir=../syn_BRIRs/tar_BRIRs_train
       	else
	       	brir_dir=../syn_BRIRs/tar_BRIRs_${set_type}
	fi
	python -m LocTools.concat_logs \
		--log ../Data/synroom/v2/${set_type}/reverb/brir.txt \
		      ${brir_dir}/dist.txt \
		--concat-log ../Data/synroom/v2/${set_type}/reverb/dist.txt
done
