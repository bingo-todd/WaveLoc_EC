#!/usr/bin/env bash 

record_dir=../Data/
TIMIT_vad_log=TIMIT_vad/TIMIT_vad.txt
for src_path in $(find $record_dir -name 'src_path.txt')
do
	echo $src_path	
	dir=$(dirname $src_path)
	python -m LocTools.concat_logs --log $src_path $TIMIT_vad_log --concat-log ${dir}/vad.txt
done
