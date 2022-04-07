#!/usr/bin/env bash
# set -e

model_script_path=
model_dir=
set_name=TIMIT
n_src=
chunksize=
parallel=0

. ./parse_options.sh

rooms=( Anechoic Room_A Room_B Room_C Room_D )
file_reader_script_path=file_reader-realroom-${n_src}src.py
model_basedir=$(dirname $model_dir)
result_in_all=${model_basedir}/${set_name}-${n_src}src-${chunksize}.txt
file_list_basedir=/Data/test

if [ $parallel -eq 0 ]; then
	for room in ${rooms[@]}; do
	       	python evaluate.py --model_script_path $model_script_path \
		       	--file_reader_script_path ${file_reader_script_path} \
		       	--model_dir ${model_dir} \
		       	--file_list ${file_list_basedir}/${room}/reverb/record1_paths.txt \
		       	--log_path ${set_name}-${n_src}src/${room}/log/1.txt
       	done
else
	for room in ${rooms[@]}; do
	       	{
		       	python evaluate.py --model_script_path $model_script_path \
			       	--file_reader_script_path ${file_reader_script_path} \
			       	--model_dir ${model_dir} \
		       		--file_list ${file_list_basedir}/${room}/reverb/record1_paths.txt \
			       	--log_path ${set_name}-${n_src}src/${room}/log/1.txt
		       	}&
       	done
       	wait
fi

for room in ${rooms[@]};  do
	loc_log_basedir=${model_basedir}/evaluations/${set_name}-${n_src}src/${room}
	
	if [ $n_src -eq 1 ]; then	
		python -m LocTools.load_loc_log --loc-log ${loc_log_basedir}/log/1.txt \
		       	--azi-pos 0 \
		       	--n-src 1 \
		       	--chunksize $chunksize
		statistic_log_dir=${loc_log_basedir}/chunksize_${chunksize}-azipos_0-nsrc_1
	elif [ $n_src -eq 2 ]; then 
		python -m LocTools.load_loc_log --loc-log ${loc_log_basedir}/log/1.txt \
		       	--azi-pos 0 1 \
		       	--n-src 1 \
		       	--chunksize $chunksize
		statistic_log_dir=${loc_log_basedir}/chunksize_${chunksize}-azipos_0_1-nsrc_1
	elif [ $n_src -eq 3 ]; then 
		python -m LocTools.load_loc_log --loc-log ${loc_log_basedir}/log/1.txt \
		       	--azi-pos 0 1 2\
		       	--n-src 1 \
		       	--chunksize $chunksize
		statistic_log_dir=${loc_log_basedir}/chunksize_${chunksize}-azipos_0_1_2-nsrc_1
	else
		echo 'n_src wrong'
	fi

	python -m LocTools.merge_logs --log ${record1_basedir}/${room}/reverb/azi.txt ${statistic_log_dir}/1.txt \
	       		--result-log ${statistic_log_dir}/1_azi_indexed.txt \
		       	--repeat-processor average 

	echo $room >> $result_in_all
	tail -n1 $statistic_log_dir/1.txt >> $result_in_all
done

python clean_up_result.py ${result_in_all}
