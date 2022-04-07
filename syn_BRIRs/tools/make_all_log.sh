#!/usr/bin/env bash
set -e

set_names=( tar_BRIRs_train tar_BRIRs_test tar_BRIRs_test_largeRT )

# make logs
for set_name in ${set_names[@]}
do
	{
	       	python make_RT_log.py ../${set_name}
	       	python make_dist_log.py ../${set_name}
	       	python make_roomvolume_log.py ../${set_name}
	}&
done
wait
