#!/usr/bin/env bash
set -e

n_bin=20

set_names=( tar_BRIRs_train tar_BRIRs_test tar_BRIRs_test_largeRT )

for set_name in ${set_names[@]}
do
	{
		python -m LocTools.plot_hist --log ../${set_name}/RT.txt \
			--n-bin $n_bin \
			--x-label 'RT(s)' \
			--fig-path ../images/RT_hist_${set_name}.png

		python -m LocTools.plot_hist --log ../${set_name}/dist.txt \
			--n-bin $n_bin \
			--x-label 'Dist(m)' \
			--fig-path ../images/dist_hist_${set_name}.png

		python -m LocTools.plot_hist --log ../${set_name}/volume.txt \
			--n-bin $n_bin \
			--x-label 'Volume(m^3)' \
			--fig-path ../images/volume_hist_${set_name}.png

		python -m LocTools.plot_hist --log ../${set_name}/drr.txt \
			--n-bin $n_bin \
			--x-label 'DRR(dB)' \
			--fig-path ../images/DRR_hist_${set_name}.png
		}&
done
wait
