#!/usr/bin/env bash

N=9
outfol=experiments/CTP/evaluation/ctp_downsample_${N}
max_belief=65536
mkdir -p ${outfol}
for i in $(seq 1 1 9); do
  pfile=ctp_downsample_eval.txt
  outfile=${outfol}/CTPInstance${N}_00${i}.txt
  build/experiments/CTP/ctp_timeseries ${pfile} --max_sim_depth $((20 * N)) --max_time_ms 3600000 --eval_interval_ms 12000000 --max_belief_samples $((max_belief * i / 10000)) --nb_particles_b0 $((100 * max_belief)) > ${outfile}
  python3 -c "from experiments.CTP.time_series import parse_file; df = parse_file('"${outfile}"'); df['Set number']=0.0"${i}"; df.to_csv('"${outfol}"/ctp_results_"${N}"_00"${i}".csv', index=False)"
done
python3 -c "import pandas as pd; results=[pd.read_csv(open(f'"${outfol}"/ctp_results_"${N}"_00{i}.csv', 'r')) for i in range(1, 9, 1)]; df=pd.concat(results, ignore_index=True, sort=False); df.to_csv('"${outfol}"/ctp_results_all.csv', index=False)"
