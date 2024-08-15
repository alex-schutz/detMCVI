N=10
problem_folder=experiments/CTP/evaluation/ctp_results_10x10_2024-08-13_12-13
outfol=experiments/CTP/evaluation/sarsop/${N}
mkdir -p ${outfol}
for i in $(seq 0 9); do
  pfile=${problem_folder}/ctp_graph_${N}_${i}.txt
  pomdpfile=${pfile}.pomdp
  outfile=${outfol}/CTPInstance${N}_${i}.txt
  start=$(date +%s.%N)
  build/experiments/CTP/ctp_to_sarsop ${pfile}
  taskset --cpu-list 1 /home/alex/ori/infjesp/third_party_dependencies/sarsop/src/pomdpsol ${pomdpfile} 2>&1 1> /dev/null
  /home/alex/ori/infjesp/third_party_dependencies/sarsop/src/polgraph ${pomdpfile} --policy-file out.policy --policy-graph policy.dot > /dev/null
  end=$(date +%s.%N)
  runtime=$(echo "$end - $start" | bc -l)
  echo $runtime
  build/experiments/CTP/ctp_evaluate_sarsop ${pfile} --max_sim_depth $(echo $N*$N*10 | bc) > ${outfile}
  python3 -c "from experiments.Maze.evaluation_set import parse_file; df = parse_file('"${outfile}"'); df['Timestamp']="${runtime}"; df['Set number']="${i}"; df.to_csv('"${outfol}"/ctp_results_"${N}_${i}".csv', index=False)"
done
python3 -c "import pandas as pd; results=[pd.read_csv(open(f'"${outfol}"/ctp_results_"${N}"_{i}.csv', 'r')) for i in range(10)]; df=pd.concat(results, ignore_index=True, sort=False); df.to_csv('"${outfol}"/ctp_results_all.csv', index=False)"
