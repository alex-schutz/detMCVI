for N in $(seq 5 5 30); do
  outfol=experiments/Maze/evaluation/sarsop/${N}x${N}
  mkdir -p ${outfol}
  for i in $(seq 0 9); do
    pfile=experiments/Maze/evaluation/${N}x${N}/${N}x${N}_${i}.txt
    pomdpfile=${pfile}.pomdp
    outfile=${outfol}/MazeInstance_${N}_${i}.txt
    start=$(date +%s.%N)
    build/experiments/Maze/maze_to_sarsop ${pfile} \
      && taskset --cpu-list 1 /home/alex/ori/infjesp/third_party_dependencies/sarsop/src/pomdpsol ${pomdpfile} --memory 16000 \
      && /home/alex/ori/infjesp/third_party_dependencies/sarsop/src/polgraph ${pomdpfile} --policy-file out.policy --policy-graph policy.dot \
      || break
    end=$(date +%s.%N)
    runtime=$(echo "$end - $start" | bc -l)
    echo $runtime
    build/experiments/Maze/evaluate_sarsop ${pfile} --max_sim_depth $(echo $N*$N*10 | bc) > ${outfile}
    python3 -c "from experiments.Maze.evaluation_set import parse_file; df = parse_file('"${outfile}"'); df['Timestamp']="${runtime}"; df['Set number']="${i}"; df.to_csv('"${outfol}"/maze_results_"${N}_${i}".csv', index=False)"
  done
  python3 -c "import pandas as pd; results=[pd.read_csv(open(f'"${outfol}"/maze_results_"${N}"_{i}.csv', 'r')) for i in range(10)]; df=pd.concat(results, ignore_index=True, sort=False); df.to_csv('"${outfol}"/maze_results_all.csv', index=False)"
done
