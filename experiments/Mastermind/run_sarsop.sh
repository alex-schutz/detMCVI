outfol=experiments/Mastermind/evaluation/sarsop/5_3
mkdir -p ${outfol}
pfile=experiments/Mastermind/evaluation/5_3/mm.txt
pomdpfile=${pfile}.pomdp
outfile=${outfol}/MMInstance.txt
start=$(date +%s.%N)
build/experiments/Mastermind/mastermind_to_sarsop ${pfile} \
  && taskset --cpu-list 1 /home/alex/ori/infjesp/third_party_dependencies/sarsop/src/pomdpsol ${pomdpfile} \
  && /home/alex/ori/infjesp/third_party_dependencies/sarsop/src/polgraph ${pomdpfile} --policy-file out.policy --policy-graph policy.dot
end=$(date +%s.%N)
runtime=$(echo "$end - $start" | bc -l)
echo $runtime
build/experiments/Mastermind/mastermind_evaluate_sarsop ${pfile} --max_sim_depth 24 > ${outfile}
python3 -c "from experiments.Maze.evaluation_set import parse_file; df = parse_file('"${outfile}"'); df['Timestamp']="${runtime}"; df['Set number']=0; df.to_csv('"${outfol}"/mastermind_results_sarsop.csv', index=False)"
