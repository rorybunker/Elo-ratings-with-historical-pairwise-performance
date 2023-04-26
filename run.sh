#!/bin/bash
source ${HOME}/workspace2/work/elo_hist_perf/venvs/elohpp/bin/activate elohpp

eloTypeArray=(std hpp)
optimizeArray=(N Y)
regressMeanArray=(N Y)

start_time=$(date +%s)

iter=0
for E in ${eloTypeArray[@]}; do
    for O in ${optimizeArray[@]}; do
        for R in ${regressMeanArray[@]}; do
            iter_start_time=$(date +%s)
            echo Iteration $iter Elo type $E optimize $O regress towards mean $R
            python -u elo_hpp_ncaa.py --elo_type $E --optimize $O --regress_towards_mean $R
            iter_end_time=$(date +%s)
            iter_running_time=$((iter_end_time-iter_start_time))
            echo "Iteration $iter running time: $iter_running_time seconds"
            ((iter++))
        done
    done
done

end_time=$(date +%s)
total_running_time=$((end_time-start_time))
echo "Total running time: $total_running_time seconds"
