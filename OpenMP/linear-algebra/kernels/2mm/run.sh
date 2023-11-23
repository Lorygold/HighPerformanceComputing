#!/bin/bash
# possible dataset_size: MINI_DATASET, SMALL_DATASET,STANDARD_DATASET, LARGE_DATASET, EXTRALARGE_DATASET
# possible STATISTICS: polybench, perf
# ./run.sh FILE_NAME DATA_SIZE N_THREADS STATISTICS

NAME_FILE="${1:-"2mm"}"
DATA_SIZE="${2:-"STANDARD_DATASET"}"
N_THREADS="${3:-"4"}"
STATISTICS="${4:-"none"}"

echo "Running $NAME_FILE with $DATA_SIZE dataset and $N_THREADS threads ($STATISTICS)"
echo "-------------------------------------"
make EXT_CFLAGS="-pg -D$DATA_SIZE -DNTHREADS=$N_THREADS -DPOLYBENCH_TIME" EXT_ARGS="" BENCHMARK=$NAME_FILE clean all run

case $STATISTICS in
    none)
        echo "No statistics"
        ;;
    perf)
        echo "Using perf"
        perf stat ./lu_acc
        ;;
    polybench)
        echo "Using Polybench"
        make benchmark
        ;;
    gprof)
        echo "Using gprof"
        NO_OPT="-O0 -g -fopenmp"
        make EXT_CFLAGS="-pg -D$DATA_SIZE -DNTHREADS=$N_THREADS" EXT_ARGS="" BENCHMARK=$NAME_FILE OPT=$NO_OPT clean all run
        gprof "${NAME_FILE}_acc" gmon.out > analysis.txt
esac