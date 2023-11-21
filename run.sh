#!/bin/bash
# possible dataset_size: MINI_DATASET, SMALL_DATASET,STANDARD_DATASET, LARGE_DATASET, EXTRALARGE_DATASET
# possible STATISTICS: polybench, perf
# ./run.sh FILE_NAME DATA_SIZE N_THREADS

NAME_FILE="${1:-"2mm"}"
DATA_SIZE="${2:-"STANDARD_DATASET"}"
N_THREADS="${3:-"4"}"

echo "Running $NAME_FILE with $DATA_SIZE dataset and $N_THREADS threads ($STATISTICS)"
echo "-------------------------------------"
make EXT_CFLAGS="-pg -D$DATA_SIZE -DNTHREADS=$N_THREADS -DPOLYBENCH_TIME" EXT_ARGS="" BENCHMARK=$NAME_FILE clean all run

