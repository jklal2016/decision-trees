#!/bin/bash
# Request 5 CPU cores
#$ -pe smp 5
# Request 4 Gigabytes of memory per core
#$ -l rmem=4G

# Load modules for spark
module load apps/java/jdk1.8.0_102/binary
module load apps/spark/2.1.0/gcc-4.8.5

# Run the scala program
# Ensure that the number of cores we try to use
# matches the number we have requested. i.e. 5
time spark-submit --master local[5] target/scala-2.11/exercisereg_2.11-1.0.jar