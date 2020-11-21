#!/bin/sh
echo N num_iterations time_per_iteration total_time
for i in 10 100 1000 10000 25000 50000 75000 100000 250000
  do
    echo `./build/GBP $i`
  done
