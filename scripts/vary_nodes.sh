#!/bin/sh
echo N num_iterations time_per_iteration total_time
for i in 100 1000 10000 100000
  do
    echo `./build/GBP $i`
  done
