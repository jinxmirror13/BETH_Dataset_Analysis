#!/bin/bash

model="svm"
for seed in {1..5}; do 
	python run_benchmark.py --vis --vis-latents --seed $seed --train --benchmark $model --dataset gaussian 
done

python run_benchmark.py --vis --vis-latents --test --benchmark $model --dataset gaussian
