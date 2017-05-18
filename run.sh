#!/bin/bash
echo "initializing weights"
echo "1 1 1 1" > weights


for i in {0..4}
do
	echo "iteration $i"
	echo "creating nbest list"
	python decode.py > train
	echo "creating new weights"
	python learn.py > weights
done

echo "outputting final translation to final.output"

python decode.py -r > final.output
