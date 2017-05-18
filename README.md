# CMPT 413 Final Project
Weight Vector Tuning in a Natural Language Filtered-Phrase Decoder
CMPT 413
Dec 14 2016

Matt Hannah
Conrad Yeung

A Chinese to English machine translation system using a filtered phrase decoding system that uses a reranking feedback loop to train weighted features for the decoder.

Full experiment results and findings located in project paper below.

https://github.com/conradyeung/413-decoder-final/blob/master/project.pdf

To run, use    
python decode.py

Change appropriate options in decode.py to use different language models, translation models, stack size, candidate numbers, etc. It will output the candidate(s) and their cumulative weighted feature phrase scores.

Use the reranker to experiment with iterative learning. It will modify the weights file. Use the shell script to automate the feedback loop.
