# 413-decoder-final

Weight Vector Tuning in a Natural Language Filtered-Phrase Decoder
CMPT 413
Dec 14 2016

Matt Hannah
mmh3@sfu.ca
Conrad Yeung
conrad_y@sfu.ca

To run, use    

python decode.py

Change appropriate options in decode.py to use different language models, translation models, stack size, candidate numbers, etc. It will output the candidate(s) and their cumulative weighted feature phrase scores.

Use the reranker to experiment with iterative learning. It will modify the weights file. Use the shell script to automate the feedback loop.