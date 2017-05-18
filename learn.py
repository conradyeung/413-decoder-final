#!/usr/bin/env python
import optparse, sys, os, math, numpy, bleu, gzip, random
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("data", "15default.train"), help="N-best file")
optparser.add_option("-t", "--tau", dest="tau", default=1000, help="tau parameter")
optparser.add_option("-x", "--xi", dest="xi", default=1000, help="xi parameter")
optparser.add_option("-a", "--alpha", dest="alpha", default=0.05, help="alpha parameter")
optparser.add_option("-e", "--eta", dest="eta", default=0.1, help="eta parameter")
optparser.add_option("-i", "--iterations", dest="iter", default=6, help="# of epochs")
optparser.add_option("-w", "--weights-model", dest="weight", default="weights",  help="initial Weight vector to use (default: weights)")
(opts, _) = optparser.parse_args()


#calculate bleu score
def bleu_calc(hyp, ref):
    stats = [0 for i in xrange(10)]
    stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(hyp,ref))]
    
    #return smoothed bleu
    return bleu.smoothed_bleu(stats)

#normalize weights
w = []
for line in open(opts.weight).readlines():
  for weight in line.split():
    w.append(float(weight))

sentence_num_counter = 0
num_lines = sum(1 for line in open('data/test/all.cn-en.en0'))
nbests = [[] for _ in xrange(num_lines)]
n_count = 0
#for each line in .nbest file
for line in open(opts.nbest):
  #parse line
  (i, sentence, features) = line.strip().split("|||")
  if(features == ""):
    print("test")
    continue
  sentence_number = int(i)
  # optionally break if limit reached
  # if(sentence_number > 300):
  # 	break
  if(sentence_number > sentence_num_counter):
  	#sys.stderr.write(str(sentence_number))
  	sentence_num_counter += 1
  	n_count = 0
  if (n_count > 15):
  	continue
  n_count += 1
  sentence = sentence.lstrip(' ')
  features = [float(h) for h in features.strip().split()]
  reference = open('data/test/all.cn-en.en0')
  reference_lines = reference.readlines()
  #calculate bleu score for sentence
  bleu_score = bleu_calc(sentence, reference_lines[sentence_number])
  #append to nbests sentence array
  nbests[sentence_number].append((sentence, bleu_score, features))
  #print("sentence number "+str(sentence_number)+" sentence and score is "+str(nbests[sentence_number][-1])+ "\n")

# while (len(nbests[-1]) == 0):
#   print "test"
#   del nbests[-1]

def train(w):
  epoch = 0
  mistakes = 0
  # until completion
  while (epoch < opts.iter):
    sys.stderr.write(str(epoch))
    # for each i sentence in nbests[i] that contains a number of hypotheses
    for nbest in nbests:
      sample = []
      for x in xrange(opts.tau):
        #tau parameter
        random.seed()
        first = random.randint(0, len(nbest)-1)
        second = random.randint(0, len(nbest)-1)
        #append results to sample
        if (math.fabs(nbest[first][1] - nbest[second][1]) > opts.alpha):
          if (nbest[first][1] > nbest[second][1]):
            sample.append((nbest[first][0], nbest[first][2], nbest[second][0], nbest[second][2], math.fabs(nbest[first][1] - nbest[second][1])))
          else:
            sample.append((nbest[second][0], nbest[second][2], nbest[first][0], nbest[first][2], math.fabs(nbest[first][1] - nbest[second][1])))
        else:
          continue
      sample.sort(key=lambda tup: tup[4], reverse = True)
      # xi parameter
      for i in range(opts.xi):
        # get features
        first_features = numpy.asarray([x for x in nbest[first][2]])
        second_features = numpy.asarray([x for x in nbest[second][2]])
        # apply weights
        first_features = w*first_features
        second_features = w*second_features
        # count mistakes
        if (sum(first_features) <= sum(second_features)):
          mistakes += 1
          feature_index = 0
          # train weights during mistakes
          for weight in w:
            add = opts.eta*(nbest[first][2][feature_index] - nbest[second][2][feature_index])
            weight += add
            w[feature_index] = weight
            feature_index += 1
    #check for convergence
    epoch +=1


x = [list(w) for i in range(5)]

train(w)
for copy in x:
  train(copy)

final = [(a+b+c+d+e+f)/5 for a,b,c,d,e,f in zip(w,x[0],x[1],x[2],x[3],x[4])]

# output 
output = ""
for weight in w:
  output += str(weight) + " "
output = output[:-1]
print(output)
