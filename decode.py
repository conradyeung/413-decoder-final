#!/usr/bin/env python
import gzip
import optparse, bleu
import sys, os, math, numpy
import random
import models
from collections import namedtuple
from random import randint

# input = data/test/all.cn-en.cn

# full
# lm = data/lm/en.gigaword.3g.arpa.gz
# tm = data/large/phrase-table/moses/phrase-table.gz

# medium
# lm = data/lm/en.gigaword.3g.filtered.train_dev_test.arpa.gz
# tm = data/medium/phrase-table/phrase-table.gz

# small
# lm = data/lm/en.gigaword.3g.filtered.dev_test.arpa.gz
# tm = data/small/phrase-table/moses/phrase-table.gz

# tiny
# lm = data/lm/en.tiny.3g.arpa
# tm = data/toy/phrase-table/phrase_table.out

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/test/all.cn-en.cn", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/toy/phrase-table/phrase_table.out", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm/en.tiny.3g.arpa", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=15, type="int", help="Number of sentences to decode (default=5)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=500, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
optparser.add_option("-r", "--translate", dest="tr", action="store_true", default=False,  help="toggle generate translation(default=off)")
optparser.add_option("-d", "--distortion-limit", dest="d", default=3, type="int", help="Distortion limit, default 4")
optparser.add_option("-w", "--weights-model", dest="wv", default="weights",  help="Weight vector to use (default: weights)")

opts = optparser.parse_args()[0]

#add unzip here

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
wv = opts.wv

silence_messages = False

weight_vector = []
for line in open(wv).readlines():
  for weight in line.split():
    weight_vector.append(float(weight))

lm_weight = sum(weight_vector)/4

#weight_vector = [1.0, 1.0, 1.0, 1.0] #default


french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, coverage, start, end, logprobarray")

def bleu_calc(hyp, ref):
    stats = [0 for i in xrange(10)]
    stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(hyp,ref))]
    
    #return smoothed bleu
    return bleu.smoothed_bleu(stats)

#populate translation model with all french words encountered
for fword in set(sum(french,())):
  if (fword,) not in tm:
    tm[(fword,)] = [models.phrase(fword, 0.0, [])]

#print_weighted_features(features_vector)
def process_phrase_logprobs(logprobs):
  result = 0
  for x in len(logprobs):
    result += logprobs[x] * weight_vector[x]
  return result

#expand hypothesis
def check_hypothesis(item, ftext, begin, end):
  if ftext[begin:end] not in tm:
      # translation not found in model, so return none
      return None 
  for i in range(begin, end):
    if item.coverage[i] is 1:
      # already translated, so return none
      return None 

  #update coverage vector
  coveredArea = [itemCoverage for itemCoverage in item.coverage]
  for i in range(begin, end):
    coveredArea[i] = 1
  coveredArea = tuple(coveredArea)       

  result = []

  p = 0
  l = 0       

  #distortion violation
  for i in coveredArea:
    if coveredArea[i] == 0:
      p = i
      break

  for i in coveredArea:
    if coveredArea[i] == 1:
      l = i

  if ((l - p) >= opts.d):
    return None

  #for each phrase/possible translation
  for phrase in tm[ftext[begin:end]]:
      
      # new logprob = sum of item and phrase logprob
      # logprob = item.logprob + weight_vector[x] * phrase.logprobarray[x]

      logprob = item.logprob
      logprobarray = phrase.logprobarray
      for x in range(len(phrase.logprobarray)):
        logprob += phrase.logprobarray[x] * weight_vector[x]

      #logprob += sum(phrase.logprobarray)

      # retrieve score from LM
      lm_state = item.lm_state
      #for each english word
      for eword in phrase.english.split():
          # populate lm state/log prob
          (lm_state, word_logprob) = lm.score(lm_state, eword)
          # add result to word's logprob
          logprob += lm_weight*word_logprob
      
      # distortion factor
      # d = abs(stackItem.end - start - 1)
      # logprob += -2 * d 
      
      # add in the probability that this is the end of the sentence, if applicable
      logprob += lm.end(lm_state) if sum(coveredArea)==len(ftext) else 0.0

      # create a new hypothesis with this phrase translation
      candidate = hypothesis(logprob, lm_state, item, phrase, coveredArea, begin, end, logprobarray)
      #logprob -
      result.append(candidate)

  return result

def print_weighted_features(features):

  features = features.split()
  feature_floats = []
  for x in range(len(features)):
    feature_floats.append(float(features[x]) * weight_vector[x])
  result = ""
  for i in feature_floats:
    result += str(i) + " "
  return result


def add_hypothesis(candidate_list, stacks):
    for candidate in candidate_list:
      # add new hypothesis with key = start, end, coverage
      candidateCoverage = ((candidate.start, candidate.end), candidate.coverage)
      stack_num = sum(candidate.coverage)

      if candidateCoverage not in stacks[stack_num] or stacks[stack_num][candidateCoverage].logprob < candidate.logprob:
        stacks[stack_num][candidateCoverage] = candidate

def extract_english(stackItem): 
  return "" if stackItem.predecessor is None else "%s%s " % (extract_english(stackItem.predecessor), stackItem.phrase.english)

def extract_logprobs(stackItem):
    if stackItem.predecessor is None:
        return [0.0, 0.0, 0.0, 0.0]
    else:
        if stackItem.logprobarray == []:
            return [x+y for x,y in zip([0.0 for _ in xrange(4)], extract_logprobs(stackItem.predecessor))]
        else:
            return [x+y for x,y in zip(stackItem.logprobarray, extract_logprobs(stackItem.predecessor))]

if not silence_messages: sys.stderr.write("Decoding %s...\n" % (opts.input,))
fcount = 0
for fword in french: # for every french sentence
  #initial coverage and hypothesis
  cover = tuple([0 for _ in fword])
  start_hypo = hypothesis(0.0, lm.begin(), None, None, cover, 0, 0, [0])

  #stacks
  stacks = [{} for _ in fword] + [{}]
  stacks[0][((0,0), cover)] = start_hypo

  #each stack
  for stack in stacks[:-1]:
    # prune to top s
    pruneSize = opts.s
    pruneSize += int(round(len(stack) / 50))
    # increase pruneSize by 2% of stack size

    for stackItem in sorted(stack.itervalues(),key=lambda stackItem: -stackItem.logprob)[:pruneSize]: 
      
      #for words before current translation
      for priorWord in range(0, stackItem.start):
        #for each remaining stack item
        for nextWord in range(priorWord+1, stackItem.start+1): 
          candidate_list = check_hypothesis(stackItem, fword, priorWord, nextWord)

          # if hypothesis exisist append it to the stack
          if candidate_list:
            add_hypothesis(candidate_list, stacks)

      #for words after translation
      for charCount in range(stackItem.end, len(fword)):
          for k in range(charCount+1, len(fword)+1):
              candidate_list = check_hypothesis(stackItem, fword, charCount, k)
              
              # if hypothesis != none, add to stack
              if candidate_list:
               add_hypothesis(candidate_list, stacks)

  while stacks[-1] == {}:
    del stacks[-1]

  if(opts.tr == False):
    nbest = sorted(stacks[-1].itervalues(),key=lambda h: -h.logprob)[:100]
    features = ""
    features_vector = []
    for i in nbest:
      features = ""
      features_vector = []
      count = 0

      #for each logprob


      for logprob in extract_logprobs(i):

        
        features_vector = []
        logprob = logprob * weight_vector[count]

        features += " " + str(logprob)
        
        count += 1
      winner_e = extract_english(i)
    print str(fcount) + " ||| " + winner_e + " ||| " + print_weighted_features(features) #str(winner.logprob)
    
  if(opts.tr == True):
    winner = max(stacks[-1].itervalues(), key=lambda stackItem: stackItem.logprob)
    print extract_english(winner)

  fcount += 1

