import numpy as np
import sys
import os
import re

beamsize = 5

input_1 = "./shallow_trans/trans_allbeams{}.sys".format(beamsize)
input_2 = "./deep_trans/trans_allbeams{}.sys".format(beamsize)
score_11 = "./shallow_trans/trans_allbeams{}_shallow_shallow.score".format(beamsize)
score_12 = "./shallow_trans/trans_allbeams{}_shallow_deep.score".format(beamsize)
score_21 = "./deep_trans/trans_allbeams{}_deep_shallow.score".format(beamsize)
score_22 = "./deep_trans/trans_allbeams{}_deep_deep.score".format(beamsize)

tmpF = "./tmp"
bleulog = "./bleu.log"

#specify the reference file (text file)
reference = './data/en-de.ref.de'
perl_script = "perl ./tiny-moses/multi-bleu.pl {} < {} >> {}".format(reference, tmpF, bleulog)

def read_file(fname):
  with open(fname, "r", encoding="utf8") as ff:
    all_lines = [x.strip() for x in ff]    
  return all_lines, np.array([len(x.split()) for x in all_lines]).astype("float32")

def read_score(fname):
  print(fname)
  all_lines, _ = read_file(fname)
  try:
    return np.array([float(x) for x in all_lines] ).astype("float32")
  except:
    return np.array([float(x) for x in all_lines[:-1]] ).astype("float32")

def process_one_file(lines, cnt, S1, S2, alpha, gamma):
  SS = alpha * S1 + (1. - alpha) * S2
  SS = SS / (cnt + 1.) ** gamma
  SS = SS.reshape((-1, beamsize))
  idx_list = SS.argmax(axis=1).tolist()
  ret_lines, ret_scores = [], []
  for ii, idx in enumerate(idx_list):
    ret_lines.append(lines[ii * beamsize + idx])
    ret_scores.append(SS[ii,idx])

  return ret_lines, ret_scores

def find_winner(ret_lines_1, ret_lines_2, score_1, score_2):
  assert len(ret_lines_1) == len(ret_lines_2) == len(score_1) == len(score_2)
  return [l1 if s1 > s2 else l2 for (l1, l2, s1, s2) in zip(ret_lines_1, ret_lines_2, score_1, score_2)]
  
# (translations, length of translations)
all_lines_1, cnt_1 = read_file(input_1)
all_lines_2, cnt_2 = read_file(input_2)

score_11 = read_score(score_11)
score_12 = read_score(score_12)

score_21 = read_score(score_21)
score_22 = read_score(score_22)

"""Alpha and Beta can be set according to the validation performance"""
alpha = 0.5 
gamma = -0.4

ret_lines_1, ret_scores_1 = process_one_file(all_lines_1, cnt_1, score_11, score_12, alpha, gamma)
ret_lines_2, ret_scores_2 = process_one_file(all_lines_2, cnt_2, score_22, score_21, alpha, gamma)
winner_lines = find_winner(ret_lines_1, ret_lines_2, ret_scores_1, ret_scores_2)
with open(tmpF, "w", encoding="utf8") as fw:
  for x in winner_lines: 
    print(re.sub("(@@ )|(@@?$)", "", x), file=fw)

os.system(perl_script)
os.system("rm " + tmpF)

