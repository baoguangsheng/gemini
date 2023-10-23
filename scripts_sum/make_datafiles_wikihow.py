import sys
import os
import hashlib
import struct
import subprocess
import collections

import regex as re
from tqdm import tqdm


# These are the number of article files we expect there to be in articles
num_expected_wikihow_articles = 180123

def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines

# Guangsheng Bao: follow the preprocessing in Contextualized Rewriting
def preprocess(line):
  if len(line) == 0:
    return []

  # restore brackets
  escapes = {'-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']', '-LCB-': '{', '-RCB-': '}'}
  for key in escapes:
    line = line.replace(key, escapes[key])

  # fix strange comma and period
  # starting with a punctuation
  if line[0] in [',', ';', '.', '?', '!']:
    line = line[1:].strip()

  # ending with two punctuations
  replaces = {'. ;': ' .', '. ,': ' .'}
  for key in replaces:
    if line.find(key) >= 0:
      # idx = line.find(key)
      # print(line[idx - 5: idx + 10])
      line = line.replace(key, replaces[key])

  # unseparated sentences
  pattern = '[a-z]+\.[A-Z]'
  all = re.findall(pattern, line)
  if len(all) > 0:
    # print(all)
    line = re.sub(pattern, lambda m: m.group(0).replace('.', ' . '), line)

  # break paragraph into sentences
  sent_ends = [".", "!", "?"]
  quot_starts = ["``", "`", "(", "[", "{"]
  quot_ends = ["''", "'", ")", "]", "}"]
  sent_end = False
  quot_in = 0
  sents = [[]]
  for w in line.split():
    if w in quot_starts:
      quot_in += 1
    elif w in quot_ends:
      quot_in = quot_in - 1 if quot_in > 0 else 0
    elif w in sent_ends:
      sent_end = True
    sents[-1].append(w)
    if sent_end and quot_in == 0:
      sents.append([])
      sent_end = False

  # `` xxx '' -> " xxx "
  sents = [['"' if w in ["``", "''"] else w for w in sent] for sent in sents if len(sent) > 0]
  sents = [' '.join(sent) for sent in sents]

  # log for checking unnormal ending
  # if len(sents) > 1:
  #   check = ['.', '!', '?']
  #   for c in check:
  #     for sent in sents:
  #       if 0 < sent.find(c) < len(sent) - 1:
  #         print(sent)
  return sents

def get_art_abs(article_file, doc_sep):
  lines = read_text_file(article_file)

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = sum([preprocess(line) for line in lines], [])

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if len(line) < 5:
      continue # empty line
    elif line.startswith("@summary"):
      next_is_highlight = True
    elif line.startswith("@article"):
      next_is_highlight = False
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article/abstract into a single string
  article = doc_sep.join(article_lines)
  abstract = doc_sep.join(highlights)
  return article, abstract


def write_to_bin(args, title_file, out_prefix, doc_sep):
  print("Making bin file for articles listed in %s..." % title_file)
  title_list = read_text_file(title_file)
  article_fnames = [s+".txt" for s in title_list]

  with open(out_prefix + '.source', 'wt') as source_file, open(out_prefix + '.target', 'wt') as target_file:
    for idx,s in enumerate(tqdm(article_fnames)):
      # Look in the article dirs to find the .txt file corresponding to this title
      if os.path.isfile(os.path.join(args.articles, s)):
        article_file = os.path.join(args.articles, s)
      else:
        print("Error: Couldn't find article file %s in either article directories %s." % (s, args.articles))
        # Check again if stories directories contain correct number of files
        print("Checking that the articles directories %s contain correct number of files..." % args.articles)
        # check_num_articles(args.articles, num_expected_wikihow_articles)
        # raise Exception("Articles directories %s contain correct number of files but article file %s found in neither." % (args.articles, s))
        continue

      # Get the strings to write to .bin file
      article, abstract = get_art_abs(article_file, doc_sep)

      # Write article and abstract to files
      source_file.write(article + '\n')
      target_file.write(abstract + '\n')

  print("Finished writing files")

def check_num_articles(articles_dir, num_expected):
  num_articles = len(os.listdir(articles_dir))
  if num_articles != num_expected:
    raise Exception("articles directory %s contains %i files but should contain %i" % (articles_dir, num_articles, num_expected))


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--wikihow', default='../shared/wikihow/')
  parser.add_argument('--articles', default='../shared/wikihow/articles_tokenized/')
  parser.add_argument("--res", default='exp_test/wikihow.tokenized')
  parser.add_argument("--sep", default=' ')
  args, unknown = parser.parse_known_args()

  check_num_articles(args.articles, num_expected_wikihow_articles)

  # Create some new directories
  if not os.path.exists(args.res):
    os.makedirs(args.res)

  # Read the stories, do a little postprocessing then write to bin files
  write_to_bin(args, "%s/all_test.txt" % args.wikihow, os.path.join(args.res, "test"), doc_sep=args.sep)
  write_to_bin(args, "%s/all_val.txt" % args.wikihow, os.path.join(args.res, "valid"), doc_sep=args.sep)
  write_to_bin(args, "%s/all_train.txt" % args.wikihow, os.path.join(args.res, "train"), doc_sep=args.sep)

if __name__ == '__main__':
  main()