import sys
import os
import hashlib
import struct
import subprocess
import collections
import make_datafiles_cnndm as cnndm
import make_datafiles_wikihow as wikihow
import make_datafiles_xsum as xsum

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='xsum', choices=['cnndm', 'wikihow', 'xsum'])
  args, unknown = parser.parse_known_args()

  processors = {'cnndm': cnndm, 'wikihow': wikihow, 'xsum': xsum}
  processors[args.dataset].main()
