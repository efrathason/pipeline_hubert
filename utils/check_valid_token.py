import json
from lhotse import CutSet
from lhotse.dataset.collation import TokenCollater

#cuts_dev = json.load(open("/export/c07/efrat/pipeline_hubert/pipeline_hubert/data/cuts/cuts_dev.jsonl"))
cuts_dev = CutSet.from_file("/export/c07/efrat/pipeline_hubert/pipeline_hubert/data/cuts/cuts_dev.jsonl.gz")
tokenizer = TokenCollater(cuts_dev)

try:
    tokens, token_lens = tokenizer(cuts_dev)
except:
  print("Something went wrong")

print ("finish")