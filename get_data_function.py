from pathlib import Path

from lhotse import CutSet
from lhotse.recipes import prepare_mgb2
from mgb2_utils.compute_fbank_mgb2 import compute_fbank_mgb2
from ASRDataset_class import ASRDataset
from lhotse.dataset import DynamicBucketingSampler
import torch

main_dir = "/home/eorenst1/pipeline_hubert"
# This function is going to get the MGBG2
def save_manifest(manifest_dir, download=False):

    # the corpous MGB2 is in the dl directory:

    dl_dir = "/data/skhudan1/corpora/mgb2"

    # Use the lhotse data prep functions to get a CutSet for the train set
    # and also the dev set.
    dict_manifest = prepare_mgb2(dl_dir, manifest_dir)

def save_data(manifest_dir, cuts_dir):

    compute_fbank_mgb2(manifest_dir, cuts_dir)

def get_data(cuts_dir):

    cuts_dir = Path(cuts_dir)
    cuts_train = CutSet.from_file(cuts_dir / f"cuts_train.jsonl.gz")
    cuts_train = cuts_train.filter(lambda c: c.duration >1)
    cuts_train = cuts_train.filter(lambda c: c.duration <40)
    #cuts_train = cuts_train.filter (lambda s: s.duration <10)
    #cuts_train = cuts_train.filter (lambda s: s.duration >5)
    cuts_train = cuts_train.subset(first=100000)
    #cuts_train = cuts_train.subset(first=10000)

    #shared_dir = Path(main_dir + "/data/train_shared3")
    #shards = [str(path) for path in sorted(shared_dir.glob("shard_0000000[12345]-0.tar"))]
    #print(shards)
    #cuts_train_webdataset = CutSet.from_webdataset(
    #   shards,
    #    shuffle_shards=True
    #)
    

    #cuts_train_webdataset = cuts_train_webdataset.filter(lambda c: c.duration <25)
    #cuts_train_webdataset.describe()
    print ("finish create cut_Set")
    cuts_dev =CutSet.from_file(cuts_dir / f"cuts_dev.jsonl.gz")
    cuts_dev = cuts_dev.filter(lambda c: c.duration <40)
    cuts_dev = cuts_dev.filter(lambda c: c.duration >1)
    #cuts_dev = cuts_dev.filter (lambda s: s.duration <10)
    #cuts_dev = cuts_dev.filter (lambda s: s.duration >5)
    #cuts_dev = cuts_dev.subset(first=100)
    cuts_dev = cuts_dev.filter(lambda c: '\\' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: 'I' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: '@' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: '+' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: '\ٌٍُ' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: 'C' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: 'B' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: 'U' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: 'M' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: 'Oِ' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: 'L' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: 'گ' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: 'W' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: '’' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: 'ﻷَ' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: 'ٱ' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: 'پْ' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: 'R' not in c.supervisions[0].text )
    cuts_dev = cuts_dev.filter(lambda c: 'ﻷ' not in c.supervisions[0].text )
    cuts_test = CutSet.from_file(cuts_dir / f"cuts_test.jsonl.gz")
    cuts_test = cuts_test.filter(lambda c: c.duration <40)
    #cuts_test = cuts_test.filter (lambda s: s.duration <10)
    #cuts_test = cuts_test.filter (lambda s: s.duration >5)
    #cuts_test = cuts_test.subset(first=50)
    # We only want a single transcript, known as a supervision in lhotse, for
    # underlying "cut" of audio

    #cuts_train = cuts_train.trim_to_supervisions(keep_overlapping=False)
    #cuts_dev = cuts_dev.trim_to_supervisions(keep_overlapping=False)
    #cuts_test = cuts_test.trim_to_supervisions(keep_overlapping=False)
    return cuts_train, cuts_dev, cuts_test

def get_dloader(cuts, tokenizer, max_duration ):
    # Define the dataset, samplers and data loaders.
    # These are responsible for batching the data during nnet training
    dataset = ASRDataset(tokenizer)
    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=max_duration,
        shuffle=True,
        num_buckets=100,
    )
    
    dloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=None, num_workers=4,
    )
    
    return dloader
