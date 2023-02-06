from pathlib import Path

from lhotse import CutSet
from lhotse.recipes import prepare_mgb2
from mgb2_utils.compute_fbank_mgb2 import compute_fbank_mgb2
# This function is going to get the MGBG2
def save_manifest(manifest_dir, download=False):

    # the corpous MGB2 is in the dl directory:

    dl_dir = "/export/c06/efrat/icefall/egs/mgb2/ASR/download/mgb2"

    # Use the lhotse data prep functions to get a CutSet for the train set
    # and also the dev set.
    dict_manifest = prepare_mgb2(dl_dir, manifest_dir)

def save_data(manifest_dir, cuts_dir):

    compute_fbank_mgb2(manifest_dir, cuts_dir)

def get_data(cuts_dir):


    cuts_dir = Path(cuts_dir)
    #cuts_train = CutSet.from_file(cuts_dir / f"cuts_train_shuf.jsonl.gz")
    #cuts_train = cuts_train.filter(lambda c: c.duration <40)
    #cuts_train = cuts_train.filter (lambda s: s.duration <10)
    #cuts_train = cuts_train.filter (lambda s: s.duration >5)
    #cuts_train = cuts_train.subset(first=10000)
    shared_dir = Path("/export/c07/efrat/pipeline_hubert/pipeline_hubert/data/train_shared3")
    shards = [str(path) for path in sorted(shared_dir.glob("shard_*-0.tar"))]
    print(shards)
    cuts_train_webdataset = CutSet.from_webdataset(
        shards,
        split_by_worker=True,
        split_by_node=True,
        shuffle_shards=True,
    )

    cuts_train_webdataset = cuts_train_webdataset.filter(lambda c: c.duration <40)
    print ("finish create cut_Set")
    cuts_dev =CutSet.from_file(cuts_dir / f"cuts_dev_deleted_slash.jsonl.gz")
    cuts_dev = cuts_dev.filter(lambda c: c.duration <40)
    #cuts_dev = cuts_dev.filter (lambda s: s.duration <10)
    #cuts_dev = cuts_dev.filter (lambda s: s.duration >5)
    cuts_dev = cuts_dev.subset(first=100)
    
    cuts_test = CutSet.from_file(cuts_dir / f"cuts_test.jsonl.gz")
    cuts_test = cuts_test.filter(lambda c: c.duration <40)
    #cuts_test = cuts_test.filter (lambda s: s.duration <10)
    #cuts_test = cuts_test.filter (lambda s: s.duration >5)
    cuts_test = cuts_test.subset(first=50)
    # We only want a single transcript, known as a supervision in lhotse, for
    # underlying "cut" of audio

    #cuts_train = cuts_train.trim_to_supervisions(keep_overlapping=False)
    #cuts_dev = cuts_dev.trim_to_supervisions(keep_overlapping=False)
    #cuts_test = cuts_test.trim_to_supervisions(keep_overlapping=False)
    return cuts_train_webdataset, cuts_dev, cuts_test