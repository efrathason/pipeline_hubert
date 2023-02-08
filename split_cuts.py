from lhotse import CutSet

path_cuts = ""
cuts = CutSet.from_file(path_cuts)

dir_split_cuts = ""
cuts_per_shard = 1000
cuts.split_lazy(dir_split_cuts, cuts_per_shard, prefix="cuts") 

