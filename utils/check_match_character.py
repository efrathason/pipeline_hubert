from lhotse import CutSet
def removeDuplicate(dup_str):
    s= set(dup_str)
    s= "".join(s)
    return s

cuts_train = CutSet.from_file("/home/eorenst1/pipeline_hubert/data/cuts/cuts_train.jsonl.gz")
#cut_train_sample = cuts_train.subset(first=10000)
merge_str_train = ""
train_list = list(cuts_train.data.items())
for item in train_list:
    for supervision in item[1].supervisions:
        merge_str_train = merge_str_train+supervision.text
        merge_str_train = removeDuplicate(merge_str_train)
print(merge_str_train)
print(len(merge_str_train))
cuts_dev =CutSet.from_file("/home/eorenst1/pipeline_hubert/data/cuts/cuts_dev.jsonl.gz")
#cut_dev_sample = cuts_dev.subset(first=1000)
merge_str_dev = ""
dev_list = list(cuts_dev.data.items())
for item in dev_list:
    for supervision in item[1].supervisions:
        merge_str_dev = merge_str_dev+supervision.text
        merge_str_dev = removeDuplicate(merge_str_dev)
print(merge_str_dev)
print(len(merge_str_dev))
problem_char=""
for c in merge_str_dev:
    if c not in merge_str_train:
        problem_char = problem_char+c
print (problem_char)