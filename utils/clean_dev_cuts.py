from lhotse import CutSet

cuts_dev =CutSet.from_file("/home/eorenst1/pipeline_hubert/data/cuts/cuts_dev.jsonl.gz")
#cuts_dev.describe()
#cuts_dev = cuts_dev.subset(first=1)
#print(cuts_dev.data)
cuts_dev_without_slash = cuts_dev.filter(lambda c: '\\' not in c.supervisions[0].text )
cuts_dev_lower_case = cuts_dev_without_slash.filter(lambda c: c.supervisions[0].text.islower() )
cuts_dev_lower_case.describe()