import os
import get_data_function
from lhotse.dataset.webdataset import export_to_webdataset
from lhotse import CutSet, Fbank
from joblib import Parallel, delayed
import multiprocessing

download = False
data_dir = "/export/c07/efrat/pipeline_hubert/pipeline_hubert/data"
shared_dir = data_dir+"/train_shared3"
print("prepare the manifests to the directory")
#get_data_function.save_manifest(data_dir+"/manifests")
def create_shared(path):
    print(f"the file is: {path}")
    cuts_train = CutSet.from_file(f"{cuts_split_dir}/{path}")
    cuts_number = path.split(".")[1]
    export_to_webdataset(
        cuts_train,
        output_path=f"{shared_dir}/shard_{cuts_number}-%d.tar",
        shard_size=1000,
        audio_format="wav",
    )

print("prepare the cuts to the directory")
#get_data_function.save_data(data_dir+"/manifests", data_dir+"/cuts")

# Get the data
#cuts_train, cuts_dev, cuts_test= get_data_function.get_data(data_dir+"/cuts")

#cuts_train.describe()
cuts_split_dir = "/export/c07/efrat/pipeline_hubert/pipeline_hubert/data/split_train_cuts"
paths = os.listdir(cuts_split_dir)
num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(create_shared)(path) for path in paths)

#for path in os.listdir(cuts_split_dir)[0:5]:
#    ceate_shared(path)


#cuts_train.to_shar(data_dir+"/train_shared", shard_size = 100, fields={"recording": "wav", "features": "lilcom"})
#count_data = len(list(cuts_train.data.items()))
#print (count_data)
#print('the first item: {}', list(cuts_train.data.items())[0])