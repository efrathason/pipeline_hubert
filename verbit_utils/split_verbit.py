import os
import random
import os.path as path
import shutil
import math
def split_to_subsets(seed: int, subset_weights, audio_dir, stm_dir):
    random.seed(seed)
    audio_files = os.listdir(audio_dir)
    
    num_audio_files = len(audio_files)
    print(num_audio_files)
    num_audio_files_per_subset={}
    for k in subset_weights.keys(): 
        num_audio_files_per_subset[k] = num_audio_files * subset_weights[k] / sum(subset_weights.values())
    
    print(num_audio_files_per_subset)
    print(audio_files[1])
    random.shuffle(sorted(audio_files))
    i = 0
    dst_dir = "/data/skhudan1/corpora/VERBIT_arabic/splited/"
    os.makedirs(dst_dir, exist_ok=True)
    for subset_audio_files in num_audio_files_per_subset.keys():
        num_subset_audio_files=num_audio_files_per_subset[subset_audio_files]
        split_k = subset_audio_files
        os.makedirs(dst_dir+split_k, exist_ok=True)
        os.makedirs(dst_dir+split_k+'/wav', exist_ok=True)
        os.makedirs(dst_dir+split_k+'/stm', exist_ok=True)
        subset = []
        print(audio_files[1])
        for audio_file in audio_files[i:math.floor(i+num_subset_audio_files)]:
            #try:
            audio_id = os.path.splitext(audio_file)[0]
            shutil.copy2(audio_dir+audio_file, dst_dir+split_k+'/wav/'+audio_file)
            shutil.copy2(stm_dir+audio_id+'.stm', dst_dir+split_k+'/stm/'+audio_id+'.stm')
            #except:
            #    print("problem:" + audio_file)
        i += math.floor(num_subset_audio_files)


    print(num_audio_files_per_subset)
	

audio_dir = '/data/skhudan1/corpora/VERBIT_arabic/mix_milestones/audio_wav_mixed_fixed/'
stm_dir = '/data/skhudan1/corpora/VERBIT_arabic/mix_milestones/stm_clean_fixed/'
subset_weights = {"train":0.8, "dev":0.15, "test":0.15}
seed = 624
subsets = split_to_subsets(seed, subset_weights, audio_dir, stm_dir)

#for subset in subsets:
