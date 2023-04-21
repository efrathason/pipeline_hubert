from os import path, makedirs, listdir
import shutil

src_dirs=[
'/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_1_and_2/',
'/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_3/',
'/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_4_1/',
'/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_4_2/',
'/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_5/',
'/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_6/'
 ]
dst_dir = '/data/skhudan1/corpora/VERBIT_arabic/mix_milestones/'
makedirs(dst_dir, exist_ok=True)
types = ['audio_wav_mixed_fixed/', 'stm_clean_fixed/']
for t in types:
    makedirs(dst_dir+t, exist_ok=True)

for src_dir in src_dirs:
    print(src_dir)
    for t in types:
        print(t)
        files = listdir(src_dir+t)
        for file_name in files:
            try:
                shutil.copy2(src_dir+t+file_name, dst_dir+t+file_name)
            except:
                print("problem:" + file_name)
