from os import path, makedirs, listdir
import shutil

src_dirs=['/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_3/audio/broadcasts_wavs/', 
'/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_3/audio/calls/']
dst_dir = '/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_3/audio_wav_mixed/'
makedirs(dst_dir, exist_ok=True)
for src_dir in src_dirs:
    files = listdir(src_dir)
    for file_name in files:
        shutil.copy2(src_dir+file_name, dst_dir+file_name)
