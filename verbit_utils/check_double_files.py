import os

audio_dir12 = '/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_1_and_2/audio_wav_mixed_fixed/'
audio_dir3 = '/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_3/audio_wav_mixed_fixed/'
audio_dir41 = '/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_4_1/audio_wav_mixed_fixed/'
audio_dir42 = '/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_4_2/audio_wav_mixed_fixed/'
audio_dir5 = '/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_5/audio_wav_mixed_fixed/'
audio_dir6 = '/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_6/audio_wav_mixed_fixed/'

all_files=[]
dirs = [audio_dir12, audio_dir3, audio_dir41, audio_dir42, audio_dir5, audio_dir6]
for audio_dir in dirs:
    files = os.listdir(audio_dir)
    for f in files:
        all_files.append(f)
print(len(all_files))
print(len(set(all_files)))