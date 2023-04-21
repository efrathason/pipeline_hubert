import os
import shutil

src_audio_dir = '/data/skhudan1/corpora/VERBIT_arabic/mix_milestones/audio_wav_mixed_fixed'
#dst_audio_dir = '/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_6/audio_wav_mixed_fixed/'
src_stm_dir = '/data/skhudan1/corpora/VERBIT_arabic/mix_milestones/stm_clean_fixed'
#dst_stm_dir = '/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_6/stm_clean_fixed/'

#os.makedirs(dst_audio_dir, exist_ok=True)
#os.makedirs(dst_stm_dir, exist_ok=True)
audio_files = os.listdir(src_audio_dir)
stm_files = os.listdir(src_stm_dir)
print("count audio:" + str(len(audio_files)))
print ("count stm:" + str(len(stm_files)))
count=0
for audio_file in audio_files:
    audio_id = os.path.splitext(audio_file)[0]
    stm_file = audio_id+".stm"
    if not (audio_id+".stm" in stm_files):
        print (audio_id)
        