from os import path, makedirs
import glob
from pydub import AudioSegment
import sys
sys.path.append('/home/eorenst1/ffmpeg')

def convert_from_mp3(src, dst):
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")

list_to_convert=glob.glob("/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_3/audio/broadcasts/*.mp3")
dst_dir='/data/skhudan1/corpora/VERBIT_arabic/milestones/milestone_3/audio/broadcasts_wavs'
makedirs(dst_dir, exist_ok=True)
dst_format='wav'
for src_path in list_to_convert:
    file_name = path.basename(src_path)
    name_withput_ext = path.splitext(file_name)[0]
    dst_path =path.join(dst_dir, name_withput_ext + '.' + dst_format)
    convert_from_mp3(src_path, dst_path)

