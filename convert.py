from os import path
from pydub import AudioSegment

# files                                                                         
src = "Datasets/arc/afrikaans1.mp3"
dst = src+".wav"

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")