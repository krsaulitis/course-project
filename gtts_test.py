import gtts.lang
from gtts import gTTS
from playsound import playsound

tts = gTTS(text='Sveiki, šis ir Google teksta sintēzes piemērs.', lang='lv')

print(gtts.lang.tts_langs())

tts.save("audios/gtts_test.mp3")

playsound("audios/gtts_test.mp3")
