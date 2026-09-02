import wave
import struct
import math

sampleRate = 44100.0 # hertz
duration = 5.0       # seconds
frequency = 440.0    # hertz
obj = wave.open('test_ses.wav','w')
obj.setnchannels(1) # mono
obj.setsampwidth(2)
obj.setframerate(sampleRate)
for i in range(99999):
   value = int(32767.0*math.sin(frequency*math.pi*float(i)/sampleRate))
   data = struct.pack('<h', value)
   obj.writeframesraw( data )
obj.close()
print("test_ses.wav başarıyla oluşturuldu.")
