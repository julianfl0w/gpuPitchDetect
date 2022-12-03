from gpuPitchDetect import *
import sounddevice as sd
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

blocksize = 512*4
sr = 44100

inStream = sd.InputStream(
    samplerate=sr,
    blocksize=blocksize,
    device=None,
    # channels=self.CHANNELS,
    channels=1,
    dtype=np.float32,
    latency=0.01,
    extra_settings=None,
    callback=None,
    finished_callback=None,
    clip_off=None,
    dither_off=None,
    never_drop_input=None,
    prime_output_buffers_using_stream_callback=None,
)

inStream.start()


# begin GPU test
instance = Instance(verbose=False)
device = instance.getDevice(0)

multiple = 30
pitchDetect = GPUPitchDetect(
    device=device,subdivisionOfSemitone=8.0, midistart=30, midiend=100, sr=sr, multiple=multiple,
)


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(pitchDetect.fprime*sr, np.zeros((len(pitchDetect.fprime))))
#line1, = ax.plot(np.zeros((2**15)))

ax.set_ylim([-1, 10])
spectrum_lpf = np.zeros((len(pitchDetect.fprime)))
while(1):
    inmic, underflow = inStream.read(blocksize)
    spectrum = pitchDetect.feed(inmic)
    inertia = 0.6
    spectrum_lpf = (inertia*spectrum_lpf) + ((1-inertia)*spectrum)
    line1.set_ydata(spectrum)
    #line1.set_ydata(pitchDetect.gpuBuffers.x.getAsNumpyArray())
    fig.canvas.draw()
    fig.canvas.flush_events()
    pitchDetect.findNote()
    #print(pitchDetect.harmonicAmplitude)
    #print(pitchDetect.gpuBuffers.offset.getAsNumpyArray())
    