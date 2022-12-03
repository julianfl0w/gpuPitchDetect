import os
import sys

# check if dev is active
import pkg_resources
gpd_home = os.path.dirname(os.path.abspath(__file__))
if "loiacono" not in [pkg.key for pkg in pkg_resources.working_set]:
    sys.path = [os.path.join(gpd_home, "..", "loiacono")] + sys.path
from loiacono_gpu import *

def freq2Note(f, A4=440.0):
    # A4, MIDI index 69
    return 12 * (np.log2(f) - np.log2(A4)) + 69


def note2Freq(note, A4=440.0):
    # A4, MIDI index 69
    return (A4 / 32) * (2 ** ((note - 9) / 12.0))


class GPUPitchDetect(Loiacono_GPU):
    def __init__(
        self,
        device,
        sr,
        midistart=30,
        midiend=110,
        subdivisionOfSemitone=4.0,
        multiple=10,
    ):
        self.midistart = midistart
        self.midiend = midiend
        self.subdivisionOfSemitone = subdivisionOfSemitone
        self.sr = sr
        self.midiRange = self.midiend - self.midistart
        self.multiple = multiple

        # find the midi indices and fprime
        self.midiIndices = np.arange(midistart, midiend, 1 / subdivisionOfSemitone)
        self.fprime = np.array([note2Freq(note) / sr for note in self.midiIndices])

        # create the note pattern
        # only need to do this once
        self.notePattern = np.zeros(int(self.midiRange / 2 * subdivisionOfSemitone))
        zerothFreq = note2Freq(0)
        self.noteIndices = []
        print("hnotes")
        for harmonic in range(1, 6):
            hfreq = zerothFreq * harmonic
            hnote = freq2Note(hfreq) * subdivisionOfSemitone
            self.noteIndices += [hnote]
            print(hnote)
            if hnote < len(self.notePattern):
                self.notePattern[int(hnote)] = 1#1/(harmonic)

        Loiacono_GPU.__init__(
            self, device, self.fprime, self.multiple,
        )

    def findNote(self):

        startTime = time.time()
        notes = np.correlate(self.absresult, self.notePattern, mode="full")
        self.maxIndex = np.argmax(notes) - len(self.notePattern) +1
        self.selectedNote = self.midistart + self.maxIndex / self.subdivisionOfSemitone
        #self.selectedAmplitude = self.notesPadded[self.maxIndex]
        endTime = time.time()
        #print("correlate runtime (s) : " + str(endTime - startTime))
        self.harmonicAmplitude = self.absresult[(self.maxIndex + self.noteIndices).astype(int)]
        # print("selectedNote " + str(selectedNote))
        # print("expected " + str([selectedNote + h for h in self.hnotes]))


if __name__ == "__main__":

    infile = sys.argv[1]
    multiple = 40
    # load the wav file
    y, sr = librosa.load(infile, sr=None)

    
    # get a section in the middle of sample for processing
    z = y[int(len(y) / 2) : int(len(y) / 2 + 2 ** 15)]

    # begin GPU test
    instance = Instance(verbose=False)
    device = instance.getDevice(0)

    pitchDetect = GPUPitchDetect(
        device=device,subdivisionOfSemitone=1.0, midistart=30, midiend=110, sr=sr, multiple=multiple,
    )

    pitchDetect.gpuBuffers.x.set(z)
    for i in range(10):
        pitchDetect.debugRun()
    # pitchDetect.dumpMemory()
    readstart = time.time()
    pitchDetect.absresult = pitchDetect.gpuBuffers.L.getAsNumpyArray()
    print("Readtime " + str(time.time() - readstart))
    v2time = time.time()
    pitchDetect.findNote()
    print("v2rt " + str(time.time() - v2time))
    print(pitchDetect.selectedNote)

    print(len(pitchDetect.absresult))

    if False:
        fig, ((ax1, ax2)) = plt.subplots(1, 2)
        #fig, ((ax1)) = plt.subplots(1, 1)
        ax1.plot(pitchDetect.midiIndices, pitchDetect.absresult)
        ax2.plot(pitchDetect.notePattern)
        plt.show()
