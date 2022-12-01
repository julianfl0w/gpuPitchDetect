def freq2Note(f, A4=440.0):
    # A4, MIDI index 69
    return 12 * (np.log2(f) - np.log2(A4)) + 69


def note2Freq(note, A4=440.0):
    # A4, MIDI index 69
    return (A4 / 32) * (2 ** ((note - 9) / 12.0))


# check if dev is active
if "loiacono" not in [pkg.key for pkg in pkg_resources.working_set]:
    sys.path = [os.path.join(here, "..", "loiacono")] + sys.path
from loiacono import *
    
class GPUPitchDetect:
    def __init__(
        self, device, midistart=30, midiend=110, subdivisionOfSemitone=4.0, sr=48000, multiple=10
    ):
        self.midistart = midistart
        self.midiend = midiend
        self.subdivisionOfSemitone = subdivisionOfSemitone
        self.sr = sr
        self.midiRange = self.midiend - self.midistart
        self.multiple = multiple

        # find the midi indices and fprime
        self.midiIndices = np.arange(midistart, midiend, 1 / subdivisionOfSemitone)
        self.fprime = [note2Freq(note) / sr for note in midiIndices]

        # create the note pattern
        # only need to do this once
        self.notePattern = np.zeros(int(midiRange / 2 * subdivisionOfSemitone))
        zerothFreq = note2Freq(0)
        hnotes = []
        for harmonic in range(1, 5):
            hfreq = zerothFreq * harmonic
            hnote = freq2Note(hfreq) * subdivisionOfSemitone
            if hnote + 1 < len(notePattern):
                hnotes += [hnote]
                self.notePattern[int(hnote)] = 1 - (hnote % 1)
                self.notePattern[int(hnote) + 1] = hnote % 1
        
        self.loiacono = Loiacono_GPU(
            device,
            self.fprime,
            self.multiple, 
        )
        
    def findNote(self, spectrum):

        startTime = time.time()
        notes = np.correlate(spectrum, self.notePattern, mode="full")
        self.maxIndex = np.argmax(notes) - len(self.notePattern)
        self.selectedNote = self.midistart + self.maxIndex / self.subdivisionOfSemitone
        self.selectedAmplitude = self.notesPadded[self.maxIndex]
        endTime = time.time()
        print("correlate runtime (s) : " + str(endTime - startTime))

        # print("selectedNote " + str(selectedNote))
        # print("expected " + str([selectedNote + h for h in self.hnotes]))
    
    def feed(self, newData):
        
    
if __name__ == "__main__":

    infile = sys.argv[1]
    multiple = 10
    # load the wav file
    y, sr = librosa.load(infile, sr=None)
    
    # get a section in the middle of sample for processing
    z = y[int(len(y) / 2) : int(len(y) / 2 + 2**15)]
    
    # begin GPU test
    instance = Instance(verbose=False)
    device = instance.getDevice(0)
    
    pitchDetect = GPUPitchDetect(
        subdivisionOfSemitone=1.0,
        midistart=30,
        midiend=110,
        sr=sr,
        multiple=multiple,
    )
    
    # generate a Loiacono based on this SR
    linst = Loiacono(
        fprime = fprime,
        multiple=multiple
    )
    
    
    # coarse detection
    linst_gpu = Loiacono_GPU(
        device = device,
        signalLength = 2**15,
        fprime = fprime,
        multiple = linst.multiple,
        constantsDict = {},
    )
    
    linst_gpu.x.setBuffer(z)
    for i in range(10):
        linst_gpu.debugRun()
    #linst_gpu.dumpMemory()
    readstart = time.time()
    linst_gpu.absresult = linst_gpu.L.getAsNumpyArray()
    print("Readtime " + str(time.time()- readstart))
    v2time = time.time()
    linst_gpu.findNote()
    linst.findNote()
    print("v2rt " + str(time.time() - v2time))
    print(linst_gpu.selectedNote)
    print(linst.selectedNote)
    
    precision = 10
    midiIndicesToCheck = np.arange(linst_gpu.selectedNote-0.5, linst_gpu.selectedNote+0.5, step = 1.0/precison)
    print(midiIndicesToCheck)
    die
    # fine detection
    linst_gpu_fine = Loiacono_GPU(
        device = device,
        signalLength = 2**15,
        fprime = fprime_fine,
        multiple = 30,
        constantsDict = {},
    )
    
    
    print(len(linst_gpu.absresult))
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    ax1.plot(linst.midiIndices, linst_gpu.notesPadded)
    ax2.plot(linst.midiIndices, linst.notesPadded)
    
    plt.show()
    
    #print(json.dumps(list(ar), indent=2))
