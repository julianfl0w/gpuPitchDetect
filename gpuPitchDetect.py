
def freq2Note(f, A4 = 440.0)):
    # A4, MIDI index 69
    return 12 * (np.log2(f) - np.log2(A4)) + 69


def note2Freq(note, A4 = 440.0):
    # A4, MIDI index 69
    return (A4 / 32) * (2 ** ((note - 9) / 12.0))

class GPUPitchDetect:

    def __init__(
            self, 
            midistart=30,
            midiend=110,
            subdivisionOfSemitone=4.0,
            sr=48000,):
            self.midistart=midistart,
            self.midiend=midiend,
            self.subdivisionOfSemitone=subdivisionOfSemitone,
            self.sr=sr,
        
        
    def getMidiFprime(
            self, 
            midistart=30,
            midiend=110,
            subdivisionOfSemitone=4.0,
            sr=48000,):

        midiRange   = midiend-midistart
        midiIndices = np.arange(midistart, midiend, 1 / subdivisionOfSemitone)
        fprime = [
            note2Freq(note) / sr for note in midiIndices
        ]
        return midiIndices, np.array(fprime)

    # function to generate note detection pattern
    def getHarmonicPattern(
            midistart=30,
            midiend=110,
            subdivisionOfSemitone=4.0,
            sr=48000,):

        midiRange   = midiend-midistart
        # create the note pattern
        # only need to do this once
        notePattern = np.zeros(int(midiRange/2 * subdivisionOfSemitone))
        zerothFreq = note2Freq(0)
        hnotes = []
        for harmonic in range(1, 5):
            hfreq = zerothFreq * harmonic
            hnote = freq2Note(hfreq) * subdivisionOfSemitone
            if hnote + 1 < len(notePattern):
                hnotes += [hnote]
                notePattern[int(hnote)] = 1 - (hnote % 1)
                notePattern[int(hnote) + 1] = hnote % 1

        return notePattern

    def findNote( 
            spectrum, 
            midiIndices,
            harmonicPattern,
            midistart=30,
            midiend=110,
            subdivisionOfSemitone=4.0,
            sr=48000,):

        startTime = time.time()
        notes = np.correlate(spectrum, harmonicPattern, mode="full")
        self.maxIndex = np.argmax(notes) - len(harmonicPattern)
        self.selectedNote = self.midistart + self.maxIndex / self.subdivisionOfSemitone
        self.selectedAmplitude = self.notesPadded[self.maxIndex]
        endTime = time.time()
        print("correlate runtime (s) : " + str(endTime-startTime))

        # print("selectedNote " + str(selectedNote))
        # print("expected " + str([selectedNote + h for h in self.hnotes]))

