import numpy as np
import pandas as pd
import pyaudio
import threading
import speech_recognition as sr
import pyttsx3
import re, sys, os, csv, keras, pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from pynput import keyboard
from pydub import AudioSegment 
from pydub.silence import split_on_silence 

MAX_SEQUENCE_LENGTH = 30 # max length of text (words) including padding
print("Loading model...")
model = load_model("best_network.h5")
print("Loading tokenizer...")
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
r = sr.Recognizer()
#r.dynamic_energy_threshold = True
r.pause_threshold = 1.5
mic = sr.Microphone()
engine = pyttsx3.init()
p = pyaudio.PyAudio()

#Generating a sine tone to player after prompt

volume = 0.5     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer
duration = 1.0   # in seconds, may be float
f = 440.0        # sine frequency, Hz, may be float

# generate samples, note conversion to float32 array
samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)

stream = p.open(format=pyaudio.paFloat32,
	                channels=1,
	                rate=fs,
	                output=True)

#Sentence Splitting Parameters (Taken from https://www.geeksforgeeks.org/audio-processing-using-pydub-and-google-speechrecognition-api/)
# Interval length at which to slice the audio file. 
# If length is 22 seconds, and interval is 5 seconds, 
# The chunks created will be: 
# chunk1 : 0 - 5 seconds 
# chunk2 : 5 - 10 seconds 
# chunk3 : 10 - 15 seconds 
# chunk4 : 15 - 20 seconds 
# chunk5 : 20 - 22 seconds 
interval = 5 * 1000
  
# Length of audio to overlap.  
# If length is 22 seconds, and interval is 5 seconds, 
# With overlap as 1.5 seconds, 
# The chunks created will be: 
# chunk1 : 0 - 5 seconds 
# chunk2 : 3.5 - 8.5 seconds 
# chunk3 : 7 - 12 seconds 
# chunk4 : 10.5 - 15.5 seconds 
# chunk5 : 14 - 19.5 seconds 
# chunk6 : 18 - 22 seconds 
overlap = 1.5 * 1000
  
# Initialize start and end seconds to 0 
start = 0
end = 0

# Flag to keep track of end of file. 
# When audio reaches its end, flag is set to 1 and we break 
flag = 0
texts = []
threads = []

debug =  False

def process_audio(audio, index):
	try:
		transcribed_text = r.recognize_google(audio, language='en-US')
		input_text = (index, transcribed_text)
		texts.append([input_text])
	except:
		if(debug):
			print("Error transcribing\n")



def analyze_sentences(sentences):
	sequences = tokenizer.texts_to_sequences(sentences)
	data = pad_sequences(sequences, padding='post', maxlen=(MAX_SEQUENCE_LENGTH))
	category = model.predict(data)
	sentence_categories = np.argmax(category,axis=1)
	sentence_categories = [int(numeric_string) for numeric_string in sentence_categories]
	num = np.argmax(sentence_categories)
	emotion = ""
	if (num == 0):
		emotion = "neutral"
	elif (num == 1):
		emotion = "joy"
	elif (num == 2):
		emotion = "anger"
	elif (num == 3):
		emotion = "sadness"
	elif (num == 4):
		emotion = "disgust"
	elif (num == 5):
		emotion = "fear"
	else:
		emotion = "surprise"
	return (emotion, sentence_categories)

def split_chunks(n,audio):
	chunks=[]
	counter = 0
	for i in range(0, 2 * n, interval): 
		# During first iteration, 
		# start is 0, end is the interval 
		if i == 0: 
			start = 0
			end = interval 
	  
		# All other iterations, 
		# start is the previous end - overlap 
		# end becomes end + interval 
		else: 
			start = end - overlap 
			end = start + interval  

		# When end becomes greater than the file length, 
		# end is set to the file length 
		# flag is set to 1 to indicate break. 
		if end >= n: 
			end = n 
			flag = 1

		# Storing audio file from the defined start to end 
		chunk = audio[start:end] 
 	
		# Store the sliced audio file to the defined path 
		# chunk.export(filename, format ="wav") 
		# chunks.append(chunk)
		filename = 'AudioChunks/chunk'+str(counter)+'.wav'
		# Store the sliced audio file to the defined path 
		chunk.export(filename, format="wav")
		# Increment counter for the next chunk 
		counter = counter + 1
	return counter

def analyze_entry(entry_path):
	audio_segment = AudioSegment.from_wav(entry_path)
	#audio_segment = AudioSegment.from_wav("AudioChunks/journal_entry_audio.wav")

		# Length of the audiofile in milliseconds 
		n = len(audio_segment) 
		# Split into chunks to send to google
		num_chunks = split_chunks(n, audio_segment)
		print("Succesfully Split Chunks (chunk size =  "  + str(num_chunks) + ")")

		for chunk_index in range(num_chunks):
			AUDIO_FILE = 'AudioChunks/chunk'+str(chunk_index)+'.wav'
			with sr.AudioFile(AUDIO_FILE) as source: 
				audio_listened = r.listen(source) 
			t = threading.Thread(target=process_audio, args=(audio_listened,index,))
			threads.append(t)
			t.start()
			index = index + 1		


		for t in threads:
			t.join()

		texts = sorted(texts, key=lambda x: x[0][0])
		texts2 = [i[0][1] for i in texts]

		(emotion, sentence_categories) = analyze_sentences(texts2)
		print(np.dstack((texts2,sentence_categories)))
		engine.say("I sense " + emotion)
		engine.runAndWait()