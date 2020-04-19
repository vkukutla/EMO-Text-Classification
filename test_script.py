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

MAX_SEQUENCE_LENGTH = 30 # max length of text (words) including padding
print("Loading model...")
model = load_model("best_network.h5")
print("Done.")
print("Loading tokenizer...")
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
print("Done")
r = sr.Recognizer()
r.pause_threshold = 0.3
r.non_speaking_duration = 1.0
mic = sr.Microphone()
engine = pyttsx3.init()

p = pyaudio.PyAudio()

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
texts = []
threads = []
break_flag = False

def processAndAdd(audio, index):
	print("Transcribing " + str(index))
	try:
		transcribed_text = r.recognize_google(audio, language='en-US')
	except:
		transcribed_text =  "[Untranscribed]"
	print(transcribed_text)
	input_text = (index, transcribed_text)
	texts.append(transcribed_text)
	if any(x in ["bye emo", "buy emo", "by emo", "bye-bye emo"] for x in [transcribed_text]):
		break_flag = True
	return

def recordLoop(index):
	
	return


while (True): 
	index = 0
	print("Hi, how was your day?")
	engine.say("Hi, how was your day?")	
	print("Starting journal entry recording, please wait... \n")
	engine.runAndWait()
	# stream.write(volume*samples)
	# stream.stop_stream()
	# stream.close()
	with mic as source:
		while(True):
			if (break_flag):
				break;
			r.adjust_for_ambient_noise(source, duration=0.2)
			print("Recording, say a sentence!")
			audio = r.listen(source, phrase_time_limit=5)
			print("Received input!")
			t = threading.Thread(target=processAndAdd, args=(audio,index,))
			threads.append(t)
			t.start()
			print("Launched thread")
			index = index + 1		


	sequences = tokenizer.texts_to_sequences(texts)
	data = pad_sequences(sequences, padding='post', maxlen=(MAX_SEQUENCE_LENGTH))
	category = model.predict(data)
	num = np.argmax(category)
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
	print("Analyzed emotion: ")
	print(emotion)
	engine.say("I sense " + emotion)
	engine.runAndWait()
	break





