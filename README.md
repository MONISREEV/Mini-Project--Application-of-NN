# Mini-Project--Application-of-NN


(Expected the following details in the report )
## Project Title:  SPEECH EMOTION RECOGNITION
## Project Description :
  The act of attempting to understand human emotion and affective states from speech is known as Speech Emotion Recognition, or SER. This takes use of the fact that tone and pitch in the voice often indicate underlying emotion. Because emotions are subjective and annotating audio is difficult, SER is difficult. It basically aids the user in determining what type of feeling they are experiencing at the moment.
  
## Algorithm:

1. Get the data from libraries.
2. Run the program in sublime tool.
3. It detect the emotions  from the played audio .
4. plot the graph.
5. Study the final output.
## Program:
```
Import  webbrowser
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import wave	
import random
def extract_feature(file_name, mfcc, chroma, mel):
with soundfile.SoundFile(file_name) as sound_file:
X = sound_file.read(dtype="float32")
sample_rate=sound_file.samplerate
if chroma:
	stft=np.abs(librosa.stft(X))
if mfcc:
mfccs=np.mean(librosa.feature.mfcc(y=X,sr=sample_rate, n_mfcc=40).T, axis=0)
result=np.hstack((result, mfccs))
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))
```

<!DOCTYPE html>
<html>
<head>
	
	<title>recommendation</title>
	
	<style type="text/css">
		#button,#button1{
			background-color: green;
			color: yellow;
			padding: 5px 10px;
			border-radius: 10px;
		}
		#button:hover{
			background-color: darkred;
		}

		#button1 :hover{
			background-color: darkred;
		}
		body{
			border: solid 2px black; 
			padding-bottom: 10px;
		}
		h1{
			text-decoration: underline;	
		}
	</style>
	
<body style="background-color: skyblue">

	<center><h1><p style="color:red">HAPPY IS A DIRECTION NOT A PLACE.</p></h1></center>

		
	</body>
		

</head>
<body>
	<center>

	<h1>
		ANGRY SONG 
	</h1>
	<form>
	<a href="https://youtu.be/TMKb8An49mk" id="link">

	<input type="button" value="click me.." id="button">
	
	</a>
	</form>
	
	
	<h1>
		HAPPY MOTIVATIONAL SPEECH
		
	</h1>
	<form>
	<a href="https://youtube.com/shorts/iuQ3zyKq-JY?feature=share">
	<input type="button" value="click me.." id="button1">
	</a>
    </form>
	</center>	



</body>
</html>

## Dataset:

![image](https://user-images.githubusercontent.com/107461059/205828921-011056f3-f587-41f1-9258-4bcfdcacc29e.png)


## Output:

![image](https://user-images.githubusercontent.com/107461059/205824700-739f2902-2847-41d7-893c-74ad5bdea241.png)

![image](https://user-images.githubusercontent.com/107461059/205825115-8ddd67ba-7eff-43b7-a5f3-24b84d0d9c4b.png)

## Graph:
![image](https://user-images.githubusercontent.com/107461059/205825310-3634636d-673d-4f77-bb1f-880cd14d100c.png)

## Advantage :

Emotion recognition provides benefits to many institutions and aspects of life.

It is useful and important for security and healthcare purposes.

Also, it is crucial for easy and simple detection of human feelings at a specific moment without actually asking them.


## Result:
Thus the speech emotion recognition using MLP classifier is implemented successfully.


