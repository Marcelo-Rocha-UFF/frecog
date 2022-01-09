import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import time
import re

import socket # conexao tcp

HOST = ''  # localhost
PORT = 3030 # porta de conexao com o servidor local

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
# from extendedmodels import Age, Gender, Race, Emotion
from extendedmodels import Emotion
from commons import functions, realtime, distance as dst

import paho.mqtt.client as mqtt

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
   print("Connected with result code "+str(rc))

client = mqtt.Client()
client.on_connect = on_connect

client.connect("broker.mqttdashboard.com", 1883, 60)

def analysis(db_path, model_name, distance_metric, enable_face_analysis = True):
	
	input_shape = (224, 224)
	text_color = (255,255,255)
	
	employees = []
	#check passed db folder exists
	if os.path.isdir(db_path) == True:
		for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
			for file in f:
				if ('.jpg' in file):
					#exact_path = os.path.join(r, file)
					exact_path = r + "/" + file
					#print(exact_path)
					employees.append(exact_path)
					
	
	#------------------------
	
	if len(employees) > 0:
		if model_name == 'VGG-Face':
			print("Using VGG-Face model backend and", distance_metric,"distance.")
			model = VGGFace.loadModel()
			input_shape = (224, 224)	
		
		elif model_name == 'OpenFace':
			print("Using OpenFace model backend", distance_metric,"distance.")
			model = OpenFace.loadModel()
			input_shape = (96, 96)
		
		elif model_name == 'Facenet':
			print("Using Facenet model backend", distance_metric,"distance.")
			model = Facenet.loadModel()
			input_shape = (160, 160)
		
		elif model_name == 'DeepFace':
			print("Using FB DeepFace model backend", distance_metric,"distance.")
			model = FbDeepFace.loadModel()
			input_shape = (152, 152)
		
		else:
			raise ValueError("Invalid model_name passed - ", model_name)
		#------------------------
		
		#tuned thresholds for model and metric pair
		threshold = functions.findThreshold(model_name, distance_metric)
		
	#------------------------
	#facial attribute analysis models
		
	if enable_face_analysis == True:
		
		tic = time.time()
		
		emotion_model = Emotion.loadModel()
		print("Emotion model loaded")
		
		#age_model = Age.loadModel()
		#print("Age model loaded")
		
		#gender_model = Gender.loadModel()
		#print("Gender model loaded")
		
		toc = time.time()
		
		print("Facial attibute analysis models loaded in ",toc-tic," seconds")
	
	#------------------------
	
	#find embeddings for employee list
	
	tic = time.time()
	
	pbar = tqdm(range(0, len(employees)), desc='Finding embeddings')
	
	embeddings = []
	#for employee in employees:
	for index in pbar:
		employee = employees[index]
		pbar.set_description("Finding embedding for %s" % (employee.split("/")[-1]))
		embedding = []
		img = functions.detectFace(employee, input_shape)
		img_representation = model.predict(img)[0,:]
		
		embedding.append(employee)
		embedding.append(img_representation)
		embeddings.append(embedding)
	
	df = pd.DataFrame(embeddings, columns = ['employee', 'embedding'])
	df['distance_metric'] = distance_metric
	
	toc = time.time()
	
	print("Embeddings found for given data set in ", toc-tic," seconds")
	
	#-----------------------

	evaluation_rounds = 5
	time_threshold = 0.1; frame_threshold = 0
	pivot_img_size = 112 #face recognition result image

	#-----------------------
	
	opencv_path = functions.get_opencv_path()
	face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
	face_cascade = cv2.CascadeClassifier(face_detector_path)
	
	#-----------------------

##### loop de captura e analise da imagem #######################################################################
	print("------------------------------------------------")
	print("- Modulo de reconhecimento facial iniciado...  -")
	print("------------------------------------------------")
	# criando o server socket
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind((HOST, PORT))
	s.listen()
	
	while(True):
	
		freeze = False
		face_detected = False
		face_included_frames = 0
		freezed_frame = 0
		tic = time.time()
	
		resultado = "indefinido"
		print("Aguardando a conexao com o EVA...")
		conn, addr = s.accept() # funcao (block) aguarda conexao
		
		cap = cv2.VideoCapture(0) #webcam
		print("Ligando a WebCam")
		
		round = 0
		guesses = []
				
		while (round != evaluation_rounds):
						
			ret, img = cap.read()
			
			raw_img = img.copy()
			resolution = img.shape
			
			resolution_x = img.shape[1]; resolution_y = img.shape[0]

			if freeze == False: 
				faces = face_cascade.detectMultiScale(img, 1.3, 5)
				
				if len(faces) == 0:
					face_included_frames = 0
			else: 
				faces = []
			
			detected_faces = []
			face_index = 0
			for (x,y,w,h) in faces:
				if w > 130: #discard small detected faces
					face_detected = True
					if face_index == 0:
						face_included_frames = face_included_frames + 1 #increase frame for a single face
					
					detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
					
					#-------------------------------------
					
					detected_faces.append((x,y,w,h))
					face_index = face_index + 1
					
					#-------------------------------------
			
			# if face_detected == True and face_included_frames == frame_threshold and freeze == False:
			if face_detected == True and freeze == False:
			
				round += 1
			
				freeze = True
				#base_img = img.copy()
				base_img = raw_img.copy()
				detected_faces_final = detected_faces.copy()
				tic = time.time()
												
			if freeze == True:
				
				toc = time.time()
				if (toc - tic) < time_threshold:

					if freezed_frame == 0:
										
						freeze_img = base_img.copy()
						#freeze_img = np.zeros(resolution, np.uint8) #here, np.uint8 handles showing white area issue	
												
						for detected_face in detected_faces_final:
							x = detected_face[0]; y = detected_face[1]
							w = detected_face[2]; h = detected_face[3]
													
							#cv2.rectangle(freeze_img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image
							
							#-------------------------------
							
							#apply deep learning for custom_face
							
							custom_face = base_img[y:y+h, x:x+w]
							
							#-------------------------------
							#facial attribute analysis
							#print(round)
														
							if enable_face_analysis == True:
								
								
								gray_img = functions.detectFace(custom_face, (48, 48), True)
								#emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
								emotion_labels = ['raiva', 'desgostoso', 'medo', 'feliz', 'triste', 'surpreso', 'neutro']
								emotion_predictions = emotion_model.predict(gray_img)[0,:]
								sum_of_predictions = emotion_predictions.sum()
							
								mood_items = []
								for i in range(0, len(emotion_labels)):
									mood_item = []
									emotion_label = emotion_labels[i]
									emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
									mood_item.append(emotion_label)
									mood_item.append(emotion_prediction)
									mood_items.append(mood_item)
								
								emotion_df = pd.DataFrame(mood_items, columns = ["emotion", "score"])
								emotion_df = emotion_df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)
								
								#print(emotion_df["emotion"][0], emotion_df["score"][0])
								#print(emotion_df)
								#print(mood_items)
								# mood = dict()
								# for item in mood_items:
									# mood[item[0]] = item[1]
								if emotion_df["score"][0] < 33:
									round -= 1
								else:
									guesses.append(emotion_df["emotion"][0])
								#print(guesses)
																	
								if round == evaluation_rounds:
									guesses_score = dict()
									for guess in guesses:
										guesses_score[guess] = 0
									for guess in guesses:
										guesses_score[guess] += 1
									ordered_guesses_score = {k: v for k, v in sorted(guesses_score.items(), key=lambda item: item[1], reverse=True)}
									#print(ordered_guesses_score)
									resultado = next(iter(ordered_guesses_score))

							tic = time.time() #in this way, freezed image can show 5 seconds
							
							#-------------------------------
					
					time_left = int(time_threshold - (toc - tic) + 1)
					
					freezed_frame = freezed_frame + 1
					
				else:
					face_detected = False
					face_included_frames = 0
					freeze = False
					freezed_frame = 0
		
		cap.release()
		
		print("Expressao detectada: " + resultado)
		conn.sendall(resultado.encode()) # envia a expressao (codificada em bytes) para o cliente
		print("Desligando a Webcam...")
		print("Fim da conexao")
		print("------------------------------------------------")
			
		conn.close()

		# fim do while	
		
	#kill open cv things		
	cap.release()
	cv2.destroyAllWindows()