import os
import imutils
import cv2
import dlib
import time
import multiprocessing
from imutils import face_utils
from scipy.spatial import distance
from playsound import playsound
from utilities import eye_aspect_ratio, mouth_aspect_ratio
from notify_run import Notify 
from playsound import playsound

def helper():
	
	# Eyes and mouth threshold value
	eyeThresh = 0.25
	mouthThresh = 0.60

	# frame to check
	frame_check_eye = 5
	frame_check_mouth = 5

	# Initializing the Face Detector object
	detect = dlib.get_frontal_face_detector()

	# Loading the trained model
	predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	# Getting the eyes and mouth index
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
	(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

	# Initializing the Video capturing object
	cap=cv2.VideoCapture(0)

	# Initializing the flags for eyes and mouth
	flag_eye=0
	flag_mouth=0

	# Calculating the Euclidean distance between facial landmark points of eyes and mouth
	while True:
		ret, frame=cap.read()
		frame = imutils.resize(frame, height = 800, width=1000)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		subjects = detect(gray, 0)
		for subject in subjects:
			shape = predict(gray, subject)
			shape = face_utils.shape_to_np(shape)
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			mouth = shape[mStart:mEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			mar = mouth_aspect_ratio(mouth)
			mouthHull = cv2.convexHull(mouth)

			# Drawing the overlay on the face
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [mouth], -1, (255, 0, 0), 1)
			cv2.putText(frame, "Eye Aspect Ratio: {}".format(ear), (5, 50),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
			cv2.putText(frame, "Mouth Aspect Ratio: {}".format(mar), (5, 80),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
		
			# Comparing threshold value of Mouth Aspect Ratio (MAR)
			
			if mar > mouthThresh:
				flag_mouth += 1
				if flag_mouth >= frame_check_mouth:
					cv2.putText(frame, "****************** YOU ARE YAWNING *******************", (10, 370),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
					time.sleep(3)
					p = multiprocessing.Process(target=playsound, args=("Alarm.wav",))
					p.start()
					time.sleep(6)
					p.terminate()
					notify = Notify()
					notify.send("HELP!!! THIS PERSON IS FEELING DROWSY ")
					
			else:
				flag_mouth = 0


			# Comparing threshold value of Eye Aspect Ratio  (EAR)
			
			if ear < eyeThresh:
				flag_eye += 1
				if flag_eye >= frame_check_eye:
					cv2.putText(frame, "******************  YOU ARE SLEEPING *******************", (10,400),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					time.sleep(3)
					p = multiprocessing.Process(target=playsound, args=("Alarm.wav",))
					p.start()
					time.sleep(6)
					p.terminate()
					notify = Notify()
					notify.send("HELP!! THIS PERSON IS FEELING DROWSY ")
			else:
				flag_eye = 0
		
		# Plotting the frame
		cv2.imshow("Frame", frame)

		# Waiting for exit key
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	
	# Destroying all windows
	cv2.destroyAllWindows()
	cap.stop()

def main():
	helper()

if __name__ == '__main__':
	main()