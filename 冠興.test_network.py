from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from imutils import paths


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

print("[INFO] loading network...")
model = load_model(args["model"])
N=0
imagePaths = sorted(list(paths.list_images(args["image"])))
for img in imagePaths:

	image = cv2.imread(img)
	orig = image.copy()

	image = cv2.resize(image, (128, 128))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	(notSanta, santa) = model.predict(image)[0]

	label = "Santa" if santa > notSanta else "No Santa"
	proba = santa if santa > notSanta else notSanta
	# label = "{}: {:.2f}%".format(label, proba * 100)

	output = imutils.resize(orig, width=400)
	if label =="Santa":
		label = label +' in the image.'
		cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
					0.7, (0, 255, 0), 2)
	else:
		label = label +' in the image.'
		cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)
	cv2.imwrite(args["image"]+'result/'+str(N)+'.jpg',output)
	N+=1

