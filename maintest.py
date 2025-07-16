import cv2
import os
import platform

if platform.system() != 'Windows':
    import fcntl
else:
    # Handle Windows-specific file control operations
    pass

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical.keras')

image=cv2.imread('C:\\Users\\cchan\\Downloads\\archive\\pred\\pred0.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result = model.predict_step(input_img)
predicted_class = np.argmax(result, axis=-1)


print(result)


predicted_class = np.argmax(result)

