# import pickle
from tensorflow.keras.models import load_model
import numpy as np
from keras.utils import load_img, img_to_array

model = load_model('model2.h5',compile=False)
model.compile()

def predict_img(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    test_image = np.expand_dims(img_array, axis=0)
    test_image = test_image / 255.0
    res=model.predict(test_image)
    cl=res[0][0]
    if cl >= 0.5:
        return 'dog'
    else:
        return 'cat'