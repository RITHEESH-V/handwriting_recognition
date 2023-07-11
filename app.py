#!/usr/bin/env python
# coding: utf-8

# In[27]:


from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import model_from_json
from keras import backend as K

app = Flask(__name__)

model_path = "G:\BYOP\deployment\model.json"
weights_path = "G:\BYOP\deployment\model.h5"
with open(model_path, 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(weights_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    def preprocess(img):
        (h, w) = img.shape
        final_img = np.ones([64, 256]) * 255

        if w > 256:
            img = img[:, :256]

        if h > 64:
            img = img[:64, :]

        final_img[:h, :w] = img
        return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

    def num_to_label(num):
        ret = ""
        for ch in num:
            if ch == -1:
                break
            else:
                ret += alphabets[ch]
        return ret

    alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
    num_of_characters = len(alphabets) + 1

    image = preprocess(image)
    image = image / 255.
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    predictions = loaded_model.predict(image)
    decoded = K.get_value(K.ctc_decode(predictions, input_length=np.ones(predictions.shape[0]) * predictions.shape[1],
                                       greedy=True)[0][0])
    predicted_text = num_to_label(decoded[0])

    return render_template('result.html', predicted_text=predicted_text)

if __name__ == '__main__':
    app.run()


