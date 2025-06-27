from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('vgg16.h5')  # Make sure this model file exists

# Ensure upload directory exists
os.makedirs("static/uploads", exist_ok=True)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/predict', methods=['GET', 'POST'])
def output():
    if request.method == 'POST':
        if 'pc_image' not in request.files:
            return render_template("portfolio-details.html", error="No file selected")

        f = request.files['pc_image']

        if f.filename == '':
            return render_template("portfolio-details.html", error="No file selected")

        if f:
            img_path = "static/uploads/" + f.filename
            f.save(img_path)

            try:
                img = load_img(img_path, target_size=(224, 224))
                image_array = img_to_array(img)
                image_array = np.expand_dims(image_array, axis=0) / 255.0

                pred = np.argmax(model.predict(image_array), axis=1)
                index = ['Biodegradable', 'Recyclable', 'Trash']  # Simplified labels
                prediction = index[int(pred[0])]

                return render_template("portfolio-details.html",
                                       predict=prediction,
                                       image_path=img_path)
            except Exception as e:
                return render_template("portfolio-details.html",
                                       error=f"Error processing image: {str(e)}")

    return render_template("portfolio-details.html")


if __name__ == '__main__':
    app.run(debug=True, port=2222)