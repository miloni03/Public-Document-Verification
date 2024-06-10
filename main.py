from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, redirect
import os
import numpy as np
import pickle
app = Flask(__name__)
import jsonify
from PIL import Image
import app as app1


app.config["IMAGE_UPLOADS"] = "C:\\LOC 5.0\\LOC_frontend\\static\\Images"
#app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG","JPEG"]


@app.route('/', methods=["GET", "POST"])
def hello():
        #return image

    return render_template('main.html')
@app.route('/', methods=["GET", "POST"])
def preprocess_image(image):

    if request.method == "POST":
        image = request.files['file']

        if image.filename == '':
            print("Image must have a file name")
            return redirect(request.url)

        filename = secure_filename(image.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))
        print(basedir)
        n = request.form.get("text")

        path = 'C:\\LOC 5.0\\LOC_frontend\\static\\Images'
        os.chdir(path)
       #n = "Miloni6"
        os.mkdir(n)
        image.save(os.path.join(basedir, app.config["IMAGE_UPLOADS"], n, filename))
        path_to_image=os.path.join(basedir, app.config["IMAGE_UPLOADS"], n, filename)
        #path_to_image.init_app(app)
        
        #return render_template("main.html", filename=filename)
    
    # Resize the image to the required size of the model
        image = path_to_image.resize((224, 224))
    # Convert the image to a numpy array
        image = np.array(image)
    # Scale the pixel values to the range [0, 1]
        image = image / 255.0
    # Add a batch dimension to the array
        image = np.expand_dims(image, axis=0)

@app.route('/display/<filename>')
def display_image(filename):
   # n = "Miloni6"
    n = request.form.get("text")
    print(filename)

    return redirect(url_for('static', filename="C:\\LOC 5.0\\LOC_frontend\\static\\Images" + n + filename), code=301)



@app.route('/predict',methods=['POST'])
def predict():
    image_file = request.files['image']
    # Open the image using PIL
    image = Image.open(image_file)
    # Preprocess the image
    image = preprocess_image(image)
    # Make a prediction using the model
    prediction = (app1.model).predict(image)
    # Get the predicted class index
    predicted_class_index = np.argmax(prediction[0])
    # Return the predicted class index as a JSON object
    return jsonify({'class_index': predicted_class_index})

if __name__ == "__main__":
    app.run(debug=True)

