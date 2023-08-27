from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import pickle

model=pickle.load(open('dl.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('upload.html')


@app.route("/upload", methods=["GET", "POST"])
def image_process():
    if request.method == "POST":
        # Check if the POST request has a file part
        if "image" not in request.files:
            return "No image file provided."

        image_file = request.files["image"]

        # Check if the file is empty
        if image_file.filename == "":
            return "No selected image file."

        # Check if the file is an allowed image format (you can customize this)
        allowed_extensions = ["jpg", "jpeg", "png", "gif"]
        if not allowed_file(image_file.filename, allowed_extensions):
            return "Invalid file format. Allowed formats are JPG, JPEG, PNG, GIF."

        # Convert the image into an array of numbers
        image = Image.open(image_file)
        image_array = np.array(image)
        #from tensorflow.keras.preprocessing import image
# import numpy as np
        from tensorflow.keras.applications.resnet50 import preprocess_input

# # Load and preprocess the image
# img_path = r"E:\Projects\Cancer_prediction\DL\input\normal\normal (122).png"# Replace with the path to your image
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
        x = np.expand_dims(image_array, axis=0)
        x = preprocess_input(x)  # Preprocess the image for ResNet50

# # Make predictions
        predictions = model.predict(x)
# predictions
# #Interpret the predictions
        val=np.argmax(predictions)
        print(val)
# category(val)

    
        # Process the image array (e.g., perform analysis)

        # For demonstration purposes, display the shape of the array

    #return render_template("final.html",data=val)
    if val==0:
        return render_template("b.html")
    elif val==1:
        return render_template("m.html")
    elif val==2:
        return render_template("n.html")


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

if __name__ == "__main__":
    app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)