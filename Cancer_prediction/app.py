from flask import Flask, request, render_template
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_image():
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

        # Process the image array (e.g., perform analysis)

        # For demonstration purposes, display the shape of the array
        return f"Image array shape: {image_array.shape}"

    return render_template("upload.html")

def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

if __name__ == "__main__":
    app.run(debug=True)
