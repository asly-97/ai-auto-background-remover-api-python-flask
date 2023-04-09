from flask import Flask, request, jsonify, send_file
import base64
import os
from ai_bg_remover import remove_background

app = Flask(__name__)


# Replace this with your own secret key
authorized_key = 'api-secret-key'
    


@app.route("/")
def welcome():
    return "Welcome To The Background Remover API"



@app.route('/api', methods=['POST'])
def process_image():
    # Check for the secret key
    if request.headers.get('Authorization') != authorized_key:
        return jsonify({'error': 'Unauthorized access'}), 401

    # Get the image file from the request
    image_file = request.files['image']

    # Save the image file to the server
    filename = 'uploaded_image.png'
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(image_path)

    # Remove the image background using AI
    output_image_path = remove_background(image_path)

    return send_file(output_image_path, mimetype='image/jpeg')




if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)











