# ai-auto-background-remover-api-python-flask
AI Auto Background Remover API with U2Net & Flask framework
- Remove backgrounds from images with u2net pre-trained model

This is a Flask API script that receives an image file via a POST request, 
checks for an authorized secret key, removes the background from the image using maching learning, 
and returns the resulting image as a response.

To use this API, you would need to send a POST request 
with the image file as the payload, along with the Authorization header 
containing your secret key. The response will be the processed image.

This repo uses the model and the pre-trained weights described in the paper: *U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection*.

Original repo: https://github.com/JaouadT/background_image_remover_python.git

How to test the API with Postman:
0. Clone this repo into your local/hosted Flask server.
1. Install requirements: `pip install -r requirements.txt`.
2. Start the server: `python index.py`
3. Launch Postman and create a new POST request.
4. Set the request URL to the endpoint of the Flask API (http://localhost:5000/api in this case).
5. Click on the "Headers" tab and add a new key-value pair to set the Authorization header with your secret key (e.g. Authorization: api-secret-key).
6. Click on the "Body" tab and select "form-data" as the request body type.
7. Add a new key-value pair to the form data to set the image parameter with the image file you want to process. Make sure the key is set to "image".
8. Click on the "Send" button to send the request to the Flask API.

Once the Flask API processes the image and returns the response, you can check the returned processed image by clicking on the "Body" tab and viewing the processed_image.

Original Image: 
![alt text](original_image.jpg "Original image")

Processed image: 
![alt text](/processed_image.png "Processed image")
