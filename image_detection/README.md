# Image detection with WebRTC and Tensorflow

## Introduction

This demo uses a Flask server and allows user to recognize images in 
device camera.

 [WebRTC Hacks Reference][https://webrtchacks.com/webrtc-cv-tensorflow/]

## Architecture

 - **Flask** will serve the HTML and JavaScript files for the browser to render. 
 - **getUserMedia.js** will grab the local video stream. 
 - **objDetect.js** will use the HTTP POST method to send images to the 
   TensorFlow Object Detection API. API will return the objects it sees 
   (what it terms classes) and their locations in the image. 
   We will wrap up this detail in a JSON object and send it back to 
   objDetect.js so we can show boxes and labels of what we see.


## Installation

Download object_detection library from Tensorflow models.

Run in folder:

```
protoc object_detection/protos/*.proto --python_out=.
```

## Customization

Modify object_recognition folder.
Get the folder from Tensorflow models.

[Tensorflow models][https://github.com/tensorflow/models/tree/master/research]

## Web server

curl -F "image=@./object_detection/test_images/image1.jpg" http://localhost:8081/image | python -m json.tool