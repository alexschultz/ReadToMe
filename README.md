# Read To Me
### Project submission for [AWS DeepLens Challenge](https://awsdeeplens.devpost.com/) 

---

<a href="http://www.youtube.com/watch?feature=player_embedded&v=fLjYKyRDDu0" target="_blank">
<img src="http://img.youtube.com/vi/fLjYKyRDDu0/0.jpg" alt="ReadToMe" />
</a>

---

#### Solution

For this project, I wanted to build an application that could read books to children. In order to achieve this, I designed a workflow which performs the following steps.

- Determine when a page with text is in the camera frame
- Clean up the image using OpenCV
- Perform OCR (Optical Character Recognition)
- Transform text into audio using AWS Polly
- Play back the audio through speakers plugged into DeepLens


#### Model Training

I used Tensorflow to create an object detection model. At the time of this writing, the onboard Intel Model Optimization library does not work for TensorFlow. Once it is fixed I will be able to optimize this model to run on the GPU on the DeepLens device.

I followed this [tutorial](https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/) which uses this [repo](https://github.com/tensorflow/models/tree/master/research/object_detection) to learn how to build my model. 

My dataset was made from a few hundred photos of my kids' books taken in various lighting conditions, orientations, and distances.
Following the tutorial, I used [labelImg](https://github.com/tzutalin/labelImg) to annotate my dataset with bounding boxes so I could train the model to identify Text Blocks on a page.

Here is my [Custom Training Dataset](https://s3.amazonaws.com/read-to-me-dataset/2018.zip).

Here is the [model](https://github.com/alexschultz/ReadToMe/blob/master/deeplens-lambda-read-to-me/tensorflow-model/frozen_inference_graph.pb) that I trained.

#### Architecture

This project is built using GreenGrass, Python 3.6, TensorFlow, OpenCV, Tesseract, and AWS Polly.


#### Instructions for testing

[Packaged Lambda](https://s3.amazonaws.com/lambdas-alex/dist.zip)

There is a [test python](https://github.com/alexschultz/ReadToMe/blob/master/deeplens-lambda-read-to-me/testModel.py) script that you can use to test the application on your development machine before deploying to the DeepLens. You will need to install a few dependancies before being able to run the application. I would recommend you create a virtual environment and pip install the following dependancies.

- opencv-python
- pillow
- pytesseract
- tensorflow 
- boto3 
- pydub

You will also need to install Tesseract and Tensorflow on your machine for this to work.

To run this project on the deeplens, you will need to install Tesseract and TensorFlow.

In order to get sound to play on the DeepLens, you will need to grant GreenGrass permission to use the Audio Card.

Green Grass requires you to explicitly authorize all the hardware that your code has access to. One way you can configure this through the Group Resources section in the AWS IOT console. Once configured, you deploy these settings to the DeepLens which results in a JSON file getting deployed greengrass directory on the to the device.

To enable Audio playback through your Lambda, you need to add two resources. The sound card on the DeepLens is located at the path **“/dev/snd/”**. You need to add both **“/dev/snd/pcmC0D0p”** and **“/dev/snd/controlC0”** in order to play sound.  

![IOT Console](https://github.com/alexschultz/ReadToMe/blob/master/iot.PNG)



