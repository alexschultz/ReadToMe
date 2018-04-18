# Read To Me
### Project submission for [AWS DeepLens Challenge](https://awsdeeplens.devpost.com/) 

---

Note: I am currently in the process of upgrading this project to use an optimized MXNet model instead of a TensorFlow SSD model. 'Watch' this repo to keep up with any updates to the code.

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

I originially used Tensorflow to create an object detection model. At the time of this writing, the onboard Intel Model Optimization library does not work for TensorFlow. Once it is fixed I will be able to optimize this model to run on the GPU on the DeepLens device.

My dataset was made from hundreds of photos of my kids' books as well as a number of library books taken in various lighting conditions, orientations, and distances.
I used [labelImg](https://github.com/tzutalin/labelImg) to annotate my dataset with bounding boxes so I could train the model to identify Text Blocks on a page.

I was finally able to figure out how to train my model using MXNet and I will be updating this repo in the coming days to reflect those changes.

#### Architecture

This project is built using GreenGrass, Python 3.6, MXNet, OpenCV, Tesseract, and AWS Polly.

To run this project on the deeplens, you will need to run the following commands.

**sudo pip3 install --upgrade mxnet**

**sudo apt-get install ffmpeg**

**sudo apt-get install tesseract**

In order to get sound to play on the DeepLens, you will need to grant GreenGrass permission to use the Audio Card.

Green Grass requires you to explicitly authorize all the hardware that your code has access to. One way you can configure this through the Group Resources section in the AWS IOT console. Once configured, you deploy these settings to the DeepLens which results in a JSON file getting deployed greengrass directory on the to the device.

To enable Audio playback through your Lambda, you need to add two resources. The sound card on the DeepLens is located at the path **“/dev/snd/”**. You need to add both **“/dev/snd/pcmC0D0p”** and **“/dev/snd/controlC0”** in order to play sound.  

![IOT Console](https://github.com/alexschultz/ReadToMe/blob/master/iot.PNG)



In order to get the Text Area cleaned up to perform OCR, it needs to go through a number of filters. This graphic shows the steps that ReadToMe goes through with each image before trying to turn the image into text.

![IOT Console](https://github.com/alexschultz/ReadToMe/blob/master/imagecleanup.PNG)
