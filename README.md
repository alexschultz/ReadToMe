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

My dataset was made from hundreds of photos of my kids' books as well as a number of library books taken in various lighting conditions, orientations, and distances.
I used [labelImg](https://github.com/tzutalin/labelImg) to annotate my dataset with bounding boxes so I could train the model to identify Text Blocks on a page.

The Model was trained using MXNet using a VGG 16 model as a base. The steps used for training are outlined in this [notebook](https://github.com/alexschultz/ReadToMe/blob/master/ReadToMe%20Model%20Training.ipynb) 


#### Architecture

This project is built using GreenGrass, Python, MXNet, OpenCV, Tesseract, and AWS Polly.

To run this project on the deeplens, you will need to first install tesseract.

**sudo apt-get update && sudo apt-get install tesseract-ocr**

The model files are located here: 
https://github.com/alexschultz/ReadToMe/tree/master/mxnet-model

You will need to tar up the files and put them in S3 when you create the project for the DeepLens.
See the official AWS instructions here:
https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-import-external-trained.html
 

In order to get sound to play on the DeepLens, you will need to grant GreenGrass permission to use the Audio Card.

Green Grass requires you to explicitly authorize all the hardware that your code has access to. One way you can configure this through the Group Resources section in the AWS IOT console. Once configured, you deploy these settings to the DeepLens which results in a JSON file getting deployed greengrass directory on the to the device.

To enable Audio playback through your Lambda, you need to add two resources. The sound card on the DeepLens is located at the path **“/dev/snd/”**. You need to add both **“/dev/snd/pcmC0D0p”** and **“/dev/snd/controlC0”** in order to play sound.  

![IOT Console](https://github.com/alexschultz/ReadToMe/blob/master/iot.PNG)


In order to get the Text Area cleaned up to perform OCR, it needs to go through a number of filters. This graphic shows the steps that ReadToMe goes through with each image before trying to turn the image into text.

![IOT Console](https://github.com/alexschultz/ReadToMe/blob/master/imagecleanup.PNG)

The lambda consists of two main files. 

* [readToMeLambda.py](https://github.com/alexschultz/ReadToMe/blob/master/deeplens-lambda-read-to-me/readToMeLambda.py) 
	* Contains main workflow for project (imports imageProcessing.py)
* [imageProcessing.py](https://github.com/alexschultz/ReadToMe/blob/master/deeplens-lambda-read-to-me/imageProcessing.py)  
	* Contains helper functions used for image and text cleanup

Because the user has no way to tell the DeepLens when a book is in front of the camera, we use the model to detect blocks of text on the page. When we find a text block, we isolate the image using the getRoi() function inside of imageProcessing.py.

Another important step that is performed is correctSkew() in imageProcessing.py. This warps/rotates the text block to try to make the text horizontal. If the text is angled or skewed, there will be problems when trying to do OCR.

Finally we remove any non utf-8 characters after doinc ocr. RemoveNonUtf8BadChars() in imageProcessing.py.  This step just attempts to clean up the text before turning the text to speach.

If you have any questions or find any issues with this project, please open an issue, Thanks!
