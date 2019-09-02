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

To run this project on the deeplens, you will need to first install a few packages from a terminal in linux.

**sudo apt-get update && sudo apt-get install tesseract-ocr && apt-get install python-gi**

In order to get the audio to work on DeepLens, I had to perform the following steps:

1. Log into the DeepLens and take any audio .mp3 file and double click it.
2. A prompt will open up asking you to install some required packages
3. You will need to enter an administrator password to proceed
![Additional Prompt](https://github.com/alexschultz/ReadToMe/blob/master/assets/additional%20prompt.png)

After performing these steps, the audio should work.

The model files are located here: 
https://github.com/alexschultz/ReadToMe/tree/master/mxnet-model

You will need to tar up the files and put them in S3 when you create the project for the DeepLens.
See the official AWS instructions here:
https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-import-external-trained.html

In order to get the Text Area cleaned up to perform OCR, the program needs perform pre-processing on the image using a number of filters in OpenCV. This graphic shows an example of the steps that ReadToMe goes through with each image before trying to turn the image into text.

![IOT Console](https://github.com/alexschultz/ReadToMe/blob/master/assets/imagecleanup.png)

The lambda consists of three main files.

* [readToMeLambda.py](https://github.com/alexschultz/ReadToMe/blob/master/lambda/readToMeLambda.py)
	* Contains main workflow for project (imports imageProcessing.py)
* [imageProcessing.py](https://github.com/alexschultz/ReadToMe/blob/master/lambda/imageProcessing.py)
	* Contains helper functions used for image and text cleanup
* [imageProcessing.py](https://github.com/alexschultz/ReadToMe/blob/master/lambda/speak.py)  
	* Contains helper functions used to call AWS Polly and synthesize the audio

Because the user has no way to tell the DeepLens when a book is in front of the camera, we use the model to detect blocks of text on the page. When we find a text block, we isolate the image using the getRoi() function inside of imageProcessing.py.

Another important step that is performed is correctSkew() in imageProcessing.py. This warps/rotates the text block to try to make the text horizontal. If the text is angled or skewed, there will be problems when trying to do OCR.

Finally we remove any non utf-8 characters after doinc ocr. RemoveNonUtf8BadChars() in imageProcessing.py.  This step just attempts to clean up the text before turning the text to speach.

If you have any questions or find any issues with this project, please open an issue, Thanks!
