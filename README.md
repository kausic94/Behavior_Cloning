<center>**Behavioral Cloning Project**</center>
---

<p> This write-up file summarises the approach I employed in completing the behavior Cloning Project.</p>

[image1]:images/example_3.jpg "Original Center Image"
[image2]:images/example_1.jpg "cropped Image"
[image3]:images/example_2.jpg "YUV Image"
[image4]:images/example_4.jpg "Model Visualisation"
[image5]:images/example_5.jpg "Original Left Image"
[image6]:images/example_6.jpg "Original Right Image"
[image7]:images/example_7.jpg "Normalized image"

The following items are included as mentioned in the project's submission itinerary:
* model.py
* model.h5
* drive.py
* writeup_report.md
* images (directory)
* video.mp4

## Model Architecture and Training Strategy

### 1. Collecting Data:
I used the simulator provided by Udacity to collect the data. I drove the car in the training mode. I drove 3 laps around track 1 in the anti-clockwise direction and 3 laps in the anti-clockwise direction. I got 33177 images in total that included left, right and center images.

![alt text][image1]
<center>_Center Camera Image_</center>

<center>![alt text][image5]</center>
<center>_Left Camera Image_</center>

<center>![alt text][image6]</center>
<center>_Right Camera Image_</center>

The above images display that the images taken from the center camera, left camera and right camera mounted on the car.

### 2. Data Augmentation:
Firstly, the driving_log.csv file was opened and the contents were read.The center, left and right images were read seperately and added to thier respective arrays. The steering angle for the corresponding center images are taken added to an array. The same steering angle is added to a correction factor for the left image and is subtracted by the same correction factor for the right images and they are added to thier corresponding arrays. This helped me in tripling my dataset. 

I plotted the data and made sure all the arrays(center, left and right data) are of all the same length to avoid any factors of bias by removing random images and the corresponding steering angle. So, the center, left and right arrays will have equal amount of data. They are consolidated into a single dataset and finally shuffled.

The Data augmentation steps can be seen in the model.py file from lines 22-79. After the end of the data augmentation step I had a total of 22140 images and theier corresponding steering angles as well.

### 3. Data Pre-Processing :
A few data pre-processing steps were carried out before they were subjected to training. I initially converted the available images to YUV colorspace (model.py line 97). This was done as this had also been followed in the Nvidia paper "End to End Learning for Self-Driving Cars". A YUV transformed Image can be seen below.
<p align="center">![alt text][image3]</center></p>
<center>_YUV image_</center>

The color-transformed image is then cropped. Only information from a particular region of interest is required.Since the camera will be in a fixed position we can eliminate the portion of the image containing the skyline and the hood of the car. This is the first step in the convolutional model(model.py line 121).Below is the cropped version of the original image.
<center>![alt text][image2]</center>
<center>_Cropped Image_</center>

In the paper they paper by Nvidia they had worked with images of a particular size. To be consistent, I also added a resizing layer to the model that resizes the images to 66x200 from 90x300.(model.py line 122)

The images are then normalized by dividing the pixel values by 255. This normalization step is followed subtracting the mean (0.5) from the images(model.py line 123) . An image subjected to such a method is shown below. 

<center>![alt text][image7]</center>
<center> _Normalized image Visualised_</center>


### 4. CNN architecture:
I implemented the architecture provided in the Nvidia paper.The model has 5 convolutional layers followed by three fully connected layers.(model.py 115-131) The first three layers uses 5x5 kernels with 2x2 strides and the other 2 layers use 3x3 kernel with 1x1 strides . Each layer is followed with a relu activation.

To avoid overfitting the data I used dropouts after every fully connected layer with a dropout rate of .25(model.py 125-130). The image below gives an illustration of the model.

<center>![alt text][image4]</center>
<center>_Model_</center>

### 5.Training and Validation:
The dataset was split into training and validation set with the training set retaing 80% of the data and the rest in the validation set. A python generator was used inorder to get the data in batches and train them accordingly. This helped to run the training procedure in my laptop. The training was done using Adam optimizer with the loss(Mean square error) as the metric . I could not configure the gpu instances in AWS with tensorflow, So I had to run the model in my laptop. The model was trained for 2 epoch with each epoch taking about 10 hours to train.

Once trained and validated the model was saved for use by other programs. The saved model can be seen as model.h5

### 6. Deploying the model and Testing in simulator:
In the pre-processing step we had converted every image to YUV space. I learned that the drive.py provided by udacity feeds RGB image directly into the model. Since, the model worked on YUV space I modified drive.py by adding a line (line 64 in drive.py) to convert the RGB image into a YUV image. I then ran the simulator and used the saved model to predict the steering angle. The car was able to stay on track. If i had had the computing power required I could have trained the model for even better performance. However, The results seem satisfactory and this can be seen in the video.mp4 file.

