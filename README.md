# Action-Recgontion-
## Motivation & Abstract 
Human action recognition is an active field in computer vision task and is highly based on the greatly developed image recognition with convolutional neural networks(CNN's). Action recognition is considered to be more challenging task compared to image recognition as a video consists of an image sequence that changes in every frame i.e the model has to deal with both spatial and temporal information from input. The current action recognition using two stream fusion methods were able to get a good performance. However, these methods are computationally expensive models for learning spatio-temporal dependencies of the action. In this paper we propose a deep neural network  architecture for learning spatial and temporal dependencies consisting of a 3D convolutional layer, fully connected(FC) layers and attention layer, which is simpler to implement and gives a competitive performance on the UFC-101 dataset. The proposed method first learns spatial and temporal features of actions through 3D-CNN, and then the attention mechanism helps the model to locate attention to essential features of recognition.

## Proposed Architecture 
The proposed model is composed of four components: preprocessing, 3D-CNN, attention gate, average pooling, and dense neural networks. In preprocessing, we divide the videos into 16-frames samples followed by data augmentation. As 3D-CNN uses a 5D tensor (batch size, channel, height, width, depth) as input, we change the number of frames accordingly. We use RGB channels with 16 number of frames as input tensors. The 3D-CNN of channel 64, followed by the ReLU activation function, extracts spatial and temporal features from input frames. These extracted features are elementwise multiplied with the output of the attention layers to get the essential features. The attention layer consists of sigmoid gating of 3D-CNN extracted features, which helps the model to focus on the more essential features for recognizing the action. After the attention layer, a 3D average pooling is utilized. Then these extracted features are passed through DNN layers followed by ReLU activation with a dropout rate of 0.25 and predicts the action for the given video sample of 16-frames.
![245956946_1289139644873070_1665360244754742956_n](https://user-images.githubusercontent.com/66351304/141077627-cc5a9910-9180-46e6-ad64-598ad02ad47d.jpg)

## Results 

![245949181_3071260209858566_2323326804956602684_n](https://user-images.githubusercontent.com/66351304/141077583-909d34ff-27b5-4b5b-aa1a-120bb2273b04.jpg)
 
 ## Evaluation with existing methods.[Right Ours]
 ![245994630_579499846606159_1521611393687931383_n](https://user-images.githubusercontent.com/66351304/141077843-e5e023d0-b7fa-458f-b365-26498add4a52.jpg)
 
 ### References 
 - Dataset 
 https://www.crcv.ucf.edu/data/UCF101.php
 - Related Paper
 C3D: Learning Spatiotemporal Features with 3D Convolutional Networks (Video Classification & Action Recognition)
 - References 
 https://github.com/DavideA/c3d-pytorch
 https://arxiv.org/abs/1412.0767
