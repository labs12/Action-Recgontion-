# Action-Recgontion-
## Motivation & Abstract 
Human action recognition is an active field in computer vision task and is highly based on the greatly developed image recognition with convolutional neural networks(CNN's). Action recognition is considered to be more challenging task compared to image recognition as a video consists of an image sequence that changes in every frame i.e the model has to deal with both spatial and temporal information from input. The current action recognition using two stream fusion methods were able to get a good performance. However, these methods are computationally expensive models for learning spatio-temporal dependencies of the action. In this paper we propose a deep neural network  architecture for learning spatial and temporal dependencies consisting of a 3D convolutional layer, fully connected(FC) layers and attention layer, which is simpler to implement and gives a competitive performance on the UFC-101 dataset. The proposed method first learns spatial and temporal features of actions through 3D-CNN, and then the attention mechanism helps the model to locate attention to essential features of recognition.

## Proposed Architecture 
![245956946_1289139644873070_1665360244754742956_n](https://user-images.githubusercontent.com/66351304/141077627-cc5a9910-9180-46e6-ad64-598ad02ad47d.jpg)

## Results 
![245949181_3071260209858566_2323326804956602684_n](https://user-images.githubusercontent.com/66351304/141077583-909d34ff-27b5-4b5b-aa1a-120bb2273b04.jpg)
 
 ## Evaluation with existing methods.[Right Ours]
 ![245994630_579499846606159_1521611393687931383_n](https://user-images.githubusercontent.com/66351304/141077843-e5e023d0-b7fa-458f-b365-26498add4a52.jpg)
