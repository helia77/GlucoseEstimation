
### Blood Glucose Level Estimation using test papers

A dataset comprising 1000 glucose test papers were used in this project.

Steps taken:
  * 1 Find the extreme points and locate the big circle
  * 2 Crop the image(s)
  * 3 Rotating the image(s)
  * 4 Draw contours
  * 5 Apply scaling
  * 6 Extracting yellow & blue pixels as 2D matrix
  * 7 Learn using Multi-layer Perceptron regressor
  * 8 Predict (prepare the new image using steps 1-6)
  
  :white_check_mark: Accuracy: 95%
  
  <p><b>Sample test paper:</b></p>
  <img src = "Images/Sample Test Paper.jpg" width=300>
  
  <p><b>Extracted RGB matrix:</b></p>
  <img src = "Images/Extracted RGB.jpg" width=300>
  
