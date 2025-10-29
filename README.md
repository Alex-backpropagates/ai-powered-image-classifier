# Basic AI-powered Image Classifier

Powered with TensorFlow, Images found on Google Images and Generated with Leonardo.ai
<br />3 labels : Cherry, Banana, Together (this means banana and cherry in the same picture)
<br />Achieved more than 90% accuracy on Cherry or Banana Alone, 60% with label together, which is normal since it is confused with Banana or Cherry
(check confusion matrix)

# How to Use

First train the program, the pictures used are in the folders cherry, banana and together
<br /> ```python ai_classifier.py train```

Then predict with the test pictures in the folder "test", it gives you accuracy and the confusion matrix : 
<br /> ```python ai_classifier.py predict```







