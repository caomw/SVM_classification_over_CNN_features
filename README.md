# metail_SVM_classification_over_CNN_features
**SVM One-vs-All Classification with CNN features** 

-----------------------------------------------------------
This project aims to do max-margin classification over CNN features. In the package shared, any type of features can be used. The package computes class-wise as well as total accuracy, and also bar plots them for visualization. We also show the average class specific CNN features, used for SVM separation. The class features which have limited number of peaks (implicit feature selection is rather easy in this case) on an average help improve accuracy with SVMs, and vice-versa. 

-----------------------------------------------------------
SVMTrainTestCNNFeat.m is the main function file, which needs to be passed with appropriate arguments as listed in the comments. All code is self-explanatory. 
