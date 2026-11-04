import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

#load the dataset
data = load_breast_cancer()
X = data.data  #matrix to include features such as size, shape and texture 
Y = data.target #labels with 0 for malignant and 1 for benign 

#see all the different features, if desired
#for i, name in enumerate(data.feature_names, 1):
 #   print(f" {i}. {name}")

#split the data into training set at 80% and test set at 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 , random_state=42, stratify = Y)

#initialize and train the Random Forest Classifier
classifer = RandomForestClassifier(n_estimators=100, random_state=42)
classifer.fit(X_train, Y_train)

#make predictions on the test dataset
Y_predict = classifer.predict(X_test)

#evaluate the model's performance
accuracy = accuracy_score(Y_test, Y_predict)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(Y_test, Y_predict, target_names=["Sick (Malignant)", "Healthy(Benign)"]))

#create confusion matrix based on true and predicted labels
disp = ConfusionMatrixDisplay.from_predictions(Y_test, Y_predict, display_labels = ["Sick(Malignant)", "Healthy(Benign)"], cmap=plt.cm.Greens)
plt.title("Cell Classification Confusion Matrix")
plt.show()

