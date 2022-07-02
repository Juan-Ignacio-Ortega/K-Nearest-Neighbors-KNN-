# KNearestNeighbors - KNN

## 1 - Code and Description
### Code
https://github.com/Juan-Ignacio-Ortega/KNearestNeighbors-KNN/blob/main/KNNCode_JIOG.py

### Description
https://github.com/Juan-Ignacio-Ortega/KNearestNeighbors-KNN/blob/main/KNNDescription_JIOG.ipynb

## 2 - Introduction
The K-nearest neighbors algorithm (K-nearest neighbors in English and KNN for its acronym in English) is a Machine Learning algorithm that belongs to the simple and easy to apply supervised learning algorithms [1].
Despite its simplicity, nearest neighbors has been successful on a large number of classification and regression problems, including handwritten digits and satellite imagery scenes.
The principle behind the KNN methods is to find a predefined number of training samples closest in distance to the new point and predict the label from them. The number of samples (K) can be a constant defined by the user or vary according to the local density of points [2].
This project presents a methodology to develop K-Nearest Neighbors (KNN) models, specifically, a class that allows generating a KNN model, which is trained, as many times as it is decided, with different odd K in each iteration with the objective of obtaining the K that allows you to obtain the model with the best performance metric and to be able to use this as a predictor for new data instances.

## 3 - Theoretical framework
3.1 Definition of deep learning algorithm
In machine learning (ML), classification is a supervised learning technique in which the computer is fed labeled data and learns from it so that in the future it can use this learning to classify new data. . The classification can be binary (only two classes) or multiclass (more than two classes) [3].

3.2 Definition of K-Nearest Neighbors
KNN is a simple classification algorithm first proposed by Evelyn Fix and Joseph Hodges in 1951 and developed by Thomas Cover in 1967. This algorithm stores all input data with their corresponding labels and classifies a new observation based on similarity. This classification is done based on the labels of its nearest neighbors [3].

3.3 Stages of the KNN algorithm
The logic behind the KNN algorithm is one of the simplest of all supervised ML algorithms.
Stage 1: Select the number of K neighbors.
Stage 2: Calculate the distance from an unclassified point to other points. For this stage it is possible to use the Euclidean distance presented in Fig. 1.

![alt text](https://github.com/Juan-Ignacio-Ortega/KNearestNeighbors-KNN/blob/main/EucDist.png?raw=true)

Figure 1. Euclidean distance formula [1].
Stage 3: Take the K nearest neighbors based on the calculated distance.
Stage 4: Among the neighboring K's, count the number of points in each category. To better understand the stages up to this stage, it can be seen graphically in the following figure.

![alt text](https://github.com/Juan-Ignacio-Ortega/KNearestNeighbors-KNN/blob/main/KNN_DifK.png?raw=true)

Figure 2. Classification based on different K values ​​[3].

Step 5: assign a new point to the most present category among the neighboring K.
Stage 6: The model is ready [1].

In Figure 2, the new observation is sorted by its K nearest neighbors, where K has two different values: K = 3 and K = 7. For K = 3, the nearest neighbors are those inside the green circle, and for K = 7, the nearest neighbors are inside the purple circle.

It can be seen that if K = 3, two of those three neighbors are female, so the new observation would be classified as female. However, if K = 7, most of the nearest neighbors are male, and then the new individual would be classified as male.

So, for the classification of this new individual, K = 7 is better than K = 3 because it classified him as male, which was his actual gender. However, this does not mean that K = 7 is the best overall value for the data set; more new observations must be classified with different values ​​of K to find the value with the best performance [3].

3.4 Selection of the best number K of nearest neighbors.
It is possible to consider a way of choosing the K that makes the classification better. One way to find it is to plot the graph of the K value and the error rate corresponding to the data set, as shown in Figure 3, for a random example where the best prediction rate is obtained with a K between 5 and 18 [1].

![alt text](https://github.com/Juan-Ignacio-Ortega/KNearestNeighbors-KNN/blob/main/TasaDError.png?raw=true)

Figure 3. Value of the error rate of a random example [1].

3.5 Advantages and disadvantages
Advantages:
The algorithm is simple and easy to apply. There is no need to create a model, configure multiple parameters, or make additional assumptions. The algorithm is versatile. It can be used for classification or regression.
Disadvantages:
The algorithm becomes slower as the number of observations increases and the independent variables increase [1].

3.6 Concepts in data management
3.6.1 Quartiles
The median divides the sample in half, the quartiles divide it, as much as possible, into fourths.

Firstquartile = 0.25 * (n + 1)
Secondquartile = 0.5 * (n + 1) –> Identical to the median
Third quartile = 0.75 * (n + 1)

The result tells you the number of the value that represents the X quartile, of the data ordered in ascending order.

Only if the result is an integer, if not, the average of the sample values ​​on either side of this value is taken, taking the sample in ascending order [5].

3.6.2 Data normalization
Some AI algorithms require all data to be centered in a specific range of values, typically -1 to 1 or 0 to 1. Even if data is not required to be within values, it is generally a good idea to make sure the values ​​are within a specific range.

Normalization of ordinal values
To normalize an ordinal set, you have to preserve the order.

Normalization of quantitative values
The first thing you have to do is observe the range in which these values ​​are found and the interval to which you want to normalize.
Not all values ​​need to be normalized.

It is necessary to perform the calculations of the following variables to find the normalized value:
1. Maximum of the data = The highest value of the observation without normalizing.
2. Minimum of the data = The lowest value of the observation without normalizing.
3. Normalized Maximum = The highest value of the normalized data.
4. Normalized Minimum = The lowest value of the normalized data.
5. DataspaceRange = Maximum of data - Minimum of data
6. NormalizedRange = Normalized Maximum - Normalized Minimum
7. D = Value to normalize - Minimum of the data
8. DPct = D / Data range
9. dNorm = Normalized Range * DPct
10. Normalized = Normalized Minimum + dNorm
In this way, the normalized value [5] is obtained.

3.7 Performance Metrics
3.7.1 MSE
It is probably the most commonly used method to be able to predict the error. It allows evaluating the performance of multiple models on a prediction problem when dealing with continuous data.

It varies in a range of [0, inf ], being lower mean square error value (abbreviated as MSE for its acronym in English), a better performance of the model.

Its formulation is as follows:
MSE = (p1-a1)^2+...+(pn-an)^2 / n

Where 'a' is the current real value and 'p' is the predicted value [5].

3.7.2 Classification rate
The simplest way to evaluate a model with nominal and discrete characteristics is through the classification rate, whose formulation is as follows [5]:

Ranking Rate = 1 - Wrong Rankings / Total Predictions Made

3.7.3 Binary confusion matrix
There can only be four different types of results that give the confusion matrix, shown in fig. 4:
1. True positive (TP) - You are expected to have a positive value of your feature and you get a positive value as well.
2. True Negative (TN) - A negative value of your characteristic is expected and a negative value is predicted as well.
3. False positive (FP) - You have a negative value despite having predicted a positive value.
4. False negative (FN) - There is a positive value despite having predicted a negative value [5].

![alt text](https://github.com/Juan-Ignacio-Ortega/KNearestNeighbors-KNN/blob/main/MatrizDConfusion.png?raw=true)

Figure 4. Confusion matrix for error calculation [5].

3.7.4 Accuracy
The classification rate can be calculated in a more precise way with the confusion matrix. It can be calculated as a model accuracy metric, whose formulation is as follows [5]:

Accuracy = (TP+TN) / (TP+TN+FP+FN)

3.7.5 Accuracy
It can be defined as the rate of the predicted samples that are relevant, it is calculated as follows [5]:

Accuracy = PT / PT+FP

3.7.6 Sensitivity (Recall)
Sensitivity, also called recall, can be defined as the rate of selected samples that are relevant to the test. It is obtained as follows:

Sensitivity = TP / TP+FN

3.7.7 F1 score
The F1 Score, F-beta score with a beta of 1, can be defined as the harmonic mean between recall and precision, and is calculated as shown in the following equation:

F1 Score = 2*TP / 2*TP+FP+FN

3.8K-Fold
K-Fold, also called cross-validation, is a procedure that consists of dividing the data into k times and performing tests with each k-partition.

The parameter K indicates how many times the data has to be partitioned. The most used K are 3, 5 and 10 partitions. The K is usually replaced by the number of k-partitions, '10-Fold'.

The method is very popular because the results are less biased and a less optimistic estimate of the performance and accuracy of an algorithm is made.

The general algorithm is as follows:
1. The data is randomly scrambled.
2. The data is separated into k groups. In this case, it is recommended to create copies of the data, each with the corresponding partition. For example, if you want to do a '5-Fold', do
five copies with the data. In the first copy, the first fifth part is removed, in the second copy, the second fifth part and so on until the five copies are completed.
3. The parts that were removed will be the test stands. The first test is done with the model from the first copy and the test data that was removed from it, and so on.
4. Quality metrics (error, sort rate, accuracy, precision, sensitivity, and F-beta score) are saved.
5. The model and the test bench with which it was made are discarded.
6. Switch to the next model with the next test bench, repeat the procedure from step four until all partitions are tested.

The data is divided as homogeneously as possible, that is, each partition contains approximately the same amount of data. As a graphic way of representing this, the example of the following figure [5] can be shown:

![alt text](https://github.com/Juan-Ignacio-Ortega/KNearestNeighbors-KNN/blob/main/KFold.jpeg?raw=true)

Figure 5. Example of the distribution of tests for a ‘10-Fold’ cross-validation [5].

3.9 Database
The 'Tumor Classification KNN' database (DB) obtained from Kaggle from [4] was used, which provides the diagnosis of whether the patient has a malignant tumor (M) or a benign tumor (B), in addition to multiple information on each patient about their analysis.

## References
[1] A. Mike, ¿Qué es el algoritmo KNN?, Formación en ciencia de datos | DataScientest.com,
28-dic-2021.
[2] 1.6. Nearest Neighbors, scikit-learn. [En línea]. Disponible en: https://scikitlearn/
stable/modules/neighbors.html. [Accedido: 04-may-2022].
[3] D. Lopez-Bernal, D. Balderas, P. Ponce, y A. Molina, Education 4.0: Teaching the basics of
KNN, LDA and Simple Perceptron algorithms for binary classification problems, Future internet,
vol. 13, n.o 8, p. 193, 2021.
[4] Shubhankitsirvaiya, Tumour Classification KNN, Kaggle.com, 16-feb-2021. [En línea].
Disponible en: https://kaggle.com/shubhankitsirvaiya06/tumour-classification-knn. [Accedido:
04-may-2022].
[5] M. A. Aceves Fernández, Inteligencia Artificial para programadores con prisa. UNIVERSO de
LETRAS, 2021.
