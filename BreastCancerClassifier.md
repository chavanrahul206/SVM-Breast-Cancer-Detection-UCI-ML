

**TOPIC: MACHINE LEARINING**

**Rahul G. Chavan**

**Mini Project: Breast Cancer Classification Using Support Vector Machine on UCI ML**

**Repository dataset.**

**Title:** Breast Cancer Classification Using SVM on UCI ML Repository Dataset – **“Breast Cancer**

**Wisconsin (Diagnostic) Data Set”**.

**Problem Definition:** Apply the Support Vector Machine for classification on a dataset obtained

from UCI ML repository.

**Prerequisite:** Knowledge of Python or R, higher Mathematical Understanding and concepts of

Machine Learning.

**Software Requirements:** Python3, Anaconda, Jupyter Notebook

**Hardware Requirements:** Functional computer system with sufficient power to run Python

environment and Internet Connectivity.

**Theory/Important Concepts Used Ahead:**

•

**Machine Learning:**

o The process of learning begins with observations or data, such as examples,

direct experience, or instruction, in order to look for patterns in data and

make better decisions in the future based on the examples that we provide.

o The primary aim is to allow the computers learn automatically without human

intervention or assistance and adjust actions accordingly**.**

o The process of learning begins with observations or data, such as examples,

direct experience, or instruction, in order to look for patterns in data and

make better decisions in the future based on the examples that we provide.

•

**Classification:**

o Classification is a process of categorizing a given set of data into classes, it

can be performed on both structured or unstructured data.

o The process starts with predicting the class of given data points. The classes

are often referred to as target, label or categories. The classification

predictive modeling is the task of approximating the mapping function from

input variables to discrete output variables.

o

The main goal is to identify which class/category the new data will fall into.

Some common terminologies used ahead about classification are:

**Classifier** – It is an algorithm that is used to map the input data to a

specific category.

**Classification Model** – The model predicts or draws a conclusion to

the input data given for training, it will predict the class or category for the

data.

•

**Modelling:**





o Depending on how long we’ve lived in a particular place and traveled to a

location, we probably have a good understanding of commute times in our

area.

o For example, we’ve traveled to work/school using some combination of the

metro, buses, trains, ubers, taxis, carpools, walking, biking, etc.

o All humans naturally model the world around them.

o Over time, our observations about transportation have built up a mental

dataset and a mental model that helps us predict what traffic will be like at

various times and locations.

o We probably use this mental model to help plan our days, predict arrival

times, and many other tasks.

o As data scientists we attempt to make our understanding of relationships

between different quantities more precise through using data and

mathematical/statistical structures. This process is called modeling.

o Models are simplifications of reality that help us to better understand that

which we observe.

o In a data science setting, models generally consist of an independent variable

(or output) of interest and one or more dependent variables (or inputs)

believed to influence the independent variable.

•

**Support Vector Machine (SVM):**

o A Support Vector Machine (SVM) is a binary linear classification whose

decision boundary is explicitly constructed to minimize generalization error.

o It is a very powerful and versatile Machine Learning model, capable of

performing linear or nonlinear classification, regression and even outlier

detection.

o SVM is well suited for classification of complex but small or medium sized

datasets.

o It’s important to start with the intuition for SVM with the special linearly

separable classification case.

o If classification of observations is “linearly separable”, SVM fits the “decision

boundary” that is defined by the largest margin between the closest points for

each class. This is commonly called the “maximum margin hyperplane

(MMH)”.





o The advantages of support vector machines are:

\1. Effective in high dimensional spaces.

\2. Still effective in cases where number of dimensions is greater than

the number of samples.

\3. Uses a subset of training points in the decision function (called

support vectors), so it is also memory efficient.

\4. Versatile: different Kernel functions can be specified for the

decision function. Common kernels are provided, but it is also

possible to specify custom kernels.

o The disadvantages of support vector machines include:

\1. If the number of features is much greater than the number of

samples, avoid over-fitting in choosing Kernel functions and

regularization term is crucial.

\2. SVMs do not directly provide probability estimates, these are

calculated using an expensive five-fold cross-validation (see

Scores and probabilities).

**Implementation:**

•

•

•

In this study, the task is to classify tumors into malignant (cancerous) or benign (non-

cancerous) using features obtained from several cell images.

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast

mass. They describe characteristics of the cell nuclei present in the image.

**Attribute Information:**

o

o

o

ID number

Diagnosis (M = malignant, B = benign)

**Ten real-valued features are computed for each cell nucleus:**

▪

▪

▪

▪

▪

▪

▪

▪

▪

▪

Radius (mean of distances from center to points on the perimeter)

Texture (standard deviation of gray-scale values)

Perimeter

Area

Smoothness (local variation in radius lengths)

Compactness (perimeter² / area — 1.0)

Concavity (severity of concave portions of the contour)

Concave points (number of concave portions of the contour)

Symmetry

Fractal dimension (“coastline approximation” — 1)

•

**Loading Python Libraries and Breast Cancer Dataset**





•

**Features (Columns) breakdown**

•

**Visualize the relationship between our features**





•

**Let’s check the correlation between our features**





o There is a strong correlation between mean radius and mean perimeter,

as well as mean area and mean perimeter

•

**From our dataset, let’s create the target and predictor matrix**

o y” = Is the feature we are trying to predict (Output). In this case we are

trying to predict if our “target” is cancerous (Malignant) or not (Benign).

i.e. we are going to use the “target” feature here.

o X” = The predictors which are the remaining columns (mean radius, mean

texture, mean perimeter, mean area, mean smoothness, etc.)





•

**Create the training and testing data**

Now that we’ve assigned values to our “X” and “y”, the next step is to import the

python library that will help us split our dataset into training and testing data.

o Training data = the subset of our data used to train our model.

o Testing data = the subset of our data that the model hasn’t seen

before (We will be using this dataset to test the performance of our

model).

o Let’s split our data using 80% for training and the remaining 20% for

testing.

•

•

**Train SVM model with our “training” dataset.**

**Use trained model to make a prediction using our testing data and compare it with**

**output(y\_test). The comparison is done using confusion matrix.**





•

**Visualize our confusion matrix on a Heatmap**

As we can see, our model did not do a good job in its predictions. It predicted that 48

healthy patients have cancer. We only achieved 34% accuracy! So, we use

normalization to improve the accuracy. Data normalization is a feature scaling process

that brings all values into range [0,1].

o **X’ = (X-X\_min) / (X\_max — X\_min)**





o Retrain your SVM model with our scaled (Normalized) datasets.

•

•

**Prediction with Scaled dataset**

**Confusion Matrix on Scaled dataset**





**Our prediction is now a lot better with only 1 false prediction (Predicted cancer**

**instead of healthy).The accuracy of the model is 98%**

**Conclusion:** We’ve thus developed a classifier model using Support Vector Machine for

classification of Cancer. The model can predict a lot better than before after normalization of the

data with an increased accuracy of 98% .

**References:**

[1]: “Breast cancer classification by using support vector machines with reduced

dimension”, <https://ieeexplore.ieee.org/document/6044334>

[2]: “Gunn, Support Vector Machines for Classification and

Regression[”,](https://www.isis.ecs.soton.ac.uk/resources/svminfo/)<https://www.isis.ecs.soton.ac.uk/resources/svminfo/>

