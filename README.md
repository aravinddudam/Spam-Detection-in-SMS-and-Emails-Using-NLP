## Spam-Detection-in-SMS-and-Emails-Using-NLP

## Overview
This project involves building a machine learning model to classify SMS messages as either "spam" or "ham" (not spam). The dataset contains SMS messages in English, labeled accordingly. The goal is to accurately predict the category of unseen messages, helping in spam detection efforts. The project includes exploratory data analysis (EDA), preprocessing of text data, feature engineering, model training, and evaluation.

## Outcome
The outcome of the project is a set of trained machine learning models capable of classifying SMS messages with high accuracy. These models were evaluated using various metrics, and the best-performing model was identified based on precision, recall, f1-score, and accuracy.

## Methodology
1. Data Preprocessing: The SMS text data were cleaned by removing punctuation and stopwords, and then converted to lowercase to standardize the input.
2. Feature Engineering: Two main features were extracted from the text data - the term frequency-inverse document frequency (TF-IDF) values and the length of the messages. Additionally, word clouds were generated to visualize the most frequent words in both spam and ham messages.
3. Model Training: Several machine learning models were trained, including Multinomial Naive Bayes (MNB), Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Stochastic Gradient Descent (SGD) Classifier, and Gradient Boosting Classifier. The models were trained using a pipeline that includes vectorization, TF-IDF transformation, and the classifier itself.
4. Model Evaluation: The models were evaluated using metrics such as accuracy, precision, recall, f1-score, and the area under the ROC curve (AUC-ROC). Grid search was performed to tune the hyperparameters of the models.

## Metrics
Accuracy: The proportion of true results (both true positives and true negatives) among the total number of cases examined.
Precision: The ratio of true positive results to all positive results, including those not identified correctly.
Recall (Sensitivity): The ratio of true positive results to the total number of positives that should have been identified.
F1-Score: The harmonic mean of precision and recall, providing a balance between them.
AUC-ROC: The area under the receiver operating characteristic curve, measuring the model's ability to distinguish between classes.

## Achievements
The project demonstrated effective text preprocessing and feature engineering strategies for NLP tasks.
The models achieved high accuracy in classifying SMS messages, with some models achieving over 98% accuracy.
The use of GridSearchCV helped in identifying the optimal parameters for each model, improving their performance.
The evaluation metrics indicated strong performance across various aspects, such as the ability to minimize false positives (high precision) and the ability to identify most positives correctly (high recall).
This comprehensive approach, from data preprocessing to model evaluation, ensures that the project not only achieves high accuracy but also balances precision and recall, making the models practical for real-world spam detection tasks.