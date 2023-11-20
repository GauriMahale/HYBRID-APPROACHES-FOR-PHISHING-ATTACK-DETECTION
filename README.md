# HYBRID APPROACHE FOR PHISHING ATTACK DETECTION 
# Introduction 
* Phishing attacks pose a major threat to cybersecurity, but detecting them is challenging due to their dynamic nature and high false positive rates.
* To address these limitations,we propose a hybrid approach for phishing attack detection that combines the strengths of both rule-based, machine learning and deep learning systems.
* Our approach has been evaluated on a dataset of realworld phishing URLs and has achieved a high detection rate with a low false positive rate.
* With our application, we aim to provide organizations with a powerful tool to protect against new and unknown phishing attacks in real-time, safeguarding users' personal and financial information.
* Our innovative approach has the potential to greatly improve phishing attack detection and help organizations better defend against cyber threats.

![image](https://github.com/GauriMahale/HYBRID-APPROACHES-FOR-PHISHING-ATTACK-DETECTION/assets/84667768/96e5e4e6-678c-4e3f-bc48-57116ecbfa98)


# Objective
* Our objective is to develop a hybrid approach for phishing attack detection that combines rule-based, machine learning, and deep learning systems to achieve high accuracy in detecting phishing attacks with low false positives.
* Specifically, we aim to improve the detection of new and unknown phishing attacks in real-time, thereby enhancing the overall security of online communication and protecting users' personal and financial information.
  
# Methodology 
* The proposed phishing attack detection system combines rule-based, machine learning, and deep learning systems. It has three components: data preprocessing, feature extraction, and classification.
* The system uses a diverse dataset of legitimate and phishing URLs to extract relevant features. Deep learning techniques like CNNs and transfer learning improve model performance and reduce overfitting.
* The code reads a CSV file with pandas, drops an unwanted column, and splits the data into training and testing sets using the sklearn train_test_split function.
  
![image](https://github.com/GauriMahale/HYBRID-APPROACHES-FOR-PHISHING-ATTACK-DETECTION/assets/84667768/d9dbfb00-9ca5-49eb-80cd-991dc3d25c7f)

Fig 1.1. System Workflow

1. In fig 1.1 . The code trains and tests various machine learning algorithms, including GaussianNB, SVC, ELM, Logistic Regression, and RandomForest.
2. It also creates an ensemble model using the VotingClassifier function from sklearn to improve classification performance.
3. The performance of the models is evaluated using confusion matrix, accuracy, and AUC.
4. Additionally, Flask, a popular Python web framework, can be used to build web-based applications for phishing attack detection.
   
    ![image](https://github.com/GauriMahale/HYBRID-APPROACHES-FOR-PHISHING-ATTACK-DETECTION/assets/84667768/a4b9a30e-4934-4b6a-83ed-1865e8bb6927)

Fig 1.2. System Architecture

![image](https://github.com/GauriMahale/HYBRID-APPROACHES-FOR-PHISHING-ATTACK-DETECTION/assets/84667768/c97e4eaf-42c0-4551-9ce2-b56119f849d6)

Fig 1.3. PROPOSED SYSTEM 


# Result
![image](https://github.com/GauriMahale/HYBRID-APPROACHES-FOR-PHISHING-ATTACK-DETECTION/assets/84667768/2e415ac5-9484-4ac9-9ef9-277b65d4a662)

# Conclusion
In conclusion, the hybrid approach of using both deep learning and machine learning techniques for phishing attack detection has proven to be highly effective, achieving an accuracy of 93% in the voting classifier. 
It provides robust and accurate detection of even sophisticated phishing attacks, improving online security and protecting users.

