# Credit risk management

This projet focus on credit risk management issues using advanced machine learning methods.  
* Understand the latest technology proposed in the literature and select a representative method of research  
* Python implements literature data and proposes improvements  
* The advantages and disadvantages of the bank’s real data comparison method  
This projet mainly implements 5 methods based on SVM. Fuzzy SVM, Bilateral Fuzzy SVM, Least Suqare Fuzzy SVM, Bilateral Least Square Fuzzy SVM, Weighted Least Square SVM.  

## Deployment  

Combine FSVM, LSFSVM with integrated learning. Improve the robustness of the model while maintaining the high accuracy of Fuzzy SVM. And use three encoding methods according to the characteristic of qualitative variables, respectively ordinal encoding, additive encoding, one-hot encoding.

## Acknowledge  

- [Bio-Inspired Credit Risk Analysis - Computational Intelligence With Support Vector Machines](https://www.researchgate.net/publication/287303068_Bio-inspired_credit_risk_analysis_Computational_intelligence_with_support_vector_machines) (2008, Lean Yu)      
- [FSVM-CIL: Fuzzy Support Vector Machines for Class Imbalance Learning](https://ieeexplore.ieee.org/abstract/document/5409611) (2010, Rukshan Batuwita and Vasile Palade)   
- [Evaluating Credit Risk with a Bilateral-Weighted Fuzzy SVM Model](https://www.researchgate.net/publication/314366058_Evaluating_Credit_Risk_with_a_Bilateral-Weighted_Fuzzy_SVM_Model?enrichId=rgreq-b7149e2ce491c19745f752b32d35f1ac-XXX&enrichSource=Y292ZXJQYWdlOzMxNDM2NjA1ODtBUzo0NzA3MDgzMzg4NjAwMzRAMTQ4OTIzNzAyMTMzNw%3D%3D&el=1_x_3&_esc=publicationCoverPdf) (2008, Lean Yu) 
- [A Least Squares Bilateral-Weighted Fuzzy SVM Method to Evaluate Credit Risk](https://www.researchgate.net/publication/224346922_A_Least_Squares_Bilateral-Weighted_Fuzzy_SVM_Method_to_Evaluate_Credit_Risk) (2008, Wei Huang and Lean Yu) 
- [https://github.com/shiluqiang/WLSSVM_python](https://github.com/shiluqiang/WLSSVM_python)  
- [A DBN-based resampling SVM ensemble learning paradigm for credit classification with imbalanced data](https://doi.org/10.1016/j.asoc.2018.04.049) (2018, Lean Yu, Rongtian Zhou, Ling Tang, Rongda Chen)  

## Description of the fichier

* FSVM.py : Fuzzy SVM
* BFSVM.py : Bilaterial Fuzzy SVM
* LS_FSVM.py : Least Square Fuzzy SVM
* BLSFSVM.py : Least Square Bilaterial Fuzzy SVM
* WLSSVM.py : Weighted Least Square SVM
* FSVM_bagging.py : Fuzzy SVM and Bagging, each estimator with best paramater
* LS_FSVM_bagging.py : Least Square Fuzzy SVM and Bagging, each estimator with best paramater
* bagging.py : SVM, FSVM and LSFSVM in bagging, Parameter customization
* GridSearch_parametre.py : Grid Search chose the best parameter
* Kernel.py : 'rbf','linear','poly' kernel
* DataDeal.py : Data input and preprocessing
* Precision.py : calculate recall, precision and accuracy
