# Diabetic Retinopathy Detection using ResNet-50

## Project Overview

This project presents a deep learning-based diagnostic solution for **early detection and classification of Diabetic Retinopathy (DR)**—a leading cause of vision loss among diabetic patients. Leveraging the power of **ResNet-50**, a pre-trained convolutional neural network (CNN), the model classifies high-resolution retinal images into five severity levels of DR:  

- 0 – No DR  
- 1 – Mild  
- 2 – Moderate  
- 3 – Severe  
- 4 – Proliferative DR  

Early detection of DR is crucial for timely medical intervention and preventing irreversible vision damage. This project serves as a decision-support tool for ophthalmologists and healthcare professionals by offering fast, automated classification of retinal images with strong accuracy.


## Key Highlights

- **Model Used:** ResNet-50 (pre-trained on ImageNet)  
- **Frameworks & Tools:** Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn  
- **Techniques:** Image preprocessing, data augmentation, transfer learning, AUC & accuracy evaluation  
- **Performance:** Achieved **Training AUC: 97.77%**, **Validation AUC: 94%**  
- **Dataset:** APTOS 2019 Blindness Detection Dataset from Kaggle (35,000+ images)  
- **Problem Type:** Multi-class image classification (5 classes)


## Problem Statement

> Develop an automated deep learning model that can accurately detect and classify the stage of Diabetic Retinopathy from retinal fundus images, minimizing manual screening time and increasing diagnostic efficiency.


## Project Workflow

1. **Data Acquisition:** High-resolution retinal fundus images from Kaggle  
2. **Data Preprocessing:**  
   - Image resizing (224x224)  
   - Normalization and noise removal  
   - Data augmentation using `ImageDataGenerator`  
3. **Model Architecture:**  
   - Feature extraction with ResNet-50  
   - Fine-tuning the classifier head  
4. **Training:**  
   - Optimizer: Adam  
   - Loss: Categorical Crossentropy  
   - Epochs: 100  
5. **Evaluation:**  
   - Accuracy  
   - Loss  
   - AUC Score  
   - Confusion Matrix  


## Results

| Metric               | Value         |
|----------------------|---------------|
| Training Accuracy    | 87%           |
| Validation AUC Score | 94%           |
| Training AUC Score   | 97.77%        |
| Classification Type  | 5-class       |

Visual plots of accuracy, loss, and AUC are provided in the results section of the code.


## File Structure

```
.
├── DR.py                        # Main Python script with ResNet-50 implementation
├── trainLabels.csv             # Annotated labels for training
├── /Dataset/                   # Directory with retinal images
├── README.md                   # This file
└── [Additional plots & model artifacts]
```


## Mentor

This project was completed as part of the B.Tech final project at **Lakireddy Bali Reddy College of Engineering**, under the guidance of Mr. N. V. Naik, Sr. Assistant Professor.

**Author**
  
- **Surya vamsi Patiballa** [ MS in Data Science at George Washington University ]
  
- Email :- svamsi2002@gmail.com
- LinkedIn :- https://www.linkedin.com/in/surya-patiballa-b724851aa/


## Future Enhancements

- Implement advanced models like **VGG16**, **InceptionV3**, or **EfficientNet**  
- Integrate hyperparameter tuning using **GridSearchCV**  
- Convert the model into a real-time **web-based diagnostic tool** using Flask/Django


## References

1. APTOS 2019 Blindness Detection – Kaggle Dataset :- (https://www.kaggle.com/competitions/aptos2019-blindness-detection)  
2. Deep learning papers cited in the included `Research Paper.pdf`
