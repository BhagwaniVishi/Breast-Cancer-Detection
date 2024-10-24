# Breast Cancer Detection using CNN
This project aims to enhance breast cancer classification by leveraging deep learning techniques, particularly using pre-trained Convolutional Neural Networks (CNNs) combined with data augmentation. This approach aims to overcome the challenges faced in distinguishing different types of cancerous and non-cancerous histopathological images.
- [Breakhis Dataset](http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz)
    
   
## PROBLEM
Histopathology is critical for diagnosing breast cancer, but manual analysis is time-consuming and subject to human error. Automated techniques using deep learning can aid in accurately classifying different types of breast cancer.

**To build a robust and efficient system for breast cancer detection using deep learning techniques. This project demonstrates the effectiveness of using transfer learning for feature extraction and classification.**

## METHOD 
- Dataset:

We use the BreakHis dataset, a public database with 7,909 histopathological images from 82 anonymous patients in Brazil. The images cover four magnification levels (40x, 100x, 200x, and 400x) and consist of eight classes: four benign (adenosis, fibroadenoma, phyllodes tumor, tubular adenoma) and four malignant (ductal carcinoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma).
- Preprocessing:

Images are resized to 256 pixels on the shorter side, then center-cropped to 224x224 pixels. Standard normalization is applied using mean and standard deviation values ([0.485, 0.456, 0.406] for the mean and [0.229, 0.224, 0.225] for the standard deviation). 
- Data Augmentation:

To reduce overfitting, geometric transformations such as horizontal flips, vertical flips, and rotations (90°, 180°, 270°) are used, along with color variations in brightness, contrast, and saturation.
- Feature Extraction:

We utilize pre-trained models (VGG16, ResNet50, DenseNet201) to extract features from the images. The pre-trained networks, trained on large datasets like ImageNet, help capture shape and texture characteristics of the images, which are then fed to a classifier.
- Classification:

The extracted features are classified using a Support Vector Machine (SVM), which performs well with the processed features and helps achieve high accuracy in differentiating between benign and malignant tumors.


#### Dataset:
- To train the model, we have used the renowned Malevis image dataset which contains over 9,100 training malware images from 26 different malware classes with balanced frequency, that is 350, of each class.

#### Model Architectures
- VGG16:

A 16-layer deep network using 3x3 convolutional layers to capture fine-grained features. It ends with three fully connected layers and a softmax layer for classification.
- ResNet50:

A 50-layer network that implements residual learning to facilitate training of deeper models. It uses 3x3 convolution filters and doubles the number of filters whenever the feature map size is halved.
- DenseNet201:

Unlike ResNet, DenseNet201 utilizes dense blocks where each layer receives inputs from all preceding layers. This architecture promotes efficient gradient flow and feature reuse.

#### Breast cancer images example
 - ![Benign](https://github.com/BhagwaniVishi/Breast-Cancer-Detection/blob/main/Images/benign.png)
  - ![Malignant](https://github.com/BhagwaniVishi/Breast-Cancer-Detection/blob/main/Images/malignant.png)



## INFERENCE

- Precision: `100%`
- Recall: `100%`
- Accuracy (test data): `94.9%`


- ####  Accuracy across magnifications.:
    - ![Accuracy curve](https://github.com/BhagwaniVishi/Breast-Cancer-Detection/blob/main/Images/accuracy_plot.png
)

- #### Composite performance summary showing the best Accuracy, GMean, and AUC scores for different models with and without augmentation.:
    - ![Performance summary](https://github.com/BhagwaniVishi/Breast-Cancer-Detection/blob/main/Images/performance_summary.png)

- #### Proposed method architecture.
    
    -![Flowchart](https://github.com/BhagwaniVishi/Breast-Cancer-Detection/blob/main/Images/flowchart.png)


## APPLICATIONS
Automated classification of breast cancer images using deep learning has significant implications for the medical field, allowing for:
- Faster Diagnosis: Reducing the time needed for pathologists to analyze samples.
- Enhanced Accuracy: Providing a second opinion to reduce human error in diagnosis.
- Scalability: Enabling analysis of large datasets that would be time-consuming for manual inspection.

## Future Directions

- Continual Learning:
Implementing models that can adapt to new data without forgetting previously learned information.
- Improved Feature Extraction Techniques:
Reducing data loss during preprocessing or exploring new image formats to retain more features for analysis.


## REFERENCES

- Maleki, Alireza, Mohammad Raahemi, and Hamid Nasiri. "Breast cancer diagnosis from histopathology images using deep neural network and XGBoost." Biomedical Signal Processing and Control 86 (2023): 105152.

- Mani, RK Chandana, and J. Kamalakannan. "The comparative study of CNN models for breast histopathological image classification." 2023 International Conference on Computer Communication and Informatics (ICCCI). IEEE, 2023.
- Atban, Furkan, Ekin Ekinci, and Zeynep Garip. "Traditional machine learning algorithms for breast cancer image classification with optimized deep features." Biomedical Signal Processing and Control 81 (2023): 104534.
