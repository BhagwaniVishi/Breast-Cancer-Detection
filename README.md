# Breast Cancer Detection using CNN

##Methods
The proposed method in this study aims to demonstrate the effect of using pre-trained CNNs for feature extraction, utilizing transfer learning networks combined with data augmentation. The extracted features are then classified using SVM. This section details the proposed procedures and experimental setups, including the datasets, preprocessing techniques, transfer learning methods, and model parameters. The study employs VGG16, ResNet50, and DenseNet for feature extraction, with SVM used as the classifier.

1)Dataset- BreakHis [49] is a public database containing 7909 biopsy images from 82 anonymous Pathological Anatomy and Cytopathology (P&D) lab patients in Brazil. The images are available at four magnification factors (40X, 100X, 200X, and 400X), with 3 RGB channels and 8 bits, and a resolution of 700 × 460 saved in PNG format. Differentiating benign tumors from malignant tumors is the most critical challenge of cancer histopathology [50]. Pathologists use different magnification factors to improve the accuracy of this distinction. However, the process is challenging due to the excessive workload and the need to analyze through different microscope lenses. Automating these processes will significantly contribute to the field of medicine. In the literature, various approaches have been applied to automate this process using datasets with different magnification factors.
The BreakHis dataset categorizes benign tumors into four subclasses: adenosis (A), fibroadenoma (F), phyllodes tumor (PT), and tubular adenoma (TA). Malignant tumors are classified into ductal carcinoma (DC), lobular carcinoma (LC), mucinous carcinoma (MC), and papillary carcinoma (PC). Fig. 4 shows the adenosis and mucinous carcinoma cancer classes at various magnification factors.

2)Preprocessing-
Preprocessing is crucial for standardizing and normalizing input images in deep learning models. In this study, the preprocessing pipeline includes several steps: resizing images to 256 pixels on the shorter side, applying a center crop of 224x224 pixels, converting the images to tensors, and normalizing them using specific mean and standard deviation values ([0.485, 0.456, 0.406] for the mean and [0.229, 0.224, 0.225] for the standard deviation). These steps ensure uniformity, improve training speed, and help achieve better model convergence.

3)Data augmentation –
Data augmentation was used in the training process to avoid the risk of overfitting [36]. Moreover, the techniques used include geometric transforms such as random horizontal flips, random vertical flips, 90-degree rotations, 180-degree rotations, 270-degree rotations, and color variations including adjustments in brightness, contrast, saturation, and hue

4)Feature extraction with pre-trained network
Analysis with a large number of features in a dataset requires significant memory space and computational power. Feature extraction aims to convert raw data into numerical features that can be processed, reducing the dataset's size and mitigating overfitting issues and resource demands. Pre-trained transfer learning networks offer a fast and effective method for feature extraction, producing more accurate features than classical techniques. Transfer learning allows models trained for one task to be reused for related tasks, enhancing learning efficiency and performance with less training data. This approach is particularly advantageous for handling large datasets typical in deep learning.

In this study, we utilize transfer learning to build our models for breast cancer classification, using pre-trained convolutional neural networks (CNNs) like VGG16, ResNet50, and DenseNet201 for feature extraction. These CNNs, trained on extensive databases, extract shape and texture characteristics from the dataset, enabling the use of traditional classifiers for the classification step. Unlike methods that train a network from scratch, transfer learning requires only fine-tuning, thus improving speed and performance with less training data.
Deep learning architectures for feature extraction

1)	Vgg16-VGG16 is a deep convolutional neural network known for its simplicity and effectiveness in image recognition. It consists of 16 layers, mainly 3 × 3 convolutional layers, stacked to capture fine-grained features. Max-pooling layers reduce spatial dimensions while preserving key features. The network ends with three fully connected layers and a softmax layer for classification. Pre-trained on large datasets like ImageNet, VGG16 excels in extracting robust features through transfer learning, making it ideal for image classification and feature extraction tasks.
	
2)	Resnet50- ResNet50 is a convolutional neural network (CNN) that is 50 layers deep and accepts an image input size of 224 × 224. Its architecture is inspired by VGG, featuring 3 × 3 filters and adhering to two straightforward design principles: (1) layers with the same output feature map size have the same number of filters, and (2) when the feature map size is halved, the number of filters is doubled to maintain the time complexity per layer.
   
3)	Densenet201- DenseNet201 is similar to ResNet, but it consists of dense blocks that are densely connected: each layer receives input from all previous layers' output feature maps. A dense block comprises batch normalization, ReLU activation, 3 × 3 convolution, and zero padding. A transition layer includes batch normalization, a 1 × 1 convolution, and average pooling.


References-
[1] Z. Momenimovahed, H. Salehiniya, Epidemiological characteristics of and risk factors for breast cancer in the world, Breast Cancer (Dove Med Press) 11 (Apr. 2019) 151–164, https://doi.org/10.2147/BCTT.S176070.
[2] WHO, “Cancer Research,” 2021 https://gco.iarc.fr/today/fact-sheets-cancers (assessed Nov. 16, 2022).
[3] World Health Organization, Breast Cancer, https://www.who.int/news- room/fact-sheets/detail/breast-cancer, access date [14 July 2022].
[4] Z. Metelko et al., Pergamon the world health organization quality of life assessment (Whoqol): position paper from the world health organization, 41 (10), 1995.
 [5] H. Sung et al., Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries, pp. 1–41, 2021.
[6] N.H. Al-Ziftawi, A.A. Shafie, M.I. Mohamed Ibrahim, Cost-effectiveness analyses of breast cancer medications use in developing countries: a systematic review, Expert Rev. Pharmacoecon. Outcomes Res. 21 (4) (Aug. 2021) 655–666, https://doi.org/ 10.1080/14737167.2020.1794826.
[7] A. Bish, A. Ramirez, C. Burgess, M. Hunter, Understanding why women delay in seeking help for breast cancer symptoms B, vol. 58, pp. 321–326, 2005.
[8] Breast Cancer Statistics | How Common Is Breast Cancer? https://www.cancer.org/cancer/breast-cancer/ about/how-common-is-breast-cancer.html#written_by.Accessed3Apr2022
[9] Newglobal breast cancer initiative highlights renewed commitment to improve survival. https://www.who. int/news/item/08-03-2021-new-global-breast-cancer-initiative-highlights-renewed-commitment-to-improve survival. Accessed 3 Apr 2022
[10] Breast cancer. https://www.who.int/news-room/fact-sheets/detail/breast-cancer. Accessed 3 Apr 2022
[11] C. Sitaula, S. Aryal, Fusion of whole and part features for the classification of histopathological image of breast tissue, Health Inf. Sci. Syst. 8 (1) (Dec. 2020) 38, https://doi.org/10.1007/s13755-020-00131-7. 
[12] E.D. Carvalho, et al., Breast cancer diagnosis from histopathological images using textural features and CBIR, Artif. Intell. Med. 105 (May 2020), 101845, https://doi. org/10.1016/j.artmed.2020.101845.
[13]  V. Bhise, S.S. Rajan, D.F. Sittig, R.O. Morgan, P. Chaudhary, H. Singh, Defining and 
measuring diagnostic uncertainty in medicine: A systematic review, J. Gen. Intern. 
Med. 33 (1) (Jan. 2018) 103–115, https://doi.org/10.1007/s11606-017-4164-1.
[14] S. Robertson, H. Azizpour, K. Smith, J. Hartman, Digital image analysis in breast pathology-from image processing techniques to artificial intelligence, Transl. Res. 194 (Apr. 2018) 19–35, https://doi.org/10.1016/j.trsl.2017.10.010.
 [15] K. G. Liakos, P. Busato, D. Moshou, S. Pearson, and D. Bochtis, “Machine Learning in Agriculture: A Review,” Sensors, vol. 18, no. 8, Art. no. 8, Aug. 2018, doi: 10.3390/s18082674. 
[16] M.H. Wilkerson, K. Lanouette, R.L. Shareff, Exploring variability during data preparation: a way to connect data, chance, and context when working with complex public datasets, Math. Think. Learn. (May 2021) 1–19, https://doi.org/ 10.1080/10986065.2021.1922838.
[17] G. Zhang, W. Wang, J. Moon, J.K. Pack, S.I. Jeon, A review of breast tissue classification in mammograms, Proc. 2011 ACM Res. Appl. Comput. Symp. RACS 2011, pp. 232–237, 2011.
[18]  Wilson ML, Fleming KA, Kuti MA, Looi LM, Lago N, Ru K:
 Access to pathology and laboratory medicine services: A crucial
 gap. The Lancet, 2018
 [19]  RobboySJ, Weintraub S, Horvath AE, Jensen BW, Alexander CB,
 Fody EP, Crawford JM, Clark JR, Cantor-Weinberg J, Joshi MG,
 Cohen MB, Prystowsky MB, Bean SM, Gupta S, Powell SZ,
 Speights VO Jr, Gross DJ, Black-Schaffer WS: Pathologist work
force in the United States: I. Development of a predictive model to
 examine factors influencing supply. Archives of Pathology and
 Laboratory Medicine 137:1723–1732, 2013
 [20] Pöllänen I, Braithwaite B, Haataja K, Ikonen T, Toivanen P:
 Current analysis approaches and performance needs for whole slide
 image processing in breast cancer diagnostics. Proc. Embedded
 Computer Systems: Architectures, Modeling, and Simulation
 (SAMOS), 2015 International Conference on: City
[21] Veta M, Pluim JP, Van Diest PJ, Viergever MA: Breast cancer
 histopathology image analysis: A review. IEEE Transactions on
 Biomedical Engineering 61:1400–1411, 2014
[22] CollinsFS,VarmusH:Anewinitiativeonprecisionmedicine.New England Journal of Medicine 372:793–795, 2015
[23] Reardon S: Precision-medicine plan raises hopes: US initiative highlights growing focus on targeted therapies. Nature 517:540 541, 2015
[24] D.Varshni,K.Thakral,L.Agarwal,R.Nijhawan,A.Mittal,Pneumoniadetection usingCNNbasedfeatureextraction, in: 2019IEEEInternationalConferenceon Electrical,ComputerandCommunicationTechnologies, ICECCT, IEEE,2019,pp. 1–7.
 [25] H. Zerouaoui, A. Idri, Deep hybrid architectures for binary classification of medicalbreastcancerimages,Biomed.SignalProcess.Control71(2022)103226.
[26] S. Hasan Abdulla, A. M. Sagheer, and H. Veisi, "Breast Cancer Classification Using Machine Learning Techniques: A Review", Turkish Journal of Computer and Mathematics Education, Vol.12 No.14 (2021), 1970- 1979.
[27] M.H. Wilkerson, K. Lanouette, R.L. Shareff, Exploring variability during data preparation: a way to connect data, chance, and context when working with complex public datasets, Math. Think. Learn. (May 2021) 1–19, https://doi.org/ 10.1080/10986065.2021.1922838.
[28] Mohammad Rahimzadeh, Abolfazl Attar, Seyed Mohammad Sakhaei, A fully automated deep learning-based network for detecting covid-19 from a new and large lung ct scan dataset, Biomed. Signal Process. Control 68 (2021) 102588.
[29] T. Liu, R. Su, C. Sun, X. Li, L.Wei, EOCSA: Predictingprognosis of epithelial ovarian cancerwithwhole slide histopathological images, Expert Syst. Appl. (2022)117643.
 [30] P. Yang, X. Yin, H. Lu, Z. Hu, X. Zhang, R. Jiang, H. Lv, CS-co: A hy brid self-supervised visual representation learning method for H&E-stained histopathological images,Med. ImageAnal.81(2022)102539.
[31] X. Dong,M. Li, P. Zhou, X. Deng, S. Li, X. Zhao, Y.Wu, J. Qin,W. Guo, Fusingpre-trainedconvolutionalneuralnetworksfeaturesformulti-differentiated subtypesof liver canceronhistopathological images, BMCMed. Inform.Decis. Mak.22(1) (2022)1–27.
[32] H. Zerouaoui, A. Idri, Reviewing machine learning and image processing based decision-making systems for breast cancer imaging, J. Med. Syst. 45 (2021) 8.
[33] R. Yan et al., A Hybrid Convolutional and Recurrent Deep Neural Network for Breast Cancer Pathological Image Classification, Proc. – 2018 IEEE Int. Conf. Bioinforma. Biomed. BIBM 2018, pp. 957–962, 2019. 
[34] F.R. Cordeiro, W.P. Santos, A.G. Silva-Filho, A semi-supervised fuzzy GrowCut algorithm to segment and classify regions of interest of mammographic images, Expert Syst. Appl. 65 (2016) 116–126.
[35] C. Zhu, F. Song, Y. Wang, H. Dong, Y. Guo, J. Liu, Breast cancer histopathology image classification through assembling multiple compact CNNs, BMC Med. Inform. Decis. Mak. 19 (1) (2019) 1–17.
[36] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional 
networks for large-scale image recognition. arXiv preprint 
arXiv:1409.1556. 
[37] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep 
residual learning for image recognition. In Proceedings of the IEEE 
conference on computer vision and pattern recognition, pages 770–778, 
2016.
[38] J. Kundale, and S. Dhage, "Classification of Breast Cancer using Histology images: Handcrafted and Pre-Trained Features Based Approach", IOP Conf. Series: Materials Science and Engineering, 1074 (2021). IOP Publishing, doi:10.1088/1757- 012008, 899X/1074/1/012008
[39] D. Bardou, K. Zhangi, and S. M. Ahmad, "Classification of Breast Cancer Based on Histology Images Using Convolutional Neural Networks", IEEE, v(6), 2018.
 [40] Mahesh Gour, Sweta Jain, and T Sunil Kumar. Residual learning based cnn for breast cancer histopathological image classification. International Journal of Imaging Systems and Technology, 30(3):621– 635, 2020.
[41] F.A. Spanhol, L.S. Oliveira, C. Petitjean, L. Heutte, Breast cancer histopathological image classification using convolutional neural networks, in: 2016 International Joint Conference on Neural Networks, IJCNN, IEEE, 2016, pp. 2560–2567.
[42] Y. Celik, M. Talo, O. Yildirim, M. Karabatak, U.R. Acharya, Automated invasive ductal carcinoma detection based using deep transfer learning with whole-slide images, Pattern Recogn. Lett. 133 (May 2020) 232–239, https://doi.org/10.1016/ j.patrec.2020.03.011.
[43] P. Agarwal, A. Yadav, P. Mathur, Breast cancer prediction on BreakHis dataset using deep CNN and transfer learning model, Lect. Notes Netw. Syst. 238 (2022) 77–88, https://doi.org/10.1007/978-981-16-2641-8_8.
[44] W. Zhi, H. W. F. Yeung, Z. Chen, S. M. Zandavi, Z. Lu, and Y. Y. Chung, “Using Transfer Learning with Convolutional Neural Networks to Diagnose Breast Cancer from Histopathological Images,” 2017. doi: 10.1007/978-3-319-70093-9_71. 
[45] Shallu and R. Mehra, “Breast cancer histology images classification: Training from scratch or transfer learning?,” ICT Express, vol. 4, no. 4, pp. 247–254, Dec. 2018, doi: 10.1016/j.icte.2018.10.007.
[46] S. Boumaraf, et al., Conventional machine learning versus deep learning for magnification dependent histopathological breast cancer image classification: A comparative study with visual explanation, Diagnostics (Basel) 11 (3) (Mar. 2021) 528, https://doi.org/10.3390/diagnostics11030528.
[47] Erkan Deniz, Abdulkadir Şengür, Zehra Kadiroğlu, Yanhui Guo, Varun Bajaj, Ümit Budak, Transfer learning based histopathologic image classification for breast cancer detection, Health Inf. Sci. Syst. 6 (1) (2018) 1–7.
[48] Hasnae Zerouaoui, Ali Idri, Deep hybrid architectures for binary classification of medical breast cancer images, Biomed. Signal Process. Control 71 (2022) 103226.
[49] D. Bardou, K. Zhangi, and S. M. Ahmad, "Classification of Breast Cancer Based on Histology Images Using Convolutional Neural Networks", IEEE, v(6), 2018.
[50] A. Ashtaiwi, Optimal histopathological magnification factors for deep learning based breast cancer prediction, Appl. Syst. Innov. 5 (5) (2022) 87.



