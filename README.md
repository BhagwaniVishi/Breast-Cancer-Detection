# Malware Classification using CNN
Addressed rising cyber threats by enhancing malware detection, overcoming limitations of traditional methods, notably in dealing with obfuscation challenges, using Deep Learning.
- [Notebook uploaded on Kaggle as well](https://www.kaggle.com/code/aakajave/malveis-simple-sgd-optmisied)
    
    `kaggle kernels pull aakajave/malveis-simple-sgd-optmisied`
## PROBLEM
The dynamic and polymorphic nature of modern malware requires more advanced techniques for timely and accurate identification.
In this context, the problem at hand is 

**To develop a robust and efficient malware detection system using deep learning (DL) techniques through CNN training model, and minimize the loss of essential features of malware images**

## METHOD 
- Conversion of malware Portable Executable header files into images and feed the images to the training model.
- But this conversion leads to variable size of images, whereas a CNN model needs fixed size of input. 
- So, to avoid maximum data loss in terms of textures or features related to malware exectubles we used bilinear interpolation to standardize the images to size (256,256,3).
- The next step involves building and training a robust model using Convolutional Neural Network. 

#### Dataset:
- To train the model, we have used the renowned Malevis image dataset which contains over 9,100 training malware images from 26 different malware classes with balanced frequency, that is 350, of each class.

#### Why PE header files ? 
- The Portable Executable (PE) file format is used by Windows executables, `object code`, and DLLs. The PE file format is a data structure that contains the information necessary for the Windows OS loader to manage the wrapped executable code.
#### Executables to images: 
- Use a compiler (e.g., GCC) to link the object file and generate an executable file.
- Use a disassembler (e.g., objdump for Linux) to disassemble the executable file and extract the opcodes.
- Convert the opcodes into a format suitable for image creation, that is, map each opcode to a visual representation in the image.
- Used a predefined color palette for mapping.
- Used PIL (Python Imaging Library) to create and manipulate images.
- For each opcode, the corresponding pixel in the image was set with the mapped color.

#### Malware images example



## INFERENCE

- Precision: `100%`
- Recall: `100%`
- Accuracy (test data): `93.62%`
- Accuracy (validation data): `84.003%`

- #### Accuracy and Validation Accuracy curve over the time:
    - ![Accuracy curve](https://github.com/kajaveaniruddha/Malware-Classification-using-CNN/assets/66174998/6f4dd393-459e-424b-8810-e0225a8a4a7c)

- #### Loss and Validation Loss curve over the time:
    - ![Loss curve](https://github.com/kajaveaniruddha/Malware-Classification-using-CNN/assets/66174998/8da8592a-c178-4255-a99b-6c7dfc7152ae)

- #### Plotting layers using `keras plot_model` library:
    
    -![Layers](https://github.com/kajaveaniruddha/Malware-Classification-using-CNN/assets/66174998/6d4bc6dd-7428-430c-ba19-3605882f9829)


## APPLICATIONS
- Malware detection using deep learning has found extensive applications in bolstering cybersecurity efforts, offering a proactive defense against the escalating sophistication of malicious software. One prominent application lies in the identification of previously unknown or polymorphic malwares.
- Deep learning’s ability to automatically learn intricate features from vast datasets is particularly advantageous in distinguishing between benign and malicious software. The models can analyze the structural and behavioral characteristics of files, enabling the identification of subtle anomalies indicative of malware activity.
- Furthermore, the real-time processing capabilities of deep learning contribute to the rapid detection of emerging threats. The speed at which deep learning models can analyze and classify data makes them well-suited for dynamic cybersecurity environments, allowing for swift response and
mitigation

## Future Directions

- Exploring the integration of continual learning and self-supervised learning techniques presents promising advantages. Continual learning enhances adaptability to evolving attack vectors, while self-supervised learning offers notable benefits in situations where acquiring labeled data proves challenging.
- There is an opportunity for improvement in the methods employed to mitigate data loss during the conversion of malware into images. Developing an approach that utilizes images without necessitating cropping is an avenue for enhancement in this context.


## REFERENCES

- [He, Ke and Kim, Dong-Seong Malware Detection with Malware Images using Deep Learning Techniques In 2019 18th IEEE International Conference On Trust, Security And Privacy In Computing And Communications](https://ieeexplore.ieee.org/document/8887303)

- [Kamran Shaukat and Suhuai Luo and Vijay Varadharajan. A novel deep learning-based approach for malware detection. In Engineering Applications of Artificial Intelligenc](https://www.sciencedirect.com/science/article/abs/pii/S0952197623002142)
