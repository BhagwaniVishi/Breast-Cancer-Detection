import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import os

# Define the base directory
base_directory = '/content/drive/MyDrive/Feature_Extractions_Data/'
input_base_directory = os.path.join(base_directory, 'With_Augmentation/Resnet50/')

# Define the magnification levels
magnification_levels = ['40x']

# Loop through each magnification level
for level in magnification_levels:
    print(f'Processing magnification level: {level}')

    # Define paths for each class
    paths = {
        'class_0': os.path.join(input_base_directory, level, f'resnet50_{level}_adenosis.csv'),
        'class_1': os.path.join(input_base_directory, level, f'resnet50_{level}_fibroadenoma.csv'),
        'class_2': os.path.join(input_base_directory, level, f'resnet50_{level}_phyllodes_tumor.csv'),
        'class_3': os.path.join(input_base_directory, level, f'resnet50_{level}_tubular_adenoma.csv'),
        'class_4': os.path.join(input_base_directory, level, f'resnet50_{level}_ductal_carcinoma.csv'),
        'class_5': os.path.join(input_base_directory, level, f'resnet50_{level}_lobular_carcinoma.csv'),
        'class_6': os.path.join(input_base_directory, level, f'resnet50_{level}_mucinous_carcinoma.csv'),
        'class_7': os.path.join(input_base_directory, level, f'resnet50_{level}_papillary_carcinoma.csv')
    }

    # Load data from CSV files
    dataframes = {label: pd.read_csv(path) for label, path in paths.items()}

    # Add labels to the data
    for i, (label, df) in enumerate(dataframes.items()):
        df['label'] = i

    # Combine the data from all classes
    data = pd.concat(dataframes.values())

    # Separate features and labels
    X = data.drop('label', axis=1)
    y = data['label']

    # Remove non-numeric columns
    X = X.select_dtypes(include=[float, int])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Define the XGBoost classifier
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Standardize the features and train the model using a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', xgb)
    ])

    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    print(f"Accuracy for {level}: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report for {level}:\n", classification_report(y_test, y_pred))

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for {level}:\n", conf_matrix)

    # Plot confusion matrix
    class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7']
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {level}')

    # Save the plot in the same directory as input files
    plot_file_name = os.path.join(input_base_directory, level, f'{level}_confusion_matrix.png')
    plt.savefig(plot_file_name)
    plt.show()

    print(f'Saved confusion matrix plot to {plot_file_name}')