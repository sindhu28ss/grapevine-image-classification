# Grapevine-Image-Classification
<p align="center">
  <img src="https://github.com/sindhu28ss/grapevine-image-classification/blob/main/images/leaves.jpeg" width="300">
</p>

## Project Overview
This project focuses on classifying different grapevine leaf species using deep learning, contributing to the viticulture and food industries. 
Grapevines are primarily cultivated for their fruit, but their leaves are also valuable as a by-product, influencing both price and taste. 
Accurate classification of grapevine leaf species is essential for quality control, market pricing, and culinary applications. 
Leveraging the CNN model, this model effectively identifies five grapevine leaf species using high-resolution images. 
The model is trained with TensorFlow and Keras, deployed via Flask, and containerized with Docker and Kubernetes, ensuring scalability and ease of deployment.

## 📌 Table of Contents  
- [Project Overview](#project-overview)  
- [The Problem](#the-problem)  
- [Solution: Project Objective](#solution-project-objective)  
- [Impact & Applications](#impact--applications)  
- [Project Directory Structure](#project-directory-structure)  
- [Dataset](#dataset)  
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
- [Model Training and Optimization](#model-training-and-optimization)  
- [Python Scripts](#python-scripts)  
- [Model Deployment](#model-deployment)  
- [Dependency and Environment Management](#dependency-and-environment-management)  
- [Docker Containerization](#docker-containerization)  
- [Deployment on Kubernetes](#deployment-on-kubernetes)  


## The Problem
Accurate classification of grapevine leaf species is essential for viticulture, agriculture and commercial processing. However, manual identification is time-consuming and prone to errors due to:

- 🔹 Visual Similarities: Many grapevine species have similar leaf structures, making differentiation difficult.
- 🔹 Image Variability: Changes in lighting, angles, and leaf conditions affect classification accuracy.
- 🔹 Lack of Expertise: Farmers and processors often lack technical knowledge, leading to misidentification and pricing inconsistencies.

An automated deep learning-based solution can streamline this process, improving accuracy, efficiency, and decision-making in viticulture.

## Solution: Project Objective
This project develops a deep learning model for automated classification of five grapevine leaf species using CNN architecture.

- 🔹 High Accuracy: Fine-tuned deep learning model for precise species classification.
- 🔹 Scalability: Adaptable to include additional grapevine species in the future.
- 🔹 Real-time Classification: Users can upload an image and get instant predictions with confidence scores.

## Impact & Applications
- ✔ Viticulture & Agriculture: Helps farmers identify species for optimized cultivation and disease management.
- ✔ Supply Chain & Pricing: Ensures accurate species-based pricing for commercial processing.
- ✔ Research & Conservation: Supports genetic studies and biodiversity preservation.
- ✔ Quality Control Automation: Enhances efficiency in large-scale sorting and classification.

## Project Directory Structure
```
|grapevine-image-classification/
│
├── images/                         # Illustrations and analysis results
├── test_images/                    # Images for model evaluation
│
├── Dockerfile                      # Docker containerization setup
│
├── Grapevine_EDA.ipynb             # Exploratory Data Analysis notebook
├── Grapevine_training.ipynb        # Model training and evaluation notebook
├── grapevine_xception_model.keras  # Saved Keras model
│
├── Pipfile                         # Pipenv virtual environment setup
├── Pipfile.lock                    # Pipenv dependency lock file
├── requirements.txt                # List of required Python dependencies
│
├── train.py                        # Script for training the model
├── predict.py                      # Flask API script for model inference
│
├── deployment.yaml                 # Kubernetes deployment configuration
├── service.yaml                    # Kubernetes service configuration
│
└── README.md                       # Project documentation
```
## Dataset

- **Source:** The dataset used for this project is available on [Kaggle](https://www.kaggle.com/datasets/muratkokludataset/grapevine-leaves-image-dataset/data).
- The dataset consists of images of grapevine leaves belonging to five distinct species:
    - **'Ak'**
    - **'Ala_Idris'**
    - **'Buzgulu'**
    - **'Dimnit'**
    - **'Nazli'**
- The images are high-resolution and categorized into respective species folders.
- Used for image classification to differentiate between grapevine species based on leaf characteristics.

## Exploratory Data Analysis (EDA)

As part of **Exploratory Data Analysis (EDA)**, several preprocessing steps were performed to ensure high-quality and clean data before model training.

### Data Cleaning & Preprocessing
- ✅ **Removed Corrupted Images**: Identified and deleted unreadable images.
- ✅ **Removed Duplicate Images**: Used hashing to detect and remove duplicate images.
- ✅ **Checked for Blank Images**: Ensured no entirely blank images were present in the dataset.
- ✅ **Removed Blurry Images**: Used the Laplacian variance method to filter out low-quality images.
- ✅ **Analyzed Image Properties**: Examined image dimensions, file formats, and color modes.
- ✅ **Converted RGBA to RGB**: Standardized all images to RGB format for consistency.
- ✅ **Saved Cleaned Dataset in a Separate Folder**: Stored the processed dataset in `Converted_Dataset` to keep the original data intact.

### Grapevine Leaf Classes
<p align="center">
  <img src="https://github.com/sindhu28ss/grapevine-image-classification/blob/main/images/GrapeVine-classes.png" width="1000">
</p>

## Model Training and Optimization

- **Dataset Preparation:** Splitting: Train (70%), Validation (20%), Test (10%).
- **Model Architecture:** Xception model (pre-trained on ImageNet) as the base model for transfer learning to classify grapevine leaf species.
- **Hyperparameter Tuning:**
  - **Learning Rate:** Tested `0.0001, 0.001, 0.01, 0.1`
  - **Dense Layer Size:** Tried `64, 128, 256, 512`
  - **Dropout Rate:** Tested `0.2, 0.5, 0.8`
  - **Best Evaluated Parameteres:** learning_rate = `0.01`; size = `128`; droprate = `0.2`
- **Final Model Training:** The Final model was trained and evaluated using the best evaluated parameteres and Checkpointing enabled to save the best model.
- **Model Evaluation on Test Data:** Loaded the best-performing model and evaluated it on unseen test data.
- **Final Accuracy:** 92% on test data

## Python Scripts
`train.py` – Script for training the model using the prepared dataset and best hyperparameters.
`predict.py` – Flask API script for making predictions on new images.

## Model Deployment
- **Flask API:** The trained model is deployed using Flask, exposing a RESTful API for predictions.
  -**Start the Flask Server:** `python predict.py`

## Dependency and Environment Management
To ensure a reproducible environment, this project uses Pipenv for dependency management. Follow the steps below to set up your environment and install dependencies.
Ensure you have Python 3.12 installed before proceeding.

- **Clone the Repository:**  `git clone https://github.com/sindhu28ss/grapevine-image-classification.git`
- **Set Up a Virtual Environment:**
  - **Install Pipenv:** `pip install pipenv`
  - **Create a virtual environment with Python 3.12.7:** `pipenv --python 3.12.7`
  - **Activate the virtual environment:** `pipenv shell`
- **Install Project Dependencies:** `pipenv install`
- **Verify Installation:** `pipenv run pip freeze`
- **Train the model:** `python train.py`
- **Run the model:** `python predict.py`

 ## Docker Containerization  
Docker containerizes the **Flask API**, ensuring consistent execution across environments, enhancing portability, and simplifying deployment.
- **`Dockerfile`** → Defines instructions for building the Docker image.  
- **`requirements.txt`** → Lists all Python dependencies required for the application.  
- **Build the Docker Image:** `docker build -t grapevine-classification .`
- **Run the Docker Container:** `docker run -p 9696:9696 grapevine-classification`


## Deployment on Kubernetes  
The containerized **Flask application** is deployed to a **Kubernetes cluster** using **Minikube** for local testing.  
- **Tag the locally built Docker image:** `docker image tag grapevine-classification sindhu0405/grapevine-classification`
- **Push the Docker image to Docker Hub:**  `docker push sindhu0405/grapevine-classification`

### Install & Set Up Kubernetes Tools:
- **Install kubectl (Kubernetes CLI):** `brew install kubectl`
- **Verify installation:** `kubectl version --client`
- **Install Minikube:** `brew install minikube`
- **Verify Minikube installation:** `minikube version`
- **Start Minikube: minikube start:** `minikube start`

### Create Kubernetes Configuration Files:
- **Create deployment.yaml** → Defines the deployment of the containerized Flask app.
- **Create service.yaml** → Exposes the application as a service.

### Deploy the Application to Kubernetes:
- **Apply the deployment configuration:** `kubectl apply -f deployment.yaml`
- **Apply the service configuration:** `kubectl apply -f service.yaml`
- **Verify if pods are running:** `kubectl get pods`
- **Verify service is running:** `kubectl get services`

### Access the Application:
- **Get the Minikube Service URL:** `minikube service grapevine-classification-service –url`
- **Test the Application by sending an image for classification:**
`curl -X POST -F "file=@/Users/sindhujaarivukkarasu/Documents/ML capstone-2/test_images/Ala_Idris (11).png" http://127.0.0.1:56393/classify`
```Expected Response:
{
  "confidence": 0.7459,
  "predicted_class": "Ala_Idris"
}
```
<p align="center">
  <img src="https://github.com/sindhu28ss/grapevine-image-classification/blob/main/images/cloud-curl.png" width="1000">
</p>

