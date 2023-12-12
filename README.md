<p align="center">
  <img src="https://github.com/TehmoorG/Generative-Hand-X-ray/blob/main/data/real_hands/000000.jpeg" alt="VAE X-ray Image" width="300"/>
</p>


# X-ray Hand Image Generation Using VAE

## Overview
This project involves the development of a Variational Autoencoder (VAE) to generate synthetic X-ray images of hands. Originally created as part of an academic assignment, the project demonstrates the application of VAEs in medical imaging, potentially benefiting areas such as radiologist training and AI in healthcare.

## Project Description
The VAE model in this project is trained on a dataset of hand X-ray images. It focuses on efficiently generating realistic images, balancing quality and computational resources. The repository includes a Jupyter notebook that details the entire process, from data preparation, model design, and training to the generation of new samples.

### Key Features
- Efficient VAE training on a dataset of 8000 images.
- Exploration of image resolutions for optimal feature capture.
- Extensive hyperparameter tuning and network enhancements.
- Comparative analysis of original and generated images for model evaluation.

## Repository Structure
- `notebooks/`: Jupyter notebooks with model development and evaluation.
- `data/`: Directory for dataset storage.
- `src/`: Source code for the project.
- `models/`: Trained model weights and architecture files.
- `data/VAE_hands/`: Folder containing generated images.

## Data Availability
Due to GitHub's file size limitations, only a subset of 1000 images from the original 'Real Hands' dataset is available in this repository. The complete dataset can be accessed [here](<https://drive.google.com/drive/folders/1uDMejVW4qjuw5iBpRqnZKwdyRCj_iYZO?usp=drive_link>).

## Usage
To run the project:
1. Clone the repository.
2. Ensure all dependencies listed in `requirements.txt` are installed.
3. Run the main Jupyter notebook to train the model or generate new X-ray hand images.

## Contributions and Feedback
Contributions to this project are welcome! If you have suggestions or feedback, please open an issue or submit a pull request.

## License
This project is open-sourced under the MIT License. See the LICENSE file for more details.
