# CNN Image Classification

This project implements a Convolutional Neural Network (CNN) for image classification. It includes scripts for training and evaluating the model, as well as preprocessing the dataset.

## Project Structure

- **data/**: Contains the training and testing images.
  - **train/**: Directory for training images.
  - **test/**: Directory for testing images.
  
- **models/**: Contains the CNN model architecture.
  - **cnn_model.py**: Defines the `CNNModel` class with methods for initialization, forward pass, and compilation.

- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis.
  - **exploration.ipynb**: Used for visualizing and understanding the dataset.

- **scripts/**: Contains scripts for training, evaluating, and preprocessing the data.
  - **train.py**: Script for training the CNN model.
  - **evaluate.py**: Script for evaluating the trained model.
  - **preprocess.py**: Script for preprocessing the images.

- **requirements.txt**: Lists the dependencies required for the project.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd cnn-image-classification
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

- To preprocess the data, run:
  ```
  python scripts/preprocess.py
  ```

- To train the model, run:
  ```
  python scripts/train.py
  ```

- To evaluate the model, run:
  ```
  python scripts/evaluate.py
  ```

## License

This project is licensed under the MIT License.