# 🧠 Deep Learning: CNN from Scratch + PyTorch UNet

This project demonstrates two deep learning implementations in a single, unified pipeline:

1. **CNN from Scratch using NumPy** – A foundational implementation of a convolutional neural network trained on a subset of the MNIST dataset.
2. **UNet with PyTorch** – A semantic segmentation model trained on the Oxford-IIIT Pet Dataset for multi-class image segmentation.

It’s designed to show a full spectrum of deep learning skills: from manually coding operations like convolution and backpropagation, to applying a modern segmentation model with PyTorch.

---

## 📁 Project Structure

```bash
cnn-unet-project/
├── main.py             # All code: NumPy CNN + PyTorch UNet
├── requirements.txt    # Dependencies to install
├── README.md           # Project overview and instructions
```
## 🚀 Getting Started

1. **Clone the repository**

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    You’ll need Python 3.8+. PyTorch with GPU support is recommended (but not required for testing).

3. **Run the project**
    ```bash
    python main.py
    ```
    This will:
    - Train a custom CNN on digits 0, 1, and 2 from the MNIST dataset using NumPy
    - Plot the training and test loss curves
    - Train a PyTorch UNet on the Oxford-IIIT Pet Dataset
    - Display a grid of predictions vs ground truth masks
    - Save a trained UNet model checkpoint

---

## 🔍 Key Highlights

### 🧮 CNN from Scratch (NumPy)
- Implemented using only NumPy
- Manual convolution, ReLU, flatten, fully connected layers, and backpropagation
- Trained on filtered MNIST dataset (classes 0, 1, 2)
- Loss tracked and plotted during training

### 🎯 PyTorch UNet
- Encoder–decoder structure with skip connections
- Handles 3-class segmentation for pet images
- Uses CrossEntropyLoss and Adam optimizer
- Visual output comparison: Original Image vs True Mask vs Predicted Mask
- Model checkpoint saved at end of training

## 📊 Example Outputs

**CNN Loss Plot**

<img src="cnn_loss_plot.png" width="400">

**UNet Prediction Grid**

---

## 🧰 Requirements

- numpy  
- matplotlib  
- scipy  
- torch  
- torchvision  
- tensorflow  

## 📚 Datasets Used

- MNIST (via TensorFlow)
- Oxford-IIIT Pet Dataset (via torchvision)

---

## 💡 Learning Outcomes

- Hands-on understanding of CNN architecture and training mechanics
- Ability to build and debug from scratch using NumPy
- Practical modeling using PyTorch with standard datasets
- Visualization of model performance

---

## 👥 Contributors

- **Julisa Delfin** – MS Data Science, DePaul University
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/julisadelfin/) 
