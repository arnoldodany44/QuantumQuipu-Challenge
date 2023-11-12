# QuantumQuipu Challenge
Reto para el Qiskit Fall Fest Latino, Escuela en Español de Computación Cuántica
### Reto: Decodificar y Clasificar
Este proyecto se enfoca en codificar los archivos [challenge_train.csv](https://github.com/arnoldodany44/QuantumQuipu-Challenge/blob/main/CodigoPennylane/challenge_train.csv) y [challenge_test.csv](https://github.com/arnoldodany44/QuantumQuipu-Challenge/blob/main/CodigoPennylane/challenge_test.csv) en al menos dos formas diferentes en un circuito cuántico (estas podrían ser codificaciones basadas en ángulos, amplitud, kernel, aleatorias o personalizadas).
Diseña un circuito cuántico variacional para cada una de las codificaciones. Utiliza la columna "Target" como el objetivo, la cual es una clase binaria 0 y 1. Debes utilizar los datos de las columnas F1, F2, F3 y F4 para tu clasificador propuesto.
Considera el ansatz que diseñes como una capa y determina cuántas capas son necesarias para alcanzar el mejor rendimiento.



## Installation

### Install Pennylane

To install PennyLane and run the quantum circuit optimizer, follow these steps:

- Ensure you have Python installed on your system.
- Install PennyLane using pip with the following command:

```bash
  pip install pennylane
```

### Install Qiskit

To install Qiskit and run the quantum circuit optimizer, follow these steps:

- Ensure you have Python installed on your system.
- Install Qiskit using pip with the following command:

```bash
  pip install qiskit
```
    
## Dependencies

**PennyLane:** A Python library that simplifies the simulation and optimization of quantum circuits. It is the only dependency required to run this project.

**Qiskit:** An open-source quantum computing software development framework provided by IBM. It allows users to create, simulate, and run quantum circuits on real quantum hardware and simulators. 


## Usage

To run this project, simply clone this repository to your local machine and, after ensuring that you have the necessary dependencies correctly installed, just run the ChallengePennylane.ipynb file in the Pennylane folder and the ChallengeQiskit.ipynb file in the Qiskit folder.





## Screenshots and Code explanation

For the development of this project, we first started coding in Pennylane because we had just finished Xanadu's challenge. Attempts were made in both technologies, creating a great combination of different coding styles, various circuits, and varying hyperparameters. We conducted deeper research into the problem, and this was the outcome.

![App Screenshot](https://arnoldodany.com/QCImages/Peny1.png)

- Data Loading:

The training and test data are loaded from CSV files. These files contain the data points and target values for the machine learning model.
- Preparing Features and Labels:

The features (input data) and labels (output results) for both training and test datasets are separated. This step is crucial for supervised learning tasks.
- Data Normalization:

The features from the training data are normalized using minimum and maximum values. Normalization is a standard practice in machine learning to improve training efficiency by scaling the input data.
- Conversion to PyTorch Tensors:

The normalized data and the labels are converted into PyTorch tensors. This conversion is necessary because the subsequent machine learning model will be implemented using PyTorch.
- Quantum Device Configuration:

The quantum device is configured with four qubits, corresponding to the four input data features. This setup with PennyLane establishes a quantum circuit that reflects the structure of the input data, allowing for the construction and training of the quantum machine learning model. The choice of four qubits is directly influenced by the dimensionality of the input data, ensuring that the quantum model is appropriately scaled to handle the data.

![App Screenshot](https://arnoldodany.com/QCImages/StronglyCircuit.png)

StronglyEntanglingLayers in PennyLane is a quantum circuit template designed for creating highly entangled states through a sequence of parameterized single-qubit rotations and two-qubit entangling gates. This template is particularly useful in quantum machine learning for classification tasks due to its ability to generate complex quantum states and capture intricate patterns in data. The entanglement and parameterization inherent in these layers allow for rich feature mapping and adaptability, similar to weight adjustments in classical neural networks. Such characteristics make StronglyEntanglingLayers effective in distinguishing between different classes, especially where classical linear methods are insufficient, enhancing the capability of quantum-classical hybrid models.

![App Screenshot](https://arnoldodany.com/QCImages/Peny22.png)

- Quantum Circuit Definition:

A quantum node (qml.qnode) is defined with a device (dev) and an interface ('torch'), creating a bridge between quantum and classical computation.
Inside the quantum circuit, AmplitudeEmbedding is used to encode the features into the quantum state of the qubits. StronglyEntanglingLayers is then applied to the qubits, using trainable weights to create a complex, entangled quantum state.
The circuit measures the expectation value of the Pauli-Z operator on the first qubit, which serves as the output of the circuit.
- Weight Initialization:

The weights for the quantum layers are initialized randomly with a specified range and shape. These weights are then wrapped in a PyTorch Variable with requires_grad=True, enabling automatic gradient computation.
- Cost Function:

A custom cost function is defined to compute the loss. It calculates the mean squared error between the predictions and the labels, adding L2 regularization (using reg_lambda) to the loss to prevent overfitting.
- Optimizer:

An Adam optimizer from PyTorch is used for optimizing the weights, with an adjusted learning rate.
- Training Loop:

The training loop iterates a specified number of steps. In each step, the gradients are reset, the cost is computed and backpropagated, and the optimizer updates the weights. The loss is printed every 10 steps.
- Prediction Function:

A function to make predictions with the trained model is defined. It applies the quantum circuit to the input features and uses a threshold to classify the outputs.
- Model Evaluation:

The model's performance is evaluated by predicting labels for the test dataset and calculating the accuracy as the percentage of correctly predicted labels.

![App Screenshot](https://arnoldodany.com/QCImages/Peny3.png)

Effective Quantum Feature Processing: The utilization of quantum circuits for feature processing, specifically using AmplitudeEmbedding and StronglyEntanglingLayers, appears to be effective in handling the complexities of the dataset. The 70% accuracy indicates that the quantum model has successfully learned and generalized some patterns from the training data.

Room for Improvement: While 70% accuracy is a promising result, especially in the context of quantum machine learning, which is still a developing field, there is room for improvement. Enhancements could be explored through further hyperparameter tuning, experimenting with different quantum circuit architectures, or increasing the dataset size and diversity.

Quantum Advantages: This result suggests that quantum computing, even at its current stage, can contribute to solving classification problems. However, it also highlights the need for more research to fully leverage quantum advantages, especially as quantum hardware and algorithms continue to evolve.

Comparative Performance: If this model's performance is compared with classical machine learning benchmarks for the same dataset, it could provide deeper insights into where quantum models stand in terms of practical utility and efficiency.

Hybrid Model Strengths: The combination of quantum and classical techniques demonstrates the strengths of hybrid approaches. While the quantum circuit processes the features in a way that could capture complex patterns, classical optimization techniques and data handling play a crucial role in the model's overall performance.

![App Screenshot](https://arnoldodany.com/QCImages/Qis1.png)

- Data Loading and Preprocessing:

The script loads training and testing data from CSV files. Features (X_train, X_test) and targets (y_train, y_test) are extracted. The features are then standardized using StandardScaler to ensure that the input data is normalized, a common practice for optimizing machine learning model performance.
- Quantum Circuit Setup:

A quantum circuit (qc) with a specified number of qubits (num_qubits) is created. The circuit includes rotation gates (rx and ry) applied to each qubit. These gates are parameterized using a ParameterVector, allowing for the adjustment of these parameters during the training process.

![App Screenshot](https://arnoldodany.com/QCImages/Qis2.png)

- Quantum Neural Network Creation:

A SamplerQNN is created using the prepared quantum circuit. This quantum neural network is designed to work with Qiskit's machine learning module, integrating quantum computations into the learning process.
- Classifier Configuration:

A NeuralNetworkClassifier is set up using the SamplerQNN and an optimizer (COBYLA). This classifier will train the model using the provided data, adjusting the quantum circuit's parameters to minimize the loss.
- Model Training and Evaluation:

The classifier is trained using the training data and then evaluated on the test data. The accuracy of the classifier is printed, indicating how well the quantum model performed on the classification task.
- Circuit Visualization:

The script concludes by drawing the quantum circuit, allowing for a visual representation of the quantum operations used in the model.

![App Screenshot](https://arnoldodany.com/QCImages/Qis3.png)

The 53% accuracy achieved by this quantum machine learning classifier indicates that while the model has some capability to process and classify the dataset, its performance is moderate and leaves room for improvement. This result highlights the challenges and complexities inherent in quantum machine learning at its current developmental stage. It suggests that further optimization of the quantum circuit design, hyperparameter tuning, and data preprocessing might enhance the model's effectiveness. Additionally, this outcome underscores the importance of ongoing research and development in quantum computing to fully realize its potential in machine learning applications.

## Authors

#### Quantum Codebreakers

- [@Arnoldo Valdez](https://www.github.com/arnoldodany44)
- [@Alejandro Monroy](https://www.github.com/AzpMon)
- [@Francisco Costa](https://www.github.com/podxboq)
- [@Sofía Salazar](https://www.github.com/nsalazard)
- [@Julio Moreno](https://www.github.com/pyspdev)


