# Simple-Feedforward-Neural-Network
This project implements a basic 2-layer feedforward neural network using only NumPy, with no deep learning frameworks.

# Features
- Manual forward & backward propagation
- Sigmoid activation & MSE loss
- Weight and bias initialization
- Model training from scratch
- Visualization and parameter saving
- Custom dataset loading via `.txt`

# Structure
- data.txt                   # Training dataset
- parameter.txt              # Model parameters
- training.py                # Core training logic (forward + backward)
- load_data.py               # Load dataset
- load_parameter.py          # Load weight & bias
- visualization.py           # Visualization of decision boundary
- test.py                    # Optional testing
- XOR/random_generator.py    # Sample data generator for non-linear tasks

# Requirements
- numpy
- matplotlib.pyplot

# Outputs
- example for circular data
  <img width="936" height="711" alt="output" src="https://github.com/user-attachments/assets/74430eb4-2d4a-425a-ad15-ee8a98b8db3f" />
