## Real-Time-ASL-Recognition

This project is a real-time computer vision system that recognizes a limited vocabulary of American Sign Language (ASL) signs from a live webcam feed. Hand landmarks are extracted with MediaPipe, then classified using k-nearest neighbors, a support vector machine, and a neural network trained on a subset of the WLASL dataset. Beyond standard accuracy, precision, recall, and F1, the project focuses on generalization: how well each model recognizes signs from people who were not in the training data, and how performance changes with lighting and camera conditions.
