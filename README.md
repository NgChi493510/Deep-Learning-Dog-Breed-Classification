# Dog Breed Classification with Deep Learning
## Project Overview
This repository contains my approach to solving a dog breed classification problem using deep learning, focusing on pre-trained models and optimizing their performance through extensive experimentation. The dataset consists of images across 70 different dog breeds. My goal was to develop a highly accurate classification model using transfer learning, model fine-tuning, and hyperparameter tuning. After a series of comprehensive experiments, the best result achieved was with ConvNeXt, scoring 0.97142 on the public leaderboard and 0.97875 on the private leaderboard on Kaggle.
## Dataset
The dataset consists of images of dog breeds, with each image labeled with its corresponding breed. Due to an issue with the dataset, one class (66) was missing, leading to a total of 70 classes.

Train images/Validation images: 8:2
Test images: Kaggle provided test set with private and public scoreboards

**Files:**

- train.csv - Training set, including class ID, filepaths, and class.
- test.csv - Test set, including the filepath to images in test.zip.
- train.zip - Training data, images pre-organized into folders.
- test.zip - Test data, images pre-organized by ID.

## Data Preprocessing
The dataset required significant preprocessing to prepare it for training:

- Image Resizing: All images were resized to a standard resolution of 224x224 pixels to match the input size of pre-trained models like VGG16, ResNet, and ConvNeXt.
- Normalization: I used normalization with mean and standard deviation based on the ImageNet dataset
- Augmentation: Various augmentation techniques were applied to increase dataset diversity and help prevent overfitting:
   - Random horizontal flips
   - Random rotations
   - Color jittering
   - Random crops and zooming

## Experimenting with Advanced Pretrained Models
To push beyond the VGG16 baseline, I explored several pre-trained models from the torchvision and timm libraries. Each model was fine-tuned using transfer learning and further optimized through hyperparameter tuning. Below is a summary of the key models I experimented with:

1. **ResNet50**: Resulted in a public score more than 0.95 after adjusting learning rate and batch size.
2. **EfficientNetB7**: Initially overfitted, but after introducing dropout and regularization techniques, it stabilized more 0.95.
3. **DenseNet**: Good performance, DenseNet201 achieved 0.96285 after several rounds of hyperparameter tuning, marking a significant improvement.
4. **GoogLeNet**: Achieved more 0.94 but overfitting was observed despite data augmentation, leading to lower accuracy.
5. **ResNeXt**: Showed good performance but still trailed behind DenseNet201.
6. **Ensemble Models**: combining DenseNet201 and GoogLeNet/Resnet/EfficientNet to leverage the strengths of both. Although this improved overall performance, it did not surpass DenseNet201's best individual result.
7. **Best Result: ConvNeXt***
The best model was ConvNeXt, achieving 0.97142 on the public leaderboard and 0.97875 on the private leaderboard. ConvNeXt is a powerful architecture, particularly for image classification tasks, and it benefited greatly from fine-tuning.

In several experiments using DenseNet and ResNet, I observed that while the public score was around 0.95875 to 0.96, the private score significantly improved, reaching up to 0.97142. This variation between public and private scores suggests that the models were well-tuned for the underlying test data distribution.

## Fine-Tuning
Fine-tuning pre-trained models played a critical role in improving performance. The key steps in the fine-tuning process are outlined below:

- Freezing Initial Layers: froze the initial layers of the pre-trained models to retain their feature extraction capabilities, allowing the model to focus on learning the specific dog breed features.
- Unfreezing Specific Layers: After the initial training phase, I selectively unfroze the final layers (such as fully connected layers) to fine-tune the model further on the target dataset. This allowed the model to adjust its higher-level representations to the dog breeds.
- Optimizer for Fine-Tuning: For fine-tuning, I used Stochastic Gradient Descent (SGD) with a reduced learning rate to allow the model to learn subtle patterns without drastically modifying the pre-trained weights:
- Learning Rate Scheduling: I implemented learning rate scheduling to gradually reduce the learning rate, which helps the model converge more smoothly during fine-tuning. I used a ReduceLROnPlateau scheduler based on validation loss:
- Early Stopping: To prevent overfitting during fine-tuning, I applied early stopping when the validation loss stopped improving.

## Hyperparameter Tuning
To further improve model performance, I conducted extensive experiments on various hyperparameters:

- Learning Rate: I experimented with different learning rates (e.g., 0.001, 0.0001) and learning rate schedulers to find the optimal convergence speed.
- Batch Size: Batch sizes of 32, 64, and 128 were tested. A smaller batch size (32) helped reduce overfitting in some cases.
- Epochs: I ran experiments ranging from 10 to 20 epochs, with early stopping implemented to avoid overfitting.
- Optimizers: Different optimizers, including SGD, Adam, and RMSprop, were tested. The best results were achieved with SGD combined with momentum.
- Data Augmentation: Extensive use of data augmentation techniques helped reduce overfitting, especially in models like ResNet50 and DenseNet201.

## Challenges and Limitations
Despite achieving a strong private score of 0.97875, several challenges limited further improvement:

- Data Imbalance: Some dog breeds had significantly fewer images than others, leading to imbalanced training and difficulties in generalizing across all breeds.
- Dataset Size: The dataset, while sufficient for initial training, lacked diversity in some classes, limiting the model's ability to generalize.
- Model Capacity: While larger models like EfficientNetB7 showed potential, they often led to overfitting due to the relatively small size of the dataset.

## Conclusion and Future Work
The experiments demonstrate that ConvNeXt outperformed other models in this dog breed classification task. However, further work is needed to push the boundaries, possibly through:

- More Advanced Data Augmentation: Techniques like CutMix, MixUp, or AutoAugment could help further diversify the dataset.
- Handling Imbalanced Data: Implementing techniques like oversampling, SMOTE, or class-weighted loss functions could address the data imbalance issue.
- Model Ensemble: A more refined ensemble approach, such as a weighted average of multiple models, could enhance the final score.


This project shows the power of modern deep learning architectures combined with systematic fine-tuning and hyperparameter tuning to achieve high-performance results on a challenging dataset.