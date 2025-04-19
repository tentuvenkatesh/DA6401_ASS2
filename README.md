In this Repo,there are two Parts(i.e Part_A and Part_B)

Wandb report link:(https://api.wandb.ai/links/tentuvenkatesh2-indian-institute-of-technology-madras/8ce5w7pa)

**Part_A - Training from Scratch**

**ass2-part-a.ipynb:**

Here I implement a 5 layer CNN network from scratch using PyTorch library and train it using inaturalist dataset.

This file contains the code of the CNN architecture and training it using our dataset.
The file has been trained in Kaggle where I have uploaded my dataset in kaggle and got the links from Kaggle.
Below are the sweep configurations,which i used for the sweep with count=20.

Wandb sweep Coonfigurations:

filters_num :[16, 32, 64]

filter_org :['same', 'double', 'half']

act_fn :['relu', 'gelu', 'silu', 'mish', 'tanh']

data_aug : [False]

batch_norm:[True, False]

dropout:[0, 0.2, 0.5]

learning_rate:[1e-3, 1e-4]

l2_reg:[0, 0.0005, 0.05]

batch_size:[16, 32]

kernel_size: [[3]*5, [3, 5, 5, 7, 7], [5]*5, [7, 5, 5, 3, 3] ]

num_neurons_dense:[64, 128, 256]

epochs: [6]

**Best hyperparameters,which gave best validation accuracy:39.46974**

act_fn: relu

batch_norm: True

batch_size: 32

data_aug: False

dropout: 0

epochs: 6

filter_org: double

filters_num: 64

kernel_size: [3, 3, 3, 3, 3]

l2_reg: 0.0005

learning_rate: 0.0001

num_neurons_dense: 256

**After increasing epochs from 6 to 10 with respect to above hyper parameters,i observed that some validation accuracy has been increases(i.e 42.27)**

**ASS2_PART_A_test.ipynb**

This file does the testing of our model using the test data. The model trained using the best hyperparameters sweeped from wandb is used for testing.

**Test accuracy: 43.15**

3 random images from each class in the test folder were plotted along with their actual and predicted name.The plot was logged into wandb.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

**Part_B:Fine-tuning a pre-trained model**

I used resnet50 model as the pre-trained model and fine-tune it using the inaturalist data.

**ass2-part-b.ipynb:**
First I loaded the resnet50 model from torchvision.models, then I finetuned it using different types of strategies. Found the best hyperparameters using wandb sweep,and The file has been trained in Kaggle.

**Wandb sweep configurations:**
lr:[1e-3, 1e-4]

freeze_percent[0.2, 0.6, 0.9]

l2_reg : [0, 0.0005, 0.05]

batch_size:[32, 64]

epochs : [5, 10]

**Best Hyper parameters i found was:**

lr:  0.0001

freeze_percent:  0.9

l2_reg : 0

batch_size:64

epochs : 10

**After 20 sweeps,with respect to above hyperparameters i got better validataion accuracy,which is 78.93948**

**ass2-art-b-test.ipynb:**

This file does the testing of our fine-tuned model using the test data. The model trained using the best hyperparameters sweeped from wandb is used for testing.

**Test accuracy: 79.30%**
