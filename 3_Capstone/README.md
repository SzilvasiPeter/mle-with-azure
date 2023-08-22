# Your Project Title Here

*TODO:* Write a short introduction to your project.

Two project idea:
- **MNIST or CIFAR-10 image recognition**
  - Use GPU for training
  - For AutoML, find the best metrics and params for training
  - For HyperDrive, experiment with multi-layer neural network. For example, first layer no or little dropout and the end of the layers bigger dropout. Inspired by the brain, lower features (edges, curves) are "hard-wired" (less dropout), the higher features (faces, concepts) are "soft-wired" (bigger dropout).
  - For Model deployment inference, rescale image (`test_img = test_img / 255.0`) and then convert images:
    - In case of MNIST: 28x28 flatten
    - In case of CIFAR: 32x32x3 tensor
- Fine-tune gpt-3.5 on alpaca dataset

## Project Set Up and Installation

The notebooks use the `azureml-sdk` package that is installed in the Azure Machine Learning (AML) studio by default. To use the SDK, you need an Azure subscription, an AML studio resource, and a compute resource inside AML. Besides the notebooks, you can run the training and testing scripts locally.

Before installing the dependencies, create a virtual environment:
```
python -m venv .venv
```

Activate it using `source .venv/bin/activate` on the Linux terminal or `source .venv/Scripts/activate` on the Windows Git Bash.

If you don't have Tensorflow, install it using the following command:
```
pip install tensorflow
```

Now, you can execute the scripts using `python <(cifar10|mnist)_(train|test).py>` commands. The models are saved in native Keras format under the `models` folder.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

- The project should make use of a dataset that is not available in the azureml framework

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

- encouraged to provide details behind your reasoning for choosing the settings and experiment configuration

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

- screenshot RunDetails widget that shows the progress of the training runs of the different experiments
- notebook about the performance of the different models on the primary metric of your experiment.
- notebook contains details of the best model and documents the parameters of that model
- screenshot of the best model with its run id
- notebook contains code showing the best model being registered

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

- Use one type of sampling method: grid sampling, random sampling or Bayesian sampling.
- Logs metrics during the training process
- Specify an early termination policy (not required in case of Bayesian sampling).

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

- The notebook code demonstrates tuning of at least 2 hyperparameters using hyperdrive
- screenshot RunDetails widget that shows the progress of the training runs of the different experiments
- notebook about the effects of the different hyperparameters on the primary metric of your model.
- screenshot of the best model with its run id
- notebook also contains code showing the best model being registered.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

- code contains: Model being registered, Model being deployed, A file containing the environment details
- screenshot showing the model endpoint as active
- code showing an inference request being sent to the deployed model: send an HTTP inference request to the model's URI

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

- Convert your model to ONNX format
- ~~Deploy your model to the Edge using Azure IoT Edge~~ Cannot create with Udacity subscription (Authorization error)
- Enable logging in your deployed web app
