Vinay Hiremath - 2021-09-10

INI-503 = NSC Master Thesis (long) and Exam

This README concerns the files in `data_files/`: everything (data, code, ...) used to produce the thesis.*


### Prerequisites
The modules listed in `environment_with_pytorch_cuda_10-1.yml` are installed in the Python environment.
Other modules may needed to be added depending on the specific system setup.

### Steps to run
The code can be run by enabling standard Hydra configurations which define the class to be initialized
and the options with which to initialize them. These are included in:
/gln-thesis/code/configs/[model, trainer, logs, etc.]

The default setup which is used unless an option is overriden to one of the above configs is located at:
/gln-thesis/code/configs/config.yaml

For example, from the `gln-thesis` directory, a standard GLN model that activates the `debug_trainer`, 
uses the MNIST dataset, logs to CSV files, and does not use dataset deskewing can be trained using:
python code/run.py trainer=debug_trainer model=gln_model_mnist datamodule=mnist_datamodule logger=csv datamodule.deskew=false


