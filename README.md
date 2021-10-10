# Dashboard for EmulatingLSTM Models over Business Processes and its Qualitative Evaluation

The project is developed as the part of Master Thesis Project at [Eindhoven University of Technology](https://www.tue.nl/en/), under the guidance of [Dr. Dirk Fahland](https://www.tue.nl/en/research/researchers/dirk-fahland/). 

The base code of this project is taken from [GenerativeLSTM v1.1.0](https://github.com/AdaptiveBProcess/GenerativeLSTM/releases/tag/v1.1.0) which is based on Camargo et al. paper [Learning Accurate LSTM Models of Business Processes](https://link.springer.com/chapter/10.1007/978-3-030-26619-6_19).

[Streamlit](https://docs.streamlit.io/) has been used to create the dashboard, and a trimmed down version of this project is kept in [Streamlit based Predictive Process Monitoring powered by LSTM](https://github.com/rhnfzl/streamlit-predictive-process-monitoring-dashboard-using-lstm) for hosting it on servers.

## Getting Started

These instructions will help you set up a development and testing copy of the project on your local machine.

### Prerequisites

- To run this code, first install [Anaconda](https://www.anaconda.com/products/individual) on your system, don't foget to check the PATH while installation.
- Change the directory to desired location where you would like to clone the repository, and then clone it.
- Create Conda virtual Environment using ```conda create -n <env name>```
- Activate the Virtual Env : ```conda activate <env name>```
- Install Python 3.7 using ```conda install python=3.7```
- Then install the packages required from the [requirement.txt](https://github.com/rhnfzl/business-process-dashboard-for-lstm/blob/master/requirements.txt) file using ```pip install -r requirements.txt```


## Running the script

Once you've established the environment, you may run the dashboard using the ```streamlit run dashboard.py``` from the root directory.


<!---
# DeepGenerator: Learning Accurate Generative Models of Business Processes with LSTM Neural Networks

The code here presented is able to execute different pre- and post-processing methods and architectures for building and using generative models from event logs in XES format using LSTM anf GRU neural networks. This code can perform the next tasks:


* Training LSTM neuronal networks using an event log as input.
* Generate full event logs using a trained LSTM neuronal network.
* Predict the remaining time and the continuation (suffix) of an incomplete business process trace. 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

To execute this code you just need to install Anaconda in your system, and create an environment using the *environment.yml* specification provided in the repository.

## Running the script

Once created the environment, you can perform each one of the tasks, specifying the following parameters in the lstm.py module, or by command line as is described below:

*Training LSTM neuronal network:* To perform this task you need to set the required activity (-a) as 'training' followed by the name of the (-f) event log, and all the following parameters:

* Implementation (-i): type of keras lstm implementation 1 cpu, 2 gpu
* lSTM activation function (-l): lSTM optimization function (see keras doc), None to set it up as the default value.
* Dense activation function (-d): dense layer activation function (see keras doc), None to set it up as the default value.
* optimization function (-p): optimization function (see keras doc).
* Scaling method (-n) = relative time between events scaling method max or lognorm.
* Model type (-m): type of LSTM model one of specialized, concatenated, shared_cat, shared_cat_gru, specialized_gru or concatenated_gru.
* N-gram size (-z): Size of the n-gram (temporal dimension)
* LSTM layer sizes (-y): Size of the LSTM layers.

```
(lstm_env) C:\sc_lstm>python lstm.py -a training -f Helpdesk.xes -i 1 -l None -d linear -p Nadam -n lognorm -m shared_cat -z 5 -y 100
```

*Predictive task:* It is possible to execute various predictive tasks with DeepGenerator, such as predicting the next event, the case continuation, and the remaining time of an ongoing case. Similarly, it is possible to generate complete event logs starting from a zero prefix size. To perform these tasks, you need to set the activity (-a) as ‘predict_next’ for the next event prediction, ‘pred_sfx’ for case continuation and remaining time, and ‘pred_log’ for the full event log generation. Additionally, it's required to indicate the folder where the predictive model is located (-c), and the name of the .h5 model (-b). Finally, you need to specify the method for selecting the next predicted task (-v) ‘random_choice’ or ‘arg_max’ and the number of repetitions of the experiment (-r). **NB! The folders and models were generated in the training task and can be found in the output_files folder:

```
(lstm_env) C:\sc_lstm>-a pred_log -c 20201001_426975C9_FAC6_453A_9F0B_4DD528CB554B -b "model_shared_cat_02-1.10.h5" -v "random_choice" -r 1"
```
*Predict the next event and role:* To perform this task the only changes with respect with the previous ones are that you need to set the required activity as 'predict_next' and its not necesary to set the maximum trace length:

```
(lstm_env) C:\sc_lstm>python lstm.py -a predict_next -c 20190228_155935509575 -b "model_rd_150 Nadam_22-0.59.h5" -x False
```
## Examples

Models examples and experimental results can be found at <a href="http://kodu.ut.ee/~chavez85/bpm2019/" target="_blank">examples</a>
## Authors

* **Manuel Camargo**
* **Marlon Dumas**
* **Oscar Gonzalez-Rojas**

-->