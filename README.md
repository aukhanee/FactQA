## Factoid Question Answering
This repo attempts to produce an implementation of "alternating stochastic gradient" descent algorithm discussed in [1]. Preprocessing is inspired from Simple-Question-Answering-With-Memory-Networks(https://github.com/Jerryzhao-z/simple-question-answering-with-memory-networks)

# Preprocessing
One has to specify location of all datasets and other local configuration information in SETTINGS.JSON file.
The vocabulary of individual words is produced with the `preprocessing/vocabulary.py` script.
Questions preprocessing `g(q)` is done with the `preprocessing/questions.py` script.
Facts processing `f(y)` is done with the `preprocessing/facts.py` script.

# Training
After preprocessing the dataset, training of facoid question answering is done using following command.
$python3 train.py
Please refer to the paper [1] for detailed understanding of how the train script trains our question-answering system. 

# Testing
A python script is provided for testing the trained system. Use `test.py` to test the system.


# TODO
Transfer learning on another dataset using the trained model.






# References
[1] Large-scale Simple Question Answering with Memory Networks (https://arxiv.org/pdf/1506.02075.pdf)
