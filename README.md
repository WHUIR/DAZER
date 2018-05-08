# DAZER
Our Tensorflow implement of ACL 2018 paper 'A deep relevance model for zero-shot document filtering'

<p align="center"> 
<img src='https://github.com/WHUIR/DAZER/blob/master/model-img.png' width="800" align="center">
</p>


### Requirements
- Tensorflow 1.2
- Numpy
- traitlets

### Guide To Use

**Prepare your dataset**: first, prepare your own data.
see below Data Preparation section.


**Configure**: then, configure the model through the config file. Configurable parameters are listed [here](#configurations)

[sample.config](https://github.com/WHUIR/DAZER/blob/master/sample.config)

in addition you need to change the zero-shot label setting in [get_label.py](https://github.com/WHUIR/DAZER/blob/master/get_label.py)


**Training** : pass the config file, training data and validation data as
```ruby
python model.py config-file\
    --train \
    --train_file: path to training data\
    --validation_file: path to validation data\
    --checkpoint_dir: directory to store/load model checkpoints\ 
    --load_model: True or False(depends on existing or not). Start with a new model or continue training
```

[sample-train.sh](https://github.com/WHUIR/DAZER/blob/master/sample-train.sh)

**Testing**: pass the config file and testing data as
```ruby
python model.py config-file\
    --test \
    --test_file: path to testing data\
    --test_size: size of testing data (number of testing samples)\
    --checkpoint_dir: directory to load trained model\
    --output_score_file: file to output documents score\

```
Relevance scores will be output to output_score_file, one score per line, in the same order as test_file.


### Data Preparation

All seed words and documents must be mapped into sequences of integer term ids. Term id starts with 1. 

**Training Data Format**

Each training sample is a tuple of (seed words, postive document, negative document)

`seed_words   \t postive_document   \t negative_document `

Example: `334,453,768   \t  123,435,657,878,6,556   \t  443,554,534,3,67,8,12,2,7,9 `


**Testing Data Format**

Each testing sample is a tuple of (seed words, document)

`seed_words   \t document`

Example: `334,453,768  \t   123,435,657,878,6,556`


**Validation Data Format**

The format is same as training data format


**Label Dict File Format**

Each line is a tuple of (label_name, seed_words)

`label_name/seed_words`

Example: `alt.atheism/atheist christian atheism god islamic`


**Word2id File Format**

Each line is a tuple of (word, id)

`word id`

Example: `world 123`

**Embedding File Format**

Each line is a tuple of (id, embedding)

`id embedding`

Example: `1 0.3 0.4 0.5 0.6 -0.4 -0.2`


### Configurations 


**Model Configurations**
- <code>BaseNN.embedding_size</code>: embedding dimension of word
- <code>BaseNN.max_q_len</code>: max query length 
- <code>BaseNN.max_d_len</code>: max document length
- <code>DataGenerator.max_q_len</code>: max query length. Should be the same as <code>BaseNN.max_q_len</code> 
- <code>DataGenerator.max_d_len</code>: max query length. Should be the same as <code>BaseNN.max_d_len</code> 
- <code>BaseNN.vocabulary_size</code>: vocabulary size
- <code>DataGenerator.vocabulary_size</code>: vocabulary size
- <code>BaseNN.batch_size</code>: batch size 
- <code>BaseNN.max_epochs</code>: max number of epochs to train
- <code>BaseNN.eval_frequency</code>: evaluate model on validation set very this epochs
- <code>BaseNN.checkpoint_steps</code>: save model very this epochs


**Data**
- <code>DAZER.emb_in</code>: path of initial embeddings file
- <code>DAZER.label_dict_path</code>: path of label dict file
- <code>DAZER.word2id_path</code>: path of word2id file


**Training Parameters**
- <code>DAZER.epsilon</code>: epsilon for Adam Optimizer 
- <code>DAZER.embedding_size</code>: embedding dimension of word
- <code>DAZER.vocabulary_size</code>: vocabulary size of the dataset
- <code>DAZER.kernal_width</code>: width of the kernel 
- <code>DAZER.kernal_num</code>: num of kernel
- <code>DAZER.regular_term</code>: weight of L2 loss
- <code>DAZER.maxpooling_num</code>: num of K-max pooling
- <code>DAZER.decoder_mlp1_num</code>: num of hidden units of first mlp in decoder part
- <code>DAZER.decoder_mlp2_num</code>: num of hidden units of second mlp in decoder part
- <code>DAZER.model_learning_rate</code>: learning rate for model instead of adversarial calssifier
- <code>DAZER.adv_learning_rate</code>: learning rate for adversarial classfier
- <code>DAZER.train_class_num</code>: num of class in training time
- <code>DAZER.adv_term</code>: weight of adversarial loss when updating model's parameters
- <code>DAZER.zsl_num</code>: num of zero-shot labels
- <code>DAZER.zsl_type</code>: type of zero-shot label setting 






