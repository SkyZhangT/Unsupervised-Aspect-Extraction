# Unsupervised Aspect Extraction

An up-to-date pytorch version model of ACL2017 paper ‘‘An unsupervised neural attention model for aspect extraction’’. [(pdf)](http://aclweb.org/anthology/P/P17/P17-1036.pdf)

The original model was implemented in Keras with python2. Many of its features are deprecated in python3. You can always check the original code repo published by the paper author. [(github)](https://github.com/ruidan/Unsupervised-Aspect-Extraction)

## Data

You can find the pre-processed datasets and the pre-trained word embeddings in [[Download]](https://drive.google.com/open?id=1L4LRi3BWoCqJt5h45J2GIAW9eP_zjiNc). The zip file should be decompressed and put in the main folder.

You can also download the original datasets of Restaurant domain and Beer domain in [[Download]](https://drive.google.com/open?id=1qzbTiJ2IL5ATZYNMp2DRkHvbFYsnOVAQ). For preprocessing, put the decompressed zip file in the main folder and run

```
python preprocess.py
python word2vec.py
```

respectively in code/ . The preprocessed files and trained word embeddings for each domain will be saved in a folder preprocessed_data/.

## Train

Under code/ and type the following command for training:

```
python train.py \
--emb ../preprocessed_data/$domain/w2v_embedding \
--domain $domain \
-o output_dir \
```

where _$domain_ in ['restaurant', 'beer'] is the corresponding domain, _--emb_ is the path to the pre-trained word embeddings, _-o_ is the path of the output directory. You can find more arguments/hyper-parameters defined in train.py with default values used in our experiments.

After training, two output files will be saved in code/output*dir/$domain/: 1) \_aspect.log* contains extracted aspects with top 100 words for each of them. 2) _model_param_ contains the saved model weights

## Evaluation

Under code/ and type the following command:

```
python evaluation.py \
--domain $domain \
-o output_dir \
```

Note that you should keep the values of arguments for evaluation the same as those for training (except _--emb_, you don't need to specify it), as we need to first rebuild the network architecture and then load the saved model weights.

This will output a file _att_weights_ that contains the attention weights on all test sentences in code/output_dir/$domain.

To assign each test sentence a gold aspect label, you need to first manually map each inferred aspect to a gold aspect label according to its top words, and then uncomment the bottom part in evaluation.py (line 136-144) for evaluaton using F scores.

One example of trained model for the restaurant domain has been put in pre_trained_model/restaurant/, and the corresponding aspect mapping has been provided in evaluation.py (line 136-139). You can uncomment line 28 in evaluation.py and run the above command to evaluate the trained model.

## Dependencies

python 3.8

- pytorch 1.8.1
- gensim 3.8.3
- numpy 1.20.0

See also requirements.txt
You can install prerequirements, using the following command.

```
pip install -r requirements.txt
```
