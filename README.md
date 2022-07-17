# RSGNN

Code for "Improving Abusive Language Detection with Online Interaction Network" submitted to Information Processing &amp; Management

For running the code, you should get HateBERT checkpoints and change the 'bert_path' parameter in the 'Community/config.py'.

Then, run datas/data_preprocess.py for cleaned datas, communities, 10-folds and random splitted datasets.

After that, run Bert/train_bert.py for the best trained features on train set for each split/fold.

Finally, run Community/train_community.py for the the model with communities (interaction networks).

Modify Community/config.py for different configuratiions.
