import pickle

import numpy as np
import os
import torch
import random
import config
import process
from transformers import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from scipy.special import expit
import logging
import torch.optim as optim
from Bert import HBert

def init_logger(log_name: str = "echo", log_file='log', log_file_level=logging.NOTSET):
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    file_handler = logging.FileHandler(log_file, encoding="utf8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

logger = init_logger()

def seed_everything(seed=1996):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

# def single_label_f1(preds, targets):
#     preds = np.array(preds).reshape(-1, config.nclass)
#     targets = np.array(targets).reshape(-1, config.nclass)
#     for i in range(config.nclass):
#         print('f1 for class {}:{}'.format(i+1, f1_score(preds[:, i], targets[:, i], average="macro")))

'''
Convert to multi-binary label
'''
def train(fold):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_iter = process.DataIter('train', fold)
    test_iter = process.DataIter('test', fold)
    torch.cuda.set_device(0)
    # if config.multi_cuda:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,0"
    model = HBert().to(device)
    # if config.multi_cuda:
    #     model = torch.nn.DataParallel(model, device_ids=[1,2,3,0])
    CE = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=0)
    epoch_steps = train_iter.get_steps(True)
    best_test_acc,best_test_f1, b_pre, b_recall = 0.0, 0.0, 0.0, 0.0
    for epoch in tqdm(range(config.epoch)):
        for step, ipt in enumerate(train_iter):
            ipt = {k: v.to(device) for k, v in ipt.items()}
            out = model(**ipt)
            # the main multi-label classification loss
            loss = CE(out, ipt["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            model.zero_grad()
            if step % 10 == 0:
                logger.info("epoch-{}, step-{}/{}, loss:{}".
                            format(epoch, step, epoch_steps, loss.data))

        targets, preds = eval(model, test_iter, device)
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro')
        pre = precision_score(targets, preds, average='macro')
        recall = recall_score(targets, preds, average='macro')
        if acc > best_test_acc:
            best_test_acc = acc
        if f1 > best_test_f1:
            best_test_f1 = f1
            b_pre = pre
            b_recall = recall
            if 'fold-{}'.format(str(fold)) not in os.listdir('../datas/{}'.format(config.dataset)):
                os.makedirs('../datas/{}/fold-{}'.format(config.dataset, str(fold)))
            model.save('../datas/{}/fold-{}'.format(config.dataset, str(fold)))   # 保存最优模型
        logger.info("epoch:{} best_acc:{},f1:{},precision:{},recall:{}".
                   format(epoch, best_test_f1, best_test_f1, b_pre, b_recall))
    return best_test_f1, best_test_f1, b_pre, b_recall

def eval(model, test_iter, device):
    model.eval()
    targets, preds = [],[]
    for step, ipt in enumerate(test_iter):
        ipt = {k: v.to(device) for k, v in ipt.items()}
        out = model(**ipt)
        target = ipt["labels"].cpu().detach().numpy()
        pred = torch.max(out, dim=-1)[1].cpu().detach().numpy()
        targets.extend(list(target))
        preds.extend(list(pred))
    model.train()
    np.save('../datas/{}/bert_preds.npy'.format(config.dataset), targets)
    # np.save('../datas/{}/labels.npy'.format(config.dataset), preds)
    return targets, preds


if __name__ == "__main__":
    b_accs, b_f1s, b_pres, b_recalls = [], [], [], []
    n = config.n_fold
    for i in range(0,n):
        print('fold======================== {}================='.format(str(i)))
        b_acc, b_f1, b_pre, b_recall = train(fold=i)
        b_accs.append(b_acc)
        b_f1s.append(b_f1)
        b_pres.append(b_pre)
        b_recalls.append(b_recall)
    means = [np.mean(b_accs), np.mean(b_f1s), np.mean(b_pres), np.mean(b_recalls)]
    stds = [np.std(b_accs), np.std(b_f1s), np.std(b_pres), np.std(b_recalls)]
    print('best-acc:{}, f1:{}, precision:{}, recall:{}'.
          format(str(means[0])+'±'+str(stds[0]), str(means[1])+'±'+str(stds[1]),
                 str(means[2])+'±'+str(stds[2]), str(means[3])+'±'+str(stds[3])))
