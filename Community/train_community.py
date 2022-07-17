import logging
import os
import random
import torch
import numpy as np
from models.GCN import GCN, RSGCN
from models.GAT import GAT

import config
from data_loader import Dataloader
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


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

# seed = random.randint(0, 10000)
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

def train(fold=0):
    dataloader = Dataloader(fold)
    features, g, tfidf_g, labels, train_index, test_index = dataloader.get_data()
    seed_everything()
    logger = init_logger()
    torch.cuda.set_device(0)
    CE = torch.nn.CrossEntropyLoss()  # 损失函数
    if config.model_name == 'GCN':
        model = GCN(config.hidden, config.drop).cuda()
    if config.model_name == 'GAT':
        model = GAT(config.hidden, heads=config.heads).cuda()
    if config.model_name == 'RSGCN':
        model = RSGCN(config.hidden, config.L, config.drop).cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-5)
    b_acc, b_f1, b_pre, b_recall = 0.0, 0.0, 0.0, 0.0
    lr = config.lr
    for epoch in range(config.epoch):
        if epoch > 400:
            lr = lr * 0.98
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
        model.train()
        preds = model(features.cuda(), g.cuda(), tfidf_g.cuda())
        assert preds.shape[0] == labels.shape[0]
        loss = CE(preds[train_index].cpu(), labels[train_index])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        model.zero_grad()
        # eval
        model.eval()
        with torch.no_grad():
            logits = model(features.cuda(), g.cuda(), tfidf_g.cuda())
            logits = torch.max(logits, dim=-1)[1]
            logits = logits.detach().cpu().numpy()
            ls = labels.detach().numpy()
            acc = accuracy_score(logits[test_index], ls[test_index])
            f1 = f1_score(logits[test_index],ls[test_index],average='macro')
            precision = precision_score(logits[test_index],ls[test_index], average='macro')
            recall = recall_score(logits[test_index],ls[test_index], average='macro')
        if acc > b_acc:
            b_acc = acc
        if f1 > b_f1:
            b_f1 = f1
            b_pre = precision
            b_recall = recall
            # best_feas = model.x1.detach().cpu().numpy()
            # np.save('../datas/{}/best_features.npy'.format(config.dataset), best_feas)
        logger.info('epoch:{}, best-acc:{}, f1:{}, precision:{}, recall:{}'.
                    format(epoch, str(b_acc), str(b_f1), str(b_pre), str(b_recall)))
    # np.save('../datas/{}/preds.npy'.format(config.dataset), logits[test_index])
    # np.save('../datas/{}/labels.npy'.format(config.dataset), labels[test_index])
    return b_acc, b_f1, b_pre, b_recall

if __name__ == '__main__':
    b_accs, b_f1s, b_pres, b_recalls = [], [], [], []
    n_fold = config.n_fold
    for i in range(n_fold):
        print('fold {}================='.format(str(i)))
        b_acc, b_f1, b_pre, b_recall = train(fold=i)
        b_accs.append(b_acc)
        b_f1s.append(b_f1)
        b_pres.append(b_pre)
        b_recalls.append(b_recall)
    means = [np.mean(b_accs), np.mean(b_f1s), np.mean(b_pres), np.mean(b_recalls)]
    stds = [np.std(b_accs), np.std(b_f1s), np.std(b_pres), np.std(b_recalls)]
    print('best-acc:{}, f1:{}, precision:{}, recall:{}'.
          format(str(means[0]) + '±' + str(stds[0]), str(means[1]) + '±' + str(stds[1]),
                 str(means[2]) + '±' + str(stds[2]), str(means[3]) + '±' + str(stds[3])))