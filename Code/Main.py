import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model
from DataHandler import DataHandler
from Utils.Utils import *
import os
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import random
import pickle
from copy import deepcopy
from tqdm import *
# Function to set random seed for reproducibility
def set_seed(seed):
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)

    if t.cuda.is_available():
        t.cuda.manual_seed(seed)
        t.backends.cudnn.benchmark = False
        t.backends.cudnn.deterministic = True

def compute_mad(embeddings):
    """
    Compute the Mean Absolute Deviation (MAD) of embeddings to measure global smoothness.

    Args:
        embeddings (torch.Tensor): Tensor of shape (N, D) where N is the number of nodes and D is the embedding dimension.

    Returns:
        float: The MAD value indicating global smoothness.
    """

    # Ensure embeddings are on the GPU
    if not embeddings.is_cuda:
        embeddings = embeddings.cuda()

    # Compute the cosine similarity matrix
    norm_embeddings = t.norm(embeddings, p=2, dim=1, keepdim=True)
    cosine_similarity = t.mm(embeddings, embeddings.t()) / (norm_embeddings * norm_embeddings.t())

    # Compute cosine distance matrix
    cosine_distance = 1 - cosine_similarity

    # Compute the mean absolute deviation
    n = embeddings.size(0)
    mad = t.sum(cosine_distance) / (n * (n - 1))

    return mad.item()

# Define the Coach class for model training and evaluation
class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('DRUG', args.drug, 'GENE', args.gene)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Acc']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    # Function to create a formatted print statement
    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    # Function to perform external testing
    def external_test_run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        return reses['Acc']

    def random_sample_nonzero(self,arr_list, k, pad_value=-1):
        # 初始化结果列表
        result_list = []

        # 遍历每一行
        for row in arr_list:
            # 将 row 转换为 numpy 数组
            row_np = np.array(row)

            # 找出非 -1 元素的索引
            valid_indices = np.where(row_np != -1)[0]

            if len(valid_indices) == 0:
                # 如果全是 -1，返回全是 pad_value 的行
                result_row = [pad_value] * k
            elif len(valid_indices) < k:
                # 如果非 -1 元素少于 k，通过重复填充来达到 k 的长度
                repeats_needed = k // len(valid_indices) + 1
                repeated_elements = np.tile(row_np[valid_indices], repeats_needed)
                result_row = repeated_elements[:k].tolist()
            else:
                # 如果非 -1 元素大于等于 k，随机选择 k 个
                result_row = np.random.choice(row_np[valid_indices], k, replace=False).tolist()

            # 添加到结果列表中
            result_list.append(result_row)

        return result_list

    # Function to train and evaluate the model
    def run(self,count):
        self.prepareModel()
        log('Model Prepared')

        aucMax = 0
        bestEpoch = 0

        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        for ep in range(stloc, args.epoch):
            temperature = max(0.05, args.init_temperature * pow(args.temperature_decay, ep))
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch(temperature , ep ,count)
            train_loss = reses
            # print('loss: ', reses['Loss'], 'preLoss: ', reses['preLoss'], 'bpr_loss: ', reses['bpr_loss'], 'regLoss: ',
            #       reses['regLoss'])
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                if  (reses['Acc'] > aucMax):
                    aucMax= reses['Acc']
                    bestEpoch = ep
                test_r = reses
                log(self.makePrint('Test', ep, reses, tstFlag))

            # wandb.log(logs)
            if(ep == 100):
                print('Best epoch : ', bestEpoch, ' , AUC : ', aucMax)
            if (ep == 150):
                print('Best epoch : ', bestEpoch, ' , AUC : ', aucMax)
            if (ep == 200):
                print('Best epoch : ', bestEpoch, ' , AUC : ', aucMax)
            if (ep == 250):
                print('Best epoch : ', bestEpoch, ' , AUC : ', aucMax)

        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        print('Best epoch : ', bestEpoch, ' , AUC : ', aucMax)
        self.save_model()
        return aucMax

    # Function to prepare the model and optimizer
    def prepareModel(self):
        self.model = Model().cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
    # Function to train a single epoch
    def trainEpoch(self ,temperature ,ep , count_ep ):
        self.model.train()
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()

        if(ep== 0 and (count_ep == 0)):
            print("make_neg_begin")
            trnLoader.dataset.negMul_gene()
            print("make_neg_gene_over")
            trnLoader.dataset.negMul_drug()
            print("make_neg_drug_over")
            trnLoader.dataset.padded_matrix()
            print("make_neg_over")
            print(args.num_neg)
            print(args.num_neg_mul)

        epLoss, epPreLoss = 0, 0
        bprLoss, bpr_loss,reg_loss ,regLoss,im_loss= 0, 0 , 0,0,0
        loss_mult        ,loss_mult_loss           = 0,0
        len__ = trnLoader.dataset.__len__()

        steps = len__ // args.batch
        for i, tem in enumerate(trnLoader):
            data = deepcopy(self.handler.torchBiAdj).cuda()

            drugs, genes, labels , negs ,negs_mul_genes_labels ,mul_genes,  negs_mul_durgs_labels ,mul_drugs= tem
            #drugs, genes, labels, negs, negs_mul_genes_labels, mul_genes = tem
            if(args.num_neg_mul != 0 ):
                #对于针对 gene
                bool_matrix = (negs_mul_genes_labels[range(len(drugs))] == labels[range(len(drugs))][:, np.newaxis])
                inverted_bool_matrix = ~bool_matrix
                # pos_mul_genes = [row[mask] for row, mask in zip(mul_genes[range(len(drugs))], bool_matrix)]
                # pos_mul_genes = self.random_sample_nonzero(pos_mul_genes, args.num_neg_mul, -1)
                # pos_mul_genes = t.tensor(pos_mul_genes).long().cuda()

                negs_mul_genes = [row[mask] for row, mask in zip(mul_genes[range(len(drugs))], inverted_bool_matrix)]
                negs_mul_genes = self.random_sample_nonzero(negs_mul_genes, args.num_neg_mul, -1)
                negs_mul_genes = t.tensor(negs_mul_genes).long().cuda()
                #
                if(args.data == 'DGIdb' or args.is_use != 0 ):
                #对于针对 drug
                    bool_matrix = (negs_mul_durgs_labels[range(len(genes))] == labels[range(len(genes))][:, np.newaxis])
                    inverted_bool_matrix = ~bool_matrix
                    # pos_mul_drugs = [row[mask] for row, mask in zip(mul_drugs[range(len(genes))], bool_matrix)]
                    # pos_mul_drugs = self.random_sample_nonzero(pos_mul_drugs, args.num_neg_mul, -1)
                    # pos_mul_drugs = t.tensor(pos_mul_drugs).long().cuda()

                    negs_mul_drugs = [row[mask] for row, mask in zip(mul_drugs[range(len(genes))], inverted_bool_matrix)]
                    negs_mul_drugs = self.random_sample_nonzero(negs_mul_drugs, args.num_neg_mul, -1)
                    negs_mul_drugs = t.tensor(negs_mul_drugs).long().cuda()


            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()
            negs = negs.long().cuda()

            #局部负采样
            if (args.num_neg_mul != 0 ):
                #BPR,为了让同一标签 的 gene 在向量空间上 更近
                usrEmbeds, itmEmbeds = self.model.forward_gcn(data)

                fixed_vector = t.tensor(np.zeros(itmEmbeds.shape[1]), device='cuda:0')#零向量
                fixed_vector_expanded = fixed_vector.unsqueeze(0)
                itmEmbeds= t.concat((itmEmbeds , fixed_vector_expanded),dim=0)
                # usrEmbeds = t.concat((usrEmbeds, fixed_vector_expanded), dim=0)

                # 获取正样本和负样本的嵌入
                ancEmbeds = usrEmbeds[drugs]  # 用户嵌入
                posEmbeds = itmEmbeds[genes]  # 正样本嵌入


                # posEmbeds_gene_elocal = itmEmbeds[pos_mul_genes]  # 所有正样本嵌入-gene
                negEmbeds_gene_elocal = itmEmbeds[negs_mul_genes]  # 所有负样本嵌入
                if (args.data == 'DGIdb' or args.is_use != 0):
                    # posEmbeds_drug_elocal = usrEmbeds[pos_mul_drugs]  # 所有正样本嵌入-gene
                    negEmbeds_drug_elocal = usrEmbeds[negs_mul_drugs]  # 所有负样本嵌入

                # 批量计算分数差异
                # ancEmbeds: [batch_size, embed_dim]
                # posEmbeds: [batch_size, embed_dim]
                # negEmbeds: [batch_size, num_neg, embed_dim]
                # 通过 unsqueeze(1) 将正样本嵌入扩展维度，使其与负样本对齐
                posScores = innerProduct(ancEmbeds.unsqueeze(1), posEmbeds.unsqueeze(1))  # [batch_size, 1]
                #gene
                # posScores_1 = innerProduct(posEmbeds.unsqueeze(1), posEmbeds_gene_elocal)  # [batch_size, num_neg
                negScores_1 = innerProduct(posEmbeds.unsqueeze(1), negEmbeds_gene_elocal)  # [batch_size, num_neg
                if (args.data == 'DGIdb' or args.is_use != 0 ):
                    # drug
                    # posScores_2 = innerProduct(ancEmbeds.unsqueeze(1), posEmbeds_drug_elocal)  # [batch_size, num_neg
                    negScores_2 = innerProduct(ancEmbeds.unsqueeze(1), negEmbeds_drug_elocal)  # [batch_size, num_neg

                #计算分数差异
                # scoreDiff = posScores + posScores_1 - negScores_1  + posScores_2 - negScores_2 - negScores_2# [batch_size, num_neg]
                if (args.data == 'DGIdb' or args.is_use != 0 ):
                    scoreDiff = posScores  - negScores_1 - negScores_2# [batch_size, num_neg]
                else:
                    scoreDiff = posScores - negScores_1  # [batch_size, num_neg]
                # 计算 BPR 损失
                loss_mult += - (scoreDiff).sigmoid().log().sum() / args.batch
                # 正则化损失
                regLoss += calcRegLoss(self.model) * args.reg
                loss = loss_mult + regLoss
                loss_mult_loss += float(loss_mult)
                reg_loss += float(regLoss)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                regLoss = 0
            # 全局负采样
            if(args.num_neg != 0):
                # BPR_0.706 + 0.03
                usrEmbeds, itmEmbeds = self.model.forward_gcn(data)
                # 获取正样本和负样本的嵌入
                ancEmbeds = usrEmbeds[drugs]  # 用户嵌入
                posEmbeds = itmEmbeds[genes]  # 正样本嵌入
                negEmbeds = itmEmbeds[negs ]  # 所有负样本嵌入
                # 批量计算分数差异
                # ancEmbeds: [batch_size, embed_dim]
                # posEmbeds: [batch_size, embed_dim]
                # negEmbeds: [batch_size, num_neg, embed_dim]
                # 通过 unsqueeze(1) 将正样本嵌入扩展维度，使其与负样本对齐
                posScores = innerProduct(ancEmbeds.unsqueeze(1), posEmbeds.unsqueeze(1))  # [batch_size, 1]
                negScores = innerProduct(ancEmbeds.unsqueeze(1), negEmbeds)  # [batch_size, num_neg
                # 计算分数差异
                scoreDiff = posScores - negScores  # [batch_size, num_neg]
                # 计算 BPR 损失
                bprLoss += - (scoreDiff).sigmoid().log().sum() / args.batch
                # 正则化损失
                regLoss += calcRegLoss(self.model) * args.reg
                loss = bprLoss + regLoss
                bpr_loss += float(bprLoss)
                reg_loss += float(regLoss)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            #交叉熵损失
            ceLoss, sslLoss = self.model.calcLosses(drugs, genes, labels, self.handler.torchBiAdj,args.keepRate)
            # sslLoss = sslLoss * args.ssl_reg

            regLoss = calcRegLoss(self.model) * args.reg
            loss = ceLoss + regLoss
            epLoss += loss.item()
            epPreLoss += ceLoss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            bprLoss = 0
            regLoss = 0
            loss_mult = 0


            # if (i == 0 and (ep == 100 or  ep == 150 or ep == 200 or ep == 250 or ep == 299)):
            #     drugEmbeds, geneEmbeds = self.model.forward_gcn(data)
            #     mad_value = compute_mad(drugEmbeds)
            #     print(f"dMAD 值: {mad_value}")
            #     mad_value = compute_mad(geneEmbeds)
            #     print(f"gMAD 值: {mad_value}")



        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        ret['bpr_loss'] = bpr_loss / steps
        ret['loss_mult_loss'] = loss_mult_loss / steps
        ret['regLoss'] = regLoss / steps
        return ret

    # Function to test a single epoch
    def testEpoch(self):
        self.model.eval()
        tstLoader = self.handler.tstLoader
        i = 0
        for tem in tstLoader:
            i += 1
            drugs, genes, labels = tem
            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()
            pre = self.model.predict(self.handler.torchBiAdj, drugs, genes)

            pre = F.log_softmax(pre, dim=1)
            pre = pre.data.max(1, keepdim=True)[1].detach().cpu()
            labels = labels.detach().cpu()
            epAcc = accuracy_score(labels, pre)
        ret = dict()
        ret['Acc'] = epAcc
        return ret

    # Function to load a pre-trained model
    def loadModel(self):
        self.model.load_state_dict(t.load('../Models/' + args.load_model + '.pkl'))

        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        log('Model Loaded')

    # Function to save the trained model
    # def save_model(self, model_path):
    #     model_parent_path = os.path.join(wandb.run.dir, 'ckl')
    #     if not os.path.exists(model_parent_path):
    #         os.mkdir(model_parent_path)
    #     t.save(self.model.state_dict(), '{}/{}_model.pkl'.format(model_parent_path, model_path))

    def save_model(self):
        if args.epoch == 0:
            return
        # with open('../../History/' + args.save_path, 'wb') as fs:
        #     pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        t.save(content, '../Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)
    def generator_generate(self, generator):
        edge_index = []
        edge_index.append([])
        edge_index.append([])
        adj = deepcopy(self.handler.torchBiAdj)
        idxs = adj._indices()

        with t.no_grad():
            view = generator.generate(self.handler.torchBiAdj, idxs, adj)

        return view

def list_txt(path, list=None):
        '''

        :param path: 储存list的位置
        :param list: list数据
        :return: None/relist 当仅有path参数输入时为读取模式将txt读取为list
                 当path参数和list都有输入时为保存模式将list保存为txt
        '''
        if list != None:
            file = open(path, 'w')
            file.write(str(list))
            file.close()
            return None
        else:
            file = open(path, 'r')
            rdlist = eval(file.read())
            file.close()
            return rdlist



# Main execution block
if __name__ == '__main__':
    if args.is_debug is True:
        print("DEBUGGING MODE - Start without wandb")
    #     wandb.init(mode="disabled")
    # else:
    #     wandb.init(project='HC', config=args)
    #     wandb.run.log_code(".")

    use_cuda = args.gpu >= 0 and t.cuda.is_available()
    device = 'cuda:{}'.format(args.gpu) if use_cuda else 'cpu'
    if use_cuda:
        t.cuda.set_device(device)
    args.device = device

    logger.saveDefault = True

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    config = dict()
    results = list()
    for i in tqdm(range(args.iteration)):
        print('{}-th iteration'.format(i + 1))
        seed = args.seed
        config['seed'] = seed
        config['iteration'] = i + 1
        set_seed(seed)
        if args.data == 'LINCS':
            result = coach.external_test_run()
        else:
            result = coach.run(i)
        results.append(result)
    # list_txt(path='savelist.txt', list=results)
    print('over!!!')
    # wandb.finish()
