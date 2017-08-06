import torch.nn as nn
import torch.autograd

class RNNLabeler(nn.Module):
    def __init__(self, hyperParams):
        super(RNNLabeler,self).__init__()

        self.hyperParams = hyperParams
        if hyperParams.wordEmbFile == "":
            self.wordEmb = nn.Embedding(hyperParams.wordNum, hyperParams.wordEmbSize)
        else:
            self.wordEmb = self.load_pretrain(hyperParams.wordEmbFile, hyperParams.wordAlpha)
        self.dropOut = nn.Dropout(hyperParams.dropProb)
        self.wordEmb.weight.requires_grad = hyperParams.wordFineTune
        self.LSTM = nn.LSTM(hyperParams.wordEmbSize, hyperParams.rnnHiddenSize // 2, batch_first=True, bidirectional=True)
        self.linearLayer = nn.Linear(hyperParams.rnnHiddenSize, hyperParams.labelSize, bias=True)

    def init_hidden(self, batch):
       return (torch.autograd.Variable(torch.randn(2, batch, self.hyperParams.rnnHiddenSize // 2)),
                torch.autograd.Variable(torch.randn(2, batch, self.hyperParams.rnnHiddenSize // 2)))

    def load_pretrain(self, file, alpha):
        f = open(file, encoding='utf8')
        allLines = f.readlines()
        indexs = []
        info = allLines[0].strip().split(' ')
        embDim = len(info) - 1
        emb = nn.Embedding(self.hyperParams.wordNum, embDim)
        oov_emb = torch.zeros(1, embDim).type(torch.FloatTensor)
        for line in allLines:
            info = line.strip().split(' ')
            wordID = alpha.from_string(info[0])
            if wordID >= 0:
                indexs.append(wordID)
                for idx in range(embDim):
                    val = float(info[idx + 1])
                    emb.weight.data[wordID][idx] = val
                    oov_emb[0][idx] += val
        f.close()
        count = len(indexs)
        for idx in range(embDim):
            oov_emb[0][idx] /= count

        unkID = self.hyperParams.wordAlpha.from_string(self.hyperParams.unk)
        print('UNK ID: ', unkID)
        if unkID != -1:
            for idx in range(embDim):
                emb.weight.data[unkID][idx] = oov_emb[0][idx]

        print("Load Embedding file: ", file, ", size: ", embDim)
        oov = 0
        for idx in range(alpha.m_size):
            if idx not in indexs:
                oov += 1
        print("OOV Num: ", oov, "Total Num: ", alpha.m_size,
              "OOV Ratio: ", oov / alpha.m_size)
        print("OOV ", self.hyperParams.unk, "use avg value initialize")
        return emb


    def forward(self, feat, batch = 1):
        sentSize = len(feat.data[0])
        wordRepresents = self.wordEmb(feat)
        wordRepresents = self.dropOut(wordRepresents)
        LSTMHidden = self.init_hidden(batch)
        LSTMOutputs, _ = self.LSTM(wordRepresents.view(batch, sentSize, -1), LSTMHidden)
        LSTMOutputs = torch.cat(LSTMOutputs, 0)

        tagHiddens = self.linearLayer(LSTMOutputs)
        return tagHiddens








