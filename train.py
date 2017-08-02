import random
import  torch
from optparse import OptionParser
import torch.nn
import torch.autograd
import torch.nn.functional
from read import Reader
from instance import Feature
from instance import Example
from hyperparams import HyperParams
from model import RNNLabeler
from eval import Eval


class Labeler:
    def __init__(self):
        self.word_state = {}
        self.label_state = {}
        self.hyperParams = HyperParams()

    def createAlphabet(self, trainInsts, devInsts, testInsts):
        print("create alpha.................")
        for inst in trainInsts:
            for w in inst.words:
                if w not in self.word_state:
                    self.word_state[w] = 1
                else:
                    self.word_state[w] += 1

            for l in inst.labels:
                if l not in self.label_state:
                    self.label_state[l] = 1
                else:
                    self.label_state[l] += 1

        print("word state:", len(self.word_state))
        self.addTestAlpha(devInsts)
        print("word state:", len(self.word_state))
        self.addTestAlpha(testInsts)
        print("word state:", len(self.word_state))

        self.word_state[self.hyperParams.unk] = self.hyperParams.wordCutOff + 1
        self.word_state[self.hyperParams.padding] = self.hyperParams.wordCutOff + 1

        self.hyperParams.wordAlpha.initial(self.word_state, self.hyperParams.wordCutOff)
        self.hyperParams.wordAlpha.set_fixed_flag(True)

        self.hyperParams.wordNum = self.hyperParams.wordAlpha.m_size

        self.hyperParams.unkWordID = self.hyperParams.wordAlpha.from_string(self.hyperParams.unk)
        self.hyperParams.paddingID = self.hyperParams.wordAlpha.from_string(self.hyperParams.padding)

        self.hyperParams.labelAlpha.initial(self.label_state)
        self.hyperParams.labelAlpha.set_fixed_flag(True)
        self.hyperParams.labelSize = self.hyperParams.labelAlpha.m_size

        print("Label num: ", self.hyperParams.labelSize)
        print("Word num: ", self.hyperParams.wordNum)
        print("Padding ID: ", self.hyperParams.paddingID)
        print("UNK ID: ", self.hyperParams.unkWordID)

    def addTestAlpha(self, insts):
        print("Add test alpha.............")
        if self.hyperParams.wordFineTune == False:
            for inst in insts:
                for w in inst.words:
                    if (w not in self.word_state):
                        self.word_state[w] = 1
                    else:
                        self.word_state[w] += 1

    def extractFeature(self, inst):
        feat = Feature()
        feat.wordIndexs = torch.autograd.Variable(torch.LongTensor(1, len(inst.words)))
        for idx in range(len(inst.words)):
            w = inst.words[idx]
            wordId = self.hyperParams.wordAlpha.from_string(w)
            if wordId == -1:
                wordId = self.unkID
            feat.wordIndexs.data[0][idx] = wordId
        return feat

    def instance2Example(self, insts):
        exams = []
        for inst in insts:
            example = Example()
            example.feat = self.extractFeature(inst)
            for l in inst.labels:
                labelId = self.hyperParams.labelAlpha.from_string(l)
                example.labelIndexs.append(labelId)

            example.labelIndexs = torch.autograd.Variable(torch.LongTensor(example.labelIndexs))
            exams.append(example)
        return exams

    def getBatchFeatLabel(self, exams):
        maxSentSize = 0
        for e in exams:
            if maxSentSize < len(e.labelIndexs):
                maxSentSize = len(e.labelIndexs)
        if maxSentSize > 40:
            maxSentSize = 40
        batch_feats = torch.autograd.Variable(torch.LongTensor(self.hyperParams.batch, maxSentSize))
        batch_labels = torch.autograd.Variable(torch.LongTensor(self.hyperParams.batch * maxSentSize))

        for idx in range(len(batch_feats.data)):
            e = exams[idx]
            for idy in range(maxSentSize):
                if idy < len(e.labelIndexs):
                    batch_feats.data[idx][idy] = e.feat.wordIndexs.data[0][idy]
                else:
                    batch_feats.data[idx][idy] = self.hyperParams.paddingID

                if idy < len(e.labelIndexs):
                    batch_labels.data[idx * maxSentSize + idy] = e.labelIndexs[idy].data[0]
                else:
                    batch_labels.data[idx * maxSentSize + idy] = 0
        return batch_feats, batch_labels

    def train(self, train_file, dev_file, test_file):
        self.hyperParams.show()
        torch.set_num_threads(self.hyperParams.thread)
        reader = Reader(self.hyperParams.maxInstance)

        trainInsts = reader.readInstances(train_file)
        devInsts = reader.readInstances(dev_file)
        testInsts = reader.readInstances(test_file)

        print("Training Instance: ", len(trainInsts))
        print("Dev Instance: ", len(devInsts))
        print("Test Instance: ", len(testInsts))

        self.createAlphabet(trainInsts, devInsts, testInsts)

        trainExamples = self.instance2Example(trainInsts)
        devExamples = self.instance2Example(devInsts)
        testExamples = self.instance2Example(testInsts)

        self.model = RNNLabeler(self.hyperParams)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.hyperParams.learningRate)

        indexes = []
        for idx in range(len(trainExamples)):
            indexes.append(idx)

        batchBlock = len(trainExamples) // self.hyperParams.batch
        for iter in range(self.hyperParams.maxIter):
            print('###Iteration' + str(iter) + "###")
            random.shuffle(indexes)
            for updateIter in range(batchBlock):
                #self.model.zero_grad()
                optimizer.zero_grad()
                exams = []
                start_pos = updateIter * self.hyperParams.batch
                end_pos = (updateIter + 1) * self.hyperParams.batch
                for idx in range(start_pos, end_pos):
                    exams.append(trainExamples[indexes[idx]])
                feats, labels = self.getBatchFeatLabel(exams)
                tag_scores = self.model(feats, self.hyperParams.batch)
                #print(tag_scores.size())
                loss = self.model.crf.neg_log_likelihood(tag_scores, labels, self.hyperParams.batch)
                loss.backward()
                optimizer.step()
                if (updateIter + 1) % self.hyperParams.verboseIter == 0:
                    print('current: ', idx + 1, ", cost:", loss.data[0])

            eval_dev = Eval()
            for idx in range(len(devExamples)):
                predictLabels = self.predict(devExamples[idx])
                devInsts[idx].evalPRF(predictLabels, eval_dev)
            print('Dev: ', end="")
            eval_dev.getFscore()

            eval_test = Eval()
            for idx in range(len(testExamples)):
                predictLabels = self.predict(testExamples[idx])
                testInsts[idx].evalPRF(predictLabels, eval_test)
            print('Test: ', end="")
            eval_test.getFscore()

    def predict(self, exam):
        tag_hiddens = self.model(exam.feat.wordIndexs)
        _, best_path = self.model.crf._viterbi_decode(tag_hiddens)
        predictLabels = []
        for idx in range(len(best_path)):
            predictLabels.append(self.hyperParams.labelAlpha.from_id(best_path[idx]))
        return predictLabels

    def getMaxIndex(self, tag_score):
        max = tag_score.data[0]
        maxIndex = 0
        for idx in range(1, self.hyperParams.labelSize):
            if tag_score.data[idx] > max:
                max = tag_score.data[idx]
                maxIndex = idx
        return maxIndex


parser = OptionParser()
parser.add_option("--train", dest="trainFile",
                  help="train dataset")

parser.add_option("--dev", dest="devFile",
                  help="dev dataset")

parser.add_option("--test", dest="testFile",
                  help="test dataset")


(options, args) = parser.parse_args()
l = Labeler()
l.train(options.trainFile, options.devFile, options.testFile)

