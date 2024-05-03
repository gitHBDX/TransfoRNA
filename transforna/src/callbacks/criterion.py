import copy
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunction(nn.Module):
    def __init__(self,main_config):
        super(LossFunction, self).__init__()
        self.train_config = main_config["train_config"]
        self.model_config = main_config["model_config"]
        self.batch_per_epoch = self.train_config.batch_per_epoch
        self.warm_up_annealing = (
            self.train_config.warmup_epoch * self.batch_per_epoch
        )
        self.num_embed_hidden = self.model_config.num_embed_hidden
        self.batch_idx = 0
        self.loss_anealing_term = 0
                

        class_weights = self.model_config.class_weights
        #TODO: use device as in main_config
        class_weights = torch.FloatTensor([float(x) for x in class_weights])

        if self.model_config.num_classes > 2:
            self.clf_loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=self.train_config.label_smoothing_clf,reduction='none')
        else:
            self.clf_loss_fn = self.focal_loss
            

    # @staticmethod
    def cosine_similarity_matrix(
        self, gene_embedd: torch.Tensor, second_input_embedd: torch.Tensor, annealing=True
    ) -> torch.Tensor:
        # if annealing is true, then this function is being called from Net.predict and
        # doesnt pass the instantiated object LossFunction, therefore no access to self.
        # in Predict we also just need the max of predictions.
        # for some reason, skorch only passes the LossFunction initialized object, only
        # from get_loss fn.
        
        assert gene_embedd.size(0) == second_input_embedd.size(0)

        cosine_similarity = torch.matmul(gene_embedd, second_input_embedd.T) 

        if annealing:
            if self.batch_idx < self.warm_up_annealing:
                self.loss_anealing_term = 1 + (
                    self.batch_idx / self.warm_up_annealing
                ) * torch.sqrt(torch.tensor(self.num_embed_hidden))

            cosine_similarity *= self.loss_anealing_term

        return cosine_similarity
    def get_similar_labels(self,y:torch.Tensor):
        '''
        This function recieves y, the labels tensor
        It creates a list of lists containing at every index a list(min_len = 2) of the indices of the labels that are similar
        '''
        # create a test array
        labels_y = y[:,0].cpu().detach().numpy()

        # creates an array of indices, sorted by unique element
        idx_sort = np.argsort(labels_y)

        # sorts records array so all unique elements are together 
        sorted_records_array = labels_y[idx_sort]

        # returns the unique values, the index of the first occurrence of a value, and the count for each element
        vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)

        # splits the indices into separate arrays
        res = np.split(idx_sort, idx_start[1:])
        #filter them with respect to their size, keeping only items occurring more than once
        vals = vals[count > 1]
        res = filter(lambda x: x.size > 1, res)

        indices_similar_labels = []
        similar_labels = []
        for r in res:
            indices_similar_labels.append(list(r))
            similar_labels.append(list(labels_y[r]))

        return indices_similar_labels,similar_labels

    def get_triplet_samples(self,indices_similar_labels,similar_labels):
        '''
        This function creates three lists, positives, anchors and negatives
        Each index in the three lists correpond to a single triplet 
        '''
        positives,anchors,negatives = [],[],[]
        for idx_similar_labels in indices_similar_labels:
            random_indices = random.sample(range(len(idx_similar_labels)), 2)
            positives.append(idx_similar_labels[random_indices[0]])
            anchors.append(idx_similar_labels[random_indices[1]])

        negatives = copy.deepcopy(positives)
        random.shuffle(negatives)
        while (np.array(positives) == np.array(negatives)).any():
            random.shuffle(negatives)

        return positives,anchors,negatives
    def get_triplet_loss(self,y,gene_embedd,second_input_embedd):
        '''
        This function computes triplet loss by creating triplet samples of positives, negatives and anchors
        The objective is to decrease the distance of the embeddings between the anchors and the positives 
        while increasing the distance between the anchor and the negatives.
        This is done seperately for both the embeddings, gene embedds 0 and ss embedds 1
        '''
        #get similar labels 
        indices_similar_labels,similar_labels = self.get_similar_labels(y)
        #insuring that there's at least two sets of labels in a given list (indices_similar_labels)
        if len(indices_similar_labels)>1:
            #get triplet samples
            positives,anchors,negatives = self.get_triplet_samples(indices_similar_labels,similar_labels)
            #get triplet loss for gene  embedds
            gene_embedd_triplet_loss = self.triplet_loss(gene_embedd[positives,:],
                                                    gene_embedd[anchors,:],
                                                    gene_embedd[negatives,:])
            #get triplet loss for ss embedds
            second_input_embedd_triplet_loss = self.triplet_loss(second_input_embedd[positives,:],
                                                    second_input_embedd[anchors,:],
                                                    second_input_embedd[negatives,:])
            return gene_embedd_triplet_loss + second_input_embedd_triplet_loss
        else: 
            return 0

    def focal_loss(self,predicted_labels,y):
        y = y.unsqueeze(dim=1)
        y_new = torch.zeros(y.shape[0], 2).type(torch.cuda.FloatTensor)
        y_new[range(y.shape[0]), y[:,0]]=1
        BCE_loss = F.binary_cross_entropy_with_logits(predicted_labels.float(), y_new.float(), reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss = (1-pt)**2 * BCE_loss
        loss = 10*F_loss.mean()
        return loss

    def contrastive_loss(self,cosine_similarity,batch_size):
        j = -torch.sum(torch.diagonal(cosine_similarity))

        cosine_similarity.diagonal().copy_(torch.zeros(cosine_similarity.size(0)))

        j = (1 - self.train_config.label_smoothing_sim) * j + (
            self.train_config.label_smoothing_sim / (cosine_similarity.size(0) * (cosine_similarity.size(0) - 1))
        ) * torch.sum(cosine_similarity)

        j += torch.sum(torch.logsumexp(cosine_similarity, dim=0))

        if j < 0:
            j = j-j
        return j/batch_size

    def forward(self, embedds: List[torch.Tensor], y=None) -> torch.Tensor:
        self.batch_idx += 1
        gene_embedd, second_input_embedd, predicted_labels,curr_epoch = embedds
        

        loss = self.clf_loss_fn(predicted_labels,y.squeeze())


            
        return loss
