import logging
import os
import pickle

import skorch
import torch
from skorch.dataset import Dataset, ValidSplit
from skorch.setter import optimizer_setter
from skorch.utils import is_dataset, to_device

logger = logging.getLogger(__name__)
#from ..tbWriter import writer


class Net(skorch.NeuralNet):
    def __init__(
        self,
        clip=0.25,
        top_k=1,
        correct=0,
        save_embedding=False,
        gene_embedds=[],
        second_input_embedd=[],
        confidence_threshold = 0.95,
        *args,
        **kwargs
    ):
        self.clip = clip
        self.curr_epoch = 0
        super(Net, self).__init__(*args, **kwargs)
        self.correct = correct
        self.save_embedding = save_embedding
        self.gene_embedds = gene_embedds
        self.second_input_embedds = second_input_embedd
        self.main_config = kwargs["module__main_config"]
        self.train_config = self.main_config["train_config"]
        self.top_k =  self.train_config.top_k
        self.num_classes = self.main_config["model_config"].num_classes
        self.labels_mapping_path = self.train_config.labels_mapping_path
        if self.labels_mapping_path:
            with open(self.labels_mapping_path, 'rb') as handle:
                self.labels_mapping_dict = pickle.load(handle)
        self.confidence_threshold = confidence_threshold
        self.max_epochs = kwargs["max_epochs"]
        self.task = '' #is set in utils.instantiate_predictor
        self.log_tb = False

        


    def set_save_epoch(self):
        ''' 
        scale best train epoch by valid size
        '''
        if self.task !='tcga':
            if self.train_split:
                self.save_epoch = self.main_config["train_config"].train_epoch
            else:
                self.save_epoch = round(self.main_config["train_config"].train_epoch*\
                    (1+self.main_config["valid_size"]))

    def save_benchmark_model(self):
        '''
        saves benchmark epochs when train_split is none 
        '''
        try:
            os.mkdir("ckpt")
        except:
            pass
        cwd = os.getcwd()+"/ckpt/"
        self.save_params(f_params= f'{cwd}/model_params_{self.main_config["task"]}.pt')


    def fit(self, X, y=None, valid_ds=None,**fit_params):
        #all sequence lengths should be saved to compute the median based 
        self.all_lengths = [[] for i in range(self.num_classes)]
        self.median_lengths = []

        if not self.warm_start or not self.initialized_:
            self.initialize()

        if valid_ds:
            self.validation_dataset = valid_ds
        else:
            self.validation_dataset = None

        self.partial_fit(X, y, **fit_params)
        return self

    def fit_loop(self, X, y=None, epochs=None, **fit_params):
        #if id then train longer otherwise stop at 0.99
        rounding_digits = 3
        if self.main_config['trained_on'] == 'full':
            rounding_digits = 2
        self.check_data(X, y)
        epochs = epochs if epochs is not None else self.max_epochs

        dataset_train, dataset_valid = self.get_split_datasets(X, y, **fit_params)

        if self.validation_dataset is not None:
            dataset_valid = self.validation_dataset.keywords["valid_ds"]

        on_epoch_kwargs = {
            "dataset_train": dataset_train,
            "dataset_valid": dataset_valid,
        }

        iterator_train = self.get_iterator(dataset_train, training=True)
        iterator_valid = None
        if dataset_valid is not None:
            iterator_valid = self.get_iterator(dataset_valid, training=False)

        self.set_save_epoch()

        for epoch_no in range(epochs):
            #save model if training only on test set
            self.curr_epoch = epoch_no
            #save epoch is scaled by best train epoch
            #save benchmark only when training on boith train and val sets
            if self.task != 'tcga' and epoch_no == self.save_epoch and self.train_split == None:
                self.save_benchmark_model()
                
            self.notify("on_epoch_begin", **on_epoch_kwargs)

            self.run_single_epoch(
                iterator_train,
                training=True,
                prefix="train",
                step_fn=self.train_step,
                **fit_params
            )

            if dataset_valid is not None:
                self.run_single_epoch(
                    iterator_valid,
                    training=False,
                    prefix="valid",
                    step_fn=self.validation_step,
                    **fit_params
                )
        
            
            self.notify("on_epoch_end", **on_epoch_kwargs)
            #manual early stopping for tcga
            if self.task == 'tcga':
                train_acc = round(self.history[:,'train_acc'][-1],rounding_digits)
                if train_acc == 1:
                    break


            
        return self

    def train_step(self, X, y=None):
        y = X[1]
        X = X[0]
        sample_weights = X[:,-1]
        if self.device == 'cuda':
            sample_weights = sample_weights.to(self.train_config.device)
        self.module_.train()
        self.module_.zero_grad()
        gene_embedd, second_input_embedd, activations,_,_ = self.module_(X[:,:-1],train=True)
        #curr_epoch is passed to loss as it is used to switch loss criteria from unsup. -> sup
        loss = self.get_loss([gene_embedd,second_input_embedd,activations,self.curr_epoch], y)
        
        ###sup loss should be X with samples weight and aggregated

        loss = loss*sample_weights
        loss = loss.mean()

        loss.backward()

        # TODO: clip only some parameters
        torch.nn.utils.clip_grad_norm_(self.module_.parameters(), self.clip)
        self.optimizer_.step()

        return {"X":X,"y":y,"loss": loss, "y_pred": [gene_embedd,second_input_embedd,activations]}

    def validation_step(self, X, y=None):
        y = X[1]
        X = X[0]
        sample_weights = X[:,-1]
        if self.device == 'cuda':
            sample_weights = sample_weights.to(self.train_config.device)
        self.module_.eval()
        with torch.no_grad():
            gene_embedd, second_input_embedd, activations,_,_ = self.module_(X[:,:-1])
            loss = self.get_loss([gene_embedd,second_input_embedd,activations,self.curr_epoch], y)

        ###sup loss should be X with samples weight and aggregated

        loss = loss*sample_weights
        loss = loss.mean()

        return {"X":X,"y":y,"loss": loss, "y_pred": [gene_embedd,second_input_embedd,activations]}

    def get_attention_scores(self, X, y=None):
        '''
        returns attention scores for a given input
        '''
        self.module_.eval()
        with torch.no_grad():
            _, _, _,attn_scores_first,attn_scores_second = self.module_(X[:,:-1])

        attn_scores_first = attn_scores_first.detach().cpu().numpy()
        if attn_scores_second is not None:
            attn_scores_second = attn_scores_second.detach().cpu().numpy()
        return attn_scores_first,attn_scores_second

    def predict(self, X):
        self.module_.train(False)
        embedds = self.module_(X[:,:-1])
        sample_weights = X[:,-1]
        if self.device == 'cuda':
            sample_weights = sample_weights.to(self.train_config.device)

        gene_embedd, second_input_embedd, activations,_,_ = embedds
        if self.save_embedding:
            self.gene_embedds.append(gene_embedd.detach().cpu())
            #in case only a single transformer is deployed, then second_input_embedd are None. thus have no detach()
            if second_input_embedd is not None:
                self.second_input_embedds.append(second_input_embedd.detach().cpu())
        
        predictions = torch.cat([activations,sample_weights[:,None]],dim=1)
        return predictions


    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        # log gradients and weights
        for _, m in self.module_.named_modules():
            for pn, p in m.named_parameters():
                if pn.endswith("weight") and pn.find("norm") < 0:
                    if p.grad != None:
                        if self.log_tb:
                            from ..callbacks.tbWriter import writer
                            writer.add_histogram("weights/" + pn, p, len(net.history))
                            writer.add_histogram(
                                "gradients/" + pn, p.grad.data, len(net.history)
                            )

        return

    def configure_opt(self, l2_weight_decay):
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p
            for n, p in self.module_.named_parameters()
            if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p
            for n, p in self.module_.named_parameters()
            if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": l2_weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        return optim_groups

    def initialize_optimizer(self, triggered_directly=True):
        """Initialize the model optimizer. If ``self.optimizer__lr``
        is not set, use ``self.lr`` instead.

        Parameters
        ----------
        triggered_directly : bool (default=True)
          Only relevant when optimizer is re-initialized.
          Initialization of the optimizer can be triggered directly
          (e.g. when lr was changed) or indirectly (e.g. when the
          module was re-initialized). If and only if the former
          happens, the user should receive a message informing them
          about the parameters that caused the re-initialization.

        """
        # get learning rate from train config
        optimizer_params = self.main_config["train_config"]
        kwargs = {}
        kwargs["lr"] = optimizer_params.learning_rate
        # get l2 weight decay to init opt params
        args = self.configure_opt(optimizer_params.l2_weight_decay)

        if self.initialized_ and self.verbose:
            msg = self._format_reinit_msg(
                "optimizer", kwargs, triggered_directly=triggered_directly
            )
            print(msg)

        self.optimizer_ = self.optimizer(args, lr=kwargs["lr"])

        self._register_virtual_param(
            ["optimizer__param_groups__*__*", "optimizer__*", "lr"],
            optimizer_setter,
        )

    def initialize_criterion(self):
        """Initializes the criterion."""
        # critereon takes train_config and model_config as an input.
        # we get both from the module parameters
        self.criterion_ = self.criterion(
            self.main_config
        )
        if isinstance(self.criterion_, torch.nn.Module):
            self.criterion_ = to_device(self.criterion_, self.device)
        return self

    def initialize_callbacks(self):
        """Initializes all callbacks and save the result in the
        ``callbacks_`` attribute.

        Both ``default_callbacks`` and ``callbacks`` are used (in that
        order). Callbacks may either be initialized or not, and if
        they don't have a name, the name is inferred from the class
        name. The ``initialize`` method is called on all callbacks.

        The final result will be a list of tuples, where each tuple
        consists of a name and an initialized callback. If names are
        not unique, a ValueError is raised.

        """
        if self.callbacks == "disable":
            self.callbacks_ = []
            return self

        callbacks_ = []

        class Dummy:
            # We cannot use None as dummy value since None is a
            # legitimate value to be set.
            pass

        for name, cb in self._uniquely_named_callbacks():
            # check if callback itself is changed
            param_callback = getattr(self, "callbacks__" + name, Dummy)
            if param_callback is not Dummy:  # callback itself was set
                cb = param_callback

            # below: check for callback params
            # don't set a parameter for non-existing callback

            # if the callback is lrcallback then initializa it with the train config,
            # which is an input to the module
            if name == "lrcallback":
                params["config"] = self.main_config["train_config"]
            else:
                params = self.get_params_for("callbacks__{}".format(name))
            if (cb is None) and params:
                raise ValueError(
                    "Trying to set a parameter for callback {} "
                    "which does not exist.".format(name)
                )
            if cb is None:
                continue

            if isinstance(cb, type):  # uninitialized:
                cb = cb(**params)
            else:
                cb.set_params(**params)
            cb.initialize()
            callbacks_.append((name, cb))

        self.callbacks_ = callbacks_

        return self
