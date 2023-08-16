
#%%
#script for computing computing OOD vs Artificial sequences (Figure 5e)
import json
import pickle

import numpy as np
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transforna.utils.file import load, save
from transforna.utils.tcga_post_analysis_utils import Results_Handler

#%%
if __name__ == "__main__":
    #######################################TO CONFIGURE#############################################
    path = '/models/tcga/TransfoRNA_ID/sub_class/Seq/embedds' #edit path to contain path for the embedds folder, for example: transforna/results/seq-rev/embedds/
    splits = ['train','ood']
    #run name
    run_name = None #if None, then the name of the model inputs will be used as the name
    #this could be for instance 'Sup Seq-Exp'
    input_col = 'Embedds' #can be 'Logits' or 'Embedds'
    ################################################################################################
    results:Results_Handler = Results_Handler(path=path,splits=splits)

    if input_col == 'Embedds':
        input_col = [col for col in results.splits_df_dict['ood_df'] if "Embedds" in col[0]]
        if '/Seq/' in path:
            input_col = input_col[:len(input_col)//2]

    neg_data = results.splits_df_dict["artificial_affix_df"][input_col].values
    neg_labels = np.zeros(neg_data.shape[0])

    if results.trained_on == 'id':
        pos_data = results.splits_df_dict["ood_df"][input_col].values
    else:
        pos_data = results.splits_df_dict["train_df"][input_col].values
    pos_labels = np.ones(pos_data.shape[0])

    data = np.append(neg_data,pos_data,axis=0)
    labels = np.append(neg_labels,pos_labels,axis=0)


    b_accs = []

    for random_seed in range(10):
        print(random_seed)
        train_x,test_x,train_y,test_y = train_test_split(data,labels,train_size=0.9,random_state=random_seed)
        #oversample training set
        oversample = SMOTE()
        train_x, train_y = oversample.fit_resample(train_x, train_y)
        
        clf = make_pipeline(StandardScaler(),
                        SGDClassifier(loss='log_loss',penalty='l2',max_iter=1000, tol=1e-3,class_weight='balanced'))
        clf.fit(train_x,train_y)
        calibrator = CalibratedClassifierCV(clf, cv='prefit')
        clf = calibrator.fit(train_x,train_y)
        predictions = clf.predict(test_x)
        b_acc = balanced_accuracy_score(test_y,predictions)
        b_accs.append(b_acc)

    b_acc_score = sum(b_accs)/len(b_accs)
    b_acc_std = np.std(b_accs)
    print(f"bal acc is {b_acc_score} +- {b_acc_std}")
    save_results = True
    if save_results:
        filename=results.analysis_path+'/id1_vs_aa0_clf.sav'
        pickle.dump(clf,open(filename,'wb'))
        id_vs_aa_metrics = {"Balanced Accuracy":b_acc_score,"Balanced Accuracy Std":b_acc_std}
        filename=results.analysis_path+'/id_vs_aa_metrics.yaml'
        id_vs_aa_metrics = eval(json.dumps(id_vs_aa_metrics)) 
        save(path=filename,data=id_vs_aa_metrics)
    #model = pickle.load(open(filename,'rb'))
    plot = False
    if plot:
        train_sizes = [0.1,0.3,0.5,0.7,0.9]
        b_accs = [0.7943406536415715,0.8691755563955033,0.8856870570496772,0.8977007139042212,0.9096356494624871]
        b_acc_stds = [0.019243640556138544,0.009122244923685772,0.00804941178571188,0.00921179808934596,0.020392253950457385]
        #use plotly to plot the results
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_sizes, y=b_accs,
                            mode='lines+markers',
                            name='Balanced Accuracy'))
        fig.add_trace(go.Scatter(x=train_sizes, y=[b_accs[i]-b_acc_stds[i] for i in range(len(b_accs))],
                            mode='lines',   
                            name='Balanced Accuracy - Std'))
        fig.add_trace(go.Scatter(x=train_sizes, y=[b_accs[i]+b_acc_stds[i] for i in range(len(b_accs))],
                            mode='lines',
                            name='Balanced Accuracy + Std'))
        fig.update_layout(title='Balanced Accuracy vs Training Set Size',
                        xaxis_title='Training Set Size',
                        yaxis_title='Balanced Accuracy')
        fig.show()

    
    






# %%
