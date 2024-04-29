#data_time for hydra output folder
get_data_time(){
    date=$(ls outputs/ | head -n 1)
    time=$(ls outputs/*/ | head -n 1)
    date=$date
    time=$time
}

train_model(){
    python src/main.py model_name=$1 trained_on=$2
    get_data_time
    #rename the folder to model_name
    mv outputs/$date/$time outputs/$date/$3
    rm -rf models/tcga/TransfoRNA_${2^^}/sub_class/$3
    mv outputs/$date/*  models/tcga/TransfoRNA_${2^^}/sub_class/
    rm -rf outputs/

}
rm -rf outputs

#train the models
train_model seq id Seq
train_model seq full Seq

train_model seq-seq id Seq-Seq
train_model seq-seq full Seq-Seq

train_model seq-rev id Seq-Rev
train_model seq-rev full Seq-Rev

train_model seq-struct id Seq-Struct
train_model seq-struct full Seq-Struct

train_model baseline id Baseline
train_model baseline full Baseline