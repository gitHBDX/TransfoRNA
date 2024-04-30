#data_time for hydra output folder
get_data_time(){
    date=$(ls outputs/ | head -n 1)
    time=$(ls outputs/*/ | head -n 1)
    date=$date
    time=$time
}

train_model(){
    python -m transforna --config-dir="/nfs/home/yat_ldap/VS_Projects/TransfoRNA-Framework/conf"\
            model_name=$1 trained_on=$2 num_replicates=$4
    get_data_time
    #rename the folder to model_name
    mv outputs/$date/$time outputs/$date/$3
    rm -rf ../models/tcga/TransfoRNA_${2^^}/sub_class/$3
    mv outputs/$date/*  ../models/tcga/TransfoRNA_${2^^}/sub_class/
    rm -rf outputs/

}
#activate transforna environment
eval "$(conda shell.bash hook)"
conda activate transforna

#remove the outputs folder
rm -rf outputs

#train the models
train_model seq id Seq 5 
train_model seq full Seq 5

train_model seq-seq id Seq-Seq 5
train_model seq-seq full Seq-Seq 5

train_model seq-rev id Seq-Rev 5
train_model seq-rev full Seq-Rev 5

train_model seq-struct id Seq-Struct 5
train_model seq-struct full Seq-Struct 5

train_model baseline id Baseline 5
train_model baseline full Baseline 5