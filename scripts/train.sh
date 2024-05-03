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
    ls outputs/$date/
    rm -rf models/tcga/TransfoRNA_${2^^}/$5/$3



    mv -f outputs/$date/$3  models/tcga/TransfoRNA_${2^^}/$5/
    rm -rf outputs/

}
#activate transforna environment
eval "$(conda shell.bash hook)"
conda activate transforna

#create the models folder if it does not exist
if [[ ! -d "models/tcga/TransfoRNA_ID/major_class" ]]; then
    mkdir -p models/tcga/TransfoRNA_ID/major_class
fi
if [[ ! -d "models/tcga/TransfoRNA_FULL/sub_class" ]]; then
    mkdir -p models/tcga/TransfoRNA_FULL/sub_class
fi
if [[ ! -d "models/tcga/TransfoRNA_ID/sub_class" ]]; then
    mkdir -p models/tcga/TransfoRNA_ID/sub_class
fi
if [[ ! -d "models/tcga/TransfoRNA_FULL/major_class" ]]; then
    mkdir -p models/tcga/TransfoRNA_FULL/major_class
fi
#remove the outputs folder
rm -rf outputs


#define models
models=("seq" "seq-seq" "seq-rev" "seq-struct" "baseline")
models_capitalized=("Seq" "Seq-Seq" "Seq-Rev" "Seq-Struct" "Baseline")


num_replicates=5


############train major_class_hico

#replace clf_target:str = 'sub_class_hico' to clf_target:str = 'major_class_hico' in ../conf/train_model_configs/tcga.py
sed -i "s/clf_target:str = 'sub_class_hico'/clf_target:str = 'major_class_hico'/g" conf/train_model_configs/tcga.py
#print the file content
cat conf/train_model_configs/tcga.py
#loop and train
for i in ${!models[@]}; do
    echo "Training model ${models_capitalized[$i]} for id on major_class"
    train_model ${models[$i]} id ${models_capitalized[$i]} $num_replicates "major_class"
    echo "Training model ${models[$i]} for full on major_class"
    train_model ${models[$i]} full ${models_capitalized[$i]} $num_replicates "major_class"
done


############train sub_class_hico

#replace clf_target:str = 'major_class_hico' to clf_target:str = 'sub_class_hico' in ../conf/train_model_configs/tcga.py
sed -i "s/clf_target:str = 'major_class_hico'/clf_target:str = 'sub_class_hico'/g" conf/train_model_configs/tcga.py

for i in ${!models[@]}; do
    echo "Training model ${models_capitalized[$i]} for id on sub_class"
    train_model ${models[$i]} id ${models_capitalized[$i]} $num_replicates "sub_class"
    echo "Training model ${models[$i]} for full on sub_class"
    train_model ${models[$i]} full ${models_capitalized[$i]} $num_replicates "sub_class"
done

