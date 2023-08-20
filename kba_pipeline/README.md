# The HBDx knowledge-based annotation (KBA) pipeline for small RNA sequences


## Install environment

```bash
cd kba_pipeline
conda env create --file environment.yml
```

## Run annotation pipeline

<b>Prerequisites:</b>
- [ ] the sequences to be annotated need to be stored as fasta format in the `data` folder
- [ ] the reference files for mapping need to be stored in the `references` folder (the required subfolders `HBDxBase`, `hg38` and `bacterial_viral` can be downloaded from XXX)

```bash
conda activate hbdx_kba
cd src
python make_anno.py --fasta_file your_sequences_to_be_annotated.fa 
```

This script calls two major functions:
- <b>map_2_HBDxBase</b>: sequential mismatch mapping to HBDxBase and genome
- <b>annotate_from_mapping</b>: generate sequence annotation based on mapping outputs

The main annotation file `sRNA_anno_aggregated_on_seq.csv` will be generated in the folder `outputs`
