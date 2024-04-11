# The HBDx knowledge-based annotation (KBA) pipeline for small RNA sequences

Most small RNA annotation tools map the sequences sequentially to different small RNA class specific reference databases, which prioritizes the distinct small RNA classes and conceals potential assignment ambiguities. The annotation strategy used here, maps the sequences to the reference sequences of all small RNA classes at the same time starting with zero mismatch tolerance. Unmapped sequences are intended to map with iterating mismatch tolerance up to three mismatches. Additionally, all small RNA sequences are checked for potential bacterial or viral origin, for genomic overlap to human transposable element loci and whether they contain potential prefixes of the 5‘ adapter.

![kba_pipeline_scheme_v05](https://github.com/gitHBDX/TransfoRNA/assets/79092907/62bf9e36-c7c7-4ff5-b747-c2c651281b42)


a) Schematic overview of the knowledge-based annotation (KBA) strategy applied for TransfoRNA. 

b) Schematic overview of the miRNA annotation of the custom annotation (isomiR definition based on recent miRNA research [1]). 

c) Schematic overview of the tRNA annotation of the custom annotation (inspired by UNITAS sub-classification [2]). 

d) Binning strategy used in the custom annotation for the remaining RNA major classes. The number of nucleotides per bin is constant for each precursor sequence and ranges between 20 and 39 nucleotides. Assignments are based on the bin with the highest overlap to the sequence of interest.

e) Filtering steps that were applied to obtain the set of HICO annotations that were used for training of the TransfoRNA models.


## Install environment

```bash
cd kba_pipeline
conda env create --file environment.yml
```

## Run annotation pipeline

<b>Prerequisites:</b>
- [ ] the sequences to be annotated need to be stored as fasta format in the `kba_pipeline/data` folder
- [ ] the reference files for mapping need to be stored in the `kba_pipeline/references` folder (the required subfolders `HBDxBase`, `hg38` and `bacterial_viral` can be downloaded together with the TransfoRNA models from https://www.dropbox.com/sh/y7u8cofmg41qs0y/AADvj5lw91bx7fcDxghMbMtsa?dl=0)

```bash
conda activate hbdx_kba
cd src
python make_anno.py --fasta_file your_sequences_to_be_annotated.fa 
```

This script calls two major functions:
- <b>map_2_HBDxBase</b>: sequential mismatch mapping to HBDxBase and genome
- <b>annotate_from_mapping</b>: generate sequence annotation based on mapping outputs

The main annotation file `sRNA_anno_aggregated_on_seq.csv` will be generated in the folder `outputs`



## References

[1] Tomasello, Luisa, Rosario Distefano, Giovanni Nigita, and Carlo M. Croce. 2021. “The MicroRNA Family Gets Wider: The IsomiRs Classification and Role.” Frontiers in Cell and Developmental Biology 9 (June): 1–15. https://doi.org/10.3389/fcell.2021.668648.

[2] Gebert, Daniel, Charlotte Hewel, and David Rosenkranz. 2017. “Unitas: The Universal Tool for Annotation of Small RNAs.” BMC Genomics 18 (1): 1–14. https://doi.org/10.1186/s12864-017-4031-9.


