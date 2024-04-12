######################################################################################################
# map sequences to HBDxBase
######################################################################################################
#%%
import os
import logging

from utils import fasta2df,log_time

log = logging.getLogger(__name__)


######################################################################################################
# paths to reference files
######################################################################################################

version = '_v4'

HBDxBase_index_path = f'../../references/HBDxBase/HBDxBase{version}'
HBDxBase_pseudo_index_path = f'../../references/HBDxBase/HBDxBase_pseudo{version}'
genome_index_path = '../../references/hg38/genome'
adapter_index_path = '../../references/HBDxBase/adapters'
TE_path = '../../references/hg38/TE.bed'
bacterial_index_path = '../../references/bacterial_viral/all_bacterial_refseq_with_human_host__201127.index' 
viral_index_path = '../../references/bacterial_viral/viral_refseq_with_human_host__201127.index' 




#%%
######################################################################################################
# specific functions
######################################################################################################

@log_time(log)
def prepare_input_files(seq_input):

    # check if seq_input is path or list
    if type(seq_input) == str:
        # get seqs in dataset
        seqs = fasta2df(seq_input)
        seqs = seqs.sequence
    elif type(seq_input) == list:
        seqs = seq_input
    else:
        raise ValueError('seq_input must be either path to fasta file or list of sequences')
    
    # add number of sequences to log file
    log_folder = "log"
    with open(f"{log_folder}/make_anno.log", "a") as ofile:
        ofile.write(f"KBA pipeline based on HBDxBase{version}\n")
        ofile.write(f"Number of sequences to be annotated: {str(len(seqs))}\n")

    if type(seq_input) == str:
        with open('seqs.fa', 'w') as ofile_1:
            for i in range(len(seqs)):
                ofile_1.write(">" + seqs.index[i] + "\n" + seqs[i] + "\n")
    else:
        with open('seqs.fa', 'w') as ofile_1:
            for i in range(len(seqs)):
                ofile_1.write(">seq_" + str(i) + "\n" + seqs[i] + "\n")

@log_time(log)
def map_seq_2_HBDxBase(
    number_mm,
    fasta_in_file,
    out_prefix
):    

    bowtie_index_file = HBDxBase_index_path

    os.system(
        f"bowtie -a --norc -v {number_mm} -f --suppress 2,6 --threads 8 -x {bowtie_index_file} {fasta_in_file} \
        --al {out_prefix + str(number_mm) + 'mm2HBDxBase__mapped.fa'} \
        --un {out_prefix + str(number_mm) + 'mm2HBDxBase__unmapped.fa'} \
        {out_prefix + str(number_mm) + 'mm2HBDxBase.txt'}"
    )
@log_time(log)
def map_seq_2_HBDxBase_pseudo(
    number_mm,
    fasta_in_file,
    out_prefix
):    

    bowtie_index_file = HBDxBase_pseudo_index_path

    os.system(
        f"bowtie -a --norc -v {number_mm} -f --suppress 2,6 --threads 8 -x {bowtie_index_file} {fasta_in_file} \
        --al {out_prefix + str(number_mm) + 'mm2HBDxBase_pseudo__mapped.fa'} \
        --un {out_prefix + str(number_mm) + 'mm2HBDxBase_pseudo__unmapped.fa'} \
        {out_prefix + str(number_mm) + 'mm2HBDxBase_pseudo.txt'}"
    )
    # -a        Report all valid alignments per read
    # --norc    No mapping to reverse strand
    # -v        Report alignments with at most <int> mismatches
    # -f        f for FASTA, -q for FASTQ; for our pipeline FASTA makes more sense
    # -suppress Suppress columns of output in the default output mode
    # -x        The basename of the Bowtie, or Bowtie 2, index to be searched

@log_time(log)
def map_seq_2_adapters(
    fasta_in_file,
    out_prefix
):    

    bowtie_index_file = adapter_index_path

    os.system(
        f"bowtie -a --best --strata --norc -v 3 -f --suppress 2,6 --threads 8 -x {bowtie_index_file} {fasta_in_file} \
        --al {out_prefix + '3mm2adapters__mapped.fa'} \
        --un {out_prefix + '3mm2adapters__unmapped.fa'} \
        {out_prefix + '3mm2adapters.txt'}"
    )
    # -a --best --strata        Specifying --strata in addition to -a and --best causes bowtie to report only those alignments in the best alignment “stratum”. The alignments in the best stratum are those having the least number of mismatches
    # --norc    No mapping to reverse strand
    # -v        Report alignments with at most <int> mismatches
    # -f        f for FASTA, -q for FASTQ; for our pipeline FASTA makes more sense
    # -suppress Suppress columns of output in the default output mode
    # -x        The basename of the Bowtie, or Bowtie 2, index to be searched


@log_time(log)
def map_seq_2_genome(
    fasta_in_file,
    out_prefix
):    

    bowtie_index_file = genome_index_path

    os.system(
        f"bowtie -a -v 0 -f -m 100 --suppress 6 --threads 8 -x {bowtie_index_file} {fasta_in_file} \
        --max {out_prefix + '0mm2genome__toomanyalign.fa'} \
        --un {out_prefix + '0mm2genome__unmapped.fa'} \
        {out_prefix + '0mm2genome.txt'}"
    )
    # -a        Report all valid alignments per read
    # -v        Report alignments with at most <int> mismatches
    # -f        f for FASTA, -q for FASTQ; for our pipeline FASTA makes more sense
    # -m        Suppress all alignments for a particular read if more than <int> reportable alignments exist for it
    # -suppress Suppress columns of output in the default output mode
    # -x        The basename of the Bowtie, or Bowtie 2, index to be searched


@log_time(log)
def map_seq_2_bacterial_viral(
    fasta_in_file,
    out_prefix
):    

    bowtie_index_file = bacterial_index_path

    os.system(
        f"bowtie -a -v 0 -f -m 10 --suppress 6 --threads 8 -x {bowtie_index_file} {fasta_in_file} \
        --al {out_prefix + '0mm2bacterial__mapped.fa'} \
        --max {out_prefix + '0mm2bacterial__toomanyalign.fa'} \
        --un {out_prefix + '0mm2bacterial__unmapped.fa'} \
        {out_prefix + '0mm2bacterial.txt'}"
    )


    bowtie_index_file = viral_index_path

    os.system(
        f"bowtie -a -v 0 -f -m 10 --suppress 6 --threads 8 -x {bowtie_index_file} {fasta_in_file} \
        --al {out_prefix + '0mm2viral__mapped.fa'} \
        --max {out_prefix + '0mm2viral__toomanyalign.fa'} \
        --un {out_prefix + '0mm2viral__unmapped.fa'} \
        {out_prefix + '0mm2viral.txt'}"
    )
    # -a        Report all valid alignments per read
    # -v        Report alignments with at most <int> mismatches
    # -f        f for FASTA, -q for FASTQ; for our pipeline FASTA makes more sense
    # -m        Suppress all alignments for a particular read if more than <int> reportable alignments exist for it
    # -suppress Suppress columns of output in the default output mode
    # -x        The basename of the Bowtie, or Bowtie 2, index to be searched





#%% 
######################################################################################################
# mapping pipeline
######################################################################################################
@log_time(log)
def main(sequence_file):
    """Executes 'map_2_HBDxBase'. Maps input sequences to HBDxBase, the human genome, and a collection of viral and bacterial genomes.

    Uses:

    - HBDxBase_index_path
    - HBDxBase_pseudo_index_path
    - genome_index_path
    - bacterial_index_path 
    - viral_index_path
    - sequence_file

    """

    prepare_input_files(sequence_file)

    # sequential mm mapping to HBDxBase
    print('-------- map to HBDxBase --------')

    print('-------- mapping seqs (0 mm) --------')
    map_seq_2_HBDxBase(
        0,
        'seqs.fa',
        'tmp_seqs'
    )

    print('-------- mapping seqs (1 mm) --------')
    map_seq_2_HBDxBase(
        1,
        'tmp_seqs0mm2HBDxBase__unmapped.fa',
        'tmp_seqs'
    )

    print('-------- mapping seqs (2 mm) --------')
    map_seq_2_HBDxBase(
        2,
        'tmp_seqs1mm2HBDxBase__unmapped.fa',
        'tmp_seqs'
    )

    print('-------- mapping seqs (3 mm) --------')
    map_seq_2_HBDxBase(
        3,
        'tmp_seqs2mm2HBDxBase__unmapped.fa',
        'tmp_seqs'
    )

    # sequential mm mapping to Pseudo-HBDxBase
    print('-------- map to Pseudo-HBDxBase --------')

    print('-------- mapping seqs (0 mm) --------')
    map_seq_2_HBDxBase_pseudo(
        0,
        'tmp_seqs3mm2HBDxBase__unmapped.fa',
        'tmp_seqs'
    )

    print('-------- mapping seqs (1 mm) --------')
    map_seq_2_HBDxBase_pseudo(
        1,
        'tmp_seqs0mm2HBDxBase_pseudo__unmapped.fa',
        'tmp_seqs'
    )

    print('-------- mapping seqs (2 mm) --------')
    map_seq_2_HBDxBase_pseudo(
        2,
        'tmp_seqs1mm2HBDxBase_pseudo__unmapped.fa',
        'tmp_seqs'
    )

    print('-------- mapping seqs (3 mm) --------')
    map_seq_2_HBDxBase_pseudo(
        3,
        'tmp_seqs2mm2HBDxBase_pseudo__unmapped.fa',
        'tmp_seqs'
    )


    # concatenate files
    print('-------- concatenate mapping files --------')
    os.system("cat tmp_seqs0mm2HBDxBase.txt tmp_seqs1mm2HBDxBase.txt tmp_seqs2mm2HBDxBase.txt tmp_seqs3mm2HBDxBase.txt tmp_seqs0mm2HBDxBase_pseudo.txt tmp_seqs1mm2HBDxBase_pseudo.txt tmp_seqs2mm2HBDxBase_pseudo.txt tmp_seqs3mm2HBDxBase_pseudo.txt > seqsmapped2HBDxBase_combined.txt")

    print('\n')

    # mapping to adapters (allowing for 3 mms)
    print('-------- map to adapters (3 mm) --------')
    map_seq_2_adapters(
        'seqs.fa',
        'tmp_seqs'
    )

    # mapping to genome (more than 50 alignments are not reported)
    print('-------- map to human genome --------')

    print('-------- mapping seqs (0 mm) --------')
    map_seq_2_genome(
        'seqs.fa',
        'tmp_seqs'
    )


    ## concatenate files
    print('-------- concatenate mapping files --------')
    os.system("cp tmp_seqs0mm2genome.txt seqsmapped2genome_combined.txt")

    print('\n')

    ## intersect genome mapping hits with TE.bed
    print('-------- intersect genome mapping hits with TE.bed --------')
    # convert to BED format
    os.system("awk 'BEGIN {FS= \"\t\"; OFS=\"\t\"} {print $3, $4, $4+length($5)-1, $5, 111, $2}' seqsmapped2genome_combined.txt > tmp_seqsmapped2genome_combined.bed")
    # intersect with TE.bed (force strandedness -> fetch only sRNA_sequence and TE_name -> aggregate TE annotation on sequences)
    os.system(f"bedtools intersect -a tmp_seqsmapped2genome_combined.bed -b {TE_path} -wa -wb -s" + "| awk '{print $4,$10}' | awk '{a[$1]=a[$1]\";\"$2} END {for(i in a) print i\"\t\"substr(a[i],2)}' > tmp_seqsmapped2genome_intersect_TE.txt")

    # mapping to bacterial and viral genomes (more than 10 alignments are not reported)
    print('-------- map to bacterial and viral genome --------')

    print('-------- mapping seqs (0 mm) --------')
    map_seq_2_bacterial_viral(
        'seqs.fa',
        'tmp_seqs'
    )

    ## concatenate files
    print('-------- concatenate mapping files --------')
    os.system("cat tmp_seqs0mm2bacterial.txt tmp_seqs0mm2viral.txt > seqsmapped2bacterialviral_combined.txt")

    print('\n')
    



