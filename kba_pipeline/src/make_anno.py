
#%%
import argparse
import os
import logging

from utils import make_output_dir,write_2_log,log_time
import map_2_HBDxBase as map_2_HBDxBase
import annotate_from_mapping as annotate_from_mapping


log = logging.getLogger(__name__)



#%%
# get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--five_prime_adapter', type=str, default='GTTCAGAGTTCTACAGTCCGACGATC')
parser.add_argument('--fasta_file', type=str, help="Required to provide: --fasta_file sequences_to_be_annotated.fa") # NOTE: needs to be stored in "data" folder
args = parser.parse_args()
if not args.fasta_file:
    parser.print_help()
    exit()
five_prime_adapter = args.five_prime_adapter
sequence_file = args.fasta_file

#%%
@log_time(log)
def main(five_prime_adapter, sequence_file):
    """Executes 'make_anno'. 
    1. Maps input sequences to HBDxBase, the human genome, and a collection of viral and bacterial genomes.
    2. Extracts information from mapping files.
    3. Generates annotation columns and final annotation dataframe.

    Uses:

    - sequence_file
    - five_prime_adapter

    """
    output_dir = make_output_dir(sequence_file)
    os.chdir(output_dir)

    log_folder = "log"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    write_2_log(f"{log_folder}/make_anno.log")
    
    # add name of sequence_file to log file
    with open(f"{log_folder}/make_anno.log", "a") as ofile:
        ofile.write(f"Sequence file: {sequence_file}\n")
                    
    map_2_HBDxBase.main("../../data/" + sequence_file)
    annotate_from_mapping.main(five_prime_adapter)
   

main(five_prime_adapter, sequence_file)
# %%
