from Bio import SeqIO
from Bio.SeqUtils import GC
import pandas as pd
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

"""________________________________Read Sequence________________________________"""

human_seq = []
Pig_seq = []

Human = open("sequence (1).fasta", "rU")
pig = open("sequence.fasta", "rU")

for record in SeqIO.parse(Human, "fasta") :
    human_seq.append(str(record.seq))
    
for record in SeqIO.parse(pig, "fasta") :
    Pig_seq.append(str(record.seq))
"""__________________________________GC Content_________________________________"""

def calc_GC(seq):
    gc_Values = sorted(GC(rec) for rec in seq)
    return gc_Values

GC_Content = calc_GC(human_seq)
print("\n", "GC Content = ", GC_Content, "\n")

"""________________________________Visualization________________________________"""

human_seq = str(human_seq) #Convert sequence to string
def Visualization(seq):
    tr = pd.DataFrame({"c":[seq.count("C")],
                     "g":[seq.count("G")],
                     "t":[seq.count("T")],
                     "a":[seq.count("A")],
                     })
    tr.plot.bar()
    return tr
    
print(Visualization(human_seq), "\n")

"""__________________________________Alignment__________________________________"""

#Convert sequence to string
human_seq = str(human_seq)
Pig_seq = str(Pig_seq)

def Alignment(seq1, seq2):
    # No parameters. Identical characters have score of 1, else 0.
    # No gap penalties.
    alignments = pairwise2.align.globalxx(seq1, seq2)

    for a in alignments:
        return format_alignment(*a)


print(Alignment(human_seq,Pig_seq))


