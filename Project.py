from Bio import SeqIO
from Bio.SeqUtils import GC
import pandas as pd
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import matplotlib.pyplot as plt
import seaborn as sns
import jellyfish as fhh

"""________________________________Read Sequence________________________________"""

human_seq = []
Pig_seq = []

Human = open("C:\\Users\\Electronica Care\\Downloads\\Ansulin\\homo.fasta", "rU")
pig = open("C:\\Users\\Electronica Care\\Downloads\\Ansulin\\PIG.fasta", "rU")

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
    sns.barplot(data=tr)
    return tr
    
print(Visualization(human_seq), "\n")



"""________________________________Visualization 2________________________________"""
human_seq = str(human_seq)
Pig_seq= str(Pig_seq)
def Visualization2(seq_human , seq_pig):
    trr = [seq_human.count("C"),seq_human.count("G"),seq_human.count("T"),
       seq_human.count("A")]

    trry = [seq_pig.count("C"),seq_pig.count("G"),seq_pig.count("T"),
       seq_pig.count("A")]
    
    plt.plot(trry,trr)

Visualization2(human_seq, Pig_seq)

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

file1 = open('Alignment.txt', 'w')
file1.write(str(Alignment(human_seq,Pig_seq)))
file1.close()

# =============================================================================
# file1 = open('Alignment.txt', 'r')
# print(file1.read())
# file1.close()
# 
# =============================================================================

print(Alignment(human_seq,Pig_seq))





"""

from Bio import SeqIO


count = len(open("C:\\Users\\Electronica Care\\Downloads\\Ansulin\\homo.fasta").readlines(  ))
print(count)


f = open("C:\\Users\\Electronica Care\\Downloads\\Ansulin\\homo.fasta", "r")
f.readline()



total=""
while count>0:
    total+=f.readline()    
    count-=1

print(total)

co=len(total)
print(co)
i=0
Kmar=""
while co>4:
    kmar=total[i:i+4]
    print(kmar)
    co-=1
    i+=1
    
    
    
def count_kmers(sequence, k_size):
    data = {}
    size = len(sequence)
    for i in range(size - k_size + 1):
        kmer = sequence[i: i + k_size]
        try:
            data[kmer] += 1
        except KeyError:
            data[kmer] = 1
    return data

print(count_kmers(total, 3))

"""

"""  Visuali
dataset = [entry.replace("\n", "") for entry in count_kmers(total, 3)]
dataset = [entry.split(" ") for entry in dataset]


def plot_coverage_chart(dataset, min_coverage=0, max_coverage=1000):
    coverage = [int(entry[0]) for entry in dataset][min_coverage:max_coverage]
    frequency = [int(entry[1]) for entry in dataset][min_coverage:max_coverage]
    higher_frequency = max(frequency)
    average_cov = coverage[frequency.index(higher_frequency)]

    plt.plot(coverage, frequency)
    plt.ylabel('Frequency')
    plt.xlabel('Coverage')
    plt.ylim(0, higher_frequency)
    plt.show()

    total_number_of_kmers = sum([c*f for c, f in zip(coverage, frequency)])
    print(int(total_number_of_kmers/average_cov))

plot_coverage_chart(dataset=dataset)
"""