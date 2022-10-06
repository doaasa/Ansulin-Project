import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


human_data = pd.read_table("C:\\Users\\Electronica Care\\Downloads\\Ansulin\\homo.fasta")
human_data.head()
print(human_data.head())


chimp_data = pd.read_table('chimp_data.txt')
dog_data = pd.read_table('dog_data.txt')
chimp_data.head()
dog_data.head()