# %% Dependecies
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, SimilarityFunction

# %% Load model
model = SentenceTransformer("all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.COSINE)

# %% Load sentences
data = pd.read_csv("processed.csv")
print(len(data))

# Drop empty rows
data = data.dropna(subset=["rephrased"])
print(len(data))

# Two lists of sentences
sentences1 = data.Instance.to_list()
sentences2 = data.rephrased.to_list()

# %% Compute embeddings

# Compute embeddings for both lists
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

# %% Compute similarities

# Compute cosine similarities
similarities = model.similarity(embeddings1, embeddings2)

# Output the pairs with their score
elems = []
for idx_i, sentence1 in enumerate(sentences1):
    s1 = sentence1
    s2 = sentences2[idx_i]
    sim_val = similarities[idx_i][sentences2.index(sentences2[idx_i])].item()
    elems.append((s1, s2, sim_val))

    print("#")
    print(s1)
    print(s2)
    print(sim_val)

df = pd.DataFrame(elems, columns=['s1', 's2', 'similarity'])

# %% Descriptive statistics
mean = np.mean(df.similarity)
median = np.median(df.similarity)
std_dev = np.std(df.similarity)
min = np.min(df.similarity)
max = np.max(df.similarity)
_25 = np.percentile(df.similarity, q=25)
_75 = np.percentile(df.similarity, q=75)

iqr = np.percentile(df.similarity, q=75) - np.percentile(df.similarity, q=25)

print(" Mean: {} \n 25%: {} \n Median: {} \n 75%: {} \n Std.Dev.: {} \n Min: {} \n Max: {} \n IQR: {} ".format(mean, _25, median, _75, std_dev, min, max, iqr))

# %% Extract samples
n = 10
df.sort_values(by="similarity", ascending=False, inplace=True)

top = df.head(n)
start = (len(df) - n) // 2
middle = df.iloc[start:start + n]
bottom = df.tail(n)

df.to_csv("similarity.csv")

# %%

top_sample = df.where(df["similarity"] > 0.65).dropna()
middle_sample = df.where((df["similarity"] < 0.65) & (df["similarity"] > 0.35)).dropna()
bottom_sample = df.where(df["similarity"] < 0.35).dropna()

top_first_sample = top_sample.sample(36)
middle_first_sample = middle_sample.sample(37)
bottom_first_sample = bottom_sample.sample(27)

sample = pd.concat([top_first_sample, middle_first_sample, bottom_first_sample], axis=0)
sample.to_csv("sample.csv")


# %%
from statsmodels.stats.inter_rater import fleiss_kappa
from statsmodels.stats import inter_rater as irr

full_analyzed = pd.read_csv("analyzed_sample.csv")

values = pd.DataFrame({
    'g': full_analyzed['Gianluca'],
    'c': full_analyzed['Costanza'],
    's': full_analyzed['Surya']

})

agg = irr.aggregate_raters(values) # returns a tuple (data, categories)

# Compute Fleiss' Kappa
kappa = fleiss_kappa(agg[0], method='fleiss')

print("Fleiss' Kappa:", kappa)

# %% Mean human rating

mean_human = np.mean(values, axis=1)

mean = np.mean(mean_human)
median = np.median(mean_human)
std_dev = np.std(mean_human)
min = np.min(mean_human)
max = np.max(mean_human)
_25 = np.percentile(mean_human, q=25)
_75 = np.percentile(mean_human, q=75)

print(" Mean: {} \n 25%: {} \n Median: {} \n 75%: {} \n Std.Dev.: {} \n Min: {} \n Max: {} \n IQR: {} ".format(mean, _25, median, _75, std_dev, min, max, iqr))


# %% Compute Burke and Dunlap ADm agreement measure

choices = values + 1

agreement_sum = 0
for index, row in choices.iterrows():
    r1, r2, r3 = row['g'], row['c'], row['s']
    mean_rating = np.mean([r1, r2, r3])
    agreement = (abs(r1 - mean_rating) + abs(r2 - mean_rating) + abs(r3 - mean_rating)) / 3
    agreement_sum += agreement

print(agreement_sum)
print(agreement_sum / len(choices))

# %% box plot of semantic similarity
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(4.5, 3))

# Create the boxplot
ax[0].boxplot([df.similarity], labels=['Cosine similarity'], patch_artist=True)
ax[1].boxplot([mean_human], labels=['Human evaluation'], patch_artist=True)

ax[1].set_ylim([0, 6])

# Add a title and labels

# Show the plot
plt.savefig("semantic_sim_plot.pdf")
plt.show()

# %%
