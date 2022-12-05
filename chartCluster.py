import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Before clustering
plt.figure()
df = pd.read_csv("GPU")
# df.columns = ["Annual income (k$)", "Spending Score (1-100)"]
# sns.scatterplot(x=df["Annual income (k$)"], 
#                 y=df["Spending Score (1-100)"])
sns.scatterplot(x=df.x, y=df.y,z=df.z             )

plt.title("Scatterplot of spending (y) vs income (x)")

# while clustering
# for i in range(6):
#     plt.figure()
#     df = pd.read_csv("train"+str(i)+".csv")
#     sns.scatterplot(x=df.x, y=df.y, 
#                     hue=df.c, 
#                     palette=sns.color_palette("hls", n_colors=5))
#     plt.xlabel("Annual income (k$)")
#     plt.ylabel("Spending Score (1-100)")
#     plt.title("Clustered: spending (y) vs income (x)")


# After clustering
# plt.figure()
# df = pd.read_csv("OutputCPU.csv")
# sns.scatterplot(x=df.x, y=df.y, 
#                 hue=df.c, 
#                 palette=sns.color_palette("hls", n_colors=5))
# plt.xlabel("Annual income (k$)")
# plt.ylabel("Spending Score (1-100)")
# plt.title("CPU")

# plt.figure()
# df = pd.read_csv("OutputGPU.csv")
# sns.scatterplot(x=df.x, y=df.y, 
#                 hue=df.c, 
#                 palette=sns.color_palette("hls", n_colors=5))
# plt.xlabel("Annual income (k$)")
# plt.ylabel("Spending Score (1-100)")
# plt.title("GPU")


# plt.figure()
# df = pd.read_csv("OutputTHRUST.csv")
# sns.scatterplot(x=df.x, y=df.y, 
#                 hue=df.c, 
#                 palette=sns.color_palette("hls", n_colors=5))
# plt.xlabel("Annual income (k$)")
# plt.ylabel("Spending Score (1-100)")
# plt.title("THRUST")

# plt.show()