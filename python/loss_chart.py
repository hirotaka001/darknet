import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('../logs/loss.txt', names=['num1', 'num2'])
plt.plot(df['num1'], df['num2'], marker="o")
plt.savefig('../logs/figure.png')
plt.show()