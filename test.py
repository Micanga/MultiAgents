import matplotlib.pyplot as plt
import numpy as np

fig= plt.figure()

# axes= fig.add_axes([0,0,1,1])

x= np.arange(0,11)

plt.plot(x, color='purple', marker='o')

plt.show()