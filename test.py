import matplotlib.pyplot as plt
import numpy as np

x = np.arange(100)
y = np.exp((np.sin(np.log(x + np.sin(x))))**2)
plt.plot(x,y)
plt.show()

