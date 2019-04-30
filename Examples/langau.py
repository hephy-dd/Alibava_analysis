import numpy as np
import pylandau
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

mpv, eta, sigma, A = 21.39605502,   10.04769147,  37.558822,   2415.89872763
x = np.arange(0, 500, 0.01)
#y = pylandau.landau(x, mpv=mpv, eta=eta, A=A)
#y_gconv = gaussian_filter1d(y, sigma=sigma / 0.01)
# Scaled means that the resulting Landau*Gauss function maximum is A at mpv
# Otherwise the underlying Landau function before convolution has the
# function maximum A at mpv
#y_gconv_2 = pylandau.langau(x, mpv, eta, sigma, A, scale_langau=False)
y_gconv_1 = pylandau.langau(x, mpv, eta, sigma, A, scale_langau=True)
y_gconv_2 = pylandau.langau(x, mpv, eta+1, sigma, A, scale_langau=True)
y_gconv_3 = pylandau.langau(x, mpv, eta+2, sigma, A, scale_langau=True)
#plt.plot(x, y, label='Landau')
#plt.plot(x, y_gconv_2, label='Langau')
#plt.plot(x, y_gconv, '--', label='Langau Scipy')
plt.plot(x, y_gconv_1, label='Langau eta= 1')
plt.plot(x, y_gconv_2, label='Langau eta= 2')
plt.plot(x, y_gconv_3, label='Langau eta= 3')
plt.legend(loc=0)
plt.show()