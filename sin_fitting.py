import numpy
import matplotlib.pyplot as plt
N = 1000
t = numpy.linspace(0,4*numpy.pi,N)
y = numpy.sin(t)
data = 3.0*y  + numpy.random.randn(t.size)
npoly = 10
A = numpy.zeros([t.size,npoly])
A[:,0] = 1.0
for i in range(1,npoly):
    A[:,i]=A[:,i-1]*t

A = numpy.matrix(A)
d = numpy.matrix(data).transpose()
lhs = A.transpose()*A
rhs = A.transpose()*d
fitp = numpy.linalg.inv(lhs)*rhs
pred = A*fitp
plt.plot(t,y,label='Sine Function'); plt.plot(t,data,'.', label='Data'); plt.plot(t,pred,'r', label='The Fit')
plt.title('Sine Function Fit')
plt.grid(True)
plt.legend()
plt.show()
