import numpy as np
import matplotlib.pyplot as plt

def read_data(file_name):
    o = open(file_name,"r")
    rl = o.readlines()
    theta,dsigma = [], []
    for i in rl:
        data = i.strip().split()
        theta.append(float(data[0]))
        dsigma.append(float(data[1]))
    return (theta,dsigma) 

def mirror_data():
	theta, dsigma = read_data("Ne_atom/q_Z.txt")
	for i in range(len(theta)):
		theta_prime = np.pi + theta[i]
		dsigma_prime = dsigma[i]
		theta.append(theta_prime)
		dsigma.append(dsigma_prime)
	return theta, dsigma

def plot_data():
    theta, dsigma = mirror_data()
    #print(theta, dsigma)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, dsigma, lw = 2.0)
    ax.set_rmax(max(dsigma)*1.02)
    ax.set_rticks([]) 
    thetatick_locs = np.linspace(0.0,315.0,8)
    thetatick_labels = [u'%i\u00b0'%np.round(x) for x in thetatick_locs]
    ax.set_thetagrids(thetatick_locs, thetatick_labels, fontsize=12)
    ax.set_rlabel_position(-30.0) 
    ax.set_title("QChem Z", va='bottom')
    plt.savefig('Ne_atom/q_Ne_Z.png')
    plt.show()
plot_data()

