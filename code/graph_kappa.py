import numpy as np
import math
import messagepassing as msg

def graph_kappa():
    kappas = np.linspace(1, 1/math.sqrt(math.e) + 0.001, num=50)
    ts = []
    threshold = 1e+15
    # Find largest t
    tn = 1
    d = 10
    while (True):
        _, mus = msg.get_poly(d, tn, kappas[-1])
        if (math.isnan(mus[-1]) or mus[-1] > threshold):
            break
        tn += 100
    tn = tn - 100
    while (True):
        _, mus = msg.get_poly(d, tn, kappas[-1])
        if (mus[-1] > threshold):
            break
        tn += 1
    ts.append(tn)
    for kappa in kappas[::-1][:-1]:
        start = 0
        stop = tn
        while (start < stop):
            _, mus = msg.get_poly(d, (start + stop) / 2, kappa)
            if mus[-1] > threshold:
                stop = (start +  stop) / 2 - 1
            elif mus[-1] < threshold:
                start = (start + stop) / 2 + 1
        tn = (start + stop) / 2
        ts.append(tn)
    print kappas.shape
    print len(ts)
    plt.plot(kappas, ts)
    plt.ylabel('t')
    plt.xlabel(r'$\lambda\kappa$')
    plt.show()

