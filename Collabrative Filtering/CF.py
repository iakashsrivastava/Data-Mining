import numpy as np
import math

a = np.zeros(shape=(944,1683))
P = np.ones(shape=(944,2))
Q = np.ones(shape=(1683,2))

def matrix_factorization(R, P, Q, K, steps=100):
    alpha=0.0002;
    beta=0.002;
    Q = Q.T
    for step in xrange(steps):
        print step
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        
        error = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    error += pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        error += (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if error < 0.00001:
            break
    return P, Q.T

lines = []
with open("u.data","r") as f:
    for line in f:
        lines.append(line);

folds =5;
foldSize = 100000/folds;

avgMAE = 0.0
avgRMAE = 0.0

for i in range(0, folds):
    start = i*foldSize;
    end = start+ foldSize -1
    train = [];
    test = [];
    for j in range(0 , 100000):
        user,movie,rating,time=lines[j].strip().split('\t')
        # testing
        if (j >=start and j<= end):
            rating =0;
            test.append(lines[j])
        else:
            a[ int(user) ] [ int(movie)] = int(rating)
        
    
    nP, nQ = matrix_factorization(a, P, Q, 2);
    nQt = np.transpose(nQ)
    k = np.dot(nP,nQt)
    absErr = 0;
    absErrs = 0
    for o in range(0,len(test)):
        user,movie,rating,time=test[o].strip().split('\t')
        ratingsCalc = k[int(user)][int(movie)]
        diff = abs(ratingsCalc - int(rating))
        diffs = math.pow(diff,2)
        absErr += diff
        absErrs += diffs
    MAE = absErr / len(test)
    RMAE = math.pow(absErrs / len(test), 0.5)
    print 'MAE after fold ' +str(i) +' : ' + str(MAE);
    print 'RMAE after fold ' +str(i) +' : ' + str(RMAE);
    avgMAE += MAE
    avgRMAE += RMAE

print 'Average MAE over 5 folds: ' + str(avgMAE / 5);
print 'Average RMAE over 5 folds: ' + str(avgRMAE / 5)
        


