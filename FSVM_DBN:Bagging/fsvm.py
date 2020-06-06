from cvxopt import matrix
import numpy as np
from numpy import linalg
import cvxopt

# 三种kernel核函数
def linear_kernel(x1, x2):
    return np.dot(x1, x2)


# 调参1 p
def polynomial_kernel(x, y, p=1.5):
    return (1 + np.dot(x, y)) ** p


# 调参2 sigmma
def gaussian_kernel(x, y, sigma=1.0):
    # print(-linalg.norm(x-y)**2)
    x = np.asarray(x)
    y = np.asarray(y)
#    return np.exp((-linalg.norm(x - y) ** 2) / (2 * (sigma ** 2)))
    return np.exp((-linalg.norm(x - y) ** 2)*sigma)





class HYP_SVM(object):
    # 初始化函数
    def __init__(self, kernel=None, C=None, P=None, sigma=None):
        self.kernel = kernel
        self.C = C
        self.P = P
        self.sigma = sigma
        if self.C is not None: self.C = float(self.C)


    def m_func(self, X_train, X_test, y):
        # 提出两个训练集的样本数和特征数
        n_samples, n_features = X_train.shape
        nt_samples, nt_features = X_test.shape

        # 开辟一个n*n的矩阵，用于存放所有计算下来的核函数的值K(i,j)
        # Creer une matrix pour mettre les kernel valeur de x
        self.K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if self.kernel == 'polynomial':
                    self.K[i, j] = polynomial_kernel(X_train[i], X_train[j],self.P)
                elif self.kernel == 'gaussian':
                    self.K[i, j] = gaussian_kernel(X_train[i], X_train[j], self.sigma)
                else:
                    self.K[i, j] = linear_kernel(X_train[i], X_train[j])

            # print(K[i,j])

        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)

        K1 = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if self.kernel == 'polynomial':
                    K1[i, j] = polynomial_kernel(X_train[i], X_train[j], self.P)
                elif self.kernel == 'gaussian':
                    K1[i, j] = gaussian_kernel(X_train[i], X_train[j], self.sigma)
                else:
                    K1[i, j] = linear_kernel(X_train[i], X_train[j])
        print(K1.shape)
  

        # P为公式中yi*yj*fi(xi)*fi(xj)
        P = cvxopt.matrix(np.outer(y, y) * self.K)
        # q为长度为训练样本数的-1向量
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        # A为将列向量y变为横向量
        A = cvxopt.matrix(y, (1, n_samples))
        A = matrix(A, (1, n_samples), 'd')  # changes done
        # b = [0.0]
        b = cvxopt.matrix(0.0)
        print(P,q,A,b)

        if self.C is None:
            # G为对角线为n*n的对角线为-1的矩阵
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            # h = [0,0,0,...,0]
            h = cvxopt.matrix(np.zeros(n_samples))

        else:
            # tmp1 为n*n的对角线为-1的对角矩阵
            tmp1 = np.diag(np.ones(n_samples) * -1)
            # tmp2 为n*n的对角线为1的对角矩阵
            tmp2 = np.identity(n_samples)
            # G为2n*n的tmp1与tmp2的纵向堆叠
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))

            # h为2n*1的上一半为0，下一半为C的列向量
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # 解决QP问题，这里没看懂，应该是一个凸优化操作
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        print(solution['status'])
        
        # Lagrange multipliers
        a= np.ravel(solution['x'])

        a_org = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers
        # 这里我的理解是，阿法不会理想化变为0，所以设置一个阈值，大于1e-5的都为有效支持向量机参数
        # 这里sv为一个向量，里面都是true或者false
        sv = a > 1e-5
 
       
        # 只挑出那些支持向量机，为sv
        # Choisir les support vector de x 
        ind = np.arange(len(a))[sv]
        self.a_org = a_org
        self.a = a[sv]
        self.sv = X_train[sv]
        self.sv_y = y[sv]
        self.sv_yorg = y
        
        X_train = np.asarray(X_train)
        
        # # Intercept
        b = 0
        for n in range(len(self.a)):
            b += self.sv_y[n]

            # print(ind[n])
            b -= np.sum(self.a * self.sv_y * self.K[ind[n], sv])
        b /= len(self.a)
        # print(self.a_org[1])
        # print(self.a_org.shape,self.sv_yorg.shape,K.shape)
       
        # # Weight vector Multiply by phi(x),公式9
        w_phi = 0
#        total = 0
        for n in range(len(self.a_org)):

            w_phi = self.a_org[n] * self.sv_yorg[n] * K1[n]

        # the functional margin
        print(w_phi + b)
        
        self.d_hyp = np.zeros(n_samples)
        
        # 这里对应的公式是21，计算d的
        for n in range(len(self.a_org)):
            self.d_hyp += self.sv_yorg[n] * (w_phi + b)
        
        
        # linear- and exponential-decaying functions to define f(x_i)
        
        func = np.zeros((n_samples))
        func = np.asarray(func)
        typ = 2

        # 这里对应公式22，linear-decaying function
        if (typ == 1):
            for i in range(n_samples):
                func[i] = 1 - (self.d_hyp[i] / (np.amax(self.d_hyp[i]) + 0.000001))
        
        
        beta = 0.8
        # 这里对应公式23 exponential-decaying function
        if (typ == 2):
            for i in range(n_samples):
                func[i] = 2 / (1 + beta * self.d_hyp[i]) 
                
        # amount of minority class is 300,amount of majority class is 700
#        r_min = 232 / 568 
        r_min = 700/700
        r_max = 1

        print("func : ", func)
        print("func shape:", len(func))
        #calculate the fuzzy membership m
        self.m = func[0:232] * r_max
        self.m = np.append(self.m, func[232:800] * r_min)
        # print(self.m.shape)

    ##############################################################################

    def fit(self, X_train, X_test, y):
        n_samples, n_features = X_train.shape
#        nt_samples, nt_features = X_test.shape
        # Gram matrix

        # print(self.K.shape)

        P = cvxopt.matrix(np.outer(y, y) * self.K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        A = matrix(A, (1, n_samples), 'd')  # changes done
        b = cvxopt.matrix(0.0)
        # print(P,q,A,b)
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))

        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # print(solution['status'])
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        a_org = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers
        for i in range(n_samples):
            # 这里加入self.m的限制条件，加入fuzzy思想
            sv = np.logical_or(self.a_org < self.m, self.a_org > 1e-5)
        # print(sv.shape)
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X_train[sv]
        self.sv_y = y[sv]
        # print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * self.K[ind[n], sv])
        self.b /= len(self.a)
        # print(self.b)

        # Weight vector
        if self.kernel == 'polynomial' or 'gaussian' or 'linear':
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

            # 预测函数

    def project(self, X):
        if self.w is None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            X = np.asarray(X)
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):

                    if self.kernel == 'polynomial':
                        s += a * sv_y * polynomial_kernel(X[i], sv, self.P)
                    elif self.kernel == 'gaussian':
                        s += a * sv_y * gaussian_kernel(X[i], sv, self.sigma)
                    else:
                        s += a * sv_y * linear_kernel(X[i], sv)


                y_predict[i] = s
            #  print(y_predict[i])
            return y_predict + self.b

    # 预测函数
    def predict(self, X):
        return np.sign(self.project(X))