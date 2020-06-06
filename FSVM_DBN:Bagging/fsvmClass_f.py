from cvxopt import matrix
import numpy as np
from numpy import linalg
import cvxopt
import fsvm_utils as utils

'''
FSVM 实现

初始化：
    Kernel：所选核种类
    C：C值
    P：poly核的P值
    sigma：gaussian核的sigma值
    bad_number: 样本中负样本的数量
    detect_noise: boolean,默认false
        是否在计算fuzzy membership中加入对点的稠密程度的判断
    calcul_distance：string型，默认 max
        选择计算membership distance 的方法，
        可选 "max" 或 "avg"
'''

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
    return np.exp((-linalg.norm(x - y) ** 2) / (2 * (sigma ** 2)))




class HYP_SVM(object):
    # 初始化函数
    def __init__(self, kernel=None, C=None, P=None, sigma=None, bad_number=None, detect_noise=None,calcul_distance=None):
        self.kernel = kernel
        self.C = C
        self.P = P
        self.sigma = sigma
        self.detectNoise = detect_noise
        self.bad_number=bad_number
        self.calcul_distance = calcul_distance
        if self.C is not None: self.C = float(self.C)
        if self.detectNoise is None:
            self.detectNoise = False
        if self.calcul_distance is None:
            self.calcul_distance = 'max'


    def m_func(self, X_train, X_test, y):
        # 提出两个训练集的样本数和特征数
        n_samples, n_features = X_train.shape

        # 开辟一个n*n的矩阵，用于存放所有计算下来的核函数的值K(i,j)
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

        # P为公式中yi*yj*fi(xi)*fi(xj)
        P = cvxopt.matrix(np.outer(y, y) * self.K)
        # q为长度为训练样本数的-1向量
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        # A为将列向量y变为横向量
        A = cvxopt.matrix(y, (1, n_samples))
        A = matrix(A, (1, n_samples), 'd')  # changes done
        # b = [0.0]
        b = cvxopt.matrix(0.0)
        # print(P,q,A,b)

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
        # print(solution['status'])
        # Lagrange multipliers
        # 将solution['x']拉为一个向量，我大胆预测这里a就是参数阿法
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        # 这里我的理解是，阿法不会理想化变为0，所以设置一个阈值，大于1e-5的都为有效支持向量机参数
        # 这里sv为一个向量，里面都是true或者false
        sv = a > 1e-5
        # print(sv.shape)

        #         print(a)
        #         print(sv)
        #        print(a[sv])
        # 只挑出那些支持向量机，为sv
        ind = np.arange(len(a))[sv]
        self.a_org = a
        self.a = a[sv]
        self.sv = X_train[sv]
        self.sv_y = y[sv]
        self.sv_yorg = y
        X_train = np.asarray(X_train)
        b = 0

        for n in range(len(self.a)):
            b += self.sv_y[n]
            b -= np.sum(self.a * self.sv_y * self.K[ind[n], sv])
        b /= len(self.a)

        w_phi_x = 0

        # 大致公式应该是对应着论文里面的公式8，weigh的更新公式
        for n in range(len(self.a_org)):
            w_phi_x = self.a_org[n] * self.sv_yorg[n] * self.K[n]

        if self.calcul_distance == 'max':
            self.d_hyp = np.zeros(n_samples)
            # 这里对应的公式是21，计算d的
            for n in range(len(self.a_org)):
                self.d_hyp += self.sv_yorg[n] * (w_phi_x + b)
        elif self.calcul_distance == 'avg':
            # 计算点到该分类中心的距离
            distance_good = utils.calcul_avg_distance(X_train[self.bad_number:len(X_train)])
            distance_bad = utils.calcul_avg_distance(X_train[0:self.bad_number])
            distance_bad.extend(distance_good)
            self.d_hyp = np.array(distance_bad)

        func = np.zeros((n_samples))
        func = np.asarray(func)


        if self.detectNoise is True:
            # Knn 计算稠密程度
            X_bad = X_train[0:self.bad_number]
            X_good = X_train[self.bad_number:len(X_train)]
            X_bad_distance, X_bad_max, X_bad_min = utils.calcul_knn_distance(X_bad,10)
            X_good_distance, X_good_max, X_good_min = utils.calcul_knn_distance(X_good, 10)

            func_a = 0.5
            for i in range(0,self.bad_number):
                func[i] = 1 - func_a * (self.d_hyp[i] / (np.amax(self.d_hyp) + 0.000001)) - (1-func_a)*(X_bad_distance[i]-X_bad_min)/(X_bad_max-X_bad_min+0.000001)
            for i in range(self.bad_number,n_samples):
                index = i - self.bad_number
                func[i] = 1 - func_a * (self.d_hyp[i] / (np.amax(self.d_hyp) + 0.000001)) - (1-func_a)*(X_good_distance[index]-X_good_min)/(X_good_max-X_good_min+0.000001)
        else:
            for i in range(n_samples):
                func[i] = 1 - (self.d_hyp[i] / (np.amax(self.d_hyp) + 0.000001))

        # 这一块很奇怪，我不知道这个数字比例是怎么设置的，有待考量
        r_max = 300 / 700
        r_min = 1

        self.m = func[0:self.bad_number] * r_min
        self.m = np.append(self.m, func[self.bad_number:n_samples] * r_max)


    ##############################################################################

    def fit(self, X_train, X_test, y):
        n_samples, n_features = X_train.shape

        # Gram matrix

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

        # Lagrange multipliers
        a = np.ravel(solution['x'])

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
        if self.kernel =='linear':
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else :
            self.w = None

        return self.w

            # 预测函数

    def project(self, X):
        if self.w is not None:
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