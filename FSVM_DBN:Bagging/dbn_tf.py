#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Peng Liu <liupeng@imscv.com>
#
# Distributed under terms of the GNU GPL3 license.

"""
This file implement a class DBN.
"""

from rbm_tf import RBM


class DBN(object):

    """Docstring for DBN. """

    def __init__(self, sizes, opts, X):
        """TODO: to be defined1.

        :sizes: TODO
        :opts: TODO

        """
        self._sizes = sizes
        self._opts = opts
        self._X = X
        self.rbm_list = []

        #这里的inputsize即为输入的每一个样本的大小。为一个拉长的像素向量
        input_size = X.shape[1]
        for i, size in enumerate(self._sizes):
            #这里循环两次，分别为i=0，size=400和i=1，size=100

            #这里构造了两层RBM，输入按顺序为name, input_size, output_size, opts
            self.rbm_list.append(RBM("rbm%d" % i, input_size, size, self._opts))
            #这里将上一层输出的个数变为下一层输入个数
            input_size = size

    def train(self):
        """TODO: Docstring for train.
        :returns: TODO

        """
        X = self._X
        #按照顺序一层一层的分别训练RBM
        for rbm in self.rbm_list:
            rbm.train(X)
            
            #这里rbmup为已经训练好参数，这里计算这一层的输出作为下一层的输入
            #Ici rbmup est les paramètres qui ont été formés.  
            # On conside la sortie de cette couche comme entrée de la couche suivante. Ensuite calculer.
            X = rbm.rbmup(X)
