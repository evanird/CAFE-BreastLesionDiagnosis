"""
Authors: David Zhu and Evani Radiya-Dixit, Beth Isreal Deaconess Medical Center and Harvard Medical School
Created: 06/22/2015
Last modified: 04/04/2017

This file generates predictions of the 51 BIDMC cases using the 116 MGH cases as the training and validation cases. The model used is logistic regression with early stopping and stochastic gradient descent optimization, 
implemented with Theano. The algorithm fits the training samples to a logistic curve by minimizing a loss function based on the feature values. The samples not used for testing are split into a training set and a 
validation set. The model trains on the former set and prevents overfitting through verification on the latter set. Training is ceased when the model no longer improves its score on the validation set. 

Input (active features) = ActiveFeatures_Label_HospitalOrder.csv
Row 1-116 = 116 MGH Cases (DCIS = 80, UDH = 36)
Row 117-167 = 51 BIDMC Cases (DCIS = 20, UDH = 31) 
"""

__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time
from PIL import Image
import glob
import random
from sklearn import cross_validation
import numpy

import theano
import theano.tensor as T
import theano.printing as P

random.seed(12345)

NUMROWS = 167

class LogisticRegression(object):
    """Multi-class Logistic Regression Class
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """
        
    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie
        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        #scores = self.p_y_given_x[:,1]
                
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        # self.y_pred = T.argmax(P.Print('scores')(self.p_y_given_x), axis=1)
        self.y_pred = T.argmax((self.p_y_given_x), axis=1)
        
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]
    
    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        .. math::
            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            #return T.mean(T.neq(P.Print('y_pred')(self.y_pred),
            #              P.Print('y')(y)))
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
    
    def values(self):
        """Return the self.p_y_given_x"""
        return self.p_y_given_x

def getnfeats(it):
    nfeats = numpy.loadtxt("../../R_directory/nfeats.csv", delimiter=',', skiprows = 1, usecols = (1,))
    return int(nfeats[it])

def get_matrix(rows, cols, addedcols, it):
    matrix = numpy.loadtxt("../../R_directory/ActiveFeatures_Label_HospitalOrder.csv", delimiter=',', skiprows = 1, usecols = range(1,cols+1))
    for col in range(0, cols):
        colmin = matrix[0][col]
        colmax = matrix[0][col]
        for row1 in range(0, rows):
            val = matrix[row1][col]
            if val < colmin:
                colmin = val
            if val > colmax:
                colmax = val
        colrange = colmax - colmin
        for row2 in range(0, rows):
            val = matrix[row2][col]
            if colrange == 0:
                newvalue = 0
            else:
                newvalue = (val - colmin) / colrange
            matrix[row2][col] = newvalue
    if addedcols == 0:
        return matrix
    zeros = numpy.zeros(shape=(rows, addedcols))
    matrix = numpy.hstack((matrix, zeros))        
    return matrix

def get_target(rows, cols, it):
    temp2 = numpy.loadtxt("../../R_directory/ActiveFeatures_Label_HospitalOrder.csv", dtype = str, delimiter=',', skiprows = 1, usecols = (cols+1,))
    target = []
    for i in range(0, rows):
        target.append(1 if temp2[i] == '"DCIS"' else 0)
    target = numpy.array(target)
    return target

def read_data(start, end, matrix, target):
    return (matrix[range(start,end),], target[start:end])

def load_data(NUMCOLS, it):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############
    
    matrix = get_matrix(NUMROWS, NUMCOLS, 0, it)
    target = get_target(NUMROWS, NUMCOLS, it)

    train_set = read_data(0, 85, matrix, target)
    valid_set = read_data(85, 116, matrix, target)
    test_set = read_data(116, 167, matrix, target)

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #which row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
    
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets us get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def sgd_optimization(it, learning_rate=0.3, n_epochs=10000, batch_size=17):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    """

    start_time = time.clock()
    NUMCOLS = getnfeats(it)
    datasets = load_data(NUMCOLS, it)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    classifier = LogisticRegression(input=x, n_in=NUMCOLS, n_out=2)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    get_values = theano.function(
        inputs=[index],
        outputs=classifier.values(),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
            (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                # go through this many
                                # minibatche before checking the network
                                # on the validation set; in this case we
                                # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                    for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                '''print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )'''
                
                #print(classifier.W.get_value())

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                    improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                for i in xrange(n_test_batches)]
                    a, b, c = [get_values(i)
                                for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    
                    predicts = numpy.concatenate((numpy.array([[0, 1]]), a, b, c), axis=0)
                    # print(predicts)
                    numpy.savetxt("../../R_directory/theanologpredicts2/theanologpredict.csv", predicts, delimiter=",")
                    '''print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )'''

            if patience <= iter:
                done_looping = True
                break

    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs' % (
        epoch)
    
    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
    t1 = time.clock()
    for it in range(0,1):
        print("This is seed #"+str(it+1))
        sgd_optimization(it)
    t2 = time.clock()
    print("The total time was "+str(t2-t1)+" seconds")
