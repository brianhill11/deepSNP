#!/usr/bin/python

import numpy as np
from sklearn.feature_extraction import DictVectorizer
import deepSNP
import deepSNP_utils

"""
Create one-hot-vector encoding of bases
>>> base_dict = [{'base': 'A'}, {'base': 'C'}, {'base': 'G'}, {'base': 'T'}]
>>> dv = DictVectorizer(sparse=False)
>>> dv.fit_transform(base_dict)
array([[ 1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  1.]])
>>> dv.transform({'base':'G'})
array([[ 0.,  0.,  1.,  0.]])
"""
base_dict = [{'base': 'A'}, {'base': 'C'}, {'base': 'G'}, {'base': 'T'}]
dv = DictVectorizer(sparse=False)
dv.fit_transform(base_dict)


def base_pair_feature_matrix(read, window_start):
    """
    Creates (WINDOW_SIZE x 4) matrix, where the 4 dimensions
    are for each possible base pair (A,C,G,T) and they are
    one-hot encoded using the vectorize_base_seq function
    and then padded with zeros around the read so that the
    final matrix size is WINDOW_SIZE wide

    :param read: pysam read
    :param window_start: starting position of feature window
    :return: (WINDOW_SIZE x 4) matrix
    """
    # check if we have only part of the read in the window
    seq_start, seq_end = deepSNP_utils.seq_start_end(read, window_start)
    # use query_alignment_sequence since we do not want soft-clipped bases
    base_pair_seq = read.query_alignment_sequence[seq_start:seq_end]
    # create the (READ_LENGTH x 4) matrix encoding base pairs
    one_hot_base_mat = vectorize_base_seq(base_pair_seq)
    # print "1-hot shape:", one_hot_base_mat.shape

    # we are padding 1st dimension on left and right with zeros.
    # (0, 0) says don't pad on 2nd dimension before or after
    base_pair_feat_matrix = np.lib.pad(one_hot_base_mat,
                                       ((deepSNP_utils.get_padding(read, window_start)), (0, 0)),
                                       'constant', constant_values=(0,))
    if base_pair_feat_matrix.shape[0] != deepSNP.WINDOW_SIZE:
        print "ERROR: base pair feat matrix not size of window"
        print "len(bps)", len(base_pair_seq)
        print "window start:", window_start
        print "read start:", read.reference_start
        print "read end:", read.reference_end
        print "read len:", read.reference_length
        print "len(read.query_seq):", len(read.query_sequence)
        print "query start:", read.query_alignment_start
        print "query end:", read.query_alignment_end
        print "len(qs):", len(read.query_alignment_sequence)
        print "shape:", base_pair_feat_matrix.shape
        print "seq_start:", seq_start
        print "seq_end:", seq_end
        print "read:", read
        exit(-101)
    # print "BP feat matrix shape: ", base_pair_feat_matrix.shape
    if deepSNP.DEBUG:
        print_base_pair_feature_matrix(base_pair_feat_matrix)
    return base_pair_feat_matrix


def print_base_pair_feature_matrix(base_pair_feat_matrix):
    """
    Uses inverse transform of one-hot encoded matrix to build
    a string representing the matrix data

    :param base_pair_feat_matrix: one-hot encoded base pair matrix
    :return: Printable string
    """
    bp_string = ""
    # get inverse transform
    bp_inv = dv.inverse_transform(base_pair_feat_matrix)
    for bp in bp_inv:
        try:
            # each dict in list looks like {'base=G', 1.0}
            bp_string += bp.items()[0][0].split('=')[1]
        except IndexError:
            bp_string += "-"
    print bp_string
    return bp_string


def vectorize_base_seq(seq):
    """
    Encodes each base letter as one-hot-vector, creating
    a matrix of one-hot-vectors representing a base sequence
    See global section for example

    :param seq: sequence string of base pairs
    :return: matrix of bases encoded as one-hot-vectors
    """
    list_of_dicts = []
    for base in seq:
        list_of_dicts.append({'base': base.upper()})
    return dv.transform(list_of_dicts)

