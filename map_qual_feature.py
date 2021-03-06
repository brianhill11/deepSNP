#!/usr/bin/python

import numpy as np
import deepSNP
import deepSNP_utils


def map_qual_feature_matrix(read, window_start):
    """
    Creates a matrix containing mapping quality value
    at positions in feature window where read overlaps

    :param read: pysam read
    :param window_start: starting position of feature window
    :return: (WINDOW_SIZE x 1) matrix containing MQAL at read positions
    """
    # check if we have only part of the read in the window
    seq_start, seq_end = deepSNP_utils.seq_start_end(read, window_start)

    # print "offset:", normalized_offset
    # print "start: ", seq_start, " end: ", seq_end
    # print "padding:", get_padding(read, window_start)
    # create vector filled with mapping quality value
    map_qual_feat_mat = np.full((seq_end - seq_start, 1), read.mapping_quality)

    # we are padding 1st dimension on left and right with zeros.
    # (0, 0) says don't pad on 2nd dimension before or after
    map_qual_feat_mat = np.lib.pad(map_qual_feat_mat,
                                       (deepSNP_utils.get_padding(read, window_start), (0, 0)),
                                       'constant', constant_values=(0,))
    return map_qual_feat_mat


def print_map_qual_feature_matrix(map_qual_feat_matrix):
    """
    Prints mapping quality feature matrix in a
    command-line friendly way (I think)

    :param map_qual_feat_matrix: matrix containing mapping quality
    :return: None
    """
    deepSNP_utils.print_two_digit_feature_matrix(map_qual_feat_matrix)
    return
