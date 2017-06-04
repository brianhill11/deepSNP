#!/usr/bin/python

import numpy as np
import deepSNP
import deepSNP_utils


def base_pos_feature_matrix(read, window_start):
    """
    Creates a (WINDOW_SIZE x 1) matrix containing the
    position of the base within the read at each location
    where the read overlaps the feature window

    :param read: pysam read
    :param window_start: starting position of feature window
    :return: (WINDOW_SIZE x 1) matrix containing position of base in read
    """
    # check if we have only part of the read in the window
    seq_start, seq_end = deepSNP_utils.seq_start_end(read, window_start)

    # read positions are zero based in pysam, so add 1
    base_positions = np.arange(seq_start+1, seq_end+1)

    # we are padding 1st dimension on left and right with zeros.
    # (0, 0) says don't pad on 2nd dimension before or after
    base_pos_feat_mat = np.lib.pad(base_positions[..., np.newaxis],
                                   (deepSNP_utils.get_padding(read, window_start), (0, 0)),
                                   'constant', constant_values=(0,))
    return base_pos_feat_mat


def print_base_pos_feature_matrix(base_pos_feat_matrix):
    """
    Prints base_pos_feature_matrix where first row printed
    is tens column of position, second row is ones column
    (Yes, I know positions can be three digits, but I'm lazy)

    :param base_pos_feat_matrix: numpy matrix
    :return: None
    """
    utils.print_two_digit_feature_matrix(base_pos_feat_matrix)
    return
