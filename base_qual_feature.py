#!/usr/bin/python

import numpy as np
import deepSNP
import deepSNP_utils



def base_qual_feature_matrix(read, window_start):
    """
    Creates a matrix containing base quality values for
    positions where a base overlaps the feature window

    :param read: pysam read
    :param window_start: starting position of feature window
    :return: (WINDOW_SIZE x 1) matrix containing QUAL at base positions
    """
    # check if we have only part of the read in the window
    seq_start, seq_end = deepSNP_utils.seq_start_end(read, window_start)
    # get base qualities of bases that are not soft-clipped
    base_qual_seq = np.asarray(read.query_alignment_qualities[seq_start:seq_end])

    # we are padding 1st dimension on left and right with zeros.
    # (0, 0) says don't pad on 2nd dimension before or after
    base_qual_feat_matrix = np.lib.pad(base_qual_seq[..., np.newaxis],
                                       ((deepSNP_utils.get_padding(read, window_start)), (0, 0)),
                                       'constant', constant_values=(0,))
    return base_qual_feat_matrix


def print_base_qual_feature_matrix(base_qual_feat_matrix):
    """
    Prints base quality feature matrix in a
    command-line friendly way (I think)

    :param base_qual_feat_matrix: matrix containing mapping quality
    :return: None
    """
    deepSNP_utils.print_two_digit_feature_matrix(base_qual_feat_matrix)
    return
