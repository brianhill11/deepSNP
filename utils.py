#!/usr/bin/python

import numpy as np
import pysam
import deepSNP


def vert_stack_matrices(top_matrix, bottom_matrix):
    """
    Add (L x F) 2D matrix to R dimension of
    (R x L x F) 3D matrix

    We are trying to make a 3D feature matrix by
    stacking 2D matrices. In this case, we have
    (L x F) matrices, where L=window width and
    F=#features and we would like to stack R of
    these matrices where R=#reads, to make an
    (R x L x F) matrix
       __________
    F/          /|
    /__________/ |
    |          | |
   R|          | |
    |          | |
    |__________|/
          L
    :param top_matrix:
    :param bottom_matrix:
    :return: stacked 3D feature matrix
    """
    # if we already have a 3D top matrix, turn bottom into 3D and merge
    if top_matrix.ndim == 3:
        new_mat = np.vstack([top_matrix, bottom_matrix[np.newaxis, ...]])
        # print "new matrix shape:", new_mat.shape
        return new_mat
    # otherwise this is the initial stacking of 2 2D matrices
    else:
        new_mat = np.vstack([top_matrix[np.newaxis, ...], bottom_matrix[np.newaxis, ...]])
        # print "new matrix shape:", new_mat.shape
        return new_mat


def concat_feature_matrices(feat_mat1, feat_mat2):
    """
    Join 2 feature matrices together column-wise

    :param feat_mat1: (WINDOW_SIZE x F1) matrix
    :param feat_mat2: (WINDOW_SIZE x F2) matrix
    :return: (WINDOW_SIZE x F1+F2) matrix
    """
    # print "mat1 shape:", feat_mat1.shape
    # print "mat2 shape:", feat_mat2.shape
    try:
        return np.concatenate((feat_mat1, feat_mat2), axis=1)
    except ValueError as e:
        print e
        print "matrix1 shape:", feat_mat1.shape
        print "matrix2 shape:", feat_mat2.shape
        print "Did you forget to return mat[..., np.newaxis] ?"
        exit(-100)


def get_padding(read, window_start):
    """
    Calculates the number of positions to the left and right
    of the read that do NOT overlap with the feature window

    Ex: window size of 10, read overlaps by 4 positions

    pos:    0123456789
    window: WWWWWWWWWW
    read:   -rrrr-----
    pad L:  1
    pad R:       12345

    # pad left = 1, # pad right = 5

    :param read: pysam read
    :param window_start: starting position of feature window
    :return: (# pad left, # pad right) tuple
    """
    window_end = window_start + deepSNP.WINDOW_SIZE
    # calculate dimensions to left and right of read for padding zeros
    num_pad_left = np.maximum(0, read.reference_start - window_start)
    # NOTE: pysam reference_length = reference_end - reference_start
    # but this does not necessarily mean the query sequence is that length
    ref_end = read.reference_start + len(read.query_sequence)
    num_pad_right = np.maximum(0, window_end - ref_end)
    return num_pad_left, num_pad_right


def get_snps_in_window(snps, window_start, window_end):
    """
    Check to see which SNPs overlap with window

    :param snps: list of dicts, [{pos -> (alleles)}, ...]
    :param window_start: starting position in contig
    :param window_end: ending position in contig
    :return: list of positions of overlapping SNPs
    """
    snps_in_window = []
    for snp in snps:
        for pos, alleles in snp.items():
            if pos >= window_start and pos <= window_end:
                snps_in_window.append(pos)
    return snps_in_window


def is_usable_read(read):
    """
    Checks read for several features to determine whether or not
    read is suitable for inclusion

    source: http://biorxiv.org/content/biorxiv/suppl/2016/12/21/092890.DC3/092890-1.pdf
    slightly modified from version found in link above

    :param read: pysam read
    :return: true if read passes all quality tests, else false
    """
    return (len(read.get_tag("MD")) < 6 and
            not (read.is_duplicate or read.is_qcfail or
                 read.is_secondary or read.is_supplementary) and
            read.is_paired and read.mapping_quality >= 10)


def decode_cigar(read, position):
    """
    Gets the cigar event for a position in a read
    :param read: the pysam read
    :param position: the position in the read to decode
    :return: the cigar event at that position
    """
    # cigar tuples have format [(operation, length)]
    # check first tuple before iterating through all
    # tuples, as most reads hopefully match fully
    cigar_pos = read.cigartuples[0][1]
    if position <= cigar_pos:
        return read.cigartuples[0][0]

    # else we have to weed through the messy by
    # checking if our position falls within range of the operation
    for op, length in read.cigartuples[1:]:
        cigar_pos += length
        if position <= cigar_pos:
            return op
