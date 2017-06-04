#!/usr/bin/python

import numpy as np
import deepSNP
import deepSNP_utils


def snp_pos_feature_matrix(read, window_start):
    """
    Creates vector of zeros, except 1 at SNP position

    :param read: pysam read
    :param window_start: starting position of feature window
    :return: (WINDOW_SIZE x 1) binary matrix marking SNP position
    """
    # if SNP exists in read, get position
    snp_pos_in_read = get_snp_pos_in_read(read)
    # create zero vector
    snp_mask_matrix = np.zeros((deepSNP.WINDOW_SIZE, 1))
    # if we have a snp, mark 1 at SNP location in read
    if snp_pos_in_read >= 0:
        snp_pos_in_matrix = (read.reference_start + snp_pos_in_read) - window_start
        # print "snp_pos_in_matrix:", snp_pos_in_matrix
        # don't mark SNP if it occurs outside of our window
        if snp_pos_in_matrix < deepSNP.WINDOW_SIZE and snp_pos_in_matrix >= 0:
            snp_mask_matrix[snp_pos_in_matrix] = 1
    return snp_mask_matrix


def print_snp_pos_feature_matrix(snp_mask_feat_matrix):
    """
    Prints a string of zeros, except 1 at SNP location

    :param snp_mask_feat_matrix: binary mask matrix with SNP location
    :return: printable string
    """
    mask_string = ""
    for val in snp_mask_feat_matrix:
        mask_string += str(int(val))
    print mask_string
    return mask_string


def get_snp_pos_in_read(read):
    """
    Use NM and MD tag to extract SNP positions in read

    :param read: pysam read
    :return: integer position of SNP in read
    """
    # get number of mismatches in read (edit distance)
    num_mismatches = read.get_tag("NM")

    # TODO: be able to handle reads w/ multiple SNPs? or toss those reads?
    # TODO: if so, replace all ACGT with X, split, then add prev val to current to get pos
    # if we have a positive number of mismatches, we have SNPs!
    if num_mismatches == 1:
        md_flag = read.get_tag("MD")
        # print md_flag
        # try to split using base character
        for b in ['A', 'C', 'T', 'G']:
            # if we can split the string, string form like [0-9]+[ACGT]
            # where the leading number is #matches before SNP and since
            # python is zero based this should give us the index of SNP

            # NOTE: len(md_flag) < 6 prevents things like 11A2T1T19 from getting through
            if len(md_flag) < 6:
                if len(md_flag.split(b)) == 2:
                    # TODO: handle deletions?
                    # check for deletion character
                    if len(md_flag.split("^")) == 2:
                        return -1
                    # if read is reversed, need to flip
                    if read.is_reverse:
                        return int(md_flag.split(b)[1])
                    else:
                        return int(md_flag.split(b)[0])
    else:
        return -1
