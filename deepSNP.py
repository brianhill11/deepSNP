#!/usr/bin/python

import pysam
import numpy as np
import sys

window_size = 200
coverage = 30
num_features = 4
feature_pos = {'qual': 0, 'mqual': 1, 'pos': 2, 'read_num': 3}

def create_feature_matrix(snp_window_pileups):
    """
    Create a (# reads) x (# bases in window) x (# features)
    feature matrix by iterating through each read at each 
    column (base) in the window around the potential SNP
    and extracting features we want to use for deep learning 
    
    :param snp_window_pileups: pysam.PileupColumns object
    :return: Numpy matrix containing features
    """
    # matrix dims : coverage rows x window_size columns x num_features
    matrix_shape = [coverage, window_size, num_features]
    # initialize matrix to -1, which indicates absence of feature
    # TODO: is this a good numeric value to use in model?
    # NOTE: order='F' stores data in Fortran-contiguous order (column-wise)
    # as we will be writing to matrix in column order
    feature_matrix = np.full(matrix_shape, -1, order='F')

    y = 0  # column number
    # for each column (genomic position), iterate through reads
    for pileup_col in snp_window_pileups:
        x = 0  # row number
        # for each read at this position, get metrics we want
        if pileup_col.nsegments > 0:
            for read in pileup_col.pileups:
                # position of base in read
                pos = read.query_position
                # quality score of base call
                qual = read.alignment.query_qualities[pos]
                # mapping quality of read
                mqual = read.alignment.mapping_quality
                # is this read the first or second of the pair?
                read_num = 0
                if read.alignment.is_read2:
                    read_num = 1

                # write features to matrix
                feature_matrix[x, y, feature_pos['qual']] = qual
                feature_matrix[x, y, feature_pos['mqual']] = mqual
                feature_matrix[x, y, feature_pos['pos']] = pos
                feature_matrix[x, y, feature_pos['read_num']] = read_num

                # increment row number
                x += 1
        # increment column number
        y += 1

    return feature_matrix


def main():
    in_bam = sys.argv[1]
    bam_f = pysam.AlignmentFile(in_bam, "rb")
    test_snp_positions = [("chr1", 10240), ("chr1", 16459)]

    # TODO: scan to get list of potential SNPs

    # for each SNP, get pileup cols in window
    # and create feature matrix
    for chrom, snp_pos in test_snp_positions:
        window_start = snp_pos - (window_size / 2)
        window_end = window_start + window_size
        snp_pileup_cols = bam_f.pileup(chrom, window_start, window_end)
        feature_matrix = create_feature_matrix(snp_pileup_cols)
        print "feature matrix dims:", feature_matrix.shape
        print feature_matrix



if __name__ == "__main__":
    main()