#!/usr/bin/python

import pysam
import numpy as np
import sys
from sklearn.feature_extraction import DictVectorizer

#######################################
# GLOBALS
#######################################
MIN_COUNT = 1
MIN_FRACTION = 1. / 20
WINDOW_SIZE = 80
NUM_ROWS = 30
NUM_FEATURES = 4
feature_pos = {'qual': 0, 'mqual': 1, 'pos': 2, 'read_num': 3}
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
        print "new matrix shape:", new_mat.shape
        return new_mat
    # otherwise this is the initial stacking of 2 2D matrices
    else:
        new_mat = np.vstack([top_matrix[np.newaxis, ...], bottom_matrix[np.newaxis, ...]])
        print "new matrix shape:", new_mat.shape
        return new_mat


def create_feat_mat_read(read, window_start, window_end):
    """
    
    each feature-creating function we call shall return a
    # (WINDOW_SIZE x F) matrix, where F is the num of features
    
    :param read: 
    :param window_start: 
    :param window_end: 
    :return: 
    """

    snp_mask_feat_mat = snp_mask_feature_matrix(read, window_start)
    print_snp_mask_feature_matrix(snp_mask_feat_mat)
    #
    bp_feat_mat = base_pair_feature_matrix(read, window_start)
    return bp_feat_mat





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
    matrix_shape = [NUM_ROWS, WINDOW_SIZE, NUM_FEATURES]
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

                print "ref seq:", read.alignment.query_alignment_sequence
                print vectorize_base_seq(read.alignment.query_alignment_sequence)
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
    window_end = window_start + WINDOW_SIZE
    # create the (READ_LENGTH x 4) matrix encoding base pairs
    one_hot_base_mat = vectorize_base_seq(read.query_sequence)
    #print "1-hot shape:", one_hot_base_mat.shape
    # calculate dimensions to left and right of read for padding zeros
    num_pad_left = np.maximum(0, read.reference_start - window_start)
    num_pad_right = np.maximum(0, window_end - read.reference_end)
    # we are padding 1st dimension on left and right with zeros.
    # (0, 0) says don't pad on 2nd dimension before or after
    base_pair_feat_matrix = np.lib.pad(one_hot_base_mat,
                      ((num_pad_left, num_pad_right), (0, 0)),
                      'constant', constant_values=(0,))
    #print "BP feat matrix shape: ", base_pair_feat_matrix.shape
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


def snp_mask_feature_matrix(read, window_start):
    """
    
    :param read: pysam read
    :param window_start: starting position of feature window
    :return: (WINDOW_SIZE x 1) binary matrix marking SNP position
    """
    # if SNP exists in read, get position
    snp_pos_in_read = get_snp_pos_in_read(read)
    # create zero vector
    snp_mask_matrix = np.zeros(WINDOW_SIZE)
    # if we have a snp, mark 1 at SNP location in read
    if snp_pos_in_read >= 0:
        snp_pos_in_matrix = (read.reference_start + snp_pos_in_read) - window_start
        snp_mask_matrix[snp_pos_in_matrix] = 1
    return snp_mask_matrix


def print_snp_mask_feature_matrix(snp_mask_feat_matrix):
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
        #print md_flag
        # try to split using base character
        for b in ['A', 'C', 'T', 'G']:
            # if we can split the string, string form like [0-9]+[ACGT]
            # where the leading number is #matches before SNP and since
            # python is zero based this should give us the index of SNP

            # NOTE: len(md_flag) < 6 prevents things like 11A2T1T19 from getting through
            if len(md_flag) < 6:
                if len(md_flag.split(b)) == 2:
                    # if read is reversed, need to flip
                    if read.is_reverse:
                        return int(md_flag.split(b)[1])
                    else:
                        return int(md_flag.split(b)[0])
    else:
        return -1


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


# source: http://biorxiv.org/content/biorxiv/suppl/2016/12/21/092890.DC3/092890-1.pdf
#def is_candidate(counts, allele):
#    allele_count = counts[allele]
#    total_count = sum(counts.values())
#    return not is_reference_base(allele) and \
#        allele_count >= MIN_COUNT and \
#        allele_count / total_count >= MIN_FRACTION

def get_candidate_snps(vcf_file):
    candidate_snps_by_chrom = {}

    # open VCF file
    vcf = pysam.VariantFile(vcf_file, 'r')
    # fetch all variants
    vcf_iter = vcf.fetch()
    for record in vcf_iter:
        chrom = record.chrom
        # if we haven't seen this chromosome yet, add to the list
        if chrom not in candidate_snps_by_chrom:
            # initialize to empty dictionary
            candidate_snps_by_chrom[chrom] = []
        # otherwise, we're inserting into existing dict
        else:
            # create list of mappings from SNP position to alleles (both ref & alternate)
            candidate_snps_by_chrom[chrom].append({record.pos : record.alleles})
    return candidate_snps_by_chrom


# source: http://biorxiv.org/content/biorxiv/suppl/2016/12/21/092890.DC3/092890-1.pdf
def is_usable_read(read):
    return (len(read.get_tag("MD")) < 6 and
            not (read.is_duplicate or read.is_qcfail or
                 read.is_secondary or read.is_supplementary) and
            (not read.is_paired or read.is_properly_placed) and
            read.mapping_quality >= 10)


def main():
    in_bam = sys.argv[1]
    in_vcf = sys.argv[2]
    in_ref = sys.argv[3]
    bam_f = pysam.AlignmentFile(in_bam, "rb")
    ref_f = pysam.Fastafile(in_ref)

    # get "chr" -> [{pos : alleles}}] mapping
    candidate_snps = get_candidate_snps(in_vcf)
    for chromosome, snp_list in candidate_snps.items():
        for snp in snp_list:

            for pos, alleles in snp.items():
                #print chromosome, pos, " -> ", alleles
                window_start = pos - (WINDOW_SIZE / 2)
                window_end = window_start + WINDOW_SIZE

                #print "win start:", window_start, " win end: ", window_end
                overlapping_snp_pos = get_snps_in_window(snp_list, window_start, window_end)
                #snp_pileup_cols = bam_f.pileup(chromosome, window_start, window_end, fastafile=ref_f)
                window_reads = bam_f.fetch(chromosome, window_start, window_end)
                ref_bases = ref_f.fetch(chromosome, window_start, window_end)
                print ref_bases.upper()
                for read in window_reads:
                    if is_usable_read(read):
                        feature_matrix = create_feat_mat_read(read, window_start, window_end)
                        #feature_matrix = create_feature_matrix(snp_pileup_cols)
                        #print "feature matrix dims:", feature_matrix.shape
                        #print feature_matrix

                exit(0)
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
