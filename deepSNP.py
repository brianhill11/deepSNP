#!/usr/bin/python

import pysam
import numpy as np
import sys
from sklearn.feature_extraction import DictVectorizer
import cPickle as pickle
import os

sys.path.append("/usr/local")
from caffe2.python import core, utils, workspace
from caffe2.proto import caffe2_pb2

#######################################
# GLOBALS
#######################################
DEBUG = 0
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
        #print "new matrix shape:", new_mat.shape
        return new_mat
    # otherwise this is the initial stacking of 2 2D matrices
    else:
        new_mat = np.vstack([top_matrix[np.newaxis, ...], bottom_matrix[np.newaxis, ...]])
        #print "new matrix shape:", new_mat.shape
        return new_mat


def concat_feature_matrices(feat_mat1, feat_mat2):
    """
    Join 2 feature matrices together column-wise
    
    :param feat_mat1: (WINDOW_SIZE x F1) matrix
    :param feat_mat2: (WINDOW_SIZE x F2) matrix
    :return: (WINDOW_SIZE x F1+F2) matrix
    """
    #print "mat1 shape:", feat_mat1.shape
    #print "mat2 shape:", feat_mat2.shape
    try:
        return np.concatenate((feat_mat1, feat_mat2), axis=1)
    except ValueError as e:
        print e
        print "matrix1 shape:", feat_mat1.shape
        print "matrix2 shape:", feat_mat2.shape
        exit(-100)

def create_feat_mat_read(read, window_start, window_end):
    """
    
    each feature-creating function we call shall return a
    (WINDOW_SIZE x F) matrix, where F is the num of features
    
    :param read: pysam read
    :param window_start: starting position of feature window
    :param window_end: ending position of feature window
    :return: (WINDOW_SIZE x F) matrix
    """

    snp_mask_feat_mat = snp_mask_feature_matrix(read, window_start)
    if DEBUG:
        print_snp_mask_feature_matrix(snp_mask_feat_mat)
    #
    bp_feat_mat = base_pair_feature_matrix(read, window_start)
    return concat_feature_matrices(bp_feat_mat, snp_mask_feat_mat)


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
    # calculate dimensions to left and right of read for padding zeros
    num_pad_left = np.maximum(0, read.reference_start - window_start)
    # NOTE: pysam reference_length = reference_end - reference_start
    # but this does not necessarily mean the query sequence is that length
    ref_end = read.reference_start + len(read.query_sequence)
    num_pad_right = np.maximum(0, window_end - ref_end)
    # check if we have only part of the read in the window
    normalized_offset = window_start - read.reference_start
    seq_start = np.maximum(normalized_offset, 0)
    seq_end = np.minimum(normalized_offset + WINDOW_SIZE, len(read.query_sequence))
    #print "[", seq_start, seq_end, "]"
    base_pair_seq = read.query_sequence[seq_start:seq_end]
    # create the (READ_LENGTH x 4) matrix encoding base pairs
    one_hot_base_mat = vectorize_base_seq(base_pair_seq)
    #print "1-hot shape:", one_hot_base_mat.shape

    # we are padding 1st dimension on left and right with zeros.
    # (0, 0) says don't pad on 2nd dimension before or after
    base_pair_feat_matrix = np.lib.pad(one_hot_base_mat,
                      ((num_pad_left, num_pad_right), (0, 0)),
                      'constant', constant_values=(0,))
    if base_pair_feat_matrix.shape[0] != WINDOW_SIZE:
        print "ERROR: base pair feat matrix not size of window"
        print "len(bps)", len(base_pair_seq)
        print "window start:", window_start
        print "window end:", window_end
        print "read start:", read.reference_start
        print "read end:", read.reference_end
        print "read len:", read.reference_length
        print "len(qs):", len(read.query_sequence)
        print "shape:", base_pair_feat_matrix.shape
        print "num_pad_left:", num_pad_left
        print "num_pad_right:", num_pad_right
        print "seq_start:", seq_start
        print "seq_end:", seq_end
        exit(-101)
    #print "BP feat matrix shape: ", base_pair_feat_matrix.shape
    if DEBUG:
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
    Creates vector of zeros, except 1 at SNP position
    
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
        #print "snp_pos_in_matrix:", snp_pos_in_matrix
        # don't mark SNP if it occurs outside of our window
        if snp_pos_in_matrix < WINDOW_SIZE and snp_pos_in_matrix >= 0:
            snp_mask_matrix[snp_pos_in_matrix] = 1
    return snp_mask_matrix[..., np.newaxis]


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


def get_candidate_snps(vcf_file):
    """
    Create {(chromosome, position): (alleles)} mapping
    of SNPS from VCF file
    
    :param vcf_file: path to VCF file
    :return: {(chromosome, position): (alleles)} SNP mapping
    """
    candidate_snps = {}

    # open VCF file
    vcf = pysam.VariantFile(vcf_file, 'r')
    # fetch all variants
    vcf_iter = vcf.fetch()
    for record in vcf_iter:
        # tuple (chromosome, position) is primary key
        location = (record.chrom, record.pos)
        # if we haven't seen this location yet, add to the dict
        if location not in candidate_snps:
            # add allele tuple
            candidate_snps[location] = record.alleles
        # otherwise, we're overwriting existing SNP
        else:
            print "Found duplicate SNP at", location, " and we're overwriting..."
            # create list of mappings from SNP position to alleles (both ref & alternate)
            candidate_snps[location] = record.alleles
    return candidate_snps


def get_real_snps(truth_file):
    """
    Reads in ground truth output (from wgsim) into dict mapping 
    chromosome to list of {pos -> alleles} maps.
    
    Ground truth file should have format like:
    Col1: chromosome
    Col2: position
    Col3: original base
    Col4: new base (IUPAC codes indicate heterozygous)
    Col5: which genomic copy/haplotype
    
    :param truth_file: output file from wgsim
    :return: {chrom1 -> [{pos -> allele}], ...}
    """
    real_snps = {}

    # open ground truth file
    with open(truth_file, 'r') as truth_f:
        for line in truth_f:
            snp = line.split('\t')
            chrom = snp[0]
            position = snp[1]
            location = (chrom, position)
            ref_allele = snp[2]
            alt_allele = snp[3]
            alleles = (ref_allele, alt_allele)
            # if we haven't seen this location yet, add to list
            if location not in real_snps:
                # initialize to empty list
                real_snps[location] = alleles
            # otherwise, we're overwriting existing SNP
            else:
                print "Found duplicate SNP at", location, " and we're overwriting..."
                real_snps[location] = alleles
        return real_snps


def write_caffe2_db(db_type, db_name, features, labels):
    """
    
    
    Based on tutorial from https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/create_your_own_dataset.ipynb
    
    
    :param db_type: 
    :param db_name: 
    :param features: 
    :param labels: 
    :return: 
    """
    db = core.C.create_db(db_type, db_name, core.C.Mode.write)
    transaction = db.new_transaction()
    # iterate through feature matrix
    for i in range(0, len(features)):
        feature_and_label = caffe2_pb2.TensorProtos()
        feature_and_label.protos.extend([
            utils.NumpyArrayToCaffe2Tensor(features[i]),
            utils.NumpyArrayToCaffe2Tensor(labels[i])
        ])
        transaction.put('train_%03d'.format(i), feature_and_label.SerializeToString())
    # close transaction and DB
    del transaction
    del db


def main():
    in_bam = sys.argv[1]
    in_vcf = sys.argv[2]
    in_ref = sys.argv[3]
    in_truth = sys.argv[4]
    bam_f = pysam.AlignmentFile(in_bam, "rb")
    ref_f = pysam.Fastafile(in_ref)

    real_snps = {}
    real_snps_pickle = os.path.splitext(in_truth)[0] + ".pickle"
    if os.path.isfile(real_snps_pickle):
        print "Loading", real_snps_pickle
        real_snps = pickle.load(open(real_snps_pickle, "rb"))
    else:
        real_snps = get_real_snps(in_truth)
        print "Creating", real_snps_pickle, " file"
        pickle.dump(real_snps, open(real_snps_pickle, "wb"))
    #print real_snps

    candidate_snps = {}
    candidate_snps_pickle = os.path.splitext(in_vcf)[0] + ".pickle"
    # get {(chr, pos): (ref, alt)} mapping
    if os.path.isfile(candidate_snps_pickle):
        print "Loading", candidate_snps_pickle
        candidate_snps = pickle.load(open(candidate_snps_pickle, "rb"))
    else:
        candidate_snps = get_candidate_snps(in_vcf)
        print "Creating", candidate_snps_pickle, " file"
        pickle.dump(candidate_snps, open(candidate_snps_pickle, "wb"))

    num_snps = 0
    feature_matrices = []
    labels = []
    for location, alleles in candidate_snps.items():
        if num_snps % 100000 == 0:
            print "Num SNPs processed:", num_snps
        # location is (chromosome, position) tuple
        chromosome = location[0]
        pos = location[1]
        # our feature window is centered on SNP position
        # max() makes sure we don't have negative index near start of contig
        window_start = max(pos - (WINDOW_SIZE / 2), 0)
        window_end = window_start + WINDOW_SIZE

        snp_feat_matrix = np.empty([WINDOW_SIZE, 1])
        first_read = True
        if DEBUG:
            print "location:", location
        #overlapping_snp_pos = get_snps_in_window(snp_list, window_start, window_end)
        #snp_pileup_cols = bam_f.pileup(chromosome, window_start, window_end, fastafile=ref_f)
        window_reads = bam_f.fetch(chromosome, window_start, window_end)
        ref_bases = ref_f.fetch(chromosome, window_start, window_end)
        if DEBUG:
            print ref_bases.upper()
        num_reads = 0
        for read in window_reads:

            if read.has_tag("MD") and is_usable_read(read):
                num_reads += 1
                read_feature_matrix = create_feat_mat_read(read, window_start, window_end)
                # if this is our first read, overwrite snp_feat_matrix
                if first_read:
                    snp_feat_matrix = read_feature_matrix
                    first_read = False
                # else, stack read's feature matrix with prev reads
                else:
                    snp_feat_matrix = vert_stack_matrices(snp_feat_matrix, read_feature_matrix)
        #print "Num reads processed:", num_reads
        if num_reads > 0:
            num_snps += 1
            feature_matrices.append(snp_feat_matrix)
            if location in real_snps:
                labels.append(1)
            else:
                labels.append(0)


        #print ""
        #print snp_feat_matrix
        #print "feature matrix dims:", snp_feat_matrix.shape
        #if num_snps == 25:
    print "Num feature matrices: ", len(feature_matrices)
    print "Num labels: ", len(labels)
    labels = np.array(labels)
    write_caffe2_db("minidb", "train.minidb", feature_matrices, labels)
    print "Sum labels:", np.sum(labels)
    exit(0)


if __name__ == "__main__":
    main()
