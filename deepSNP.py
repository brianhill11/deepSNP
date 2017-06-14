#!/usr/bin/python

import pysam
import numpy as np
import sys
import cPickle as pickle
import os
import argparse
import time
import random

sys.path.append("/usr/local")
from caffe2.python import core, utils, workspace
from caffe2.proto import caffe2_pb2

# local imports
from deepSNP_utils import *
from base_feature import *
from snp_pos_feature import *
from map_qual_feature import *
from base_pos_feature import *



#######################################
# GLOBALS
#######################################
DEBUG = 0
# feature matrix dimensions
WINDOW_SIZE = 100
NUM_ROWS = 30
# required minimum number of reads in a feature window
MIN_NUM_READS = 4
# number of classes to predict (in this case, 2, SNP/not SNP)
NUM_CLASSES = 2
# number of training examples
NUM_TRAINING_EXAMPLES = 200000
# we gather batches of training examples, evenly distributed
# between classes, and write batches to disk to prevent
# generating gigantic lists containing all training examples
NUM_TRAINING_EXAMPLES_PER_BATCH = 1024
NUM_TRAINING_EX_PER_CLASS = NUM_TRAINING_EXAMPLES_PER_BATCH / NUM_CLASSES
# number of testing examples
NUM_TESTING_EXAMPLES = 100000

def create_feat_mat_read(read, window_start, window_end):
    """
    
    each feature-creating function we call shall return a
    (WINDOW_SIZE x F) matrix, where F is the num of features
    
    :param read: pysam read
    :param window_start: starting position of feature window
    :param window_end: ending position of feature window
    :return: (WINDOW_SIZE x F) matrix
    """
    # FEATURE 1: base character
    bp_feat_mat = base_pair_feature_matrix(read, window_start)
    # FEATURE 2: mapping quality
    map_qual_mat = map_qual_feature_matrix(read, window_start)
    if DEBUG > 1:
        print_map_qual_feature_matrix(map_qual_mat)
    feat_mat = concat_feature_matrices(bp_feat_mat, map_qual_mat)
    # FEATURE 3: SNP position (marked by 1)
    snp_pos_feat_mat = snp_pos_feature_matrix(read, window_start)
    if DEBUG > 0:
        print_snp_pos_feature_matrix(snp_pos_feat_mat)
    feat_mat = concat_feature_matrices(feat_mat, snp_pos_feat_mat)
    # FEATURE 4: base position within read
    base_pos_feat_matrix = base_pos_feature_matrix(read, window_start)
    if DEBUG > 1:
        print_base_pos_feature_matrix(base_pos_feat_matrix)
    feat_mat = concat_feature_matrices(feat_mat, base_pos_feat_matrix)
    return feat_mat


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


def ref_feature_matrix(ref_f, chromosome, window_start, window_end):
    """
    Create feature matrix representing reference sequence. We do this by
    one-hot encoding reference sequence (as we do with alignment sequences)
    and fill in quality scores as very high value since we're fairly confident
    in the reference sequence. We will set the rest of the matrix to zeros,
    since many of the other features are not useful for the reference sequence.

    :param ref_f: reference Fasta file
    :param chromosome: chromosome string
    :param window_start: starting position of feature window
    :param window_end: ending position of feature window
    :return: (WINDOW_SIZE x F) matrix, (F is number of features)
    """
    # get reference sequence
    ref_bases = ref_f.fetch(chromosome, window_start, window_end)
    # create feature matrix by one-hot-encoding reference sequence, and use super-high Qual score
    ref_vector = vectorize_base_seq(ref_bases)
    # we will use 60 as a very high quality score
    qual_vector = np.full([WINDOW_SIZE, 1], 60)
    # stack feature vectors to make feature matrix
    feat_mat = concat_feature_matrices(ref_vector, qual_vector)
    # pad the rest of the matrix with zeros to make it the same size as a read feature matrix
    # currently, this dimension is 7
    zero_vector = np.zeros((WINDOW_SIZE, 7 - feat_mat.shape[1]))
    feat_mat = concat_feature_matrices(feat_mat, zero_vector)
    #if deepSNP.DEBUG:
    #    print print_base_pair_feature_matrix(ref_vector)
    return feat_mat


def write_caffe2_db(db_type, db_name, features, labels, snp_num):
    """
    
    
    Based on tutorial from https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/create_your_own_dataset.ipynb
    
    
    :param db_type: 
    :param db_name: 
    :param features: 
    :param labels: 
    :return: 
    """
    db = core.C.create_db(db_type, db_name, core.C.Mode.write)


    assert(len(features) == len(labels))
    # use random permutation to shuffle examples in batch 
    random_indices = np.random.permutation(len(features))
    # iterate through feature matrix
    for i in random_indices:
        transaction = db.new_transaction()
        feature_and_label = caffe2_pb2.TensorProtos()
        feature_and_label.protos.extend([
            utils.NumpyArrayToCaffe2Tensor(features[i]),
            utils.NumpyArrayToCaffe2Tensor(labels[i])
        ])
        #print("tensor proto:")
        #print(str(feature_and_label))
        transaction.put('train_%03d'.format(snp_num + i), feature_and_label.SerializeToString())
        del transaction
    # close transaction and DB
    del db


def main():
    parser = argparse.ArgumentParser(
        description="Creates a training/testing feature/label mapping from an \
         input BAM file using a VCF file containing potential SNP locations")
    parser.add_argument('-b', '--bam-file', dest='bam', required=True, help='Input BAM file')
    parser.add_argument('-r', '--reference-file', dest='ref', required=True, help='Reference fasta file for input BAM')
    parser.add_argument('-v', '--vcf-file', dest='vcf', required=True, help='Input VCF file containing potential SNPs')
    parser.add_argument('-t', '--truth-file', dest='truth', required=True, help='Output of wgsim containing ground truth SNPs')
    parser.add_argument('-d', '--db-file', dest='db_file', required=True, help='Output Caffe DB file name')
    args = parser.parse_args()

    in_bam = args.bam
    in_vcf = args.vcf
    in_ref = args.ref
    in_truth = args.truth
    db_file = args.db_file
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
    num_positive_train_ex = 0
    num_negative_train_ex = 0
    total_num_reads = 0
    num_snps_in_batch = 0
    feature_matrices = []
    labels = []
    print "Num candidate SNPs:", len(candidate_snps)
    print "Num true SNPs:", len(real_snps)
    start_time = time.time()
    #for location, alleles in candidate_snps.items():
    #for location, alleles in real_snps.items():
    num_iters = np.minimum(NUM_TRAINING_EXAMPLES, len(real_snps))
    for i in range(num_iters):
        # switch class every other iteration
        if i % 2 == 0:
            location = random.choice(list(real_snps))
            snp_label = 1
            num_positive_train_ex += 1
        else:
            location = random.choice(list(candidate_snps))
            if location not in real_snps:
                snp_label = 0
                num_negative_train_ex += 1
            else:
                print "WARNING: randomly chosen SNP is true SNP"
                snp_label = 1
                num_positive_train_ex += 1

        if num_snps % 100000 == 0:
            cur_time = time.time()
            print "Num SNPs processed:", num_snps
            print "Elapsed time:", cur_time - start_time 
            print "Num positive training examples:", num_positive_train_ex
            print "Num negative training examples:", num_negative_train_ex
        if num_snps > NUM_TRAINING_EXAMPLES:
            print "Reached max number of training examples"
            break
        # location is (chromosome, position) tuple
        chromosome = location[0]
        pos = int(location[1])
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
        # get reference feature matrix using reference sequence
        ref_feat_matrix = ref_feature_matrix(ref_f, chromosome, window_start, window_end)

        if DEBUG:
            ref_bases = ref_f.fetch(chromosome, window_start, window_end)
            print ref_bases.upper()
        num_reads = 0
        for read in window_reads:
            # check to make sure read is useful and we haven't hit max num reads
            if read.has_tag("MD") and is_usable_read(read) and num_reads < (NUM_ROWS - 1):
                read_feature_matrix = create_feat_mat_read(read, window_start, window_end)
                # if this is our first read, stack read feature matrix under reference feature matrix
                if first_read:
                    snp_feat_matrix = vert_stack_matrices(ref_feat_matrix, read_feature_matrix)
                    first_read = False
                # else, stack read's feature matrix with prev reads
                else:
                    snp_feat_matrix = vert_stack_matrices(snp_feat_matrix, read_feature_matrix)
                num_reads += 1
        #print "Num reads processed:", num_reads
        if num_reads > MIN_NUM_READS:
            # calculate number of empty rows we need to add to matrix (minus one for Ref seq)
            num_empty_rows = NUM_ROWS - num_reads - 1

            if num_empty_rows > 0:
                # if we only have one read, snp_feat_matrix is still 2D matrix
                # TODO: should we just throw out SNPs with only 1 read covering?
                if snp_feat_matrix.ndim != 3:
                    snp_feat_matrix = snp_feat_matrix[np.newaxis, ...]
                # empty_rows matrix should have dims: (#emptyrows x WINDOW_SIZE x #features)
                empty_rows = np.zeros((num_empty_rows, WINDOW_SIZE, snp_feat_matrix.shape[2]))

                snp_feat_matrix = np.vstack([snp_feat_matrix, empty_rows])

            assert snp_feat_matrix.shape[0] == NUM_ROWS, "SNP feature matrix shape[0]: %r" % snp_feat_matrix.shape[0]
            assert snp_feat_matrix.shape[1] == WINDOW_SIZE, "SNP feature matrix shape[1]: %r" % snp_feat_matrix.shape[1]

            # for GPU processing, we need to change matrix shape from HWC -> CHW, where 
            # H: height (#rows), W: width (#cols), C: channels (#features)
            snp_feat_matrix = snp_feat_matrix.swapaxes(1, 2).swapaxes(0, 1)
            
            # case: True SNP
            # make sure our class distributions are even
            #if num_positive_train_ex < NUM_TRAINING_EX_PER_CLASS:
            feature_matrices.append(snp_feat_matrix)
            
            labels.append(snp_label)
            #write_caffe2_db("minidb", db_file, snp_feat_matrix, np.array([1]), num_snps)
            #num_positive_train_ex += 1
            num_snps_in_batch += 1
            num_snps += 1
            total_num_reads += num_reads
            # case: False SNP
            #if num_negative_train_ex < NUM_TRAINING_EX_PER_CLASS:
            #feature_matrices.append(snp_feat_matrix)
            #labels.append(0)
            ##write_caffe2_db("minidb", db_file, snp_feat_matrix, np.array([0]), num_snps)
            #num_negative_train_ex += 1
            #num_snps_in_batch += 1
            #num_snps += 1
            #total_num_reads += num_reads
            
            #print "num pos:", num_positive_train_ex
            #print "num neg:", num_negative_train_ex
            # if we have a full batch of evenly distributed examples, write to file
            if num_snps_in_batch == NUM_TRAINING_EXAMPLES_PER_BATCH:
                print "writing batch to minidb"
                cur_time = time.time()
                print "Num SNPs processed:", num_snps
                print "Elapsed time:", cur_time - start_time
            write_caffe2_db("minidb", db_file, feature_matrices, np.array(labels), num_snps)
                # reset 
                num_snps_in_batch = 0
                num_positive_train_ex = 0
                num_negative_train_ex = 0
                feature_matrices = []
                labels = []


        #print ""
        #print snp_feat_matrix
        #print "feature matrix dims:", snp_feat_matrix.shape
        #if num_snps == 25:
    #print "Num feature matrices: ", len(feature_matrices)
    #print "Num labels: ", len(labels)
    #labels = np.array(labels)
    #write_caffe2_db("minidb", "train.minidb", feature_matrices, labels)
    #print "Sum of labels:", np.sum(labels)
    #print "Avg #reads per SNP:", total_num_reads / len(feature_matrices)
    print "Total number of SNPs:", num_snps
    print "Avg #reads per SNP:", total_num_reads / num_snps
    exit(0)


if __name__ == "__main__":
    main()
