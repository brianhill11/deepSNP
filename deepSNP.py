#!/usr/bin/python

import pysam
import numpy as np
import sys
import cPickle as pickle
import os
import argparse
import time
import random
import logging
from collections import OrderedDict

sys.path.append("/usr/local")
from caffe2.python import core, utils, workspace
from caffe2.proto import caffe2_pb2

# local imports
from deepSNP_utils import *
from base_feature import *
from snp_pos_feature import *
from map_qual_feature import *
from base_pos_feature import *
from base_qual_feature import *


#######################################
# GLOBALS
#######################################
DEBUG = 0
# feature matrix dimensions
WINDOW_SIZE = 200
NUM_ROWS = 30
# depth of feature matrix
FEATURE_DEPTH = 7
# required minimum number of reads in a feature window
MIN_NUM_READS = 4
# number of classes to predict (in this case, 2, SNP/not SNP)
NUM_CLASSES = 2
# number of training examples
NUM_TRAINING_EXAMPLES = 1000000
# we gather batches of training examples, evenly distributed
# between classes, and write batches to disk to prevent
# generating gigantic lists containing all training examples
NUM_TRAINING_EXAMPLES_PER_BATCH = 1024*8
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
    #snp_pos_feat_mat = snp_pos_feature_matrix(read, window_start)
    #if DEBUG > 0:
    #    print_snp_pos_feature_matrix(snp_pos_feat_mat)
    #    print read.query_alignment_qualities
    #feat_mat = concat_feature_matrices(feat_mat, snp_pos_feat_mat)
    # FEATURE 4: base position within read
    base_pos_feat_matrix = base_pos_feature_matrix(read, window_start)
    if DEBUG > 1:
        print_base_pos_feature_matrix(base_pos_feat_matrix)
    feat_mat = concat_feature_matrices(feat_mat, base_pos_feat_matrix)
    # FEATURE 5: base quality score for each base
    base_qual_feat_matrix = base_qual_feature_matrix(read, window_start)
    if DEBUG > 1:
        print_base_qual_feature_matrix(base_qual_feat_matrix)
    feat_mat = concat_feature_matrices(feat_mat, base_qual_feat_matrix)
    return feat_mat


def get_candidate_snps(vcf_file):
    """
    Create {(chromosome, position): (alleles)} mapping
    of SNPs from VCF file

    :param vcf_file: path to VCF file
    :return: {(chromosome, position): (alleles)} SNP mapping
    """
    candidate_snps = OrderedDict()

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
    real_snps = OrderedDict()

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
    zero_vector = np.zeros((WINDOW_SIZE, FEATURE_DEPTH - feat_mat.shape[1]))
    feat_mat = concat_feature_matrices(feat_mat, zero_vector)
    #if deepSNP.DEBUG:
    #    print print_base_pair_feature_matrix(ref_vector)
    return feat_mat


def write_caffe2_db(db_type, db_name, features, labels, snp_num):
    """
    Writes pairs of feature matrices and labels as protobuf objects
    to a minidb file to be used as input to Caffe2

    The order of the feature matrices/labels are shuffled randomly

    Based on tutorial from https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/create_your_own_dataset.ipynb

    :param db_type: type of database file to write (ex. minidb, leveldb, lmdb)
    :param db_name: name of the database file
    :param features: list of Numpy feature matrices
    :param labels: list of Numpy labels
    :return: None
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
    parser.add_argument('-l', '--log', dest='log_level', required=False, default="INFO", help='Set log level (ex:, DEBUG, INFO, WARNING, etc.)')
    args = parser.parse_args()

    # set logging config
    log_level_num = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level_num, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    log = logging.getLogger('deepSNP')
    log.setLevel(log_level_num)

    in_bam = args.bam
    in_vcf = args.vcf
    in_ref = args.ref
    in_truth = args.truth
    db_file = args.db_file
    bam_f = pysam.AlignmentFile(in_bam, "rb")
    ref_f = pysam.Fastafile(in_ref)

    real_snps = OrderedDict()
    real_snps_pickle = os.path.splitext(in_truth)[0] + ".pickle"
    if os.path.isfile(real_snps_pickle):
        log.info("Loading %s",real_snps_pickle)
        real_snps = pickle.load(open(real_snps_pickle, "rb"))
    else:
        #TODO: change get_candidate_snps to name not tied to candidate SNPs, instead VCF
        # if we have VCF file, parse VCF
        if os.path.splitext(in_truth)[1] == '.vcf':
            # get_candidate_snps parses VCF
            real_snps = get_candidate_snps(in_truth)
        # else assume BED-like file
        else:
            real_snps = get_real_snps(in_truth)
        log.info("Creating file: %s", real_snps_pickle)
        pickle.dump(real_snps, open(real_snps_pickle, "wb"))

    candidate_snps = OrderedDict()
    candidate_snps_pickle = os.path.splitext(in_vcf)[0] + ".pickle"
    # get {(chr, pos): (ref, alt)} mapping
    #if os.path.isfile(candidate_snps_pickle):
    #    log.info("Loading %s", candidate_snps_pickle)
    #    candidate_snps = pickle.load(open(candidate_snps_pickle, "rb"))
    #else:
    #    candidate_snps = get_candidate_snps(in_vcf)
    #    log.info("Creating file: %s", candidate_snps_pickle)
    #    pickle.dump(candidate_snps, open(candidate_snps_pickle, "wb"))

    num_snps = 0
    num_positive_train_ex = 0
    num_negative_train_ex = 0
    total_num_reads = 0
    total_num_positive_train_ex = 0
    total_num_negative_train_ex = 0
    num_reads_positive_train_ex = 0
    num_reads_negative_train_ex = 0
    num_snps_in_batch = 0
    feature_matrices = []
    labels = []
    log.info("Num candidate SNPs: %s" % len(candidate_snps))
    log.info("Num true SNPs: %s", len(real_snps))
    start_time = time.time()

    num_iters = np.minimum(NUM_TRAINING_EXAMPLES, len(real_snps))
    #for i in range(num_iters):
    vcf_iter = pysam.VariantFile(in_vcf, 'r')
    # fetch all variants
    for record in vcf_iter.fetch():
        # tuple (chromosome, position) is primary key
        location = (record.chrom, record.pos)
        if location in real_snps:
            snp_label = 1
        else: 
            snp_label = 0
        # if we already have too many negative training examples for this batch, continue loop
        if snp_label == 0 and num_negative_train_ex == NUM_TRAINING_EXAMPLES_PER_BATCH / 2: 
            continue
    # switch class every other iteration
    #    if i % 2 == 0:
    #        location = random.choice(list(real_snps))
    #        snp_label = 1
    #    else:
    #        location = random.choice(list(candidate_snps))
    #        if location not in real_snps:
    #            snp_label = 0
    #        else:
    #            log.warning("randomly chosen potential SNP is true SNP! Location: (%s, %s)", location[0], location[1])
    #            snp_label = 1

        if num_snps % 5000 == 0 and num_snps > 0:
            cur_time = time.time()
            avg_time_per_snp = (cur_time - start_time) / float(num_snps)
            log.info("============ Num SNPs processed: %s ============", num_snps)
            log.info("Elapsed time: %.2f mins", (cur_time - start_time) / 60.0)
            log.info("Estimated time remaining: %.2f mins", ((len(real_snps) - num_snps) * avg_time_per_snp) / 60.0)
            log.info("Avg time per SNP: %.2f sec", avg_time_per_snp)
            log.info("Avg num reads per SNP: %s", total_num_reads / float(num_snps))
            log.info("Num positive training examples: %s", total_num_positive_train_ex)
            log.info("Num reads per positive training example: %s", float(num_reads_positive_train_ex) / total_num_positive_train_ex)
            log.info("Num negative training examples: %s", total_num_negative_train_ex)
            log.info("Num reads per negative training example: %s", float(num_reads_negative_train_ex) / total_num_negative_train_ex)
            log.info("============ %.2f percent complete ============", (float(num_snps) / NUM_TRAINING_EXAMPLES) * 100.0)
        if num_snps > NUM_TRAINING_EXAMPLES:
            log.info("Reached max number of training examples")
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
        log.debug("location: %s", location)

        window_reads = bam_f.fetch(chromosome, window_start, window_end)
        # get reference feature matrix using reference sequence
        ref_feat_matrix = ref_feature_matrix(ref_f, chromosome, window_start, window_end)

        if DEBUG:
            ref_bases = ref_f.fetch(chromosome, window_start, window_end)
            print ref_bases.upper()
        num_reads = 0
        for read in window_reads:
            # check to make sure read is useful and we haven't hit max num reads
            if is_usable_read(read) and num_reads < (NUM_ROWS - 1):
            #if read.has_tag("MD") and is_usable_read(read) and num_reads < (NUM_ROWS - 1):
                read_feature_matrix = create_feat_mat_read(read, window_start, window_end)
                # if this is our first read, stack read feature matrix under reference feature matrix
                if first_read:
                    snp_feat_matrix = vert_stack_matrices(ref_feat_matrix, read_feature_matrix)
                    first_read = False
                # else, stack read's feature matrix with prev reads
                else:
                    snp_feat_matrix = vert_stack_matrices(snp_feat_matrix, read_feature_matrix)
                num_reads += 1

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

            feature_matrices.append(snp_feat_matrix)
            
            labels.append(snp_label)

            if snp_label == 1:
                num_positive_train_ex += 1
                total_num_positive_train_ex += 1
                num_reads_positive_train_ex += num_reads
            else:
                num_negative_train_ex += 1
                total_num_negative_train_ex += 1
                num_reads_negative_train_ex += num_reads
            num_snps_in_batch += 1
            num_snps += 1
            total_num_reads += num_reads

            # if we have a full batch of evenly distributed examples, write to file
            if num_snps_in_batch == NUM_TRAINING_EXAMPLES_PER_BATCH:
                log.info("writing batch to minidb")
                cur_time = time.time()
                log.info("Num SNPs processed: %s", num_snps)
                log.info("Elapsed time: %s", cur_time - start_time)
                write_caffe2_db("minidb", db_file, feature_matrices, np.array(labels), num_snps)
                # reset 
                num_snps_in_batch = 0
                num_positive_train_ex = 0
                num_negative_train_ex = 0
                feature_matrices = []
                labels = []

    #print "Num feature matrices: ", len(feature_matrices)
    #print "Num labels: ", len(labels)
    #labels = np.array(labels)
    #write_caffe2_db("minidb", "train.minidb", feature_matrices, labels)
    #print "Sum of labels:", np.sum(labels)
    #print "Avg #reads per SNP:", total_num_reads / len(feature_matrices)


    print "##########################################################"
    print "## FINAL SUMMARY STATISTICS"
    print "##########################################################"
    cur_time = time.time()
    print "Elapsed time:", cur_time - start_time
    print "Total number of SNPs:", num_snps
    print "Avg #reads per SNP:", total_num_reads / float(num_snps)
    print "Num positive training examples:", total_num_positive_train_ex
    print "Num reads per positive training example:", float(num_reads_positive_train_ex) / total_num_positive_train_ex
    print "Num negative training examples:", total_num_negative_train_ex
    print "Num reads per negative training example:", float(num_reads_negative_train_ex) / total_num_negative_train_ex
    print "##########################################################"
    print "Done!"
    exit(0)


if __name__ == "__main__":
    main()
