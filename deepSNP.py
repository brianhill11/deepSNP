#!/usr/bin/python

import pysam
import numpy as np
import sys
import cPickle as pickle
import os
import argparse

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
WINDOW_SIZE = 80
NUM_ROWS = 30
# number of training examples
NUM_TRAINING_EXAMPLES = 1000000
NUM_TRAINING_EX_PER_CLASS = NUM_TRAINING_EXAMPLES / 2
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
    # FEATURE 2: SNP position (marked by 1)
    snp_pos_feat_mat = snp_pos_feature_matrix(read, window_start)
    if DEBUG:
        print_snp_pos_feature_matrix(snp_pos_feat_mat)
    #
    feat_mat = concat_feature_matrices(bp_feat_mat, snp_pos_feat_mat)
    # FEATURE 3: mapping quality
    map_qual_mat = map_qual_feature_matrix(read, window_start)
    if DEBUG:
        print_map_qual_feature_matrix(map_qual_mat)
    feat_mat = concat_feature_matrices(feat_mat, map_qual_mat)
    # FEATURE 4: base position within read
    base_pos_feat_matrix = base_pos_feature_matrix(read, window_start)
    if DEBUG:
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

    # iterate through feature matrix
    #
    transaction = db.new_transaction()
    feature_and_label = caffe2_pb2.TensorProtos()
    feature_and_label.protos.extend([
        utils.NumpyArrayToCaffe2Tensor(features),
        utils.NumpyArrayToCaffe2Tensor(labels)
    ])
    transaction.put('train_%03d'.format(snp_num), feature_and_label.SerializeToString())
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
    args = parser.parse_args()

    in_bam = args.bam
    in_vcf = args.vcf
    in_ref = args.ref
    in_truth = args.truth
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
    feature_matrices = []
    labels = []
    for location, alleles in candidate_snps.items():
        if num_snps % 100000 == 0:
            print "Num SNPs processed:", num_snps
        if num_snps > NUM_TRAINING_EXAMPLES:
            print "Reached max number of training examples"
            break
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
        if DEBUG:
            ref_bases = ref_f.fetch(chromosome, window_start, window_end)
            print ref_bases.upper()
        num_reads = 0
        for read in window_reads:
            # check to make sure read is useful and we haven't hit max num reads
            if read.has_tag("MD") and is_usable_read(read) and num_reads < NUM_ROWS:
                read_feature_matrix = create_feat_mat_read(read, window_start, window_end)
                # if this is our first read, overwrite snp_feat_matrix
                if first_read:
                    snp_feat_matrix = read_feature_matrix
                    first_read = False
                # else, stack read's feature matrix with prev reads
                else:
                    snp_feat_matrix = vert_stack_matrices(snp_feat_matrix, read_feature_matrix)
                num_reads += 1
        #print "Num reads processed:", num_reads
        if num_reads > 0:
            # calculate number of empty rows we need to add to matrix
            num_empty_rows = NUM_ROWS - num_reads

            if num_empty_rows > 0:
                # if we only have one read, snp_feat_matrix is still 2D matrix
                # TODO: should we just throw out SNPs with only 1 read covering?
                if snp_feat_matrix.ndim != 3:
                    snp_feat_matrix = snp_feat_matrix[np.newaxis, ...]
                # empty_rows matrix should have dims: (#emptyrows x WINDOW_SIZE x #features)
                empty_rows = np.zeros((num_empty_rows, WINDOW_SIZE, snp_feat_matrix.shape[2]))

                snp_feat_matrix = np.vstack([snp_feat_matrix, empty_rows])

            assert(snp_feat_matrix.shape[0] == NUM_ROWS)
            assert(snp_feat_matrix.shape[1] == WINDOW_SIZE)

            # case: True SNP
            if location in real_snps:
                # make sure our class distributions are even
                if num_positive_train_ex < NUM_TRAINING_EX_PER_CLASS:
                    #feature_matrices.append(snp_feat_matrix)
                    #labels.append(1)
                    write_caffe2_db("minidb", "train.minidb", snp_feat_matrix, np.array([1]), num_snps)
                    num_positive_train_ex += 1
                    num_snps += 1
                    total_num_reads += num_reads
            # case: False SNP
            else:
                if num_negative_train_ex < NUM_TRAINING_EX_PER_CLASS:
                    #feature_matrices.append(snp_feat_matrix)
                    #labels.append(0)
                    write_caffe2_db("minidb", "train.minidb", snp_feat_matrix, np.array([0]), num_snps)
                    num_negative_train_ex += 1
                    num_snps += 1
                    total_num_reads += num_reads


        #print ""
        #print snp_feat_matrix
        #print "feature matrix dims:", snp_feat_matrix.shape
        #if num_snps == 25:
    print "Num feature matrices: ", len(feature_matrices)
    print "Num labels: ", len(labels)
    labels = np.array(labels)
    #write_caffe2_db("minidb", "train.minidb", feature_matrices, labels)
    print "Sum of labels:", np.sum(labels)
    print "Avg #reads per SNP:", total_num_reads / len(feature_matrices)
    exit(0)


if __name__ == "__main__":
    main()
