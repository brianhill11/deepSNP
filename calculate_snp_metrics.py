#!/usr/bin/python

import os
import pysam
from deepSNP import *
from collections import OrderedDict

WINDOW_SIZE = 100


def overlaps(read, position):
    return read.reference_start <= position and read.reference_end >= position


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
    if os.path.isfile(candidate_snps_pickle):
        log.info("Loading %s", candidate_snps_pickle)
        candidate_snps = pickle.load(open(candidate_snps_pickle, "rb"))
    else:
        candidate_snps = get_candidate_snps(in_vcf)
        log.info("Creating file: %s", candidate_snps_pickle)
        pickle.dump(candidate_snps, open(candidate_snps_pickle, "wb"))

    # total number of reads we see
    total_num_reads = 0
    # total number of usable reads
    total_num_usable_reads = 0
    # total number of reads overlapping SNPs
    total_num_overlapping_reads = 0
    # total number of usable reads overlapping SNPs
    total_num_usable_overlapping_reads = 0

    start_time = time.time()
    num_snps = 0
    for location in real_snps:
        num_snps += 1
        if num_snps % 100000 == 0 and num_snps > 0:
            cur_time = time.time()
            log.info("Num real SNPs processed: %s (%s %)", num_snps, float(num_snps)/len(real_snps))
            log.info("Elapsed time: %s", cur_time - start_time)
            log.info("Avg num reads per window:", float(total_num_reads)/float(num_snps))
            log.info("Avg num usable reads per window:", float(total_num_usable_reads)/float(num_snps))
            log.info("Avg num overlapping reads:", float(total_num_overlapping_reads)/float(num_snps))
            log.info("Avg num usable overlapping reads:", float(total_num_usable_overlapping_reads)/float(num_snps))
        # location is (chromosome, position) tuple
        chromosome = location[0]
        pos = int(location[1])
        # our feature window is centered on SNP position
        # max() makes sure we don't have negative index near start of contig
        window_start = max(pos - (WINDOW_SIZE / 2), 0)
        window_end = window_start + WINDOW_SIZE

        log.debug("location: %s", location)

        window_reads = bam_f.fetch(chromosome, window_start, window_end)

        if DEBUG:
            ref_bases = ref_f.fetch(chromosome, window_start, window_end)
            print ref_bases.upper()
        # number of reads in this window
        num_reads = 0
        # number of usable reads in this window
        num_usable_reads = 0
        # number of reads overlapping SNP
        num_overlapping_reads = 0
        # number of usable overlapping reads
        num_usable_overlapping_reads = 0
        for read in window_reads:
            num_reads += 1
            if overlaps(read, pos):
                num_overlapping_reads += 1
            # check to make sure read is useful and we haven't hit max num reads
            if read.has_tag("MD") and is_usable_read(read):
                num_usable_reads += 1
                if overlaps(read, pos):
                    num_usable_overlapping_reads += 1

        total_num_reads += num_reads
        total_num_usable_reads += num_usable_reads
        total_num_overlapping_reads += num_overlapping_reads
        total_num_usable_overlapping_reads += num_usable_overlapping_reads

    print "##########################################################"
    print "## FINAL SUMMARY STATISTICS - REAL SNPS"
    print "##########################################################"
    print "Total num reads:", total_num_reads
    print "Total num usable reads:", total_num_usable_reads
    print "Total num overlapping reads:", total_num_overlapping_reads
    print "Total num usable overlapping reads:", total_num_usable_overlapping_reads
    print "Avg num reads per window:", float(total_num_reads)/float(num_snps)
    print "Avg num usable reads per window:", float(total_num_usable_reads)/float(num_snps)
    print "Avg num overlapping reads:", float(total_num_overlapping_reads)/float(num_snps)
    print "Avg num usable overlapping reads:", float(total_num_usable_overlapping_reads)/float(num_snps)
    print "##########################################################"

    # total number of reads we see
    total_num_reads = 0
    # total number of usable reads
    total_num_usable_reads = 0
    # total number of reads overlapping SNPs
    total_num_overlapping_reads = 0
    # total number of usable reads overlapping SNPs
    total_num_usable_overlapping_reads = 0

    start_time = time.time()
    num_snps = 0
    for location in candidate_snps:
        num_snps += 1
        if num_snps % 100000 == 0 and num_snps > 0:
            cur_time = time.time()
            log.info("Num candidate SNPs processed: %s (%s %)", num_snps, float(num_snps)/len(candidate_snps))
            log.info("Elapsed time: %s", cur_time - start_time)
            log.info("Avg num reads per window:", float(total_num_reads)/float(num_snps))
            log.info("Avg num usable reads per window:", float(total_num_usable_reads)/float(num_snps))
            log.info("Avg num overlapping reads:", float(total_num_overlapping_reads)/float(num_snps))
            log.info("Avg num usable overlapping reads:", float(total_num_usable_overlapping_reads)/float(num_snps))
        # location is (chromosome, position) tuple
        chromosome = location[0]
        pos = int(location[1])
        # our feature window is centered on SNP position
        # max() makes sure we don't have negative index near start of contig
        window_start = max(pos - (WINDOW_SIZE / 2), 0)
        window_end = window_start + WINDOW_SIZE

        log.debug("location: %s", location)

        window_reads = bam_f.fetch(chromosome, window_start, window_end)

        if DEBUG:
            ref_bases = ref_f.fetch(chromosome, window_start, window_end)
            print ref_bases.upper()
        # number of reads in this window
        num_reads = 0
        # number of usable reads in this window
        num_usable_reads = 0
        # number of reads overlapping SNP
        num_overlapping_reads = 0
        # number of usable overlapping reads
        num_usable_overlapping_reads = 0
        for read in window_reads:
            num_reads += 1
            if overlaps(read, pos):
                num_overlapping_reads += 1
            # check to make sure read is useful and we haven't hit max num reads
            if read.has_tag("MD") and is_usable_read(read):
                num_usable_reads += 1
                if overlaps(read, pos):
                    num_usable_overlapping_reads += 1

        total_num_reads += num_reads
        total_num_usable_reads += num_usable_reads
        total_num_overlapping_reads += num_overlapping_reads
        total_num_usable_overlapping_reads += num_usable_overlapping_reads

    print "##########################################################"
    print "## FINAL SUMMARY STATISTICS - CANDIDATE SNPS"
    print "##########################################################"
    print "Total num reads:", total_num_reads
    print "Total num usable reads:", total_num_usable_reads
    print "Total num overlapping reads:", total_num_overlapping_reads
    print "Total num usable overlapping reads:", total_num_usable_overlapping_reads
    print "Avg num reads per window:", float(total_num_reads)/float(num_snps)
    print "Avg num usable reads per window:", float(total_num_usable_reads)/float(num_snps)
    print "Avg num overlapping reads:", float(total_num_overlapping_reads)/float(num_snps)
    print "Avg num usable overlapping reads:", float(total_num_usable_overlapping_reads)/float(num_snps)
    print "##########################################################"

if __name__ == "__main__":
    main()
