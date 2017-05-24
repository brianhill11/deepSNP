#!/usr/bin/python

import argparse
import os
import subprocess

def check_files_exist(file_list):
    for f in file_list:
        if not os.path.isfile(f):
            print "ERROR:", f, " does not exist!"
            exit(1)

def main():
    parser = argparse.ArgumentParser(description="Creates VCF file containing all genomic positions containing at least one alternate alelle")
    parser.add_argument('-b', '--bam-file', dest='bam_file', required=True, help='Input BAM file')
    parser.add_argument('-f', '--reference-file', dest='reference_file', required=True, help='Reference fasta file for input BAM')
    parser.add_argument('-o', '--vcf-file', dest='vcf_file', required=True, help='Output VCF file')
    parser.add_argument('-r', '--region', dest='region', required=False, help='Region of genome to process (format = chr:start-end)')
    args = parser.parse_args()
    
    # make sure BAM file and reference file exist 
    check_files_exist([args.bam_file, args.reference_file])

    bam_file = args.bam_file
    reference_file = args.reference_file
    vcf_file = args.vcf_file

    # check that we were given a BAM file
    if os.path.splitext(bam_file)[1] != '.bam':
        print "ERROR: input BAM file missing .bam extension!"
        exit(1)

    # if supplied VCF file name does not have .vcf extension, add it
    if os.path.splitext(vcf_file)[1] != '.vcf':
        vcf_file += '.vcf'
    
    # -u : generate uncompressed VCF output
    # -f is flag for reference file
    # --skip-indels : do not perform indel calling
    # --output-tags : optional tags to output (AD = allele depth)
    # we pipe this output into bcftools, to filter out non-SNPs
    # -v snps : select snp types
    samtools_cmd_list = ["samtools", "mpileup", "-uf", reference_file, \
            "--skip-indels", "--output-tags", "AD", bam_file]
    bcftools_cmd_list = ["bcftools", "view", "-v", "snps", "-o", vcf_file] 

    print "Calling command:", ' '.join(samtools_cmd_list)
    samtools = subprocess.Popen(samtools_cmd_list, stdout=subprocess.PIPE)
    bcftools = subprocess.check_output(bcftools_cmd_list, stdin=samtools.stdout)
    samtools.wait()

    exit(0)


if __name__ == "__main__":
    main()
