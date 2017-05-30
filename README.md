# deepSNP


# Plan

- design fetures better reflecting technology
- better training data
- better deep learning method


# Datasets 
## GIB
## WXS/WGS
## Simulated data 

Simulate from chr1 

to calculate number of reads (N)
N=(cov * G)/rl

For chr17, cov=30, rl=200 2x100bp
N=(30 * 83M)/ 200==8M reads to be simulated using this command 

Using WGS
```
/u/scratch2/n/ngcrawfo/ERROR_CORRECTION/wgs_simulation/wgsim/wgsim -r 0.03 -R 0.005 -e 0.02 -1 100 -2 100 -A 0 -N 8000000 /u/home/s/serghei/project/Homo_sapiens/Ensembl/GRCh37/Sequence/Chromosomes/17.fa WGS_chr17_1.fastq WGS_chr17_2.fastq >log
```

Reads are simulated here
/u/home/s/serghei/scratch/WGS_deepSNP


Now i am mapping using BWA

```
bwa mem ~/project/Homo_sapiens/Ensembl/GRCh37/Sequence/Chromosomes/17.fa WGS_chr17_1.fastq WGS_chr17_2.fastq | samtools view -bS - > WGS_chr17.bam
qsub -cwd -V -N bwa -l h_data=16G,time=10:00:00 run_BWA.sh 
/u/home/s/serghei/scratch/WGS_deepSNP

```



# To try
- Train on simulated apply on real? 
