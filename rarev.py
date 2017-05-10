import pysam
import argparse
import csv
import sys


ap = argparse.ArgumentParser()
ap.add_argument('inBam', help='Mapped reads in bam format')
ap.add_argument('rl', help='read length')
ap.add_argument('out', help='out')
ap.add_argument('chr', help='chr')

args = ap.parse_args()



cigar_full=str(args.rl)+"M"


#SRR062635.19983732	163	2	60058	60	100M	2	60129	100	TGGTAGGGGACAAAACAGATACAGCACCAAACATGAAAAGCATTGATTTATCTCTCTCTGCTAATGCATGTGAAGTGTTGATCCTGGGGTTGAAAGTCGG	array('B', [15, 35, 36, 32, 36, 36, 36, 34, 29, 37, 32, 37, 37, 37, 37, 34, 37, 36, 37, 36, 37, 34, 37, 36, 36, 37, 34, 36, 35, 37, 37, 32, 37, 36, 34, 37, 37, 37, 33, 34, 36, 37, 34, 37, 36, 33, 36, 37, 37, 36, 36, 36, 37, 36, 35, 36, 37, 32, 37, 33, 36, 37, 37, 38, 33, 34, 36, 34, 37, 36, 34, 35, 32, 35, 35, 32, 35, 32, 34, 34, 33, 32, 33, 33, 34, 33, 34, 34, 34, 33, 32, 34, 35, 30, 32, 33, 31, 29, 23, 30])	[('X0', 1), ('X1', 1), ('MD', '0G99'), ('RG', 'SRR062635'), ('AM', 23), ('NM', 1), ('SM', 23), ('MQ', 60), ('XT', 'U'), ('BQ', 'ARP@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')]	0	0	14636624	0	1	0	0



#MD tag will be always in this form : 0G99, Since we allow for only one mistmath. number{ACTG}number



snpSet=set()
dict={}


n=0

out=open(args.out,"w")

samfile = pysam.AlignmentFile(args.inBam, "rb" )
for pileupcolumn in samfile.pileup(args.chr):
    
   
    	if pileupcolumn.n>0:
            n+=1
            
            if n%1000000==0:
                print n


            for pileupread in pileupcolumn.pileups:
			
			
                if not pileupread.is_del and not pileupread.is_refskip:
                    
                    
                        ed=pileupread.alignment.get_tag("NM")
                        mq=pileupread.alignment.mapping_quality
                        cigar=pileupread.alignment.cigarstring
                        phreadQ=pileupread.alignment.query_qualities[pileupread.query_position]
                        
                        if ed==1 and cigar==cigar_full and mq==60 and phreadQ>17:

                        
                        
                        
                            pos=int(pileupread.query_position)




                                

                            for b in ['A','C','T','G']:
                                
                                    if len(pileupread.alignment.get_tag("MD").split(b))==2:

                                        if int(pileupread.alignment.get_tag("MD").split(b)[0])==pos:
                                            #print pileupread, pos
                                            #print pos, phreadQ,pileupread.alignment.get_tag("MD"),pileupread.alignment.query_sequence[pileupread.query_position]
                                            
                                            ALT=pileupread.alignment.query_sequence[pileupread.query_position]
                                            
                                            #out.write(str(pileupcolumn.pos+1)+","+str(b)+","+str(ALT)+","+str(phreadQ))
                                            #out.write("\n")
                                            
                                            posRef=pileupcolumn.pos+1
                                            
                                            
                                            if posRef not in snpSet:
                                                dict[posRef]=[]
                                                dict[posRef].append(ALT)
                                            else:
                                                dict[posRef].append(ALT)

                                            snpSet.add(posRef)
                                            #print pileupcolumn.pos+1,b,pileupread.alignment.query_sequence[pileupread.query_position], phreadQ





print "Number of SNPs", len(snpSet)


for k,v in dict.items():
    t=[]
    t[:]=[]
    t.append(v.count('A'))
    t.append(v.count('C'))
    t.append(v.count('T'))
    t.append(v.count('G'))
    if max(t)!=sum(t):
        print args.inBam,"ERROR",k, v.count('A'),v.count('C'),v.count('T'),v.count('G')
    out.write(str(k)+","+str(v.count('A'))+","+str(v.count('C'))+","+str(v.count('T'))+","+str(v.count('G')) )
    out.write("\n")


sys.exit(1)

import pandas as pd
import numpy as np
import pysam

def GetPileup(bam_file, region=None, start_pos=None, pileup_len=100, reference=None):
    region_str = region + ':' + str(start_pos) + '-' + str(start_pos + pileup_len)
    pileup_sam = pysam.mpileup(bam_file, '-r', region_str, '-f', reference, '-d', '20000000')
    
    parsed_pileup = [dict(zip(['ref_id', 'ref_pos', 'ref_base', 'coverage', 'read_base'],
                              x.split()[:-1])) for x in pileup_sam]
                              
    return pd.DataFrame(parsed_pileup)

def ParsePileup(pileup_df):
    alphabet = 'ACGT'
    for nuc in alphabet:
        pileup_df.loc[:, nuc] = np.where(pileup_df.ref_base.str.upper() != nuc, pileup_df.read_base.str.count(nuc), np.nan)
    for f_base in alphabet:
        for t_base in alphabet:
            if f_base != t_base:
                pileup_df.loc[:, f_base + '>' + t_base] = np.where(pileup_df.ref_base.str.upper() == f_base, pileup_df[t_base] / pileup_df.coverage * 100, np.nan)
    return pileup_df


pileup = GetPileup(args.inBam, region, start_pos, ref_file_str)
pileup = ParsePileup(pileup)




samfile.close()



