import pandas as pd
import numpy as np
import sys
from collections import defaultdict

# TODO: Clean these two functions taken from nanom6a
def pare_annotation2(fl):
	geneids=defaultdict(dict)
	#NM_001197123.2	52	1650	13d4649d-79d3-4593-9cfc-14fac5bee959.fast5	0	+	52	1650	255,0,0	2	175,1020	0,578
	for i in open(fl, "r"):
		ele=i.rstrip().split()
		# ~ for item in ele[-1].split(","):
			# ~ item=item.split(";")[0]
		geneids[ele[3]]=ele[0]
	return geneids

def pare_sam_site(fl,geneids):
	store=defaultdict(dict)
# ~ Chr06	GXB01149_20180715_FAH87828_GA10000_sequencing_run_20180715_NPL0183_I1_33361_read_61758_ch_424_strand.fast5	-	19873404	19873404	230|19874613|GGACA	532|19873617|GGACA	646|19873503|AAACT	714|19873435|GGACC
	for i in open(fl,"r"):
		ele=i.rstrip().split()
		ref,ids,strand=ele[0],ele[1],ele[2]
		if ids in geneids:
			genename=geneids[ids]
		else:
			genename="NA"
		for item in ele[5:]:
			spos,gpos,gbase=item.split("|")
			store[ids][spos]=gpos,ref,gbase,genename
	return store

def get_y_pred(bed_file_path, sam_parse_path, total_mod_file_path, site):
    geneids = pare_annotation2(bed_file_path)
    sites = pare_sam_site(sam_parse_path, geneids)

    mod_df_rows = []

    with open(total_mod_file_path, "r") as total_mod_file:
        for l in total_mod_file.readlines():
            ele = l.rstrip().split()
            unmod, mod, mark = ele

            fast_5_name, spos, seq = mark.split("|")
            mod = float(mod)

            if fast_5_name in sites and spos in sites[fast_5_name]:
                gpos, ref, gbase, genename = sites[fast_5_name][spos]
                read_name = fast_5_name[:-6]
                mod_df_rows.append([gpos, read_name, mod])

    mod_df = pd.DataFrame(np.array(mod_df_rows), columns=["transcript_position", "read_index", "probability"])
    mod_df = mod_df.set_index(["transcript_position", "read_index"])
    read_name_prob_df = mod_df.loc[str(site + 1)]
    return read_name_prob_df
