import os
import sys
import subprocess
import pandas as pd

def read_is_in_dir(read, path):
    test_path = os.path.join(path, f"{read}.fast5")
    return os.path.exists(test_path)

def copy_needed_reads(read_dir1, read_dir2, X_test, work_dir):
    read_paths = []

    for read in X_test.index:
        if read_is_in_dir(read, read_dir1):
            read_path = os.path.join(read_dir1, f"{read}.fast5\n")
        elif read_is_in_dir(read, read_dir2):
            read_path = os.path.join(read_dir2, f"{read}.fast5\n")
        else:
            continue

        read_paths.append(read_path)

    reads_file_path = os.path.join(work_dir, "reads.txt")
    with open(reads_file_path, "w") as reads_file:
        reads_file.writelines(read_paths)

    copy_reads_script_path = "./copy_reads.sh"
    destination_path = os.path.join(work_dir, "fast5s")

    if not os.path.exists(destination_path):
        subprocess.run([copy_reads_script_path, work_dir, reads_file_path, destination_path])

def prepare(read_dir1, read_dir2, X_test, name):
    work_dir = os.path.join(".", name)
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    reads_file_path = os.path.join(work_dir, "reads.txt")
    if not os.path.exists(reads_file_path):
        copy_needed_reads(read_dir1, read_dir2, X_test, work_dir)
    # run m6anet pipeline on those reads

def get_y_pred(y_pred_path, summary_path, site):
    # ctrl9kb
    df = pd.read_csv(y_pred_path)
    df.drop(["transcript_id"], axis=1, inplace=True)
    df = df.set_index(["transcript_position", "read_index"])
    read_prob_df = df.loc[site] # read_index -> probability_modified

    # read_index -> read_name
    alignment_summary_df = pd.read_csv(summary_path, sep="\t")
    as_s = alignment_summary_df.set_index(["read_index"])["read_name"]
    prob_read_name_df = read_prob_df.join(as_s)
    read_name_prob_df = prob_read_name_df.set_index("read_name")

    return read_name_prob_df

def train(X_train, y_train):
    # Not implemented
    return None
