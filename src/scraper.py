from lm_dataformat import Archive, Reader
from datasets import load_dataset
import multiprocessing as mp
import logging
from tqdm import tqdm
import os
import datasets
import subprocess
from pathlib import Path
from unidiff import PatchSet
import time
import argparse

logging.basicConfig(
    level = logging.INFO,
    format=  '%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename="/logs/scraper.log")
logger = logging.getLogger(__name__)


#Shell utils
def run_in_shell(cmd:str,cwd=None):
    completed = subprocess.run([cmd], capture_output=True,shell=True,cwd=cwd)
    return completed
def get_diff_cmd(commit_id:str,cwd=None):
    """
    Given a commit_id gives the diff string corresponding to it.
    """
    commit_format_fn = f"git diff {commit_id}^ {commit_id}"
    diff_result = run_in_shell(commit_format_fn,cwd=cwd).stdout
    diff_result = diff_result.decode() #returns the diff string
    return diff_result


def get_git_clone(repo_name:str,output_path:str):
    """
    Given a repo name and owner name  it clones to the output_path
    """
    clone_format_fn = f"git clone https://github.com/{repo_name}.git {output_path}/{repo_name.split('/')[1]}"
    clone_result = run_in_shell(clone_format_fn)


def get_before_file(file_diff: dict, commit_hash: str, repo_name: str, length_threshold: int, clone_path:str) -> str:
    repo_name = repo_name.split("/")[1]
    if file_diff["src_file"] == "/dev/null":
        raw_file: any = ["ADDFILE"]
    elif file_diff["tgt_file"] == "/dev/null":
        # If file is deleted, get before file from the raw diff, which will be the full file.
        raw_file = [line[1:] + "\n" for line in file_diff["hunks_process"][0][3:]]
    else:
        try:
            file_raw_url = (f"{clone_path}/{file_diff['tgt_file'][2:]}")
            raw_file = open(file_raw_url,"r")
            raw_file = [line for line in raw_file.readlines()]
            if length_threshold > 0 and len(raw_file) > length_threshold:
                return ""
        except Exception as e:
            return ""
        # Iterate over hunks for this file and apply the reverse patch.
        for hunk in file_diff["hunks_process"]:
            hunk_list = []
            for line in hunk[3:]:
                if line.startswith("-") or line.startswith(" "):
                    hunk_list.append(line[1:] + "\n")
            raw_file[hunk[0][0] - 1:hunk[0][0] + hunk[1][1] - 1] = hunk_list
    del file_diff["hunks_process"]  # Deletes this item from the dict in parent functions

    return "".join(raw_file)



class config:
    diff_length_threshold = 10000
    code_length_threshold = 10000
    ignore_deletions = True
    python_only = False

def process_ind_patch(patch_diff) -> dict:
    """Process patch to get diff data."""
    patch_parsed_diff: dict = {
        "hunks": [],
        "hunks_process": [],
        "addition_count" : [],
        "deletion_count" : [],
        "src_file" : [],
        "tgt_file" : [],
        "file_extension" : []
            }

    patch_parsed_diff["addition_count"] = patch_diff.added
    patch_parsed_diff["deletion_count"] = patch_diff.removed
    patch_parsed_diff["src_file"] = patch_diff.source_file
    patch_parsed_diff["tgt_file"] = patch_diff.target_file
    if patch_parsed_diff["tgt_file"] == "/dev/null":
        patch_parsed_diff["file_extension"] = Path(patch_diff.source_file).suffix
    else:
        patch_parsed_diff["file_extension"] = Path(patch_diff.target_file).suffix
    for patch_diff_ind in patch_diff:
        patch_diff_ind = str(patch_diff_ind)
        patch_diff_split = patch_diff_ind.split("@@")
        patch_diff_line = patch_diff_split[2].split("\n")
        patch_diff_line_numbers = [list(map(int, hunk.strip("-+").split(",")))
                                   for hunk in patch_diff_split[1].strip().split(" ")]
        patch_parsed_diff["hunks_process"].append(patch_diff_line_numbers + patch_diff_line[:-1])
        patch_parsed_diff["hunks"].append(patch_diff_ind)
    patch_parsed_diff["hunks"] = "".join(patch_parsed_diff["hunks"])
    return patch_parsed_diff





def process_commit_hf(commit_data: dict,clone_path:str) -> list[dict]: #HF centric process
    """
    Process a commit dictionary to get the before files and diff dict.
    Args:
        commit_data (dict): Dictionary containing commit hash, repo name, and
        commit message.
    Returns:
        list[dict]: A list of dicts, where each dict contains the data for a
        change to a single file.
    """
    
    
    master_dict={k:None for k in ['hunks', 'addition_count', 'deletion_count', 'src_file', 'tgt_file', 'file_extension', 'before_file', 'commit', 'message', 'repo_name', 'language_name', 'author_name', 'license']}

    #diff_url = f"https://github.com/{commit_data['repo_name']}/commit/{commit_data['commit']}.diff"
    try:
        diff = get_diff_cmd(commit_data["commit"],cwd=clone_path)
        patch = PatchSet(diff)
        if len(patch) == 0:
            return master_dict
    except Exception as e:
        return master_dict
    #dummy_diff_dict = {k:None for k in ["commit","message","repo_name","language_name","author_name","license"]}
 
    # Iterate over files within the diff.
    for patch_ind in patch:
        if config.ignore_deletions and patch_ind.target_file == "/dev/null":
            # logger.info("Deletion")
            continue
        if config.diff_length_threshold > 0 and sum(len(hunk) for hunk in patch_ind) > config.diff_length_threshold:
            # logger.info("Length Threshold")

            continue
        # Filter non-text files.
        if patch_ind.added == 0 and patch_ind.removed == 0:
            # logger.info("Non-Text File")

            continue
        diff_dict: dict = process_ind_patch(patch_ind)
        diff_dict["before_file"] = get_before_file(diff_dict, commit_data["commit"], commit_data["repo_name"],
                                                     length_threshold=config.code_length_threshold,clone_path = clone_path)
        if not diff_dict["before_file"]:
            # Happens if exception is thrown or file is too long.
            continue
        diff_dict["commit"] : str = diff#stringify
        diff_dict["message"] = commit_data["commit_message"]
        diff_dict["repo_name"] :str = commit_data["repo_name"]
        diff_dict["language_name"] = commit_data["language_name"]
        diff_dict["author_name"] = commit_data["committer"]["name"]
        diff_dict["license"] = commit_data["license"]
        
        master_dict.update(diff_dict)
    return master_dict




 
def get_all_parquet_files(directory:str):
    return [ os.path.join(directory,i) for i in os.listdir(directory)]


        
def find_last_ind(output_path):
    file_names = [int(i.split(".parquet")[0].replace("github_diff_proc_","")) for i in os.listdir(output_path)]
    logger.info(file_names)
    if len(file_names) == 0:
        return 0
    else:
        file_max_num = len(file_names)#max(file_names)
        return file_max_num+1

def load_diff_batch_processed_dataset(file_path):
    gh_diff_dataset = datasets.load_dataset("parquet",data_files=batch,cache_dir="/fsx/home-reshinth/work/github_diff_data/Code-Pile/cache_gh_diff_proc")
    return gh_diff_dataset





if __name__ == "__main__":
    input_config = {
        "input_path" : "/fsx/home-reshinth/work/github_diff_data/filtered_dataset_gh_diff",
        "output_path" : "/fsx/home-reshinth/work/clean_github_diff/dataset_output_02",
        "clone_path" : "/fsx/home-reshinth/work/clean_github_diff/clone_hash"
    }
    cpu_count = mp.cpu_count()
    logger.info(f"The available cores are {cpu_count}")
    logger.info(f"#####START#####")

    inp_parquet_file_names = get_all_parquet_files(input_config["input_path"])
    inp_parquet_file_names.sort()
    file_path_start_ind = find_last_ind(input_config["output_path"])
    print(file_path_start_ind)
    inp_parquet_file_names = inp_parquet_file_names[4500+file_path_start_ind:]
    clone_path = input_config["clone_path"]
    
    total_clone_count = 0
    clone_count = 0
    run_len_count = 0
    for ind_file in tqdm(inp_parquet_file_names): 
        try:
            file_pointer =   Path(ind_file).stem.replace("github_diff_","")
            chunked_dataset = datasets.load_dataset("parquet",data_files=ind_file,split="train",cache_dir="/fsx/home-reshinth/work/github_diff_data/Code-Pile/cache_gh_diff_proc_1")
            chunked_repos : list[str] = list(set(chunked_dataset["repo_name"]))
            clone_path_files : list[str] = os.listdir(input_config["clone_path"])
            for repo in  chunked_repos:
                repo_name_without_user = repo.split("/")[1]
                if repo_name_without_user not in clone_path_files:
                    logger.info(f"{repo_name_without_user} not found in clone path. going to clone...")
                    get_git_clone(repo,clone_path)
                    clone_count += 1
                    total_clone_count +=1

            clone_repo_count_message = f"Cloned {clone_count} number of repos out of {total_clone_count} global clones..."
            logger.info(clone_repo_count_message)
            logger.info(f"Currently {len(clone_path_files)} many repos..")
            clone_count = 0
            def process_local_fn(example,clone_path_files=clone_path_files,clone_path=clone_path):
                #preparation
                repo_name_without_user = example["repo_name"].split("/")[1]
                # if repo_name_without_user not in clone_path_files:
                #     logger.info(f"{example['repo_name']} not found in clone path. going to clone...")
                #     get_git_clone(example["repo_name"],clone_path)
        
                git_folder_path = os.path.join(clone_path,repo_name_without_user)
                check_ls = run_in_shell(f"pwd",cwd=git_folder_path)
                #done preparation
                return process_commit_hf(example,git_folder_path)
            
            chunked_dataset_processed = chunked_dataset.map(process_local_fn,batch_size=10_000,num_proc=cpu_count).filter(lambda example: example["language_name"] != None,batch_size=5000,num_proc=cpu_count)
            output_file_name = os.path.join(input_config["output_path"],f"github_diff_proc_{file_pointer}.parquet")
            if len(chunked_dataset_processed) > 1:
                chunked_dataset_processed.to_parquet(output_file_name)

            file_path_start_ind += 1
            run_len_count += len(chunked_dataset_processed)
            print(f"Going for sleep, GLOBAL : {run_len_count} {len(chunked_dataset_processed)} rows written... after {file_path_start_ind} chunks... written in {output_file_name}") 
            logger.info(f"Going for sleep, GLOBAL : {run_len_count} {len(chunked_dataset_processed)} rows written... after {file_path_start_ind} chunks... written in {output_file_name}") 
            if total_clone_count > 1000:
                time.sleep()
        except:
            pass
            
