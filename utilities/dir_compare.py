#!/usr/bin/env python3

# dir_compare.py
#
# Checks for duplicate files
# Existing == old archive
# Current  == new archive

import os
import shutil
import sys
import hashlib
import logging
import pickle
from typing import Dict

# Possible levels : DEBUG, INFO, WARNING, ERROR, CRITICAL
logging.basicConfig(
    level="DEBUG",
    format="%(asctime)s <%(threadName)s> (%(filename)s.%(funcName)s.%(lineno)d)::%(levelname)s:%(message)s",
)


def md5_for_file_name(file_name):
    """
    MD5 a file using a sane block_size multiplier : (64 * 128 == 8192 bytes).

    MD5 has 128 byte digest block, so use a multiplier of 128.
    """
    md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def hashes_for_dir(root_dir) -> Dict[str, str]:
    results = {}
    counter = 0
    for folder, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(folder, file)
            try:
                md5 = md5_for_file_name(file_path)
                if md5 in results:
                    results[md5].append(file_path)
                else:
                    results[md5] = [file_path]
                counter += 1
                if counter % 250 == 0:
                    log.debug("hashed " + str(counter) + " files in " + root_dir)
            except IOError:
                log.error("ERROR: Could not read file at : " + file_path)
    return results


def hashes_for_dir_use_cache(root_dir, cache_file):
    if os.path.exists(cache_file):
        return pickle.load(open(cache_file, "rb"))
    else:
        hashes = hashes_for_dir(root_dir)
        pickle.dump(hashes, open(cache_file, "wb"))
        return hashes


def make_dir_hide_exc(dir):
    """Deletes dir, ignoring exceptions."""

    full_path = os.path.abspath(dir)
    if os.path.exists(dir):
        try:
            shutil.rmtree(full_path)
        except Exception as e:
            log.info("exception removing tree.")
            log.info(e)
    os.mkdir(full_path)
    log.info("created directory at " + full_path)


def get_duplicates(hashes):
    duplicate_hashes = {}
    for k, v in hashes.items():
        if len(v) > 1:
            duplicate_hashes[k] = v
    return duplicate_hashes


def gen_next_filename(proposed_name):
    """Returns the next sequential file name.

    If `proposed_name` does not end in a digit, a "1" is appended.
    Otherwise, the last digit is appended to.

    Example:
        "test.txt" -> "test1.txt"
        "test1.txt" -> "test2.txt"
        "my.test.txt" -> "my.test1.txt"
        "my.test9.txt" -> "my.test10.txt"
        "my.test10.txt" -> "my.test11.txt"
        "my.test19.txt" -> "my.test110.txt" (this is a bug)
    """
    file_name, ext = os.path.splitext(proposed_name)
    last = file_name[-1:]
    append_digit = 1

    if str.isdigit(last):
        append_digit = int(last) + 1
        return_name = file_name[:-1] + str(append_digit)
    else:
        return_name = file_name + str(append_digit)
    return return_name + ext


def next_filename(base_dir, proposed_name):
    available = False
    dir_contents = os.listdir(base_dir)
    next_name = proposed_name
    while not available:
        next_name = gen_next_filename(next_name)
        if not next_name in os.listdir(base_dir):
            available = True
    return next_name


if __name__ == "__main__":
    log = logging.getLogger("root")
    log.debug("Python version : " + sys.version)
    log.debug("os.name :: " + os.name)
    log.debug("environ['HOME'] :: " + os.environ["HOME"])
    log.debug("Current working directory:" + os.getcwd())

    os.chdir("/tmp")
    log.debug("Current working directory :: " + os.getcwd())

    log.debug(f"sys.argv: {sys.argv}")

    if len(sys.argv) < 3:
        log.debug("usage: python dir_compare.py /dir1 /dir2")
        sys.exit(1)

    existing_dir = "/Volumes/homebase/backup/music"
    current_dir = "/Volumes/homebase/backup/music-copy"

    if not os.path.exists(existing_dir):
        log.error("Could not find existing dir (path does not exist)")
        sys.exit()
    if not os.path.exists(current_dir):
        log.error("Could not find current dir (path does not exist)")
        sys.exit()

    tmp_dir = "/Volumes/homebase/backup/music-temp"
    verify_dir = os.path.join(tmp_dir, "verify")

    # don't sweep the temp directory, it may contained hashed contents.
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    make_dir_hide_exc(verify_dir)  # it's ok to sweep verify_dir

    CURRENT_PICKLE_FILE = os.path.join(tmp_dir, "current.pickle")
    EXISTING_PICKLE_FILE = os.path.join(tmp_dir, "existing.pickle")

    log.debug("obtaining existing hashes from " + existing_dir)
    # existing_hashes = hashes_for_dir_use_cache(
    #     existing_dir, EXISTING_PICKLE_FILE)
    existing_hashes = hashes_for_dir(existing_dir)
    # for hash, files in existing_hashes.items():
    #     print(hash, ":", files)

    log.debug("obtaining current hashes from" + current_dir)
    # current_hashes = hashes_for_dir_use_cache(current_dir, CURRENT_PICKLE_FILE)
    current_hashes = hashes_for_dir(current_dir)

    # for hash, files in current_hashes.items():
    #     print(hash, ":", files)

    duplicate_hashes = get_duplicates(current_hashes)
    if len(duplicate_hashes) == 0:
        log.debug("no duplicate hashes")
    for k, v in duplicate_hashes.items():
        log.debug("Found duplicate hash : {" + str(k) + " : " + str(v) + "}")

    # Find files in existing that are not in current

    count = 0
    for k, v in existing_hashes.items():
        if not k in current_hashes.keys():
            log.debug("---------------------------------------------------")
            log.debug("file not in current:" + k + str(v))
            existing_file_name = os.path.split(v[0])[1]
            proposed_name = os.path.join(
                verify_dir, next_filename(verify_dir, existing_file_name)
            )
            log.debug("proposed_name:" + proposed_name)
            os.path.exists()
            if os.path.exists(proposed_name):
                os.remove(proposed_name)
            log.debug("copying '" + v[0] + "' to '" + proposed_name + "'")
            shutil.copy(v[0], proposed_name)

    log.debug("done")
