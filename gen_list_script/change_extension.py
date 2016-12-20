#!/usr/bin/env python

import file_io
import sys

def check_ext(ext):
    assert(len(ext) > 1)
    if (ext[0] != "."):
        ext = "." + ext
        return ext


if __name__ == "__main__":
    file_list_name = sys.argv[1]
    org_ext = check_ext(sys.argv[2])
    new_ext = check_ext(sys.argv[3])

    file_list = file_io.read_file(file_list_name)
    file_list = [file_.replace(org_ext, new_ext) for file_ in file_list]
    file_io.save_file(file_list, file_list_name)
