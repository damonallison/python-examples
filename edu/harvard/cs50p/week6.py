"""
Week 6: File I/O

poetry run pytest edu/harvard/cs50p/week6.py

* with == context manager will call __enter__ and __exit__ on the context object

"""

import pathlib
import pytest


def test_io_basics(tmp_path: pathlib.Path) -> None:
    """Test basic write / read"""

    fname = tmp_path / "names.txt"
    # r == read
    # w == overwrite
    # a == append
    with open(fname, "w") as f:
        f.writelines(["hello\n", "world\n"])

    with open(fname, "r") as f:
        # for line in f
        #    # do something with f
        assert [line.rstrip() for line in sorted(f.readlines())] == ["hello", "world"]


def test_csv(tmp_path: pathlib.Path) -> None:
    fname = tmp_path / "students.csv"
    with open(fname, "w") as f:
        f.writelines(["damon,test\n", "sam,test2\n"])

    with open(fname, "r") as f:
        lines = [line.rstrip() for line in f]
        students: list[dict[str, str]] = []
        for line in lines:
            name, house = line.split(",")
            students.append({"name": name, "house": house})

        def get_name(d: dict[str, str]) -> str:
            return d["name"]

        # key is used for sorting. each list element is passed to get_name
        sorted_students = sorted(students, key=get_name)
        assert sorted_students[0]["name"] == "damon"
        assert sorted_students[1]["name"] == "sam"
