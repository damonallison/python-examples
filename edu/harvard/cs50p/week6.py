"""
Week 6: File I/O

poetry run pytest edu/harvard/cs50p/week6.py

* with == context manager will call __enter__ and __exit__ on the context object
* Use good data structures (i.e., parse strings into structured types, read /
  write structured types as opposed to positional types)
* Use lambda for simple functions only (lambda is a single line function)

"""

import csv
import pathlib


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


def test_pre_csv(tmp_path: pathlib.Path) -> None:
    """manually parsing a csv file without csvfile"""
    fname = tmp_path / "students.csv"
    with open(fname, "w") as f:
        f.writelines(["damon,test\n", "sam,test2\n"])

    with open(fname, "r") as f:
        lines = [line.rstrip() for line in f]
        students: list[dict[str, str]] = []
        for line in lines:
            name, house = line.split(",")
            students.append({"name": name, "house": house})

        # key is used for sorting. each list element (dictionary) is sent to the
        # `key` function.
        sorted_students = sorted(students, key=lambda s: s["name"])
        assert sorted_students[0]["name"] == "damon"
        assert sorted_students[1]["name"] == "sam"


def test_csv(tmp_path: pathlib.Path) -> None:

    fname = tmp_path / "students.csv"
    with open(fname, "w") as f:
        w = csv.DictWriter(f, fieldnames=["name", "greeting"])
        w.writerows(
            [
                {"name": "damon", "greeting": "hello, test"},
                {"name": "sam", "greeting": "test2"},
            ]
        )

    students: list[dict[str, str]] = []
    with open(fname, "r") as f:
        for row in csv.DictReader(f):
            students.append({"name": row["name"], "greeting": row["greeting"]})

    # csv.reader() is smart enough to parse the second column as a string and
    # not split on the "," in "hello, test"
    assert students[0]["greeting"] == "hello, test"
