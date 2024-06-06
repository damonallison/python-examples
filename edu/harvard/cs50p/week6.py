"""
Week 6: File I/O

poetry run pytest edu/harvard/cs50p/week6.py


* with == context manager
    *  call __enter__ and __exit__ on the context object
    * Allows the context object to "wrap" the block and do instiantiation and
      cleanup.
    * Used to ensure resources (file handles / network sockets) are closed.

* Use good data structures
    * read / write structured types as opposed to simple string concatenation
    * For example, read and write "first_name" and "last_name" fields rather
      than a single "name".
"""

import csv
import os
import pathlib
import tempfile

from PIL import Image, ImageOps


def test_io_basics(tmp_path: pathlib.Path) -> None:
    """Test basic write / read"""

    fname = tmp_path / "names.txt"
    # r == read
    # w == overwrite
    # a == append
    with open(fname, "w") as f:
        # note that `.writelines()` does not add newlines to each str.
        f.writelines([f"hello{os.linesep}", f"world{os.linesep}"])

    with open(fname, "r") as f:
        for line in f:
            assert line.endswith(os.linesep)

    with open(fname, "r") as f:
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
    """Read and write to a csv.

    csv accounts for string quoting (escaping `,` in strings), handling headers,
    etc.

    DictReader() / DictWriter() is more robust than reading / writing fields by
    ordinal. It's easier to add / remove columns without breaking ordinals for
    remaining fields.
    """
    fname = tmp_path / "students.csv"
    with open(fname, "w") as f:
        w = csv.DictWriter(f, fieldnames=["name", "greeting"])
        w.writeheader()  # note the file needs a header to read back column names
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


#
# Problem Set 6
#


def lines_of_code(p: pathlib.Path) -> int:
    """Outputs the number of lines in a file, excluding blank lines and strings
    that start with # (comments)."""
    lines = 0
    with open(p, "r") as f:
        for line in f:
            trimmed = line.strip()
            if len(trimmed) > 0 and not trimmed.startswith("#"):
                lines += 1

    return lines


def pizza_py() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        tmp_file = tmp_path / "menu.csv"
        with open(tmp_file, "w") as f:
            w = csv.DictWriter(f, fieldnames=["Sicilian Pizza", "Small", "Large"])
            w.writeheader()
            w.writerows(
                [
                    {"Sicilian Pizza": "Cheese", "Small": "$25.50", "Large": "$39.95"},
                    {"Sicilian Pizza": "Special", "Small": "$33.50", "Large": "$47.95"},
                    # 1 item,$27.50,$41.95
                    # 2 items,$29.50,$43.95
                    # 3 items,$31.50,$45.95
                ]
            )

        with open(tmp_file, "r") as f:
            import tabulate

            r = csv.DictReader(f)
            rows: list[dict[str:str]] = []
            for row in r:
                rows.append(row)
            print(tabulate.tabulate(rows, headers="keys", tablefmt="grid"))


def scourgify() -> list[dict[str, str]]:
    """Read a file, separate out first and last name"""

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        tmp_file = tmp_path / "names.csv"

        with open(tmp_file, "w") as f:
            w = csv.DictWriter(f, fieldnames=["name", "house"])
            w.writeheader()
            w.writerows(
                [
                    {"name": "allison, damon", "house": "maple grove"},
                    {"name": "raiche, sam", "house": "plymouth"},
                ]
            )

        tmp_out = tmp_path / "split-names.csv"

        students: list[dict[str, str]] = []
        with open(tmp_file, "r") as f:
            r = csv.DictReader(f)
            with open(tmp_out, "w") as fout:
                w = csv.DictWriter(fout, ["first", "last", "house"])
                w.writeheader()
                for row in r:
                    fullname = row["name"]
                    l_name, f_name = [name.strip() for name in fullname.split(",")]
                    student = {"first": f_name, "last": l_name, "house": row["house"]}
                    students.append(student)
                    w.writerow(student)

        with open(tmp_out, "r") as final:
            print(final.read())
        # To write this back to a file
        return students


def cs50_tshirt() -> None:
    with Image.open(
        "edu/harvard/cs50p/week6-images/before.png"
    ) as original, Image.open("edu/harvard/cs50p/week6-images/shirt.png") as overlay:
        original = ImageOps.fit(original, size=overlay.size)
        original.paste(overlay, box=None, mask=overlay)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = f"{tmp_dir}/out.png"
            original.save(tmp_file)
            # with Image.open(tmp_file) as final:
            #     final.show()
