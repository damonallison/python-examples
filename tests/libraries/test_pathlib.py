from pathlib import Path


def test_path(tmp_path: Path) -> None:
    assert tmp_path.is_dir()
    assert tmp_path.is_absolute()

    tmp_file = tmp_path / "test.txt"
    assert not tmp_file.exists()
    assert tmp_file.parent == tmp_path

    print(f"parent: {tmp_file.parent}")
    assert tmp_file.stem == "test"
    assert tmp_file.name == "test.txt"
    assert tmp_file.suffix == ".txt"

    # change file extension
    ext_file = Path("dir/test.txt")
    assert ext_file.suffix == ".txt"
    ext_file = ext_file.with_suffix(".t")
    assert ext_file.suffix == ".t"

    # i/o
    with tmp_file.open(mode="w") as f:
        f.writelines(["test", "line2"])
    assert tmp_file.read_text() == "testline2"

    # read / write (note this will overwrite all previous file contents)
    tmp_file.write_text("hello")
    assert tmp_file.read_text() == "hello"

    # iterating a directory
    assert [x.name for x in tmp_path.iterdir()] == ["test.txt"]

    # directory trees
    tmp_child = tmp_path / "child" / "grandchild"
    tmp_child.mkdir(parents=True)
    for i in range(10):
        gc = tmp_child / f"ggc{i}.txt"
        gc.touch()

    assert len(list(tmp_child.glob("*.txt"))) == 10

    # ** == "this directory and all subdirectories" (recursive globbing)
    assert len(list(tmp_child.parent.glob("**/*.txt"))) == 10

    # tmp_grandchild1.parent.mkdir(parents=True, exist_ok=True)
    # # tmp_grandhhild2 = tmp_path / "child" / "test2.txt"

    # tmp_grandchild1.write_text("hello")

    p2 = Path(tmp_path, "test", "test2", "test3")
    p2.mkdir(parents=True, exist_ok=True)
    print(p2)
