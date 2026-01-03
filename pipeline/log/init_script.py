import os
import shutil
from pathlib import Path


def copy_to_workflow(src_dir: str = ".", dst_dir: str = "./DEST"):
    src = Path(src_dir).resolve()
    dst = Path(dst_dir).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    for child in src.iterdir():
        if not child.is_dir():
            continue
        if child.name == dst.name and child.resolve() == dst:
            continue

        csv_files = sorted(child.glob("*.csv"))
        if not csv_files:
            continue

        base = child.name

        for i, csv_path in enumerate(csv_files, start=1):
            suffix = "" if i == 1 else f"_{i}"
            new_name = f"{base}{suffix}.csv"
            target = dst / new_name

            shutil.copy2(str(csv_path), str(target))

    print(f"[\033[32mInfo\033[0m] Done. CSVs moved to: {dst}")


if __name__ == "__main__":
    current_dir = os.getcwd()
    dest_dir = '../../data_process/post_process/filter_workflow/init'
    dirs = [name for name in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, name))]
    list1 = [name for name in os.listdir(current_dir) if
             os.path.isdir(os.path.join(current_dir, name)) and "_M" in name]
    list2 = sorted({name.split("_M", 1)[0] for name in list1})

    print(f"[\033[32mInfo\033[0m] Current dirs:")
    print("All logs:\n", list1)
    print("Log sources:\n", list2)
    print("Destination dir:", dest_dir)
    input("Continue? (press any key)")
    copy_to_workflow(".", dest_dir)
