from __future__ import annotations

from flame.training.flex_data import current_file_names

FOLDER = "/Home/psi-data/PSI-25/metadata"
OLD_NAME = "PSI-25_metadata_PSI-101_BASS-II.zip"
NEW_NAME = "PSI-25_metadata_PSI_metadata_PSI-101_BASS-II.zip"
RENAME = {"file_name": OLD_NAME, "new_file_name": NEW_NAME}
TABLE = {"file_name": "PSI-25_Experimental table_BASS-II.csv",
         "remote_path": "/Home/psi-data/PSI-25/Experimental table/PSI-25_Experimental table_BASS-II.csv"}


def stored(name: str) -> dict:
    return {"file_name": name, "remote_path": f"{FOLDER}/{name}"}


def version(number: int, **lists: list[dict]) -> dict:
    return {"version": number, "added": [], "modified": [], "removed": [], **lists}


def test_a_rename_without_remote_path_keeps_the_folder_and_takes_the_new_name():
    versions = [version(1, added=[stored(OLD_NAME), TABLE]), version(2, modified=[RENAME])]
    assert current_file_names(versions) == sorted([NEW_NAME, TABLE["file_name"]])


def test_a_renamed_file_listed_again_with_its_path_appears_once():
    versions = [version(1, added=[stored(OLD_NAME)]), version(2, modified=[RENAME]),
                version(3, modified=[stored(NEW_NAME)])]
    assert current_file_names(versions) == [NEW_NAME]


def test_versions_are_applied_in_order():
    versions = [version(2, removed=[TABLE]), version(1, added=[stored(OLD_NAME), TABLE])]
    assert current_file_names(versions) == [OLD_NAME]
