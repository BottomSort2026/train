"""current_files: versions applied in order, renames applied, removals respected."""
from flame.corpus.inventory import current_files

P = "/Home/psi-data/PSI-1/Raw Data/"


def test_versions_are_applied_in_order_with_renames():
    versions = {"versions": [  # deliberately out of order
        {"version": 3, "added": [], "removed": [],
         "modified": [{"file_name": "C.txt", "new_file_name": "C_final.txt", "remote_path": P + "C.txt"}]},
        {"version": 1, "added": [{"file_name": n, "remote_path": P + n} for n in ("A.pdf", "B.pdf", "C.txt")],
         "modified": [], "removed": []},
        {"version": 2, "added": [], "modified": [], "removed": [{"file_name": "B.pdf", "remote_path": P + "B.pdf"}]},
    ]}
    files = {f["file_name"]: f for f in current_files(versions)}
    assert set(files) == {"A.pdf", "C_final.txt"}  # B stays removed; C is renamed
    assert files["C_final.txt"]["remote_path"] == P + "C_final.txt"
    assert files["C_final.txt"]["previous_file_name"] == "C.txt"
