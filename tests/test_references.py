"""Bibliography removal in PDFs: citation lists go, normal prose stays."""
from flame.corpus.extract import is_citation_paragraph, strip_references

# Real paragraph from the BASS fact sheet (references page) that polluted vector search.
BASS_REFS = ("Extinguishment Mechanisms of Co-flow Diffusion Flames in a Cup-Burner Apparatus. Proceedings of The "
             "Combustion Institute. Vol. 31 (2007). • Takahashi F, Linteris GT, Katta VR. Suppression of Cup-Burner "
             "Flames. Fire Safety Science. Vol. 8, pp. 1239-1250 (2005).")
PROSE = ("BASS-II tests produced data on how ambient oxygen, ventilation, and fuel affect combustion and burning. "
         "Theoretical formulas and data on flame spread rates in microgravity were compared with results from "
         "normal-gravity tests, and the flames burned at lower oxygen concentrations than expected.")
PROSE_WITH_ONE_CITATION = ("Olson et al. showed that thin PMMA samples burned in still air in 2015, which changed "
                           "how material flammability limits are interpreted for spacecraft fire safety.")


def test_citation_paragraph_detection():
    assert is_citation_paragraph(BASS_REFS)
    assert not is_citation_paragraph(PROSE)
    assert not is_citation_paragraph(PROSE_WITH_ONE_CITATION)


def test_everything_after_references_heading_is_dropped():
    pages = [PROSE, "\n\n".join([PROSE, "References", "Smith J. A plain-looking reference line", BASS_REFS])]
    out, removed = strip_references(pages)
    assert out[0] == PROSE
    assert out[1] == PROSE
    assert removed == 3
