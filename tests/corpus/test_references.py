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


# Reference lists embedded in PSI metadata fields (investigation cards).
PSI21_STYLE = ("This investigation gathered benchmark data, detailed in the following publications:\n\n"
               "    1)  Smooke, M.D., Hall, R.J., Colket, M.B., Investigation of the transition towards heavily sooting\n"
               "  flames. Combustion Theory and Modelling, 2004. 8(3): p. 593-606. https://doi.org/10.1088/1364-7830/8/3/009\n"
               "    2)  Luque, J., Jeffries, J.B., CH(A-X) and OH(A-X) optical emission in a laminar diffusion flame.\n"
               "  Combustion and Flame, 2000. 122(1/2): p. 172-175. https://doi.org/10.1016/S0010-2180(00)00112-7")
PSI142_STYLE = (PROSE + "\n\nSpecific research details are outlined in ''Dynamics of cool flames'' "
                "[https://doi.org/10.1016/j.pecs.2019.100787] and ''Understanding Cool Flames'', "
                "[https://doi.org/10.1016/j.proci.2020.09.019].")


def test_numbered_reference_list_in_a_field_is_removed():
    from flame.corpus.extract import drop_citations
    out = drop_citations(PSI21_STYLE)
    assert "benchmark data" in out and "doi.org" not in out


def test_prose_paragraph_survives_next_to_a_reference_paragraph():
    from flame.corpus.extract import drop_citations
    out = drop_citations(PSI142_STYLE)
    assert out == PROSE
