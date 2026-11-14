"""HTML extraction: whole page inside <form>, hidden modal, link menus, duplicated paragraphs."""
from flame.corpus.extract import html_text

PAGE = """
<html><body><form id="aspnetForm">
  <header><a href="/">Home</a></header>
  <div class="menu"><a href="/h">History</a> <a href="/c">Crew</a> <a href="/a">Accident</a></div>
  <h2>Apollo 1 History</h2>
  <p>In November 1966, after the Gemini program, NASA turned its focus to the first crewed Apollo mission.
     The crew trained for months on the Command Module, and the agency set ambitious goals for 1967,
     including human-rating the Saturn IB and Saturn V rockets.</p>
  <p>In November 1966, after the Gemini program, NASA turned its focus to the first crewed Apollo mission.
     The crew trained for months on the Command Module, and the agency set ambitious goals for 1967,
     including human-rating the Saturn IB and Saturn V rockets.</p>
  <div class="reveal" data-reveal=""><p>Text from another page that the site never shows.</p></div>
  <ul><li><a href="/x">Related article one</a></li><li><a href="/y">Related article two</a></li></ul>
  <h3>Related Stories</h3>
</form></body></html>
"""


def test_html_keeps_content_and_drops_junk():
    text, tag = html_text(PAGE)
    assert tag == "body"  # no <main> or <article>: falls back to <body>
    assert "first crewed Apollo mission" in text  # content inside <form> is kept
    assert text.count("first crewed Apollo mission") == 1  # duplicate paragraph removed
    assert "never shows" not in text  # hidden modal removed
    assert "History Crew" not in text and "Related article" not in text  # link menus removed
    assert "Related Stories" not in text  # trailing heading without body removed
    assert "Apollo 1 History" in text  # heading followed by content is kept
