import re
from fractions import Fraction

from ownership.renderer import (FRACTION_WORDS, percent_from_fraction_word,
                                percent_from_shares, render)
from tests.test_generator import dataset


def test_every_edge_stated_exactly_once_at_its_sentence_index():
    for s in dataset():
        docs, statements = render(s)
        by_id = {d.doc_id: d for d in docs}
        stated = [e for st in statements for e in st.edges]
        assert sorted(map(repr, stated)) == sorted(map(repr, s.edges))
        for st in statements:
            doc = by_id[st.doc_id]
            assert doc.sentences[st.sentence_index] == st.text
            assert st.text in doc.text


def test_share_count_phrasings_round_trip():
    seen = 0
    for s in dataset():
        for st in render(s)[1]:
            if st.template == "hard_shares":
                held, total = (int(x.replace(",", "")) for x in
                               re.search(r"holds ([\d,]+) of the ([\d,]+) outstanding", st.text).groups())
                assert percent_from_shares(held, total) == st.edges[0].percent
                seen += 1
    assert seen > 0


def test_fraction_phrasings_round_trip():
    seen = 0
    for s in dataset():
        for st in render(s)[1]:
            if st.template.startswith("hard_fraction"):
                word = next(w for w in FRACTION_WORDS.values() if w in st.text)
                assert percent_from_fraction_word(word) == st.edges[0].percent
                seen += 1
    assert seen > 0
    for percent, word in FRACTION_WORDS.items():
        assert percent_from_fraction_word(word) == percent


def test_remainder_sentences_cover_a_full_100_percent():
    seen = 0
    for s in dataset():
        for st in render(s)[1]:
            if st.template == "hard_remainder":
                assert sum(e.percent for e in st.edges) == 100 and len(st.edges) >= 2
                seen += 1
    assert seen > 0


def test_officers_are_mentioned_but_own_nothing():
    for s in dataset():
        docs, _ = render(s)
        owner_names = {s.entities[e.owner].name for e in s.edges}
        text = "".join(d.text for d in docs)
        for name, role, company in s.officers:
            assert name not in owner_names
        assert any(name in text for name, _, _ in s.officers)


def test_easy_uses_direct_percents_only():
    for s in dataset():
        if s.difficulty == "easy":
            assert all(st.template.startswith("easy") for st in render(s)[1])
