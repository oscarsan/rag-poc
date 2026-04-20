from app.services.language import detect_language


def test_detects_finnish_by_umlauts():
    assert detect_language("Onko mökissä sauna?") == "fi"


def test_detects_finnish_by_common_words():
    assert detect_language("Paljonko husky safari maksaa") == "fi"


def test_defaults_to_english():
    assert detect_language("How much is the husky safari?") == "en"


def test_empty_defaults_to_english():
    assert detect_language("") == "en"
