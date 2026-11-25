from src.evaluate import safe_mean, safe_median


def test_safe_mean():
    assert safe_mean([1, 2, 3]) == 2


def test_safe_median():
    assert safe_median([1, 3, 2]) == 2