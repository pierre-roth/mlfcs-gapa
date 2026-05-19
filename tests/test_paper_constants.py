from mlfcs_gapa.paper.constants import PAPER, PAPER_STOCKS, PAPER_TRADING_DAYS_201911


def test_paper_lob_shape() -> None:
    assert PAPER.lob_levels == 10
    assert PAPER.window_length == 50
    assert PAPER.lob_width == 40
    assert PAPER.lob_window_shape == (50, 40)


def test_paper_inventory_and_action_constants() -> None:
    assert PAPER.minimum_trade_unit == 100
    assert PAPER.omega_inventory_units == 10
    assert PAPER.max_inventory == 1000
    assert PAPER.max_bias == 0.05
    assert PAPER.max_spread == 0.1


def test_paper_dataset_scope() -> None:
    assert set(PAPER_STOCKS) == {"000001", "000858", "002415"}
    assert len(PAPER_TRADING_DAYS_201911) == 21
