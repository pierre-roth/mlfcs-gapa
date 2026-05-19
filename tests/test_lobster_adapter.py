import polars as pl

from mlfcs_gapa.data.lobster import load_lobster_csv
from mlfcs_gapa.data.schema import lob_columns


def test_load_lobster_csv_maps_message_and_orderbook_schema(tmp_path) -> None:
    message_path = tmp_path / "message.csv"
    orderbook_path = tmp_path / "orderbook.csv"
    messages = pl.DataFrame(
        [
            [34200.0, 1, 1, 100, 100000, 1],
            [34200.1, 1, 2, 200, 100100, -1],
            [34200.2, 4, 2, 50, 100100, -1],
            [34200.3, 2, 1, 25, 100000, 1],
        ],
        orient="row",
    )
    orderbook_rows = []
    for i in range(4):
        row = []
        for level in range(1, 11):
            row.extend(
                [100100 + 100 * level + i, 1000 + level, 100000 - 100 * level - i, 900 + level]
            )
        orderbook_rows.append(row)
    orderbook = pl.DataFrame(orderbook_rows, orient="row")
    messages.write_csv(message_path, include_header=False)
    orderbook.write_csv(orderbook_path, include_header=False)

    dataset = load_lobster_csv(
        message_path=message_path,
        orderbook_path=orderbook_path,
        stock="AAPL",
        day="2012-06-21",
        levels=10,
    )

    assert dataset.stock == "AAPL"
    assert dataset.orderbook.height == 4
    assert dataset.orderbook.select(lob_columns()).width == 40
    assert dataset.orderbook["ask1_price"][0] == 10.02
    assert dataset.messages["limit_buy_volume"].to_list() == [100, 0, 0, 0]
    assert dataset.messages["limit_sell_volume"].to_list() == [0, 200, 0, 0]
    assert dataset.messages["market_buy_volume"].to_list() == [0, 0, 50, 0]
    assert dataset.messages["withdraw_buy_volume"].to_list() == [0, 0, 0, 25]
    assert dataset.trades["trade_price_max"].to_list() == [0.0, 0.0, 10.01, 0.0]
