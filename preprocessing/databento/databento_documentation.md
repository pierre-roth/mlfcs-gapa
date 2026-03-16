What's a schema?

Databento supports over 15 different data formats of market data. When you make a request from Databento, you must usually specify which data format—also called a schema—to receive your data in.

Supported schemas and their fields

A schema represents a collection of data fields. The following is a summary of schemas supported by Databento. Click on any schema below for its details, the fields included, and a data dictionary that defines each field.

Schema	Schema IDs	Common names used by third parties
MBO	mbo	L3, Market by order, full order book, tick data
MBP-10	mbp-10	L2, Market by price, market depth
MBP-1 / CMBP-1	mbp-1 / cmbp-1	L1, Top of book, trades and quotes
BBO / CBBO	bbo-1s, bbo-1m / cbbo-1s, cbbo-1m	L1, Top of book sampled in time space, subsampled BBO and trades
TBBO / TCBBO	tbbo / tcbbo	L1, Top of book sampled in trade space
Trades	trades	L1, Last sale, time and sales, tick-by-tick trades
OHLCV	ohlcv-1s, ohlcv-1m, ohlcv-1h, ohlcv-1d	L0, OHLCV bars, aggregates
Definition	definition	L0, Security definitions, reference data, symbol list
Imbalance	imbalance	L3, Auction imbalance, order imbalance, NOII
Statistics	statistics	L0, Session or daily statistics, end-of-day summary, open interest
Status	status	L0, Market or trading state/status
Market by order (MBO) provides every order book event across every price level, keyed by its order ID. This allows you to determine the queue position of each order and provides the highest level of granularity available.

Market by price (MBP-10) provides every order book event across the top ten price levels, keyed by price. This includes every trade and changes to aggregate market depth, alongside total size and order count at the top ten price levels.

Market by price (MBP-1) provides every order book event that updates the top price level, also known as the best bid and offer (BBO). This includes every trade and changes to book depth, alongside total size and order count at the BBO.

Consolidated market by price (CMBP-1) provides every order book event that updates the top price level across all venues in the dataset, also known as the consolidated best bid and offer (CBBO). This includes every trade and changes to book depth, alongside total size and publisher attribution at the CBBO.

BBO on trade (TBBO) provides every trade event alongside the BBO immediately before the effect of each trade. This is a subset of MBP-1.

Consolidated BBO on trade (TCBBO) provides every trade event alongside the consolidated BBO immediately before the effect of each trade. This is a subset of CMBP-1.

BBO on interval (BBO) provides the last best bid, best offer, and sale at 1-second or 1-minute intervals. This is a subset of MBP-1.

Consolidated BBO on interval (CBBO) provides the consolidated last best bid, best offer, and sale at 1-second or 1-minute intervals. This is a subset of CMBP-1.

Trades provides every trade event. This is a subset of MBO.

Aggregate bars (OHLCV) provide open, high, low, and close prices and total volume aggregated from trades at 1-second, 1-minute, 1-hour, or 1-day intervals.

Instrument definitions provide reference information about each instrument, which includes properties like symbol, instrument name, expiration date, listing date, tick size, and strike price.

Imbalance provides auction imbalance data such as paired quantity, total quantity, and auction status.

Statistics provides official summary statistics of each instrument that's published by the venue. This generally includes properties like daily volume, open interest, preliminary and final settlement prices, and official open, high, and low prices.

Status provides updates about the trading session, such as halts, pauses, short-selling restrictions, auction start, and other matching engine statuses. The granularity and frequency of these updates vary by publisher and dataset.

See also
See also
The MBP-1, BBO and TBBO schemas, as well as the CMBP-1, CBBO, and TCBBO schemas, all provide top of book data with different update space and sampling intervals. Learn more about their differences in our MBP-1 vs. BBO vs. TBBO schemas guide.
Why are Databento's naming conventions different from third parties?

Databento avoids terms like Level 1 (L1) or Level 2 (L2) due to their inconsistent application. For example, some vendors refer to both MBO and MBP data as L2, while others strictly refer to MBP data as L2. More misleadingly, many vendors refer to MBO data as L3, even though this doesn't follow any major trading venue's naming convention.

Likewise, the term tick originates from the concept of a ticker tape and only refers to trades, not resting limit orders. This becomes a source of confusion when vendors use the term tick data to refer to either MBO or MBP data when it should be strictly reserved for trades data.

See also
See also
For more Databento naming conventions and key terminology, visit our FAQs.
Special cases

Our MBO, MBP-1 and MBP-10 schemas adopt the following conventions in these special cases:

Combining MBO with trades feed: Typically, MBO data provides the highest granularity, but certain venues enhance their trades feed with additional information like trades not reflected in the MBO feed, consolidated NBBO, and more. In these cases, we document the exception in our Venues and datasets section and recommend that you request both our MBO and trades schemas if you need the highest level of granularity.
Deriving one schema from another

Databento captures market data directly from the source and is only subscribed to the most granular feed(s) available from each publisher. Order book feeds are usually normalized into our MBO schema and top-of-book feeds are usually normalized into our MBP-1 schema.

To ensure consistency between schemas, Databento doesn't source the less granular schemas from separate feeds. Instead, Databento derives all of the less granular schemas starting from the most granular schema available. As you may have noticed from our schema's descriptions, the majority of them—MBP, BBO, TBBO, trades, and OHLCV—are simply derived from MBO data.

Likewise, you can also derive one schema from another losslessly on the client side, and you should expect your self-derived data to be consistent with ours. For example:

MBP-1, BBO, and Trades can be derived from MBP-10.
BBO, TBBO, and Trades can be derived from MBP-1.
Trades and OHLCV can be derived from TBBO.
OHLCV can be derived from Trades.
Deriving your own schema is useful for various reasons:

The data needs to be defined differently for your application.
Our derivation differs from those of another vendor and you want transparency.
You can cut down the number of API requests made to Databento by getting the most granular schema that you need and deriving the rest yourself.
Databento provides MBP-10 merely as a convenience feature. You can reduce bandwidth requirements, latency, and transfer time significantly by deriving MBP-10 yourself from MBO.
This is especially relevant for OHLCV, which can vary depending on how trade breaks or market halts are managed, how the start and end of each time interval are determined, and how illiquid instruments are handled if there are no trades over a given time interval. If these considerations are trivial for your use case, Databento offers OHLCV data in multiple time intervals (seconds, minutes, hours, and daily) for your convenience.

The table below summarizes which schemas can be derived from the another. Each row represents the original schema, and each column represents schemas that you can derive from the original schema.

Schema	MBO	MBP-10	MBP-1	CMBP-1	TBBO	TCBBO	BBO-1s	BBO-1m	CBBO-1s	CBBO-1m	Trades	OHLCV-1s	OHLCV-1m	OHLCV-1h	OHLCV-1d
MBO	✓	✓	✓		✓		✓	✓			✓	✓	✓	✓	✓
MBP-10		✓	✓		✓		✓	✓			✓	✓	✓	✓	✓
MBP-1			✓		✓		✓	✓			✓	✓	✓	✓	✓
CMBP-1				✓		✓			✓	✓	✓	✓	✓	✓	✓
TBBO					✓						✓	✓	✓	✓	✓
TCBBO						✓					✓	✓	✓	✓	✓
BBO-1s							✓	✓							
BBO-1m								✓							
CBBO-1s									✓	✓					
CBBO-1m										✓					
Trades											✓	✓	✓	✓	✓
OHLCV-1s												✓	✓	✓	✓
OHLCV-1m													✓	✓	✓
OHLCV-1h														✓	✓
OHLCV-1d															✓
See also
See also
Learn how to resample trades data to other intervals, such as 5-minute intervals, from our bar aggregation tutorial.
You can also learn how to generate MBP-10 from MBO data using an order book, as seen in our limit order book construction tutorial.




Market by order (MBO)

Market by order (MBO) provides every order book event across every price level, keyed by its order ID. This allows you to determine the queue position of each order and provides the highest level of granularity available.

MBO data includes all trades, fills, adds, cancels, modifies (or replaces), book clear events, and, depending on the venue and dataset, other special order events. It is often called "L3 data".

Fields (mbo)

Field	Type	Description
ts_recv	uint64_t	The capture-server-received timestamp expressed as the number of nanoseconds since the UNIX epoch. See ts_recv.
ts_event	uint64_t	The matching-engine-received timestamp expressed as the number of nanoseconds since the UNIX epoch. See ts_event.
rtype	uint8_t	A sentinel value indicating the record type. Always 160 in the MBO schema. See Rtype.
publisher_id	uint16_t	The publisher ID assigned by Databento, which denotes the dataset and venue. See Publishers.
instrument_id	uint32_t	The numeric instrument ID. See Instrument identifiers.
action	char	The event action. Can be Add, Cancel, Modify, cleaR book, Trade, Fill, or None. See Action.
side	char	The side that initiates the event. Can be Ask for a sell order (or sell aggressor in a trade), Bid for a buy order (or buy aggressor in a trade), or None where no side is specified. See Side.
price	int64_t	The order price where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001. See Prices.
size	uint32_t	The order quantity.
channel_id	uint8_t	The channel ID assigned by Databento as an incrementing integer starting at zero.
order_id	uint64_t	The order ID assigned by the venue.
flags	uint8_t	A bit field indicating event end, message characteristics, and data quality. See Flags.
ts_in_delta	int32_t	The matching-engine-sending timestamp expressed as the number of nanoseconds before ts_recv. See ts_in_delta.
sequence	uint32_t	The message sequence number assigned at the venue.
Snapshots

For the convenience of managing state and recovery, Databento provides a synthetic snapshot of the order book at the start of each UTC day in our historical MBO data and periodic book snapshots in our real-time MBO data. The mechanics of these snapshots is detailed here.

See also
See also
Learn more about the different action types and how to manage order state with respect to each action from our State management of resting orders tutorial.
Also learn how to construct a limit order book from MBO data from our limit order book construction tutorial.
MBO data normalization differs slightly from one venue or dataset to another. Edge cases and differences are documented separately for each venue in the Venues and datasets section.


Trades

Trades provides every trade event. This is a subset of MBO.

This is often referred to as "time and sales", "last sale," or "tick data."

Fields (trades)

Field	Type	Description
ts_recv	uint64_t	The capture-server-received timestamp expressed as the number of nanoseconds since the UNIX epoch. See ts_recv.
ts_event	uint64_t	The matching-engine-received timestamp expressed as the number of nanoseconds since the UNIX epoch. See ts_event.
rtype	uint8_t	A sentinel value indicating the record type. Always 0 in the trades schema. See Rtype.
publisher_id	uint16_t	The publisher ID assigned by Databento, which denotes the dataset and venue. See Publishers.
instrument_id	uint32_t	The numeric instrument ID. See Instrument identifiers.
action	char	The event action. Always Trade in the trades schema. See Action.
side	char	The side that initiates the trade. Can be Ask for a sell aggressor in a trade, Bid for a buy aggressor in a trade, or None where no side is specified. See Side.
depth	uint8_t	The book level where the update event occurred.
price	int64_t	The order price where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001. See Prices.
size	uint32_t	The order quantity.
flags	uint8_t	A bit field indicating event end, message characteristics, and data quality. See Flags.
ts_in_delta	int32_t	The matching-engine-sending timestamp expressed as the number of nanoseconds before ts_recv. See ts_in_delta.
sequence	uint32_t	The message sequence number assigned at the venue.



BBO on trade (TBBO)

BBO on trade (TBBO) provides every trade event alongside the BBO immediately before the effect of each trade. This is a subset of MBP-1.

Fields (tbbo)

Field	Type	Description
ts_recv	uint64_t	The capture-server-received timestamp expressed as the number of nanoseconds since the UNIX epoch. See ts_recv.
ts_event	uint64_t	The matching-engine-received timestamp expressed as the number of nanoseconds since the UNIX epoch. See ts_event.
rtype	uint8_t	A sentinel value indicating the record type. Always 1 in the TBBO schema. See Rtype.
publisher_id	uint16_t	The publisher ID assigned by Databento, which denotes the dataset and venue. See Publishers.
instrument_id	uint32_t	The numeric instrument ID. See Instrument identifiers.
action	char	The event action. Always Trade in the TBBO schema. See Action.
side	char	The side that initiates the trade. Can be Ask for a sell aggressor in a trade, Bid for a buy aggressor in a trade, or None where no side is specified. See Side.
depth	uint8_t	The book level where the update event occurred.
price	int64_t	The order price where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001. See Prices.
size	uint32_t	The order quantity.
flags	uint8_t	A bit field indicating event end, message characteristics, and data quality. See Flags.
ts_in_delta	int32_t	The matching-engine-sending timestamp expressed as the number of nanoseconds before ts_recv. See ts_in_delta.
sequence	uint32_t	The message sequence number assigned at the venue.
bid_px_00	int64_t	The bid price at the top level where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001. See Prices.
ask_px_00	int64_t	The ask price at the top level where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001. See Prices.
bid_sz_00	uint32_t	The bid size at the top level.
ask_sz_00	uint32_t	The ask size at the top level.
bid_ct_00	uint32_t	The bid order count at the top level.
ask_ct_00	uint32_t	The ask order count at the top level.
Implementation differences between clients and encodings

Bid and ask depth messages (fields starting with bid_ and ask_) are structured differently in the C++ and Rust clients, the Python record interface, and JSON data. Instead of using the _N suffix, they're stored in an array of structures named levels, with the top-of-book at index 0.

For example, in C++, levels[5].bid_px corresponds to bid_px_05 in the Python DataFrame API and CSV format.

See also
See also
TBBO has many similarities to the MBP-1 and BBO schemas. The main distinction is that TBBO is in trade space, while MBP-1 is in book update space, and BBO is in time space. In particular, the action type of TBBO is always Trade. Learn about the differences between each in our MBP-1 vs. TBBO vs. BBO schemas guide.
Consolidated BBO on trade (TCBBO)

Consolidated BBO on trade (TCBBO) provides every trade event alongside the consolidated BBO immediately before the effect of each trade. This is a subset of CMBP-1.

Fields (tcbbo)

Field	Type	Description
ts_recv	uint64_t	The capture-server-received timestamp expressed as the number of nanoseconds since the UNIX epoch. See ts_recv.
ts_event	uint64_t	The matching-engine-received timestamp expressed as the number of nanoseconds since the UNIX epoch. See ts_event.
rtype	uint8_t	A sentinel value indicating the record type. Always 194 in the TCBBO schema. See Rtype.
publisher_id	uint16_t	The publisher ID indicating the venue where the trade executed on. See Publishers.
instrument_id	uint32_t	The numeric instrument ID. See Instrument identifiers.
action	char	The event action. Always Trade in the TCBBO schema. See Action.
side	char	The side that initiates the event. Can be Ask for a sell aggressor in a trade, Bid for a buy aggressor in a trade, or None where no side is specified. See Side.
price	int64_t	The order price where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001. See Prices.
size	uint32_t	The order quantity.
flags	uint8_t	A bit field indicating event end, message characteristics, and data quality. See Flags.
ts_in_delta	int32_t	The matching-engine-sending timestamp expressed as the number of nanoseconds before ts_recv. See ts_in_delta.
bid_px_00	int64_t	The bid price at the top level where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001. See Prices.
ask_px_00	int64_t	The ask price at the top level where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001. See Prices.
bid_sz_00	uint32_t	The bid size at the top level.
ask_sz_00	uint32_t	The ask size at the top level.
bid_pb_00	uint16_t	The publisher ID indicating the venue containing the best bid. See Publishers.
ask_pb_00	uint16_t	The publisher ID indicating the venue containing the best ask. See Publishers.
TCBBO publisher

The publisher_id field will correspond to the venue the trade executed on.

bid_pb_00 and ask_pb_00 will represent the individual venues showing the NBBO.

Implementation differences between clients and encodings

Bid and ask depth messages (fields starting with bid_ and ask_) are structured differently in the C++ and Rust clients, the Python record interface, and JSON data. Instead of using the _N suffix, they're stored in an array of structures named levels, with the top-of-book at index 0.

For example, in C++, levels[5].bid_px corresponds to bid_px_05 in the Python DataFrame API and CSV format.

See also
See also
TCBBO has many similarities to the CMBP-1 and CBBO schemas. The main distinction is that TCBBO is in trade space, while CMBP-1 is in book update space, and CBBO is in time space. In particular, the action type of TBBO is always Trade. Learn about the differences between each in our CMBP-1 vs. TCBBO vs. CBBO schemas guide.


Market by price (MBP-10)

MBP-10 (market by price) provides every order book event across the top ten price levels, keyed by price. This includes every trade and changes to aggregate market depth, alongside total size and order count at the top ten price levels.

This is often called "L2 data".

Fields (mbp-10)

Field	Type	Description
ts_recv	uint64_t	The capture-server-received timestamp expressed as the number of nanoseconds since the UNIX epoch. See ts_recv.
ts_event	uint64_t	The matching-engine-received timestamp expressed as the number of nanoseconds since the UNIX epoch. See ts_event.
rtype	uint8_t	A sentinel value indicating the record type. Always 10 in the MBP-10 schema. See Rtype.
publisher_id	uint16_t	The publisher ID assigned by Databento, which denotes the dataset and venue. See Publishers.
instrument_id	uint32_t	The numeric instrument ID. See Instrument identifiers.
action	char	The event action. Can be Add, Cancel, Modify, cleaR book, or Trade. See Action.
side	char	The side that initiates the event. Can be Ask for a sell order (or sell aggressor in a trade), Bid for a buy order (or buy aggressor in a trade), or None where no side is specified. See Side.
depth	uint8_t	The book level where the update event occurred.
price	int64_t	The order price where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001. See Prices.
size	uint32_t	The order quantity.
flags	uint8_t	A bit field indicating event end, message characteristics, and data quality. See Flags.
ts_in_delta	int32_t	The matching-engine-sending timestamp expressed as the number of nanoseconds before ts_recv. See ts_in_delta.
sequence	uint32_t	The message sequence number assigned at the venue.
bid_px_N	int64_t	The bid price at level N (top level if N = 00) where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001. See Prices.
ask_px_N	int64_t	The ask price at level N (top level if N = 00) where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001. See Prices.
bid_sz_N	uint32_t	The bid size at level N (top level if N = 00).
ask_sz_N	uint32_t	The ask size at level N (top level if N = 00).
bid_ct_N	uint32_t	The bid order count at level N (top level if N = 00).
ask_ct_N	uint32_t	The ask order count at level N (top level if N = 00).
Implementation differences between clients and encodings

Bid and ask depth messages (fields starting with bid_ and ask_) are structured differently in the C++ and Rust clients, the Python record interface, and JSON data. Instead of using the _N suffix, they're stored in an array of structures named levels, with the top-of-book at index 0.

For example, in C++, levels[5].bid_px corresponds to bid_px_05 in the Python DataFrame API and CSV format.

See also
See also
It is possible to construct MBP-10 yourself from MBO data if you want more price levels or prefer to reduce your bandwidth use. Learn how to construct a limit order book from MBO data from our limit order book construction tutorial.


Common fields, enums and types

Publishers, datasets, and venues

We use a few different terms to describe our data:

A dataset is a source of data.
A venue is an exchange, OTC market (e.g., ATS, ECN) or reporting entity.
A publisher is a specific venue from a specific dataset.
See also
See also
Read our Venues and publishers section for a more detailed explanation.
Publisher identifiers

All of our schemas include a publisher_id field, which is a unique numeric ID assigned by Databento to each publisher. A full list of publishers can be found using the metadata.list_publishers endpoint.

Venue and dataset identifiers

Each publisher is also assigned a string identifier (e.g., OPRA.PILLAR.XCBO), composed of two parts:

Dataset ID (e.g., OPRA.PILLAR). This is used as the dataset argument in any API or client method. Dataset IDs can be found on the Databento portal on each dataset's details page or via the metadata.list_datasets endpoint.
Venue (e.g., XCBO). For most markets, this is its ISO 10383 MIC code, which is guaranteed to be four characters long. For entities without a MIC code, this string is arbitrarily assigned by Databento and will also be four characters long.
Instrument identifiers

All of our schemas contain an instrument_id field which is a numeric ID that maps to a given instrument. In most cases, this numeric ID is assigned by the publisher. For publishers that do not assign this value, we create a synthetic mapping for it.

instrument_id is only guaranteed to be unique within a given day. Some publishers provide a different instrument ID on different days for the same underlying instrument. Other publishers may use the same instrument ID for different underlying instruments at different points in time.

Depending on the use case, it may be easier to work with other symbology types such as raw_symbol. Our symbology documentation outlines the various symbology types we support.

Timestamps

All the timestamps in our data are expressed as the number of nanoseconds since the UNIX epoch, i.e. UNIX timestamps. All timestamp fields are prefixed with ts_. Some of our timestamps are encoded as the difference, i.e. delta, relative to another timestamp. Such timestamp fields are suffixed with _delta.

We provide four types of timestamps, through the following fields:

Event timestamps, ts_event
Publisher sending timestamp, ts_in_delta
Databento receive timestamp, ts_recv
Databento sending timestamp (live only), ts_out
UNDEF_TIMESTAMP (UINT64_MAX, 18446744073709551615) is used to denote a null or undefined timestamp.

The event and publisher sending timestamps are provided by the publisher (or market), and we provide their original values without any adjustment.

ts_event

Most users will only need the event timestamp, i.e. ts_event. For market data, this represents the time that the event is received by the matching engine (tag 60 in FIX encoding).

The exact location where this timestamp is taken varies with matching engine architecture of each market. Some markets will handle different subsets of instruments on independent order gateways, while other markets will load balance the same subset of instruments across independent order gateways. Some markets take the event timestamp at the time it is received on the independent order gateways, while others may take this timestamp at the time it reaches a FIFO matching queue. In the former case, the clocks on independent order gateways are often not properly synchronized to the same clock source. Since we do not adjust the publisher's timestamps, any non-monotonicity in the original data will remain.

ts_in_delta

The publisher sending timestamp represents the time when the data message associated with an event is sent (tag 52 in FIX encoding). We encode this information in ts_in_delta, which expresses the number of nanoseconds between the Databento receive timestamp (ts_recv) and the publisher sending timestamp. To get the sending timestamp itself, simply subtract ts_in_delta from ts_recv. Since the publisher and Databento are not necessarily synchronized to the same clock source, ts_in_delta may be negative.

ts_in_delta is a 32-bit signed integer. The minimum will clamp to INT32_MIN and the maximum will clamp to INT32_MAX, even if the true value exceeds these limits.

Some markets do not provide both match event timestamps and sending timestamps. Often, they will provide only one of the two. In such cases, we take it that the event timestamp and sending timestamp assume the same value. As such, ts_event will be the provided timestamp and ts_in_delta will be equal to the difference between ts_recv and ts_event.

ts_recv

Unless otherwise specified, Databento receive timestamps, i.e. ts_recv, are synchronized against UTC with sub-microsecond accuracy. Moreover, these receive timestamps are always guaranteed to be monotonic for any given symbol.

These receive timestamps rely on hardware timestamping on the network adapter and are synchronized against a GPS clock source using PTP. The clock is corrected by slewing the time, which prevents discrete jumps backwards in time. In other words, our local receive timestamps are guaranteed to be monotonic for any given symbol.

ts_recv is also adjusted for leap seconds. The local receive timestamp is not immediately adjusted intraday when a leap second is introduced. Instead, the leap second update is applied at the end of the market session.

ts_out

For live data, we optionally include a timestamp of our data before it leaves our data gateways. This information is encoded as ts_out. Both ts_out and ts_recv are synced to the same GPS clock source. Subtracting ts_recv gives the number of nanoseconds spent in our system.

Index timestamp

All schemas have a primary timestamp that should be used for sorting records as well as indexing into any symbology data structure. This index timestamp will be ts_recv if it exists in the schema, otherwise it will be ts_event.

When requesting historical data, the data will be filtered based on the index timestamp.

When requesting data in CSV and JSON encodings, the first field will be set to this index timestamp. Additionally, for schemas that contain ts_recv, the second field will be set to ts_event.

Encodings

We support DBN, CSV, and JSON encodings for our data. DBN is an extremely fast message encoding and storage format for normalized market data. All official Databento client libraries use DBN under the hood, both as a data interchange format and for in-memory representation of data. DBN is also the default encoding for all Databento APIs, including live data streaming, historical data streaming, and batch flat files.

Our batch download system also supports CSV and JSON encodings.

Time zone

By default, all of our data is set in UTC time zone. Likewise, our site displays all dates and times in UTC by default.

Dates and times

We use the ISO 8601 date and time format to express dates and times used as parameters to our APIs. All dates and times used as parameters are in UTC by default.

The "reduced precision" concept in the ISO 8601 standard allows for dates and times to be represented with varying levels of detail. Any number of values may be dropped from any of the date and time representations, but in the order from the least to the most significant. For example, "2024-05" corresponds to "2024-05-01T00:00:00".

Any parameter that takes an ISO 8601 timestamp can instead be given a timestamp in nanoseconds since the UNIX epoch, as described in the above section.

All of our timestamp parameters are start-inclusive and end-exclusive.

Forward filling end parameters

For our APIs that take an optional end parameter as an ISO 8601 string, we will implement the following behavior when the end parameter is not provided:

We will forward fill any date or time components of the associated start parameter that are omitted. This "rounds up" the start timestamp for use as the end timestamp, and is done for more concise usage.

Examples of this behavior are shown below.

Info
Info
We will only forward fill timestamps with less than one-second resolution.
Start timestamp	Effective start timestamp	Forward filled end timestamp
"2024"	"2024-01-01T00:00:00"	"2025-01-01T00:00:00"
"2024-03"	"2024-03-01T00:00:00"	"2024-04-01T00:00:00"
"2024-03-10"	"2024-03-10T00:00:00"	"2024-03-11T00:00:00"
"2024-03-10T01"	"2024-03-10T01:00:00"	"2024-03-10T02:00:00"
"2024-03-10T00:01"	"2024-03-10T00:01:00"	"2024-03-10T00:02:00"
For example, a query for the entire month of March 2024 can be specified with start="2024-03" without an end.

rtype

An rtype or record type is an unsigned 8-bit discriminant in the header of every DBN record that indicates the type of record structure. Each schema has one rtype and by extension one record structure associated with it.

Info
Info
Some rtypes are not associated with a schema and are only present in live data.
Name	Hex	Decimal	Description
MBP-0	0x00	0	A market-by-price record with a book depth of 0. Used for the trades schema.
MBP-1	0x01	1	A market-by-price record with a book depth of 1. Used for the TBBO and MBP-1 schemas.
MBP-10	0x0A	10	A market-by-price record with a book depth of 10.
Status	0x12	18	An exchange status record.
Definition	0x13	19	An instrument definition record.
Imbalance	0x14	20	An order imbalance record.
Error	0x15	21	An error record from the live gateway.
Symbol mapping	0x16	22	A symbol mapping record from the live gateway.
System	0x17	23	A non-error record from the live gateway.
Statistics	0x18	24	A statistics record from the publisher.
OHLCV-1s	0x20	32	An OHLCV record at a 1-second cadence.
OHLCV-1m	0x21	33	An OHLCV record at a 1-minute cadence.
OHLCV-1h	0x22	34	An OHLCV record at an hourly cadence.
OHLCV-1d	0x23	35	An OHLCV record at a daily cadence.
MBO	0xA0	160	A market-by-order record.
CMBP-1	0xB1	177	A consolidated market-by-price record with a book depth of 1.
CBBO-1s	0xC0	192	A consolidated market-by-price record with a book depth of 1 at a 1-second cadence.
CBBO-1m	0xC1	193	A consolidated market-by-price record with a book depth of 1 at a 1-minute cadence.
TCBBO	0xC2	194	A consolidated market-by-price record with a book depth of 1 with only trades.
BBO-1s	0xC3	195	A market-by-price record with a book depth of 1 at a 1-second cadence.
BBO-1m	0xC4	196	A market-by-price record with a book depth of 1 at a 1-minute cadence.
Prices

Prices are expressed as signed integers in fixed-precision format, whereby every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001. For example, a price of 5411750000000 corresponds to 5411.75 (decimal format).

When requesting data via batch download in CSV and JSON encodings, you can optionally choose for prices to be returned in decimal format. If you are requesting data using the online portal, you can select Decimal prices in the Advanced customization section. Otherwise, you can specify the pretty_px parameter in batch.submit_job using the client libraries.

Additionally, our client libraries support functionality to view prices in decimal format.

In certain scenarios—such as calendar spreads in futures—prices can be negative.

UNDEF_PRICE is used to denote a null or undefined price. It will be equal to 9223372036854775807 (INT64_MAX) when using the fixed-precision integer format. When expressed in decimal format, it will be equal to null in JSON, or "" (an empty string) in CSV.

Side

The side field contains information about the side of an order event. It's meaning will vary depending on the action field.

When action is Trade:

A - The trade aggressor was a seller
B - The trade aggressor was a buyer
N - No side specified
When action is Fill:

A - A resting sell order was filled
B - A resting buy order was filled
N - No side specified
When action is Add, Modify, and Cancel:

A - A resting sell order updated the book
B - A resting buy order updated the book
N - No side specified
When action is cleaR book, side will always be N

side can be N in the following cases:

The source does not disseminate a side for trades.
Trades happening during opening and closing auctions
Trades against non-displayed orders
Trades involving implied orders
Off-exchange trades
The Venues and datasets section provides more information regarding the specific cases for each dataset where no side will be specified.

Action

The action field contains information about the type of order event contained in the message.

Name	Value	Action
Add	A	Insert a new order into the book.
Modify	M	Change an order's price and/or size.
Cancel	C	Fully or partially cancel an order from the book.
Clear	R	Remove all resting orders for the instrument.
Trade	T	An aggressing order traded. Does not affect the book.
Fill	F	A resting order was filled. Does not affect the book.
None	N	No action: does not affect the book, but may carry flags or other information.
Flags

The flags field is a bit field that contains information about the message. Multiple flags can be set on a single message.

The meaning of each bit is as follows:

Flag	Value	Decimal	Description
F_LAST	1 << 7	128	Marks the last record in a single event for a given  instrument_id.
F_TOB	1 << 6	64	Top-of-book message, not an individual order.
F_SNAPSHOT	1 << 5	32	Message sourced from a replay, such as a snapshot server.
F_MBP	1 << 4	16	Aggregated price level message, not an individual order.
F_BAD_TS_RECV	1 << 3	8	The ts_recv value is inaccurate due to clock issues or packet reordering.
F_MAYBE_BAD_BOOK	1 << 2	4	An unrecoverable gap was detected in the channel.
F_PUBLISHER_SPECIFIC	1 << 1	2	Semantics depend on the publisher_id. Refer to the relevant dataset supplement for more details.
1 << 0	1	Reserved for internal use can safely be ignored. May be set or unset.
Top-of-book datasets

Some datasets are built on feeds from vendors that only provide top-of-book information (best bid and offer). Top-of-book messages are normalized into a pair of MBO records with the Add action and the F_TOB flag (0x40, 64) set. Typically for these datasets, there is no information available about the passive side of trades, so there are no Fill records and the side of the Trade record is always set to None.

The removal of a price level is normalized as an Add action with a size of 0 and price of UNDEF_PRICE (INT64_MAX, 9223372036854775807) or NaN in Python. This indicates there's currently no quotes for that side.

Other schemas, such as MBP-1, Trades, and OHLCV remain the same for top-of-book datasets.

Market-by-price datasets

Some datasets are built on feeds from vendors that only provide market-by-price information (with limited depth).

Messages adding/modifying/deleting a price level are normalized into an MBO record with the Add/Modify/Cancel action, with the size field containing the full quantity at that level and the F_MBP flag (0x10, 16) set. A price level can be identified from the combination of the side and the price. The order_id field should be ignored for those messages.

If the upstream feed has a maximum depth, an additional record with Cancel action will be sent whenever a price level falls outside the maximum depth - even if there are still orders at that price level.

Typically for these datasets, there is no information available about the passive side of trades, so there are no Fill records.

MBP-10 will only include depth up to the depth provided by the publisher. The remaining levels will always be empty.

Other schemas, such as Trades and OHLCV are otherwise unaffected.

Normalization

Normalization refers to the process of exporting data in their various source formats to a single, unified format. Such a unified format is often called a normalization format (or normalized schema). The primary reason for normalizing market data is to abstract away differences between source formats, making the data easier to work with.

The normalization process is one of the most likely places where inaccuracies or data errors are introduced. This article describes these issues, the trade-offs for addressing them, and the reasons behind the design of Databento's normalization schema.

Examples of normalized data

For example, Nasdaq's proprietary TotalView data feed has a protocol with its own message format and provides market-by-order data, while IEX's proprietary TOPS data feed has a completely different protocol with another message format and provides top-of-book data. These are examples of raw data formats.
When you consume market data from a data redistributor's feed, the redistributor will have its own protocol and message format, distinct from the venues'. This is an example of normalized data.
The most sophisticated trading firms will generally collect data directly from their sources and normalize them to a proprietary format.
Common issues found in normalized market data

There are many ways in which normalization can introduce data errors, lossiness or performance issues.

Issue	Definition	Examples
Incompatible schema	The source schema and normalized schema are mismatched.	A direct market feed with an order-based schema is normalized to a vendor's schema that only provides aggregated market depth.
Truncated timestamps	A direct market feed which originally includes nanosecond-resolution timestamps is normalized to a schema that truncates the timestamps to a lower resolution.	Some vendors have a legacy data schema designed for older FIX dialects, forcing them to truncate nanosecond-resolution timestamps found in modern markets to millisecond resolution.
Discarded timestamps	A direct market feed which originally includes more than one timestamp field is normalized to a schema that discards that timestamp. This introduces imprecision when the normalized data is used for strategy backtesting.	A proprietary exchange feed may include both match (Tag 60) and sending (Tag 52) timestamps but a vendor's schema may preserve only one of the two.
Discarded or remapped sequence numbers	The normalized schema either discards the original message sequence numbers or remaps them to a vendor's own message sequence numbers.	This creates problems if you need to resolve post-trade issues with the market or your broker, as it makes it harder to identify the exact event.
Loss of price precision	The normalized schema represents prices in a type that loses precision.	Many vendors use floating point representation for prices, losing precision past 6 decimal places. This can create issues for trading Japanese yen spot rates, fixed income instruments and cryptocurrencies.
Loss of null semantics	The normalized schema represents null values in a way that changes the meaning.	Some data feeds will represent null prices with zeros or a negative value like -1. This can introduce errors downstream if the price is interpreted to be non-null. This is also problem if your application needs to handle both asset classes that can have negative prices (such as futures and spreads) and asset classes which cannot.
Loss of packet structure	The normalized schema does not preserve packet-level structure.	Many markets publish multiple events within a packet. Without packet-level structure, it may create the appearance of artificial trading opportunities between any two events within a packet.
Lossy or irreversible symbology mappings	The normalized schema adopts a proprietary symbology that is different from the original source's symbology. Sometimes, such proprietary symbology cannot be mapped back to the original.	Some vendors adopt a symbology system that only includes lead months of futures contracts, causing the far month contracts to be discarded.
Lossy abstraction	The normalized schema does not adequately standardize information across multiple datasets, resulting in the end user needing to understand the specifications of the various source schemas anyway in order to determine the lost information.	This often happens when normalizing less commonly used features such as matching engine statuses or instrument definitions. This puts significant burden on the user to study the specifications of various data feeds to understand the lost information.
Statelessness	The normalized schema provides incremental changes but does not provide snapshots or replay of order book state.	This presents an issue when using the normalized data in real-time, as the user loses information in the event of a disconnection or late join.
Coalescing	The normalized schema aggregates the information at a lower granularity.	A vendor may coalesce a feed of tick data with second bar aggregates or subsample a source feed.
Conflation	A normalized feed batches multiple updates into one at some lower frequency, to alleviate bandwidth limitations. Often present along with coalescing.	This is a common practice for retail brokerages, whose data feeds are designed more for display use and consumption over sparse WAN links.
Dropped packets	A normalized data feed deliberately discards data when the network or system is unable to keep up.	This is often present when the source feed or upstream parts of the vendor's infrastructure uses UDP for transmission.
Buffering	The data server sends stale data either because there is insufficient network bandwidth or the client is reading too slowly. The client misinterprets the stale data, either obscuring this effect or injecting incorrect timestamps.	This often manifests when the data feed uses TCP for transmission - which is a common practice when disseminating data over WAN links.
Ex post cleaning	A data source is cleaned, during the normalization process, using future information. This enhances the historical data with artificial information that may not have been actionable in real-time.	The data may be reordered; trades that were canceled after the end of market session may be removed, or prices may be adjusted with information from a future rollover or dividend event.
Schema bloat	A normalized schema represents some data fields with types that take up unnecessary space or make the data more difficult to compress, which increases storage costs and reduces application performance.	Common cases of this include representing timestamps as ISO 8601 strings or prices as strings, especially on vendor feeds that use JSON encoding.
Our normalization schema is designed to mitigate most of these issues.

Why use normalized data?

Though it may seem counterintuitive, some degree of lossiness introduced during normalization can be preferable.

A normalized schema that has too many data fields, as a result of trying to preserve information from too many different source schemas, is hard to use.

Here are some ways in which lossiness can be useful:

Discarding unnecessary data fields can reduce storage and bandwidth requirements, and improve application performance. For example, many strategies execute at time scales where extra timestamps are unnecessary.
Most status or reference data events are irrelevant for any given business use case. For example, many users only trade during the regular market session and their applications do not need to be aware of special matching conditions that are more typically found outside of regular hours or during pre-market.
Floating point prices can be easier to use and the modeling error introduced by them could be negligible compared to other, more likely sources of error for the given use case.
Order book snapshots can be unnecessary on liquid products whose orders turnover very quickly, as pre-existing orders in the snapshot will eventually be filled or canceled - a process which is commonly referred to as natural refresh.
It should be noted that normalized data is not necessarily going to be smaller or have a simpler specification than the source data. If your use case only requires a single dataset, there may be complexity using normalized data whose schema was designed to accommodate differences between multiple datasets.

We normalize to our proprietary Databento Binary Encoding DBN.

Symbology

Financial datasets usually contain symbols or product identifiers. The mapping of symbols to their corresponding product names can be extracted from our definition schema as well as the metadata packaged with our data.

Databento supports four symbology types, also referred to as stypes. They are: raw_symbol, instrument_id, parent, and continuous.

We include methods for mapping between symbology types and resolving symbols under the symbology family of methods.

We do not retroactively reassign symbols in our historical data. Symbols found in our historical data are exactly as they appeared in the live data at the original event time. For example, if a stock symbol was changed due to a corporate action, we preserve the original symbol for data before the event and the new symbol for data after the event. This approach guarantees that the historical data looks identical to the live data at the original time, and encourages our users to write their integration in a manner that handles historical and live data in the same way.

A symbol can be reused and point to two different instruments on two different dates.

Supported symbology combinations

When requesting data, such as with the timeseries.get_range or batch.submit_job endpoints, an input (stype_in) and output (stype_out) symbology type are specified. Not all symbology types are supported for output and some symbology types are not available in certain datasets.

stype_in ↓ / stype_out →	instrument_id	raw_symbol	parent	continuous
instrument_id	✓	✓		
raw_symbol	✓			
parent	✓			
continuous	✓			
All datasets support bidirectional conversion between raw_symbol and instrument_id.

The table below outlines the datasets that support parent symbology and continuous contract symbology (futures contracts only).

stype_in	parent	continuous
GLBX.MDP3	✓	✓
IFEU.IMPACT	✓	✓
IFLL.IMPACT	✓	✓
IFUS.IMPACT	✓	✓
NDEX.IMPACT	✓	✓
OPRA.PILLAR	✓	
XEEE.EOBI	✓	✓
XEUR.EOBI	✓	✓
Raw symbol

Raw symbols are the original string symbols used by the publisher in the source data. This can be useful for environments with direct market connectivity. Examples of raw symbols include AAPL, ESH3, etc.

This symbology is used by setting the stype_in=raw_symbol parameter in the API.

Instrument ID

Instrument IDs are the unique numeric ID assigned to each instrument by the publisher. Most venues use such numeric IDs under the hood. Numeric IDs have the benefit of taking less space than most string symbols. However, numeric IDs can be difficult to work with, especially as some publishers remap them daily.

This symbology is used by setting the stype_in=instrument_id parameter in the API.

Parent

Parent symbology is a smart symbology feature that allows you to easily refer to groups of related symbols using a single root symbol. The root symbols are sourced from the asset field of the definition schema. All futures for a root symbol can be referenced using the parent symbol [ROOT].FUT, for options: [ROOT].OPT. For example, ES.FUT refers to all E-mini S&P 500 futures and futures spreads and ES.OPT refers to all quarterly E-mini S&P 500 options and option spreads.

The type of instrument will be specified in the instrument_class field. When requesting data using futures parent symbology, this field will indicate whether the instrument is a future or futures spread. When requesting data using options parent symbology, it will indicate whether the instrument is a call or put. A full list of variants can be found in the instrument class documentation.

This symbology is used by setting the stype_in=parent parameter in the API.

Continuous

Info
Info
Our continuous contract symbology is a notation that maps to an actual, tradable instrument on any given date. The continuous contract prices returned are the original, unadjusted prices. We don't create a synthetic time series by back-adjusting the prices to remove jumps during rollovers.
Continuous contract symbology is a smart symbology feature that allow a single symbol to refer to different instruments over time. For example, continuous contract symbology allows you to query a single symbol that changes or rolls forward before expiration.

For futures outrights, we use the format [ROOT].[ROLL_RULE].[RANK] to refer to continuous contracts that change over time according to a roll rule and rank. Like with parent symbology, the root symbol corresponds with the asset field of the definition schema.

RANK is a zero-indexed integer, and ROLL_RULE is either c, n, or v from the table below.

This symbology is used by setting the stype_in=continuous parameter in the API. It is not currently possible to select continuous contracts through our web portal.

Roll rule	Code	Overview	Example
Calendar	c	Refers to the offset from the closest expiration or front month.	On September 28, 2022 NG.c.0 referred to the October NG future (NGV2) and NG.c.1 referred to the November future (NGX2). However, because the October contract expired at the end of trading on September 28 and the continuous smart symbol would be rolled forward, on September 29, 2022, NG.c.0 then referred to the November future (NGX2) and NG.c.1 referred to the December future (NGZ2).
Open interest	n	Will rank the expirations by the open interest at the previous day's close.	CL.n.1 refers to the CL future with the second-highest open interest.
Volume	v	Will rank the expirations by the previous day's trading volume.	ZN.v.0 refers to the ZN future with the most volume.
All symbols

It is possible to request all symbols within a dataset without providing them explicitly. This is done by specifying ALL_SYMBOLS with stype_in=raw_symbol or stype_in=parent in the API.

When requesting all symbols using timeseries.get_range symbology data is not provided. This means that for the CSV and JSON encodings the parameter map_symbols=True is not allowed. For the DBN encoding, the metadata header will not contain symbology mappings.

When requesting all symbols using batch.submit_job, the symbology.json support file will not contain symbology mappings.

Symbology.resolve endpoint

Databento offers symbology resolution for free in our symbology.resolve endpoint and in our client libraries. This endpoint can be used to request mappings from one symbology type to another and contains all the data necessary to perform these conversions.

Field	Description
result	The symbology mapping result. For each requested symbol, a list of symbology mappings is provided.
symbols	The requested symbols.
stype_in	The requested input symbology type.
stype_out	The requested output symbology type.
start_date	The requested symbology start date, as an ISO 8601 date string.
end_date	The requested symbology end date, as an ISO 8601 date string.
partial	The list of symbols, if any, that partially resolved inside the start date and end date interval.
not_found	The list of symbols, if any, that failed to resolve inside the start date and end date interval.
message	A short message indicating the overall symbology result. Can be one of: "OK", "Not found", or "Partially resolved".
status	A numerical status field indicating the overall symbology result. Can be one of: 0 (OK), 1 (Partially resolved), or 2 (Not found).
Symbology support file

For some batch downloads, symbology information for the job is contained in *.symbology.json support files. This file is automatically included when the batch job files do not contain symbology information, such as when requesting CSV or JSON encodings when symbol mapping is not requested. Below is a sample file. It's contents are directly obtained from the symbology.resolve endpoint:


{
   "result": {
      "ES.c.0": [
         { "d0": "2023-01-01", "d1": "2023-03-19", "s": "206299"},
         { "d0": "2023-03-19", "d1": "2023-06-01", "s": "95414"}
      ]
   },
   "symbols": ["ES.c.0"],
   "stype_in": "continuous",
   "stype_out": "instrument_id",
   "start_date": "2023-01-01",
   "end_date": "2023-06-01",
   "partial": [],
   "not_found": [],
   "message": "OK",
   "status": 0
}
Examining this sample we can see a requested mapping of the stype_in ("continuous") symbol ("ES.c.0") to stype_out ("instrument_id") over the date range start_date ("2023-01-01") to the end_date ("2023-06-01").

We can check the message ("OK") and status (0) fields to confirm that our request was successful over the entire date interval. Additionally, the not_found and partial fields are empty.

Most importantly, the result field contains our symbology mappings keyed by each input symbol. Each symbol entry in the result mapping will contain a list of entries. These entries contain a start date in the d0 field, and an end date in the d1 field for the mapping. The s field contains the output symbol.

Continuous contract	Start date (d0)	End date (d1)	Instrument ID (s)
ES.c.0	2023-01-01	2023-03-19	206299
ES.c.0	2023-03-19	2023-06-01	95414
SymbolMappingMsg

Databento's live data publishes symbology information using the SymbolMappingMsg. This message will always contain the input symbol and the resolved output symbol. The record header of the SymbolMappingMsg will always contain the instrument_id. See our DBN encoding article for more information on our binary format.

Field	Type	Description
stype_in	uint8_t	The input symbology type (DBN version 2 only).
stype_in_symbol	char[symbol_cstr_len]	The input symbol from the subscription, where symbol_cstr_len is specified in the Metadata.
stype_out	uint8_t	The output symbology type (DBN version 2 only). Will always be raw_symbol.
stype_out_symbol	char[symbol_cstr_len]	The output symbol from the subscription, where symbol_cstr_len is specified in the Metadata.
start_ts	uint64_t	The start of the mapping interval expressed as the number of nanoseconds since the UNIX epoch.
end_ts	uint64_t	The end of the mapping interval expressed as the number of nanoseconds since the UNIX epoch.


Zstandard (zstd)

Zstandard is a fast, lossless compression algorithm that offers high compression ratios, making it suitable for storing and transmitting large amounts of data.

This compression algorithm is recommended by Databento for all historical streaming and batch downloads, as it minimizes the amount of data transmitted over the network and stored on file systems. You can select Zstandard compression by using the zstd option through the client libraries or the HTTP API.

Once you have obtained Zstandard compressed data from Databento, you have a number of options for decompressing:

Databento client libraries. Decompression from Zstandard is handled internally. Refer to the API reference for usage details
dbn-cli. The Databento CLI tool for working with DBN data also includes support for decompressing .zst files
7-Zip. To decompress using 7-Zip for Windows, you'll need to install the 7-Zip Zstandard edition. Then right-click on the .zst file, navigate to the 7-Zip menu, and select 'Extract here' or 'Extract to <folder>'
zstd CLI. Used directly from the command line (described below), or via your own bash scripts
Installing dbn-cli

To install the dbn-cli (dbn) library, ensure you have Cargo (the Rust package manager) installed. Then, run the following command:


cargo install dbn-cli
For more details, visit dbn-cli on crates.io.

Installing Zstandard

The Zstandard (zstd) library can be installed on most operating systems, as detailed below:

macOS


brew install zstd
Linux

Debian/Ubuntu:


sudo apt-get update
sudo apt-get install zstd
CentOS/RHEL:


sudo dnf install zstd
Arch:


sudo pacman -S zstd
Windows

Using Chocolatey:


choco install zstd
Alternatively, you can also download the Zstandard binaries from the official releases page on GitHub and add the directory to your PATH. You can also install from source by following the instructions in the Zstandard GitHub repository.

Verify your installation by checking the version of Zstandard:


zstd --version
Decompressing

Once Zstandard is installed on your machine, you can utilize its command line interface (CLI) for various operations, including decompression and recompression. Below, we'll explore commands specifically for decompressing Zstandard-compressed files in different scenarios.

To view all available options in the Zstandard CLI, you can run the help command:


zstd -h
Decompress to a file

To decompress a file that has been compressed with Zstandard, use the -d (or --decompress) option followed by the filename. For example, if you have a file named data.zst, you can decompress it with the following command:


zstd -d data.zst
This will create a decompressed file named data in the same directory.

Tip
Tip
You can decompress a Zstandard-compressed Databento batched data file by running the following command:
zstd -d glbx-mdp3-20231201.trades.csv.zst
This will result in the decompressed file, glbx-mdp3-20231201.trades.csv, being created in the current directory.
If you want to decompress the file to a specific output file, you can use the -o option:


zstd -d data.zst -o decompressed_data.txt
This command will decompress data.zst into a file named decompressed_data.txt.

By default, zstd will not overwrite existing files. If you need to decompress and overwrite any existing files, you can use the -f (force) option:


zstd -d -f data.zst
This will decompress data.zst and overwrite any existing file with the same name as the output.

Decompressing multiple files

You can also decompress multiple files at once by specifying multiple filenames:


zstd -d file1.zst file2.zst file3.zst
Using wildcards for batch decompression

In a directory with multiple .zst files, you can decompress all of them using a wildcard (*). This is particularly useful for batch processing:


zstd -d *.zst
This command will decompress all files in the current directory with the .zst extension.

Tip
Tip
Running this command in a batch download directory will decompress all .zst files at once.
Decompressing to standard output

If you want to decompress a file and output the contents directly to the terminal (standard output), use the --stdout option:


zstd -d --stdout data.zst
This could be useful for piping the decompressed output to other programs in bash scripts.


MBO snapshots

An MBO snapshot represents the order book at a specific point in time, including all outstanding buy and sell orders. The snapshot is streamed as a sequence of MBO records to insert new orders in the book (a sequence of Add Actions). The snapshot records preserve the priority order (per instrument) at each price level, enabling accurate reconstruction of the order book.

All snapshot records are marked with the F_SNAPSHOT and F_BAD_TS_RECV flags (ts_recv is set to the snapshot generation timestamp). An instrument's snapshot starts with a cleaR action, followed by zero or more Add Actions.

The snippet below shows the result of a snapshot request for 2 instruments.


MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 4916, ts_event: 1718539204593519451 }, order_id: 0, price: UNDEF_PRICE, size: 0, flags: SNAPSHOT | BAD_TS_RECV (40), channel_id: 0, action: 'R', side: 'N', ts_recv: 1718582400000000000, ts_in_delta: 0, sequence: 0 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 4916, ts_event: 1718539204593519451 }, order_id: 6413364814613, price: 5560.000000000, size: 1, flags: SNAPSHOT | BAD_TS_RECV (40), channel_id: 0, action: 'A', side: 'B', ts_recv: 1718582400000000000, ts_in_delta: 0, sequence: 752 }
...
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 4916, ts_event: 1718582343502504063 }, order_id: 6413384952694, price: 5655.000000000, size: 2, flags: SNAPSHOT | BAD_TS_RECV (40), channel_id: 0, action: 'A', side: 'B', ts_recv: 1718582400000000000, ts_in_delta: 0, sequence: 175066 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 4916, ts_event: 1718582343502957759 }, order_id: 6413384952695, price: 5691.000000000, size: 2, flags: LAST | SNAPSHOT | BAD_TS_RECV (168), channel_id: 0, action: 'A', side: 'A', ts_recv: 1718582400000000000, ts_in_delta: 0, sequence: 175067 }

MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 5002, ts_event: 1718539204593519451 }, order_id: 0, price: UNDEF_PRICE, size: 0, flags: SNAPSHOT | BAD_TS_RECV (40), channel_id: 0, action: 'R', side: 'N', ts_recv: 1718582400000000000, ts_in_delta: 0, sequence: 0 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 5002, ts_event: 1718539204593519451 }, order_id: 6413256341927, price: 4650.000000000, size: 1, flags: SNAPSHOT | BAD_TS_RECV (40), channel_id: 0, action: 'A', side: 'B', ts_recv: 1718582400000000000, ts_in_delta: 0, sequence: 519 }
...
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 5002, ts_event: 1718582366242085265 }, order_id: 6413384879630, price: 5612.500000000, size: 1, flags: SNAPSHOT | BAD_TS_RECV (40), channel_id: 0, action: 'A', side: 'B', ts_recv: 1718582400000000000, ts_in_delta: 0, sequence: 175633 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 5002, ts_event: 1718582366260973309 }, order_id: 6413384879629, price: 5632.500000000, size: 1, flags: LAST | SNAPSHOT | BAD_TS_RECV (168), channel_id: 0, action: 'A', side: 'A', ts_recv: 1718582400000000000, ts_in_delta: 0, sequence: 175760 }
The snapshot for an empty order book has a single MBO record with cleaR action, marked with the flags F_SNAPSHOT and F_LAST, as in the snippet below.


MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 14160, ts_event: 1718117495809255541 }, order_id: 0, price: UNDEF_PRICE, size: 0, flags: LAST | SNAPSHOT | BAD_TS_RECV (168), channel_id: 0, action: 'R', side: 'N', ts_recv: 18446744073709551615, ts_in_delta: 0, sequence: 0 }
MBO snapshots are available from both the historical and live APIs.

Historical API

We offer MBO snapshot through the Historical API for venues that follow a weekly session structure (CME Globex MDP 3.0), and for venues whose daily trading sessions cross 00:00:00 UTC (ICE Europe Commodities iMpact). The order book snapshot is generated at 00:00:00 UTC each weekday (Monday-Friday).

Snapshot records are streamed through the Historical API when the requested interval includes midnight UTC for a given weekday. The snippet below shows an example of a timeseries request in Python whose result includes an MBO snapshot.


client.timeseries.get_range(
    dataset="GLBX.MDP3",
    symbols="ES.c.2",
    stype_in="continuous",
    schema="mbo",
    start="2024-06-16T23:58:50",
    end="2024-06-17T00:00:10",
)

MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 183748, ts_event: 1718582336782043745 }, order_id: 6413384952478, price: 5564.750000000, size: 1, flags: LAST (130), channel_id: 0, action: 'A', side: 'A', ts_recv: 1718582336782569129, ts_in_delta: 12306, sequence: 174718 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 183748, ts_event: 1718582336782045281 }, order_id: 6413384879627, price: 5560.500000000, size: 1, flags: LAST (130), channel_id: 0, action: 'M', side: 'B', ts_recv: 1718582336782613211, ts_in_delta: 12838, sequence: 174721 }
...
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 183748, ts_event: 1718582388400474957 }, order_id: 6413384953470, price: 5562.750000000, size: 1, flags: LAST (130), channel_id: 0, action: 'A', side: 'B', ts_recv: 1718582388400587693, ts_in_delta: 14247, sequence: 176276 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 183748, ts_event: 1718539204593519451 }, order_id: 0, price: UNDEF_PRICE, size: 0, flags: SNAPSHOT | BAD_TS_RECV (40), channel_id: 0, action: 'R', side: 'N', ts_recv: 1718582400000000000, ts_in_delta: 0, sequence: 0 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 183748, ts_event: 1718539204593519451 }, order_id: 6413383913050, price: 5500.000000000, size: 1, flags: SNAPSHOT | BAD_TS_RECV (40), channel_id: 0, action: 'A', side: 'B', ts_recv: 1718582400000000000, ts_in_delta: 0, sequence: 936 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 183748, ts_event: 1718539204593519451 }, order_id: 6413373623913, price: 5475.000000000, size: 1, flags: SNAPSHOT | BAD_TS_RECV (40), channel_id: 0, action: 'A', side: 'B', ts_recv: 1718582400000000000, ts_in_delta: 0, sequence: 936 }
...
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 183748, ts_event: 1718582388400474957 }, order_id: 6413384953470, price: 5562.750000000, size: 1, flags: LAST | SNAPSHOT | BAD_TS_RECV (168), channel_id: 0, action: 'A', side: 'B', ts_recv: 1718582400000000000, ts_in_delta: 0, sequence: 176276 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 183748, ts_event: 1718582400066482471 }, order_id: 6413384953619, price: 5564.750000000, size: 1, flags: LAST (130), channel_id: 0, action: 'A', side: 'A', ts_recv: 1718582400066580869, ts_in_delta: 14145, sequence: 176515 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 183748, ts_event: 1718582401163834877 }, order_id: 6413384953619, price: 5564.750000000, size: 1, flags: LAST (130), channel_id: 0, action: 'C', side: 'A', ts_recv: 1718582401163946845, ts_in_delta: 13326, sequence: 176774 }
...
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 183748, ts_event: 1718582409155703017 }, order_id: 6413384953440, price: 5413.250000000, size: 1, flags: LAST (130), channel_id: 0, action: 'M', side: 'B', ts_recv: 1718582409155796258, ts_in_delta: 12560, sequence: 178707 }
Live API

Users can request an MBO snapshot through the live API to obtain the recent order book state without replaying the whole trading session. The following sequence of messages is streamed from a live session after a snapshot subscription.

Symbol mapping messages
Snapshot records (clear book and outstanding orders at each price level)
Real-time records
Warning
Warning
The order book of a given instrument is not guaranteed to be in a complete and valid state after the last snapshot record because it might not correspond to the final event record (indicated with the F_LAST flag). If the last snapshot record does not have the F_LAST flag set, the order book will not be valid until the next MBO record with the F_LAST flag set.
The example below shows the live stream for a snapshot subscription for a single instrument, followed by a real-time record.


SymbolMappingMsg { hd: RecordHeader { length: 44, rtype: SymbolMapping, publisher_id: 0, instrument_id: 118, ts_event: 1721732190956747490 }, stype_in: 255, stype_in_symbol: "ES.c.0", stype_out: 255, stype_out_symbol: "ESU4", start_ts: 18446744073709551615, end_ts: 18446744073709551615 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 118, ts_event: 1721732152358684229 }, order_id: 0, price: UNDEF_PRICE, size: 0, flags: SNAPSHOT | BAD_TS_RECV (40), channel_id: 0, action: 'R', side: 'N', ts_recv: 1721732152358684229, ts_in_delta: 0, sequence: 0 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 118, ts_event: 1721579296070204729 }, order_id: 6413514530616, price: 5547.500000000, size: 8, flags: SNAPSHOT | BAD_TS_RECV (40), channel_id: 0, action: 'A', side: 'B', ts_recv: 1721732152358684229, ts_in_delta: 0, sequence: 1022 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 118, ts_event: 1721579296070204729 }, order_id: 6413514530330, price: 5547.250000000, size: 7, flags: SNAPSHOT | BAD_TS_RECV (40), channel_id: 0, action: 'A', side: 'B', ts_recv: 1721732152358684229, ts_in_delta: 0, sequence: 1022 }
...
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 118, ts_event: 1721579296070204729 }, order_id: 6413506441952, price: 5628.000000000, size: 1, flags: SNAPSHOT | BAD_TS_RECV (40), channel_id: 0, action: 'A', side: 'A', ts_recv: 1721732152358684229, ts_in_delta: 0, sequence: 1101 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 118, ts_event: 1721579296070204729 }, order_id: 6413500305436, price: 5628.250000000, size: 6, flags: SNAPSHOT | BAD_TS_RECV (40), channel_id: 0, action: 'A', side: 'A', ts_recv: 1721732152358684229, ts_in_delta: 0, sequence: 1058 }
...
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 118, ts_event: 1721732152290217707 }, order_id: 6413522310536, price: 5613.250000000, size: 3, flags: SNAPSHOT | BAD_TS_RECV (42), channel_id: 0, action: 'A', side: 'A', ts_recv: 1721732152358684229, ts_in_delta: 0, sequence: 13076493 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 118, ts_event: 1721732152290632121 }, order_id: 6413522309988, price: 5512.500000000, size: 6, flags: LAST | SNAPSHOT | BAD_TS_RECV (170), channel_id: 0, action: 'A', side: 'B', ts_recv: 1721732152358684229, ts_in_delta: 0, sequence: 13076496 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 118, ts_event: 1721732152365904111 }, order_id: 6413522310535, price: 5613.250000000, size: 1, flags: LAST (130), channel_id: 0, action: 'C', side: 'A', ts_recv: 1721732152366001513, ts_in_delta: 14406, sequence: 13076497 }
MboMsg { hd: RecordHeader { length: 14, rtype: Mbo, publisher_id: GlbxMdp3Glbx, instrument_id: 118, ts_event: 1721732152365916169 }, order_id: 6413522310537, price: 5613.000000000, size: 1, flags: LAST (130), channel_id: 0, action: 'A', side: 'B', ts_recv: 1721732152366014389, ts_in_delta: 12865, sequence: 13076498 }
...
Timestamp fields on snapshot messages

The timestamp fields for snapshot messages are described below.

ts_event: unchanged. The cleaR book MBO record from the Live API snapshot contains the snapshot generation timestamp instead
ts_in_delta: always 0
ts_recv: set to the snapshot generation timestamp (indicated with the F_BAD_TS_RECV flag)
ts_out: unchanged
Public client support for MBO snapshot

Live and historical snapshot features are available on all our official client libraries (Python, C++, Rust), as well as through our Raw API.

An example of a live snapshot subscription using our client libraries can be found in this article.


Reference data enums

Exchanges, event types, other fields included in Databento's corporate actions data and reference data API. Over 60 event types, including splits, dividends, adjustment factors, and listings.

The following tables specify the descriptions for all reference data enums found in responses.

See also
See also
Corporate actions dataset guide for further details.
ACTION

Value
Description
C	Cancelled
D	Deleted
I	Inserted
P	Payment details Cancelled by Issuer
Q	Payment details Deleted by Data Supplier
U	Updated

