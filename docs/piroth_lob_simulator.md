# `piroth` Agent-Based LOB Simulator

This document describes the current simulator on the `piroth` branch. The simulator is implemented in [piroth/simulator.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/simulator.py) and is now an agent-based, order-level synthetic market rather than the older parametric top-10 depth generator.

## Design Goals

The simulator is built to support:

- continuous market-making experiments
- realistic queue competition and FIFO matching
- explicit, interpretable agent populations
- synthetic data that still fits the existing pipeline format
- richer diagnostics than raw mid-price simulation alone

It is still a stylized research simulator. It is not a full exchange replica.

## Main Idea

Instead of directly sampling top-10 depths and trade events from a compact parametric rule, the simulator now generates the book from interacting agents:

- noise takers
- informed takers
- competing market makers
- liquidity providers

These agents submit market orders, limit orders, and cancellations into an explicit order book. The top 10 levels are then snapshot from that book and written to disk.

## Core State

The main simulator state is held by `AgentBasedLOB` in [piroth/simulator.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/simulator.py).

It contains:

- `bids`: bid-side queues by price
- `asks`: ask-side queues by price
- `fair_value`: latent fair value
- `signal`: persistent directional latent state
- `regime`: latent regime in `{-1, 0, 1}`
- `signed_flow_state`: smoothed signed order-flow state
- `event_seq`: event counter

Resting orders are stored explicitly as `RestingOrder` objects with:

- `order_id`
- `owner`
- `side`
- `price`
- `size`
- `created_event`

So there is now explicit queue composition by agent type.

## Agent Populations

### Noise takers

Noise takers submit market buys and sells with weak dependence on current imbalance and signed flow. They provide uninformed liquidity demand.

### Informed takers

Informed takers submit market orders in the direction of the latent fair-value edge. If fair value is above the displayed mid, informed buy pressure rises; if below, informed sell pressure rises.

### Competing market makers

Competing MMs add liquidity near the touch and cancel stale orders. They create queue competition for the RL agent and make fill quality depend on who is already resting at the best price.

### Liquidity providers

Liquidity providers refill deeper levels and keep the book from collapsing. They are not meant to be alpha-seeking; they provide structural depth.

## Matching Model

The simulator now has explicit FIFO matching.

Market orders walk the opposing book:

- buy market orders consume the ask book from best ask upward
- sell market orders consume the bid book from best bid downward
- within each price level, the oldest resting order is matched first

Each match produces a `TradeRecord` with:

- execution price
- matched size
- aggressor side
- taker agent type
- maker agent type
- maker order ID
- queue-ahead proxy

This is the main interpretability improvement over the old simulator.

## Book Initialization

At the start of each day:

- the book is seeded around the symbol base price
- 10 price levels are initialized on each side
- each level receives several `competing_mm` orders and one `liquidity_provider` order

This gives a populated book before the first event and avoids the degenerate “empty book until first add” problem.

## Latent State Dynamics

Latent state is updated in `_step_latent()`.

There are three components:

### Regime

- discrete regime `-1 / 0 / 1`
- occasional regime switches after a minimum persistence period

### Signal

- persistent directional latent state
- mean reverts toward a regime-dependent target
- perturbed by noise

### Fair value

- latent fair value
- moves with the signal plus noise

This fair value is not directly observed by the RL agent. It only affects the behavior of the simulated informed traders and, indirectly, the displayed book.

## Event Types

Each event is chosen from:

- `noise_market_buy`
- `noise_market_sell`
- `informed_market_buy`
- `informed_market_sell`
- `maker_add_bid`
- `maker_add_ask`
- `maker_cancel_bid`
- `maker_cancel_ask`
- `refill_bid`
- `refill_ask`

Event probabilities depend on:

- fair-value edge versus displayed mid
- top-of-book imbalance
- smoothed signed flow
- symbol-specific agent-rate parameters

This produces endogenous book dynamics rather than pre-baked top-level summaries.

## Spread and Mid Formation

In the old simulator, displayed mid and spread were controlled by an explicit recentering rule.

In the new simulator:

- displayed best bid and best ask come directly from the current order book
- displayed mid is `(best_bid + best_ask) / 2`
- spread is endogenous to the current queue state

So the displayed book now moves because agents add, cancel, and consume orders, not because a separate recentering step forces it to move.

## Per-Event Outputs

The simulator still writes the same core files the rest of the branch expects:

- `ask.csv`
- `bid.csv`
- `price.csv`
- `msg.csv`
- `trades.csv`
- `latent.csv`

### `ask.csv` / `bid.csv`

These are top-10 snapshots derived from the explicit book after each event.

### `price.csv`

Contains:

- `ask1_price`
- `bid1_price`
- `midprice`

### `msg.csv`

Per-event aggregate message counts/volumes for:

- market buy/sell
- limit buy/sell
- cancel buy/sell

### `trades.csv`

Now contains richer fields than before:

- `price`
- `size`
- `aggressor_side`
- `taker_agent`
- `maker_agent`
- `maker_order_id`
- `queue_ahead`

### `latent.csv`

Now contains both latent market state and interpretability metadata, including:

- `fair_value`
- `efficient_price`
- `latent_alpha`
- `regime`
- `signed_flow_state`
- `spread_ticks`
- `top_imbalance`
- `queue_pressure`
- `event_type`
- `event_name`
- `event_side`
- `event_actor`
- `maker_agent`
- `regime_shift`
- `efficient_move`
- `trade_count`
- `traded_volume`
- `best_bid_depth`
- `best_ask_depth`

## Symbol Profiles

Each symbol gets its own `SymbolProfile`:

- base price
- fair-value persistence
- signal noise
- noise taker rate
- informed taker rate
- competing maker add/cancel rates
- liquidity refill rate
- maker join-touch probability
- depth scale

These profiles let the same simulator represent different liquidity regimes without changing its structure.

## Configuration Knobs

The simulator is mainly controlled through [piroth/config.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/config.py).

Important knobs still used by the agent-based simulator include:

- `events_per_day`
- `base_prices`
- `alpha_signal_scale`
- `price_noise_scale`
- `market_order_impact_scale`
- `market_order_tick_impact`
- `market_order_alpha_impact`
- `touch_replenish_fraction`

The rest of the branch’s training and reward configuration remains separate from the simulator itself.

## Interpretability Advantages

The agent-based simulator is more interpretable than the previous one because:

- trades have explicit taker and maker agent identities
- queue competition is explicit
- depth is generated by visible agent actions
- price movement is tied to fair-value pressure plus endogenous order flow
- you can attribute PnL and fills to particular agent populations

This makes it much easier to ask:

- Is the RL agent making money mainly against noise takers?
- Is it losing to informed takers?
- Is queue competition from other market makers killing its fill rate?
- Does it earn spread capture or drift/speculation?

## Remaining Limitations

This simulator is still not a full real-market reconstruction.

Key limitations:

- no cross-day carry of book state or fair value
- no hidden liquidity
- no multiple venues
- no exchange-specific fee table by default
- no explicit RL-agent queue placement in the data generator itself
- stylized agent population rather than calibrated real participant taxonomy

So this is a significantly more realistic and interpretable simulator than the old `piroth` version, but it is still a research simulator, not a full historical market replica.
