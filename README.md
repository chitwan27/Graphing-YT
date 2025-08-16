# Mapping the YouTube Network

## Preliminary Research

* [Not David's Excellent Video](https://www.youtube.com/watch?v=o879xRxmwmU): Inspiration and technical guidance.
* [Co-Citation Matrix Video](https://www.youtube.com/watch?v=iSyuvk94tlk): Derivation of the co-citation result.
* [Louvain Algorithm Video](https://www.youtube.com/watch?v=QfTxqAxJp0U): Intro to Louvain community detection.

## Data Collection (`Data Collection.py`)

* Configure OAuth 2.0 credentials and obtain YouTube Data API v3 access for subscription data retrieval.
* Manually add initial channels to the database to bootstrap crawling (popular channels from different niches).
* Run `Data Collection.py` to crawl public subscription lists, expanding from seeds to discovered channels.
* Stores relationships as directed edges in `youtube_graph.db` (from_channel → to_channel), with crawl timestamps and channel metadata.
* Handles YouTube API daily limits (10,000 units) via quota tracking and resumable crawling.
* Robust to private subscriptions (403 errors), deleted channels, and API failures.

## Network Analysis & Community Detection (`Com Clusterer.py`)

* Loads data from `youtube_graph.db` and filters channels by popularity.
* Builds co-subscription network using sparse matrix operations.
* Runs community detection (Louvain or Leiden algorithms) with configurable parameters.
* Exports results as timestamped CSVs in `output/` for further visualization.
* Prints analysis summary and statistics.

## Visualization (`A Visualizer.py`)

* Loads latest or specific analysis results from `output/`.
* Builds interactive network visualizations using Plotly.
* Assigns colors to communities and supports multiple layout algorithms.
* Displays network statistics and community summaries.
* Can save visualizations as HTML files for sharing or further exploration.

## Workflow Summary

1. **Collect Data:** Run `Data Collection.py` to crawl and store YouTube subscription data.
2. **Analyze & Cluster:** Run `Com Clusterer.py` to build the co-subscription network and detect communities.
3. **Visualize:** Use `A Visualizer.py` to explore and present the network and community structure interactively.

## Notes & Discoveries

* Youtube API limit for fetching subscriptions of a user stops at 997 channels.
* Community detection is good for realizing the local structure of the network but not global.
