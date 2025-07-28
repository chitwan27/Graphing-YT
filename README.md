# Can we map the whole of youtube?

## Preliminary Research

* [Not David's Excellent Video](https://www.youtube.com/watch?v=o879xRxmwmU): Inspired me and taught me most of what I need to do.
* [Louvain Algorithm Video](https://www.youtube.com/watch?v=QfTxqAxJp0U): A really good intro to the community detection approach of louvain.

## Data Collection

* Configure OAuth 2.0 credentials and obtain YouTube Data API v3 access for subscription data retrieval.
* Manually add initial channels to the database to bootstrap the crawling process (popular channels from different niches work well).
* Use a subscription crawler script to systematically fetch public subscription lists from channels, starting with seeds and expanding to discovered channels.
* Store subscription relationships as directed edges in SQLite database (from_channel → to_channel), tracking crawl timestamps and channel metadata.
* Respect the YouTube API daily limits (10,000 units) by implementing quota tracking and resumable crawling across multiple days.
* The crawler should handle private subscriptions (403 errors), deleted channels, and other API failures.

## First Pass (for feasability check)

* Take the raw collected data and convert it into proper csv format.
* Convert the network into a co-subscriber network using matrix multiplication.
* Apply the Louvain community detection algorithm on the network and export the results.
* Use the Gephi software to visualize the complete graph and tinker with the options available.
