# Fetcher Architecture Guide

## Overview

This document describes the architecture for data fetchers in the MiniMeAI system. All data sources (email, web, chat, etc.) must use a consistent approach for handling source types and metadata to ensure proper attribution and timestamp handling.

## BaseFetcher Class

All data fetchers must inherit from the `BaseFetcher` abstract base class located in `src/utils/base_fetcher.py`. This class ensures:

1. Consistent source type attribution for all documents
2. Reliable timestamp handling and normalization
3. Common error handling and logging patterns

## Creating a New Fetcher

To create a new fetcher for a data source, follow these steps:

1. Inherit from `BaseFetcher`
2. Specify the source type in the constructor (e.g., "email", "web", "whatsapp")
3. Implement the abstract `fetch_data()` method
4. Use the `enrich_metadata()` method to ensure consistent metadata

### Example:

```python
from utils.base_fetcher import BaseFetcher

class WebsiteFetcher(BaseFetcher):
    def __init__(self):
        super().__init__(source_type="website", logger=my_logger)
        
    def fetch_data(self, limit=100):
        # Implement website-specific fetching logic
        websites = self._fetch_websites(limit)
        
        # Return list of enriched items
        return [self.enrich_metadata(website) for website in websites]
        
    def _fetch_websites(self, limit):
        # Website-specific implementation
        # ...
```

## Important Metadata Fields

The following metadata fields are standardized across all data sources:

| Field | Description | Required |
|-------|-------------|----------|
| `source_type` | Data source identifier (e.g., "email") | Yes |
| `id` | Unique identifier for the document | Yes |
| `timestamp` | Unix timestamp of the document | Yes |
| `date` | Original date string (format varies by source) | Recommended |

## Processing and Ingestion

The processor module will use the source type field to determine how to properly chunk and process the documents. This ensures that each document type is handled appropriately based on its structure and content.

## Querying by Source Type

You can filter search results by source type using the API's filter parameter:

```json
{
  "query_text": "my search query",
  "filter": {
    "source_type": "email"
  }
}
``` 