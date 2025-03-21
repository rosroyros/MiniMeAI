# Vector Database Timestamp Utilities

This directory contains a collection of utilities for working with timestamps in vector database collections. These tools help analyze, normalize, validate, and fix timestamp-related issues.

## Available Tools

### 1. Timestamp CLI (timestamp_cli.py)

The main command-line interface that integrates all timestamp utilities into a single tool.

```bash
python timestamp_cli.py [--host HOST] [--port PORT] [--collection COLLECTION] [--remote] [--output OUTPUT] {analyze,normalize,validate,search}
```

#### Commands:

- **analyze**: Analyze timestamp quality in a collection
- **normalize**: Add timestamps to documents missing them
- **validate**: Validate existing timestamps and optionally fix issues
- **search**: Test recency-based search to ensure timestamps are working

#### Global Options:

- `--host`: Database host (default: localhost)
- `--port`: Database port (default: 8000)
- `--collection`: Collection name (default: emails)
- `--remote`: Use the remote server at 10.0.0.58
- `--output`: Output file for results in JSON format

### 2. Individual Utilities

Each utility can also be used independently:

#### Analyze Collection Timestamps (analyze_collection_timestamps.py)

Analyzes a collection and provides detailed statistics about timestamp fields.

```bash
python analyze_collection_timestamps.py --host HOST --port PORT --collection COLLECTION [--remote]
```

#### Normalize Timestamps (normalize_timestamps.py)

Adds timestamps to documents that have a date field but no timestamp field.

```bash
python normalize_timestamps.py --host HOST --port PORT --collection COLLECTION [--remote] [--dry-run]
```

#### Validate Timestamps (validate_timestamps.py)

Validates existing timestamps and can fix issues such as null, zero, or future timestamps.

```bash
python validate_timestamps.py --host HOST --port PORT --collection COLLECTION [--remote] [--fix]
```

## Examples

### Analyzing a Collection

```bash
# Local analysis
python timestamp_cli.py analyze --collection emails

# Remote analysis
python timestamp_cli.py analyze --collection emails --remote
```

### Normalizing Missing Timestamps

```bash
# Dry run first to see what would be updated
python timestamp_cli.py normalize --collection emails --remote --dry-run

# Actually perform the updates
python timestamp_cli.py normalize --collection emails --remote
```

### Validating and Fixing Timestamps

```bash
# Validate without fixing
python timestamp_cli.py validate --collection emails --remote

# Validate and fix issues
python timestamp_cli.py validate --collection emails --remote --fix
```

### Testing Recency-Based Search

```bash
python timestamp_cli.py search --collection emails --remote
```

## Technical Details

### Vector Database API Requirements

These utilities require the following API endpoints in the vector database:

1. `/api/v1/collections/{name}/query` - For querying documents with filtering
2. `/api/v1/collections/{name}/update` - For updating document metadata
3. `/api/v1/collections/{name}/timestamp_analysis` - For analyzing timestamps
4. `/api/v1/collections/{name}/count` - For getting collection document count

### Timestamp Format

All timestamps are stored as Unix timestamps (seconds since epoch) as integer values.

## Common Issues and Solutions

1. **Missing Timestamps**: Documents without a timestamp field cannot be included in recency-based searches. Use the `normalize` command to add timestamps based on date fields.

2. **Invalid Timestamps**: Null, zero, or non-numeric timestamp values cause issues with sorting and filtering. Use the `validate --fix` command to correct these issues.

3. **Future Timestamps**: Timestamps that are in the future can cause incorrect sorting. The validation tool identifies and can fix these issues.

4. **Very Old Timestamps**: Timestamps that are unreasonably old (before 1990) are likely errors and can be fixed with the validation tool.

## Best Practices

1. **Run Analysis First**: Always start with the `analyze` command to understand the scope of timestamp issues.

2. **Use Dry Run**: When normalizing or fixing timestamps, use the `--dry-run` flag first to preview changes.

3. **Batch Processing**: For large collections, the normalization tool uses batching to avoid overwhelming the server.

4. **Test After Changes**: After making changes, use the `search` command to test that recency-based searching works as expected.

## Troubleshooting

If you encounter issues:

1. Check that the vector database server is running and accessible
2. Verify that you have specified the correct host, port, and collection
3. For remote access, use the `--remote` flag
4. Check the output JSON files for detailed error information 