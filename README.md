# MiniMeAI

A microservices-based AI application for processing and analyzing email data.

## Architecture

The application consists of several containerized microservices:

- **API Server**: Handles external requests and provides access to the processed data
- **Vector Database**: Stores and manages vector embeddings
- **Email Processor**: Handles email collection and preprocessing
- **Web Interface**: Provides a user interface for interaction
- **Processing Service**: Handles data processing and AI tasks

## Setup and Deployment

The application is containerized using Docker and orchestrated with Docker Compose.

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Gmail API credentials (for email processing)

### Configuration

1. Copy the `.env.save` file to `.env` and update the environment variables
2. Configure your Google API credentials in the appropriate config files

### Running the Application

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

## Development

The project is structured as follows:

- `src/` - Source code for all services
- `config/` - Configuration files
- `Dockerfile.*` - Docker configuration for each service
- `requirements.*.txt` - Python dependencies for each service

## Project Updates

For details on recent major updates to the project, see [Project Updates](docs/project_updates.md).

Recent major updates:
- **2025-03-21**: [Timestamp Normalization Project](docs/project_updates.md#timestamp-normalization-project---2025-03-21) - Fixed timestamp handling and standardized source type attribution.
- **2025-03-21**: [Timestamp Standardization](docs/project_updates.md#additional-improvements-2025-03-21) - Created unified timestamp handling module and standardized across all components.

## License

[Your chosen license] 