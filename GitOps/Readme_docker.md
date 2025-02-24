# Multi-Stage Docker Build Guide for Python API Testing

## Overview

This guide explains a sophisticated multi-stage Docker setup for a Python API project with comprehensive testing capabilities. The configuration includes production deployment, unit testing, and integration testing stages.

## Dockerfile Structure

The Dockerfile uses multi-stage builds to optimize different scenarios. Let's break down each stage:

### Base Stage

```dockerfile
# Experiment Object API Dockerfile.
#
# Copyright 2025 DFX5
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG PYTHON_VERSION=3.9.19
FROM python:${PYTHON_VERSION}-slim as base

WORKDIR /src

# Non-privileged user setup for security
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app/project" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser \
    && chown appuser /app/project \
    && chown appuser /src
```

Key features:
- Uses Python 3.9.19 slim image
- Creates a non-privileged user for security
- Sets up working directory and permissions

### System Dependencies

```dockerfile
RUN apt-get update \
  && apt-get -y install gcc g++ unixodbc unixodbc-dev libgssapi-krb5-2 \
  && apt-get clean
```

These dependencies are required for:
- C/C++ compilation capabilities
- ODBC database connectivity
- Kerberos authentication support

### Python Environment Configuration

```dockerfile
ENV PYTHONUSERBASE=/app/project/.local/bin
ENV PATH=$PYTHONUSERBASE/bin:$PATH
ENV PIP_USER=yes
ENV PYTHONPATH="${PYTHONPATH}:/app/project/.local/bin"
```

Environment variables ensure:
- Proper Python package installation location
- Correct PATH settings
- User-level package installation

### Production Stage

```dockerfile
FROM base AS production
USER appuser
COPY ./src /src/app
EXPOSE 8000
CMD ["uvicorn", "app.main:eor_api", "--host=0.0.0.0", "--port=8000"]
```

Features:
- Builds upon base image
- Runs as non-privileged user
- Exposes port 8000 for API access
- Uses Uvicorn ASGI server

### Testing Stages

The Dockerfile includes multiple testing stages:

```dockerfile
FROM base as testing
WORKDIR /
USER appuser
RUN pip install httpx pytest pytest-dotenv pytest-cov pytest-xdist pytest-mock

FROM testing as unit_test
USER root
CMD bash -c "pytest --cov-config=.coveragerc src/tests/unit_test --junitxml=reports/unittest-${PYTHON_VERSION}.xml --cov-report=xml:reports/coverage-${PYTHON_VERSION}.xml --cov-report=html:reports/coverage_report-${PYTHON_VERSION} -n 3 --cov=src"
```

Testing capabilities include:
- Unit tests with coverage reporting
- Integration tests
- Parallel test execution
- Multiple report formats (XML, HTML)

## Docker Compose Configuration

The docker-compose.yml file orchestrates different services:

```yaml
services:
  eor_api:
    build:
      target: production
      context: .
    container_name: eor_api
    environment:
      DEBUG: 1
    restart: always
    env_file:
      - .env
    volumes:
      - ./src:/src/app
    ports:
      - 8000:8000
    command: uvicorn app.main:eor_api --reload --host 0.0.0.0 --port 8000
```

### Development Service
The main API service features:
- Hot-reload capability
- Environment variable configuration
- Volume mounting for source code
- Port mapping

### Testing Services

```yaml
  eor_api_unit_test:
    build:
      target: unit_test
      context: .
    container_name: eor_api_unit_test
    env_file:
      - .env
    volumes:
      - ./reports/:/src/reports
      - ./tests:/src/tests
      - ./reports/:/reports
      - ./tests:/tests
```

Test service features:
- Separate containers for unit and integration tests
- Report volume mounting
- Parallel test execution
- Coverage reporting

## Usage Instructions

### Running the API

```bash
# Build and start the API
docker-compose build && docker-compose up eor_api
```

### Running Tests

```bash
# Run unit tests
docker-compose build && docker-compose up eor_api_unit_test

# Run integration tests
docker-compose build && time docker-compose up --force-recreate eor_api_integration_test
```

## Best Practices Implemented

1. **Security**
   - Non-root user execution
   - Minimal base image
   - Proper permission settings

2. **Efficiency**
   - Multi-stage builds
   - Layer optimization
   - Minimal dependencies

3. **Testing**
   - Comprehensive test coverage
   - Multiple report formats
   - Parallel test execution

4. **Development Experience**
   - Hot reload capability
   - Volume mounting
   - Environment configuration

## Common Operations

### Modifying Python Version

```bash
# Build with specific Python version
docker-compose build --build-arg PYTHON_VERSION=3.9.19
```

### Accessing Test Reports

Reports are available in the `./reports` directory:
- Unit test results: `unittest-3.9.19.xml`
- Coverage reports: `coverage-3.9.19.xml`
- HTML coverage: `coverage_report-3.9.19/`

### Debugging

1. Access container logs:
```bash
docker-compose logs eor_api
```

2. Execute interactive shell:
```bash
docker-compose exec eor_api bash
```

## Troubleshooting

Common issues and solutions:

1. **Permission Issues**
   - Check volume mount permissions
   - Verify UID matches local user

2. **Missing Dependencies**
   - Review requirements.txt
   - Check system dependencies in Dockerfile

3. **Test Failures**
   - Examine test reports
   - Check coverage requirements
   - Verify environment variables

