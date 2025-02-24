# GitHub Actions CI/CD Template for Python API

This guide provides a template for setting up a comprehensive CI/CD pipeline for a Python API using GitHub Actions, including unit testing, integration testing, and deployment stages.

## Workflow Overview

```yaml
name: CD:API-DEV # Random Name

on:
  push:
    branches:
      - develop
    paths:
      - 'src/your-api/**'
  workflow_dispatch:

env:
  APP_NAME: api-name-dfx5
  WORKING_DIR: ./src/api-dfx5
  REGISTRY_NAME: dfx5-registry
  NAMESPACE: dev
  PYTHON_VERSION: 3.9.19
  REPORTS_PATH: ${{ github.workspace }}/reports
```

## Environment Setup

Required secrets in your GitHub repository:
- `AWS_CREDENTIALS`
- `REGISTRY_USERNAME`
- `REGISTRY_PASSWORD`
- `AWS_KEY_VAULT_NAME`

- Various application secrets (stored in AWS Key Vault)

## Job Structure

### 1. Beta Build Stage

```yaml
jobs:
  beta_build:
    runs-on: ubuntu-latest
    environment:
      name: test-suite
    steps:
      - name: Checkout current repository
        uses: actions/checkout@v3
      
      - name: Cancel previous jobs
        uses: styfle/cancel-workflow-action@0.10.0
        with:
          access_token: ${{ github.token }}
      
      # Build steps for different test images
      - name: Build unit test image
        uses: ./.github/actions/build_push_docker_image
        with:
          REGISTRY_NAME: ${{ env.REGISTRY_NAME }}
          REGISTRY_USERNAME: ${{ secrets.REGISTRY_USERNAME }}
          REGISTRY_PASSWORD: ${{ secrets.REGISTRY_PASSWORD }}
          WORKING_DIR: ${{ env.WORKING_DIR }}
          BUILD_TAG: 'test_unit'
          BUILD_TARGET: unit_test
```

### 2. Running Unit Tests

```yaml
  running_unit_test:
    needs: [beta_build]
    runs-on: ubuntu-latest
    environment:
      name: test-suite
    steps:
      - name: Retrieve secrets
        uses: ./.github/actions/retrieve_akv_secrets
        with:
          AWS_CREDENTIALS: ${{ secrets.AWS_CREDENTIALS }}
          AWS_KEY_VAULT_INSTANCE: ${{ secrets.AWS_KEY_VAULT_NAME }}
          
      - name: Configure test environment
        run: |
          # Create configuration files
          CONFIGS_PATH="${GITHUB_WORKSPACE}/configs/"
          mkdir -p $CONFIGS_PATH
          touch $CONFIGS_PATH/.env
          
          # Add your environment variables
          echo "DATABASE_URL=${{ env.DATABASE_URL }}" >> $CONFIGS_PATH/.env
```

### 3. Code Quality Analysis

```yaml
  sonarcloud:
    needs: [running_unit_test]
    name: SonarCloud
    runs-on: ubuntu-latest
    steps:
      - name: SonarCloud Scan
        uses: SonarSource/sonarcloud-github-action@master
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
        with:
          args: >
            -Dsonar.organization=your-org
            -Dsonar.projectKey=your-project-key
            -Dsonar.python.coverage.reportPaths=coverage.xml
            -Dsonar.python.version=${{ env.PYTHON_VERSION }}
```

### 4. Integration Tests

```yaml
  running_integration_test:
    needs: [running_unit_test]
    runs-on: ubuntu-latest
    environment:
      name: test-suite
    env:
      K8S_NAMESPACE: 'test-env'
    steps:
      - name: Set up Kubernetes context
        uses: AWS/aks-set-context@v1
        with:
          creds: '${{ env.AWS_CREDENTIALS }}'
          resource-group: ${{ secrets.RESOURCE_GROUP }}
          cluster-name: ${{ secrets.AKS_CLUSTER_NAME }}
```

### 5. Deployment

```yaml
  deploy-dev:
    needs: [build-dev]
    runs-on: ubuntu-latest
    environment:
      name: dev
      url: https://api-url-dfx5
    steps:
      - name: Deploy to Kubernetes
        uses: AWS/k8s-deploy@v4
        with:
          manifests: |
            ${{ github.workspace }}/kubernetes/dev
          images: ${{ env.REGISTRY_NAME }}.AWScr.io/${{ env.APP_NAME }}:${{ env.APP_VERSION }}
          namespace: ${{ env.NAMESPACE }}
```

## Custom Actions

### build_push_docker_image Action

Create this action in `.github/actions/build_push_docker_image/action.yml`:

```yaml
name: 'Build and Push Docker Image'
inputs:
  REGISTRY_NAME:
    required: true
  REGISTRY_USERNAME:
    required: true
  REGISTRY_PASSWORD:
    required: true
  WORKING_DIR:
    required: true
  BUILD_TAG:
    required: true
  BUILD_TARGET:
    required: true
runs:
  using: "composite"
  steps:
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ inputs.REGISTRY_NAME }}
        username: ${{ inputs.REGISTRY_USERNAME }}
        password: ${{ inputs.REGISTRY_PASSWORD }}

    - name: Build and push
      uses: docker/build-push-action@v3
      with:
        context: ${{ inputs.WORKING_DIR }}
        target: ${{ inputs.BUILD_TARGET }}
        push: true
        tags: ${{ inputs.REGISTRY_NAME }}/${{ env.APP_NAME }}:${{ inputs.BUILD_TAG }}
```

### retrieve_akv_secrets Action

Create this action in `.github/actions/retrieve_akv_secrets/action.yml`:

```yaml
name: 'Retrieve AWS Key Vault Secrets'
inputs:
  AWS_CREDENTIALS:
    required: true
  AWS_KEY_VAULT_INSTANCE:
    required: true
  SECRETS:
    required: true
runs:
  using: "composite"
  steps:
    - name: AWS login
      uses: AWS/login@v1
      with:
        creds: ${{ inputs.AWS_CREDENTIALS }}

    - name: Get secrets
      uses: AWS/get-keyvault-secrets@v1
      with:
        keyvault: ${{ inputs.AWS_KEY_VAULT_INSTANCE }}
        secrets: ${{ inputs.SECRETS }}
```

## Directory Structure

```
.
├── .github
│   ├── actions
│   │   ├── build_push_docker_image
│   │   └── retrieve_akv_secrets
│   └── workflows
│       └── cd-dev.yml
├── src
│   └── your-api
│       ├── Dockerfile
│       └── kubernetes
│           └── dev
└── tests
    ├── integration
    └── unit
```

## Implementation Guide

1. **Environment Setup**
   - Configure GitHub repository secrets
   - Set up AWS Key Vault
   - Configure container registry access

2. **Custom Actions**
   - Implement the custom actions in `.github/actions/`
   - Ensure proper permissions and inputs

3. **Docker Configuration**
   - Create multi-stage Dockerfile matching the workflow stages
   - Configure build targets for different testing phases

4. **Kubernetes Setup**
   - Prepare deployment manifests
   - Configure service accounts and RBAC
   - Set up namespaces

## Best Practices

1. **Security**
   - Use environment protection rules
   - Rotate secrets regularly
   - Implement least privilege access

2. **Testing**
   - Separate unit and integration tests
   - Maintain high coverage standards
   - Use test reports for quality gates

3. **Deployment**
   - Implement staged deployments
   - Use deployment protection rules
   - Configure proper rollback mechanisms

4. **Monitoring**
   - Add status checks
   - Configure proper logging
   - Set up alerts for failures

## Common Issues and Solutions

1. **Docker Build Failures**
   ```bash
   # Check build context
   docker build -t test . --target unit_test
   ```

2. **Secret Access Issues**
   - Verify AWS credentials
   - Check Key Vault access policies
   - Ensure proper secret names

3. **Test Failures**
   - Check test environment configuration
   - Verify dependency versions
   - Review test logs

## Customization Points

1. **Environment Variables**
   - Adjust `APP_NAME` and `WORKING_DIR`
   - Configure custom secrets
   - Set appropriate Python version

2. **Build Stages**
   - Add/remove build targets
   - Modify test configurations
   - Customize deployment steps

3. **Quality Gates**
   - Configure SonarCloud rules
   - Adjust coverage thresholds
   - Add custom quality checks

