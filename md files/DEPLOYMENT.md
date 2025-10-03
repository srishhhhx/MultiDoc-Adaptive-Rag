# 🚢 Deployment Guide

This guide covers different deployment strategies for the Advanced RAG system.

## Table of Contents
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Checklist](#production-checklist)

## Docker Deployment

### Prerequisites
- Docker installed
- Docker Compose installed

### Quick Start with Docker

1. **Ensure `.env` file exists with your API keys**

2. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

3. **Access the application**
- Frontend: http://localhost
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Individual Container Builds

**Build Backend:**
```bash
docker build -f Dockerfile.backend -t advanced-rag-backend .
docker run -p 8000:8000 --env-file .env advanced-rag-backend
```

**Build Frontend:**
```bash
cd frontend
docker build -t advanced-rag-frontend .
docker run -p 80:80 advanced-rag-frontend
```

### Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up --build

# Remove volumes (WARNING: deletes data)
docker-compose down -v
```

## Cloud Deployment

### AWS Deployment

#### Option 1: AWS Elastic Beanstalk

**Backend:**
```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.11 advanced-rag-backend

# Create environment
eb create advanced-rag-prod

# Set environment variables
eb setenv OPENAI_API_KEY=your_key TAVILY_API_KEY=your_key

# Deploy
eb deploy
```

**Frontend:**
```bash
# Build
cd frontend
npm run build

# Deploy to S3 + CloudFront
aws s3 sync dist/ s3://your-bucket-name/
```

#### Option 2: AWS ECS (Docker)

1. Push images to ECR
2. Create ECS cluster
3. Define task definitions
4. Create services

### Google Cloud Platform

#### Cloud Run (Recommended)

**Backend:**
```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/advanced-rag-backend

# Deploy
gcloud run deploy advanced-rag-backend \
  --image gcr.io/PROJECT_ID/advanced-rag-backend \
  --platform managed \
  --set-env-vars OPENAI_API_KEY=your_key,TAVILY_API_KEY=your_key
```

**Frontend:**
```bash
# Build
cd frontend
npm run build

# Deploy to Firebase Hosting
firebase deploy
```

### Microsoft Azure

#### Azure Container Instances

```bash
# Create resource group
az group create --name advanced-rag-rg --location eastus

# Deploy backend
az container create \
  --resource-group advanced-rag-rg \
  --name backend \
  --image your-registry/advanced-rag-backend \
  --dns-name-label advanced-rag-backend \
  --ports 8000

# Deploy frontend
az container create \
  --resource-group advanced-rag-rg \
  --name frontend \
  --image your-registry/advanced-rag-frontend \
  --dns-name-label advanced-rag-frontend \
  --ports 80
```

### Heroku

**Backend:**
```bash
# Login
heroku login

# Create app
heroku create advanced-rag-backend

# Set config vars
heroku config:set OPENAI_API_KEY=your_key
heroku config:set TAVILY_API_KEY=your_key

# Deploy
git push heroku main
```

**Frontend:**
```bash
# Build locally
npm run build

# Deploy to Netlify or Vercel
netlify deploy --prod --dir=dist
# or
vercel --prod
```

### DigitalOcean

#### App Platform

1. Connect your GitHub repository
2. Configure build settings
3. Set environment variables
4. Deploy

## Production Checklist

### Security

- [ ] Use HTTPS/TLS certificates
- [ ] Implement authentication and authorization
- [ ] Set up rate limiting
- [ ] Use secrets management (AWS Secrets Manager, etc.)
- [ ] Enable CORS only for trusted domains
- [ ] Implement API key rotation
- [ ] Set up Web Application Firewall (WAF)

### Performance

- [ ] Enable caching (Redis, Memcached)
- [ ] Use CDN for frontend assets
- [ ] Implement connection pooling
- [ ] Set up load balancing
- [ ] Optimize database queries
- [ ] Enable gzip compression
- [ ] Implement request/response compression

### Monitoring

- [ ] Set up logging (ELK stack, CloudWatch)
- [ ] Configure error tracking (Sentry, Rollbar)
- [ ] Implement health checks
- [ ] Set up uptime monitoring
- [ ] Configure alerting
- [ ] Track performance metrics
- [ ] Monitor API usage

### Reliability

- [ ] Set up auto-scaling
- [ ] Implement circuit breakers
- [ ] Configure retry logic
- [ ] Set up database backups
- [ ] Create disaster recovery plan
- [ ] Document runbooks
- [ ] Set up staging environment

### Configuration

**Environment Variables for Production:**
```env
# API Keys
OPENAI_API_KEY=your_production_key
TAVILY_API_KEY=your_production_key

# Database
DATABASE_URL=your_database_url

# Security
SECRET_KEY=your_secret_key
ALLOWED_HOSTS=your-domain.com

# CORS
CORS_ORIGINS=https://your-frontend-domain.com

# Monitoring
SENTRY_DSN=your_sentry_dsn

# Feature Flags
DEBUG=false
LOG_LEVEL=info
```

## SSL/TLS Setup

### Using Let's Encrypt (Certbot)

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### Using Cloudflare

1. Point domain to Cloudflare
2. Enable "Full (strict)" SSL
3. Create origin certificate
4. Configure on server

## Database Setup (Optional)

For persistent storage beyond ChromaDB:

### PostgreSQL

```bash
# Docker
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=advancedrag \
  -p 5432:5432 \
  postgres:15

# Update api.py to use PostgreSQL for session/user data
```

### Redis (Caching)

```bash
# Docker
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:7-alpine

# Use for caching API responses
```

## Backup Strategy

### Automated Backups

**ChromaDB Data:**
```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf backup_chroma_$DATE.tar.gz .chroma/
aws s3 cp backup_chroma_$DATE.tar.gz s3://your-backup-bucket/
```

**Configuration:**
```bash
# Add to crontab
0 2 * * * /path/to/backup_script.sh
```

## Scaling Considerations

### Horizontal Scaling

- Use load balancer (Nginx, HAProxy, AWS ALB)
- Stateless backend design
- Shared file storage (S3, NFS)
- Distributed caching

### Vertical Scaling

- Increase CPU/RAM for compute-intensive tasks
- Use GPU instances for embedding generation
- Optimize chunk sizes and batch processing

## Cost Optimization

- [ ] Use reserved instances (AWS/GCP)
- [ ] Implement request caching
- [ ] Optimize API calls to external services
- [ ] Use spot instances for non-critical tasks
- [ ] Set up budget alerts
- [ ] Monitor and optimize resource usage

## Troubleshooting Production Issues

### Backend Issues

**High Memory Usage:**
- Check ChromaDB size
- Implement pagination
- Clear old embeddings

**Slow Response Times:**
- Enable caching
- Optimize database queries
- Use async operations

**API Rate Limits:**
- Implement queuing
- Use exponential backoff
- Cache responses

### Frontend Issues

**Slow Load Times:**
- Enable CDN
- Optimize bundle size
- Implement code splitting
- Use lazy loading

**Connection Issues:**
- Check CORS configuration
- Verify API endpoint URLs
- Check SSL certificates

## Rollback Procedure

1. **Identify issue** - Check logs and monitoring
2. **Stop deployment** - Halt any ongoing deployments
3. **Revert to previous version**
   ```bash
   # Docker
   docker-compose down
   docker-compose up -d --force-recreate
   
   # Kubernetes
   kubectl rollout undo deployment/advanced-rag-backend
   ```
4. **Verify rollback** - Test critical functionality
5. **Post-mortem** - Document what went wrong

## Support

For production support:
- Check logs first
- Review monitoring dashboards
- Consult documentation
- Open GitHub issue for bugs

---

Deploy with confidence! 🚀
