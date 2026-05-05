pipeline {
  agent any

  environment {
    PROJECT      = "heart-mlops-2026"
    REGION       = "asia-south1"
    REGISTRY     = "${REGION}-docker.pkg.dev/${PROJECT}/heart-app"
    IMAGE        = "${REGISTRY}/heart-api"
    CLUSTER      = "heart-cluster"
    CLUSTER_ZONE = "${REGION}-a"
    MLFLOW_URI   = credentials('mlflow-tracking-uri')
    GCP_KEY      = credentials('gcp-service-account-key')
  }

  stages {

    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Train') {
      steps {
        sh '''
          python3 -m venv .venv
          . .venv/bin/activate
          pip install -q -r requirements.txt

          MLFLOW_TRACKING_URI=$MLFLOW_URI \
          MLFLOW_EXPERIMENT_NAME=heart-disease-ci \
          PYTHONPATH=src \
          python3 -m heart.train \
            --data  data/processed.cleveland.data \
            --val-data data/processed.hungarian.data \
            --artifacts-dir artifacts \
            --output-model models/model.pkl
        '''
      }
      post {
        always {
          archiveArtifacts artifacts: 'artifacts/**', allowEmptyArchive: true
        }
      }
    }

    stage('Unit Tests') {
      steps {
        sh '''
          . .venv/bin/activate
          . .venv/bin/activate
          PYTHONPATH=src pytest tests/ -v --tb=short
        '''
      }
    }

    stage('Build & Push Docker Image') {
      steps {
        sh '''
          gcloud auth activate-service-account --key-file=$GCP_KEY
          gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

          docker build \
            -f docker/Dockerfile \
            -t ${IMAGE}:${GIT_COMMIT} \
            -t ${IMAGE}:latest \
            .

          docker push ${IMAGE}:${GIT_COMMIT}
          docker push ${IMAGE}:latest
        '''
      }
    }

    stage('Deploy to Non-Prod') {
      steps {
        sh '''
          gcloud container clusters get-credentials $CLUSTER \
            --zone $CLUSTER_ZONE --project $PROJECT

          kubectl set image deployment/heart-api \
            heart-api=${IMAGE}:${GIT_COMMIT} -n nonprod

          kubectl rollout status deployment/heart-api \
            -n nonprod --timeout=120s
        '''
      }
    }

    stage('Deploy to Prod (Blue/Green)') {
      when { branch 'main' }
      steps {
        input message: "Promote to Production?", ok: "Deploy"
        sh '''
          gcloud container clusters get-credentials $CLUSTER \
            --zone $CLUSTER_ZONE --project $PROJECT

          # Deploy new image to green slot
          kubectl set image deployment/heart-api-green \
            heart-api=${IMAGE}:${GIT_COMMIT} -n prod

          kubectl rollout status deployment/heart-api-green \
            -n prod --timeout=180s

          # Cut traffic to green
          kubectl patch service heart-api-svc -n prod \
            --type=json \
            -p="[{\"op\":\"replace\",\"path\":\"/spec/selector/slot\",\"value\":\"green\"}]"
        '''
      }
    }
  }

  post {
    failure {
      echo "Pipeline failed — check logs above"
    }
    success {
      echo "Pipeline succeeded — image: ${env.IMAGE}:${env.GIT_COMMIT}"
    }
  }
}
