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
          python3 -m venv $WORKSPACE/.venv
          $WORKSPACE/.venv/bin/pip install -q -r requirements.txt

          MLFLOW_TRACKING_URI=$MLFLOW_URI \
          MLFLOW_EXPERIMENT_NAME=heart-disease-ci \
          PYTHONPATH=$WORKSPACE/src \
          $WORKSPACE/.venv/bin/python3 -m heart.train \
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
          PYTHONPATH=$WORKSPACE/src $WORKSPACE/.venv/bin/pytest tests/ -v --tb=short
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
          gcloud auth activate-service-account --key-file=$GCP_KEY
          gcloud container clusters get-credentials $CLUSTER \
            --zone $CLUSTER_ZONE --project $PROJECT

          kubectl set image deployment/heart-api \
            heart-api=${IMAGE}:${GIT_COMMIT} -n nonprod

          kubectl rollout status deployment/heart-api \
            -n nonprod --timeout=120s
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
