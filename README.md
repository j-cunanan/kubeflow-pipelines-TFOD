
# kubeflow-pipelines-TFOD

The compiled yaml file will work on any deploed Kubeflow pipelines with GPU attached.
The setup below is an example when using Google Cloud Platform.

## Pre-Setup
- Make sure you have at least **1 GPU** resource in your project. Verify by going to **Quotas** page in the GCP console.
- Choose **Limit name** in the filter drop-down then type **GPU**
- Choose **GPUs (all regions)**
- If you have **Limit: 0** then you should first request an increase of quota.
  - Tick the **Global** box then click **EDIT QUOTAS**.
  - Enter your desired **New limit** and complete the submission of your request.
  - The result and time to recieve your request depends on your account. 
    - In my case, my new account and project needs 2 days waiting time before I can resubmit my qouta requests which then got the increase within a few minutes.
## Setup

Open the `Cloud Shell` from you GCP console.
Type the following.

```
export PROJECT_ID=<your_project_id>
gcloud config set project ${PROJECT_ID}
```

### Create a storage bucket

```
export BUCKET_NAME=kubeflow-${PROJECT_ID}
gsutil mb gs://${BUCKET_NAME}
```

### Deploy AI Platform Pipelines (Kubeflow Pipelines)

Follow the instructions in the ‘Before you begin' and ‘Set up your instance' sections [here](https://cloud.google.com/ai-platform/pipelines/docs/getting-started).
Pick a zone which support the GPU of your choice. The default **us-central1-a** works for all GPU types.
After the deployment as completed type the following in the `Cloud Shell`.
```
export ZONE=<your zone>
export CLUSTER_NAME=<your cluster name>
```

### Set up kubectl to use your new GKE cluster's credentials

```
gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --project ${PROJECT_ID} \
  --zone ${ZONE}
```
Verify **kubectl** is working as expected.
`kubectl get nodes -o wide`
You should see nodes listed with a status of "`Ready`", and other information.

### Configure the cluster to install the Nvidia driver on gpu-enabled node pools

Next, we'll apply a daemonset to the cluster, which will install the Nvidia driver on any GPU-enabled cluster nodes:
`kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml`

Then run the following command, which gives the KFP components permission to create new Kubernetes resources:
`kubectl create clusterrolebinding sa-admin --clusterrole=cluster-admin --serviceaccount=kubeflow:pipeline-runner`

### Create a GPU node pool

Setup a GPU node pool of size 1:

```
gcloud container node-pools create gpu-pool \
    --cluster=${CLUSTER_NAME} \
    --zone ${ZONE} \
    --num-nodes=1 \
    --machine-type n1-highmem-8 \
    --scopes cloud-platform --verbosity error \
    --accelerator=type=nvidia-tesla-k80,count=1
```
