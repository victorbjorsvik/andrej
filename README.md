# MINI-GPT
> Personal repo for follow-along with Andrej Karapathy's nano-GPT2. In this repo I have tried to implement alll corrections from *errata* and also several elements from PR (e.g. shuffling fineweb to avoid periodization issue)




---
## Instructions on setting up lambda cluster
1. Make sure to have your conda env config file (environment.yml) up to date.
2. Initialize a GPU cluster on [Lambda websites](https://cloud.lambdalabs.com/instances) 
3. Hop on a terminal and SSH into the cluster:
```bash
ssh -i C:/Users/<USERNAME>/.ssh/LAMBDA_SSH.pem ubuntu@<XXX.XXX.XXX.XXX>
```
4. Clone this repo:
```bash
git clone https://github.com/victorbjorsvik/andrej.git
```
5. Install MiniConda:
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
```bash
sh Miniconda3-latest-Linux-x86_64.sh
```
```bash
source ~/.bashrc
```
6. Create the environment:
```bash
conda env create -f environment.yml
```
7. Rock n' Roll

---
## Instructions on setting up Google Cloud CLI (for uploading model states)
[*Link to google cloud storage*](https://console.cloud.google.com/storage/browser)


1. Check that Ubuntu is up to date and has necessary packages:
```bash
sudo apt-get update
```
```bash
sudo apt-get install apt-transport-https ca-certificates gnupg curl
```
2. Import Google Cloud public key:
```bash
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
```
3. Add the gcloud CLI distribution URI as a package source
```bash
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
```
4. Update and install the gcloud CLI:
```bash
sudo apt-get update && sudo apt-get install google-cloud-cli
```
5. Run `gcloud init` to get started
```bash
gcloud init
```
> This step will prompt you to autorize with your google account via browser. Upon verification you will receive a verification code that you can input in the terminal. After this step you will be able to transfer files from your local directory to a google cloud storage destination e.g.:
 ```bash
gcloud storage cp <file_to_transfer> gs://<destination_bucket_name>
```