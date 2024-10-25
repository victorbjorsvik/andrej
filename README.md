# andrej
> Personal repo for follow-along with Andrej Karapathy's Zero to Hero series




---
## Instructions on setting up lambda cluster
1. Make sure to have your conda env config file (environment.yml) up to date.
2. Initialize a GPU cluster on [Lambda websites](https://cloud.lambdalabs.com/instances) 
3. Hop on a terminal and SSH into the cluster:
```bash
ssh -i C:\Users\<USERNAME>\.ssh\LAMBDA_SSH.pem ubuntu@<111.111.111.111>
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