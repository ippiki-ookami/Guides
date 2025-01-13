# GitHub Setup

## Initial Setup

First, download your class repository from GitHub, and upload the folder named GH-ML4T-Fall24 into your Google Drive. 

You will need a Personal Access Token (PAT) from your GT GitHub account to authenticate. Below are the instructions for creating one.

https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic

## Ran At Beginning of Each Session

```python
# Mount your Google Drive to access the repository
from google.colab import drive
drive.mount('/content/drive')

# Navigate to the repository folder in Google Drive
%cd /content/drive/MyDrive/GH-ML4T-Fall24

# Configure Git (required for each new Colab session)
!git config --global user.name "<user-name>"
!git config --global user.email "<user-name>@gatech.edu"

# Set the remote URL with your Personal Access Token (replace <your-PAT>)
!git remote set-url origin https://<user-name>:<your-PAT>@github.gatech.edu/<user-name>/GH-ML4T-Fall24.git

# Pull the latest changes from the remote repository
!git pull origin main

```
## Ran as Needed

### Update and Push Changes

```python
# Navigate to the repository folder
%cd /content/drive/MyDrive/GH-ML4T-Fall24

# Add all changes to the staging area (or use specific files/folders)
!git add .

# Commit changes with a descriptive message (example with multiline notes)
!git commit -m """
Refactor and update repository structure

- Added new preprocessing scripts
- Updated Assignment1 notebook with analysis
- Improved README documentation
"""

# Push changes to the remote repository
!git push origin main

```
