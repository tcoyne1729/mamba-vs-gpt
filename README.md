# mamba-vs-gpt

  1. Setup
``` bash
  curl -fsSL https://raw.githubusercontent.com/tcoyne1729/mamba-vs-gpt/refs/heads/main/runners/setup.sh | bash
                                                                                        ```
  2. Enter the repo dir and activate the venv

  ``` bash
  cd mamba-vs-gpt
  source .venv/bin/activate
  ```
  3. Launch the dev run
``` bash
  DEV_RUN=1 python gpt_train.py
```

  Or launch the prod run with

``` bash
python gpt_train.py
```

## Post-training: secure artifact export

After training, use `post_train.py` to encrypt and upload artifacts to GCS.

**Zero-trust design:** the host machine (vast.ai) cannot read your data even if it copies everything. Encryption uses [age](https://github.com/FiloSottile/age) with a public key embedded in the script — the private key never touches the instance. The GCS service account is write-only, so the host can't read back the bucket.

### One-time local setup

```bash
# 1. Install age
brew install age   # macOS — see https://github.com/FiloSottile/age/releases for Linux

# 2. Generate a keypair — keep this file safe, never commit it
age-keygen -o ~/mamba-key.txt
# Copy the "public key:" line into AGE_PUBLIC_KEY in post_train.py

# 3. Create a GCS bucket and write-only service account
export PROJECT=your-gcp-project BUCKET=your-bucket-name
gcloud storage buckets create gs://$BUCKET --location=US
gcloud iam service-accounts create vast-uploader --project=$PROJECT
gcloud storage buckets add-iam-policy-binding gs://$BUCKET \
  --member="serviceAccount:vast-uploader@$PROJECT.iam.gserviceaccount.com" \
  --role="roles/storage.objectCreator"
gcloud iam service-accounts keys create vast-sa-key.json \
  --iam-account=vast-uploader@$PROJECT.iam.gserviceaccount.com

# 4. Set in vast.ai instance config (not in code — never commit credentials):
#    GCS_BUCKET=your-bucket-name
#    GCS_SA_KEY_FILE=/root/mamba-vs-gpt/vast-sa-key.json
```

### On the instance after training

```bash
python post_train.py --model mamba   # or gpt
```

### Decrypt locally

```bash
age --decrypt --identity ~/mamba-key.txt mamba-sqale-*.tar.gz.age | tar -xz
```
