def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def download_dataset(bucket_name: str, prefix: str, dest_dir: str):
    from google.cloud import storage
    from google.oauth2 import service_account
    import os

    credentials = service_account.Credentials.from_service_account_file(
        "sandbox-project-462110-215ba3072b18.json"
    )
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)

    blobs = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        rel_path = os.path.relpath(blob.name, prefix)
        dest_path = os.path.join(dest_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob.download_to_filename(dest_path)
        print(f"Downloaded {blob.name} to {dest_path}")
