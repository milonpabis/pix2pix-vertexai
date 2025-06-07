def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def download_dataset(bucket_name: str, prefix: str, dest_dir: str = "data", credentials = None):
    from google.cloud import storage
    import os

    if credentials:
        client = storage.Client(credentials=credentials)
    else:
        client = storage.Client()
    print("FETCHING THE DATA") # slow version -> placeholder

    def fetch_data(subset: str):
        subset_prefix = os.path.join(prefix, subset)
        blobs = client.list_blobs(bucket_name, prefix=subset_prefix)
        for blob in blobs:
            rel_path = os.path.relpath(blob.name, subset_prefix)
            dest_path = os.path.join(dest_dir, subset, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            blob.download_to_filename(dest_path)
            print(f"Downloaded {blob.name} to {dest_path}")

    fetch_data("train")
    fetch_data("val")