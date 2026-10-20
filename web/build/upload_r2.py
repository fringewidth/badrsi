"""Put the weights and tables in a Cloudflare R2 bucket, laid out for a CDN.

Each export goes into a folder named after its version and is cached for a day at most; manifest.json and meta.json sit at
fixed paths, are cached for a minute, and name the folder. A new export is live a minute after upload, and a browser
never mixes pieces of two exports.

    R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
    uv run --with boto3 python web/build/upload_r2.py <bucket> https://your.site

The second argument is the origin the page is served from; the bucket only answers browsers on that origin.
"""

import json
import os
import sys
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig

DIST = Path(__file__).resolve().parent.parent / "dist"
DAY = "public, max-age=86400"
MINUTE = "public, max-age=60"


def put(s3, bucket: str, key: str, path: Path | None = None, body: bytes | None = None, cache: str = DAY) -> None:
    kind = "application/json" if key.endswith(".json") else "application/octet-stream"
    extra = {"ContentType": kind, "CacheControl": cache}
    if body is not None:
        s3.put_object(Bucket=bucket, Key=key, Body=body, **extra)
    else:
        s3.upload_file(str(path), bucket, key, ExtraArgs=extra, Config=TransferConfig(multipart_chunksize=64 * 2**20, max_concurrency=4))
    print(f"  {key}")


def main() -> None:
    bucket, origin = sys.argv[1], sys.argv[2]
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    s3.put_bucket_cors(Bucket=bucket, CORSConfiguration={"CORSRules": [{
        "AllowedOrigins": [origin],
        "AllowedMethods": ["GET", "HEAD"],
        "AllowedHeaders": ["*"],
        "ExposeHeaders": ["Content-Length"],
        "MaxAgeSeconds": 86400,
    }]})

    manifest = json.loads((DIST / "model" / "manifest.json").read_text())
    folder = manifest["version"] + "/"
    files = [manifest["graph"]] + [d["file"] for d in manifest["data"]]
    print(f"model {manifest['version']}")
    for f in files:
        put(s3, bucket, f"model/{folder}{f}", DIST / "model" / f)
    put(s3, bucket, "model/manifest.json", body=json.dumps({**manifest, "dir": folder}).encode(), cache=MINUTE)

    meta = json.loads((DIST / "data" / "meta.json").read_text())
    folder = meta["version"] + "/"
    print(f"data {meta['version']}")
    for f in [f["file"] for f in meta["files"].values()] + list(meta["lexicon"].values()):
        put(s3, bucket, f"data/{folder}{f}", DIST / "data" / f)
    put(s3, bucket, "data/meta.json", body=json.dumps({**meta, "dir": folder}, ensure_ascii=False).encode(), cache=MINUTE)


if __name__ == "__main__":
    main()
