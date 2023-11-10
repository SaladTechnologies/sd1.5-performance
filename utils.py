import boto3
import json
import pandas as pd
import re
import requests
from PIL import Image
from io import BytesIO
from typing import Iterable, Dict
import time
import uuid
import asyncio
import aioboto3
from aioboto3.session import Session
import os
import dotenv

dotenv.load_dotenv(".env", override=True)

salad_api_key = os.getenv("SALAD_API_KEY")
salad_org_id = os.getenv("SALAD_ORG")
salad_project_name = os.getenv("SALAD_PROJECT_NAME")
reporting_api_key = os.getenv("REPORTING_API_KEY")
reporting_url = os.getenv("REPORTING_URL")
queue_service_url = os.getenv("QUEUE_SERVICE_URL")

salad_headers = {
    "accept": "application/json",
    "Salad-Api-Key": salad_api_key,
}

reporting_headers = {
    "accept": "application/json",
    "Benchmark-Api-Key": reporting_api_key,
}

salad_api_base_url = "https://api.salad.com/api/public"

dynamodb = boto3.client("dynamodb", region_name="us-east-2")


def query_dynamodb_table(benchmark_id):
    # Initial query parameters
    query_params = {
        "TableName": "benchmark-data",  # replace with your table name
        "KeyConditionExpression": "benchmark_id = :benchmarkValue",
        "IndexName": "benchmark_id-timestamp-index",
        "ExpressionAttributeValues": {":benchmarkValue": {"S": benchmark_id}},
    }

    while True:
        # Execute the query
        response = dynamodb.query(**query_params)
        # Yield each item
        for item in response["Items"]:
            yield item

        # If there's more data to be retrieved, update the ExclusiveStartKey
        if "LastEvaluatedKey" in response:
            query_params["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        else:
            break


def get_rows_for_pd(benchmark_id):
    for item in query_dynamodb_table(benchmark_id):
        timestamp = item["timestamp"]["N"]
        data = json.loads(item["data"]["S"])
        system = data["system_info"]
        del data["system_info"]

        row = {**data, **system, "timestamp": timestamp}
        yield row


def get_df_for_benchmark(benchmark_id):
    df = pd.DataFrame(get_rows_for_pd(benchmark_id))
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def performance_score(gpu_name):
    # Extract the number part using regex
    match = re.search(r"(\d+)", gpu_name)
    if match:
        number = int(match.group(1))
    else:
        return 0  # Default performance score in case no number is found

    # Check for 'Ti', 'Laptop', and combinations
    if "Ti" in gpu_name and "Laptop" in gpu_name:
        return number + 0.3
    elif "Ti" in gpu_name:
        return number + 0.5
    elif "Laptop" in gpu_name:
        return number
    else:
        return number + 0.1


def shorten_gpu_name(full_name):
    shortened = []
    for name in full_name.split("\n"):
        # Extract the GPU model number, any 'Ti' suffix, and "Laptop GPU" distinction
        match = re.search(r"(RTX|GTX) (\d{3,4})( Ti)?( Laptop GPU)?", name)
        if match:
            shortened.append(
                match.group(1)
                + " "
                + match.group(2)
                + (match.group(3) or "")
                + (" Laptop" if match.group(4) else "")
            )
        else:
            shortened.append(name)
    return " & ".join(shortened)


# A function to load an image from a url
def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


def dict_to_md_list(dictionary: dict):
    if dictionary is None:
        return None
    return "\n".join(["- **{}**: {}".format(k, v) for k, v in dictionary.items()])


def dict_to_html_list(dictionary: dict):
    if dictionary is None:
        return None
    return (
        "<ul>"
        + "".join([f"<li><b>{k}</b>: {v}</li>" for k, v in dictionary.items()])
        + "</ul>"
    )


async def send_messages(client, queue_url, messages):
    response = await client.send_message_batch(QueueUrl=queue_url, Entries=messages)
    return response


def get_queue_name(queue_id: str):
    return f"benchmark-{queue_id}.fifo"


def get_job_from_queue_service(queue_id: str):
    response = requests.get(
        f"{queue_service_url}${queue_id}", headers=reporting_headers
    )
    if response.status_code == 200:
        return response.json()
    else:
        return None


async def queue_jobs(
    queue_id: str, jobs: Iterable[Dict], concurrency: int = 10, delay: int = 1
):
    # Each job is a json-serializable dictionary that should be sent as the message body.
    # We want to send these in batches of 10, with a configurable concurrency, and a configurable
    # delay between each batch of batches. A concurrency of 10 means we send 10 batches of 10 jobs each
    # before waiting for the delay. The delay is in seconds.
    batch_size = 10

    batch = []
    batches = []
    total = 0

    # Initialize the queue by requesting a job. There won't be one, so we
    # don't need to do anything with it.
    get_job_from_queue_service(queue_id)

    async with Session().client("sqs", region_name="us-east-2") as client:
        queue_url = client.get_queue_url(QueueName=get_queue_name(queue_id))["QueueUrl"]
        for job in jobs:
            job_id = str(uuid.uuid4())
            job["id"] = job_id
            batch.append(
                {
                    "Id": job_id,
                    "MessageDeduplicationId": job_id,
                    "MessageGroupId": job_id,
                    "MessageBody": json.dumps(job),
                }
            )
            if len(batch) == batch_size:
                # asyncronously send the batch of jobs
                batches.append(send_messages(client, queue_url, batch))
                batch = []
                if len(batches) == concurrency:
                    # wait for all the batches to finish
                    await asyncio.gather(*batches)
                    total += sum([len(b["Successful"]) for b in batches])
                    print(f"Sent {total} jobs so far.", flush=True, end="\r")
                    batches = []
                    # wait for the delay
                    await asyncio.sleep(delay)
        # send the last batch
        if len(batch) > 0:
            batches.append(send_messages(client, queue_url, batch))
        # wait for the last batches to finish
        await asyncio.gather(*batches)
        total += sum([len(b["Successful"]) for b in batches])
        print(f"Sent {total} jobs in total.", flush=True, end="\r")
        print()
        return True


def list_all_objects(s3: boto3.client, bucket: str, prefix: str = ""):
    """List all objects in an S3 bucket."""
    paginator = s3.get_paginator("list_objects_v2")
    params = {"Bucket": bucket}

    if prefix != "":
        params["Prefix"] = prefix

    for page in paginator.paginate(**params):
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            yield obj


def get_file_from_bucket(s3: boto3.client, bucket: str, key: str):
    """Get a file from an S3 bucket."""
    response = s3.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def list_all_container_groups():
    url = f"https://api.salad.com/api/public/organizations/{salad_org_id}/projects/{salad_project_name}/containers"
    response = requests.get(url, headers=salad_headers)
    return response.json()["items"]


def stop_container_group(container_group_name: str):
    url = f"https://api.salad.com/api/public/organizations/{salad_org_id}/projects/{salad_project_name}/containers/{container_group_name}/stop"

    response = requests.post(url, headers=salad_headers)
    if response.status_code == 202:
        return True
    else:
        return False


def start_container_group(container_group_name: str):
    url = f"https://api.salad.com/api/public/organizations/{salad_org_id}/projects/{salad_project_name}/containers/{container_group_name}/start"

    response = requests.post(url, headers=salad_headers)
    if response.status_code == 202:
        return True
    else:
        return False


def delete_container_group(container_group_name: str):
    url = f"https://api.salad.com/api/public/organizations/{salad_org_id}/projects/{salad_project_name}/containers/{container_group_name}"

    response = requests.delete(url, headers=salad_headers)
    if response.status_code == 202:
        return True
    else:
        return False


def stop_all_container_groups():
    for container_group in list_all_container_groups():
        stop_container_group(container_group["name"])


def start_all_container_groups():
    for container_group in list_all_container_groups():
        start_container_group(container_group["name"])


def delete_all_container_groups():
    for container_group in list_all_container_groups():
        delete_container_group(container_group["name"])
