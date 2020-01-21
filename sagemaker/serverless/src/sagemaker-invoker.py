import os
import io
import boto3
import json
import csv
import numpy as np

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    try:
        data = json.loads(event['body'])
        
        payload = [data[x] for x in data]

        f = io.BytesIO()
        np.save(f, payload)

        response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                        ContentType='application/x-npy',
                                        Body=f.getvalue())

        result = json.loads(response['Body'].read().decode())
        print(result)
    
        response = {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:

        response = {
            'statusCode': 500,
            'body': json.dumps(e)
        }
    return response