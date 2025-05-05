"""
S3 integration for storing and retrieving guideline documents and assets.
"""

import logging
import boto3
from botocore.exceptions import ClientError
from typing import Optional, BinaryIO, Dict, Any, List, Union
import io

from app.config import get_settings

logger = logging.getLogger(__name__)

class S3Client:
    """Client for interacting with AWS S3."""
    
    def __init__(self):
        """Initialize S3 client with credentials from config."""
        settings = get_settings()
        
        # Get S3 settings from config
        self.bucket_name = settings.s3.bucket_name
        self.region = settings.s3.region
        
        # Initialize S3 client
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=settings.s3.access_key_id,
            aws_secret_access_key=settings.s3.secret_access_key,
            region_name=self.region,
            endpoint_url=settings.s3.endpoint_url
        )
        
        logger.info(f"Initialized S3 client for bucket: {self.bucket_name}")
    
    def upload_file(
        self, 
        file_obj: Union[BinaryIO, bytes], 
        object_key: str, 
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Upload a file to S3.
        
        Args:
            file_obj: File-like object or bytes to upload
            object_key: S3 object key (path)
            content_type: Optional content type
            metadata: Optional metadata dictionary
            
        Returns:
            Boolean indicating success
        """
        try:
            extra_args = {}
            
            if content_type:
                extra_args['ContentType'] = content_type
                
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Handle both file-like objects and bytes
            if isinstance(file_obj, bytes):
                # Convert bytes to file-like object
                file_obj = io.BytesIO(file_obj)
            
            self.s3.upload_fileobj(
                file_obj,
                self.bucket_name,
                object_key,
                ExtraArgs=extra_args
            )
            
            logger.info(f"Successfully uploaded file to s3://{self.bucket_name}/{object_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error uploading file to S3: {str(e)}")
            return False
    
    def download_file(self, object_key: str) -> Optional[bytes]:
        """
        Download a file from S3.
        
        Args:
            object_key: S3 object key (path)
            
        Returns:
            File content as bytes, or None if not found
        """
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
            return response['Body'].read()
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"File not found in S3: {object_key}")
            else:
                logger.error(f"Error downloading file from S3: {str(e)}")
            return None
    
    def get_presigned_url(self, object_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for an S3 object.
        
        Args:
            object_key: S3 object key (path)
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL or None if error
        """
        try:
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            return url
            
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {str(e)}")
            return None
    
    def list_objects(self, prefix: str) -> List[Dict[str, Any]]:
        """
        List objects in the S3 bucket with the given prefix.
        
        Args:
            prefix: S3 object key prefix
            
        Returns:
            List of objects with their metadata
        """
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                return response['Contents']
            return []
            
        except ClientError as e:
            logger.error(f"Error listing objects in S3: {str(e)}")
            return []
    
    def delete_file(self, object_key: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            object_key: S3 object key (path)
            
        Returns:
            Boolean indicating success
        """
        try:
            self.s3.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            
            logger.info(f"Successfully deleted file from s3://{self.bucket_name}/{object_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error deleting file from S3: {str(e)}")
            return False


# Create a singleton instance
s3_client = S3Client()

def get_s3_client() -> S3Client:
    """Get the S3 client instance."""
    return s3_client