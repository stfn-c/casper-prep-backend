import boto3
import tempfile
from pathlib import Path
from typing import BinaryIO, Optional
from app.config import get_settings_sync


class R2Service:
    """Service for interacting with Cloudflare R2 storage."""

    _client = None
    _bucket_name = None

    @property
    def client(self):
        """Lazy-load the S3 client."""
        if self._client is None:
            settings = get_settings_sync()
            self._client = boto3.client(
                's3',
                endpoint_url=settings.r2_endpoint,
                aws_access_key_id=settings.r2_access_key_id,
                aws_secret_access_key=settings.r2_secret_access_key,
                region_name='auto'  # R2 uses 'auto' region
            )
        return self._client

    @property
    def bucket_name(self):
        """Lazy-load the bucket name."""
        if self._bucket_name is None:
            settings = get_settings_sync()
            self._bucket_name = settings.r2_bucket_name
        return self._bucket_name

    def download_video(self, key: str) -> Path:
        """
        Download a video from R2 to a temporary file.

        Args:
            key: The S3 key (path) of the video in R2

        Returns:
            Path to the temporary file containing the downloaded video

        Raises:
            Exception: If download fails
        """
        # Create a temporary file with .webm extension
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.webm',
            prefix='casper_video_'
        )

        try:
            # Download the file from R2
            self.client.download_fileobj(
                Bucket=self.bucket_name,
                Key=key,
                Fileobj=temp_file
            )
            temp_file.close()
            return Path(temp_file.name)
        except Exception as e:
            # Clean up temp file if download fails
            temp_file.close()
            Path(temp_file.name).unlink(missing_ok=True)
            raise Exception(f"Failed to download video from R2: {str(e)}")

    def upload_video(self, file: BinaryIO, key: str, content_type: str = "video/webm") -> str:
        """
        Upload a video to R2.

        Args:
            file: File-like object to upload
            key: The S3 key (path) where the video will be stored
            content_type: MIME type of the video

        Returns:
            The key of the uploaded file

        Raises:
            Exception: If upload fails
        """
        try:
            self.client.upload_fileobj(
                Fileobj=file,
                Bucket=self.bucket_name,
                Key=key,
                ExtraArgs={'ContentType': content_type}
            )
            return key
        except Exception as e:
            raise Exception(f"Failed to upload video to R2: {str(e)}")

    def generate_video_key(self, user_id: str, attempt_id: int, question_index: int) -> str:
        """
        Generate a unique key for video storage following the established pattern.

        Path: videos/user-{userId}/scenario-attempt-{attemptId}/q{index}.webm

        Args:
            user_id: User ID
            attempt_id: Scenario attempt ID
            question_index: Question index

        Returns:
            The generated S3 key
        """
        return f"videos/user-{user_id}/scenario-attempt-{attempt_id}/q{question_index}.webm"


# Global R2 service instance
r2_service = R2Service()
