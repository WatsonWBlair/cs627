# AWS CLI Setup

Command-line setup for AWS resources and EC2 training instances.

## Prerequisites

Install and configure AWS CLI:
```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)
```

## 1. IAM Role Setup

```bash
# Create trust policy
cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

# Create IAM role
aws iam create-role \
  --role-name cs627-ec2-role \
  --assume-role-policy-document file://trust-policy.json

# Attach S3 access policy
aws iam attach-role-policy \
  --role-name cs627-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create and configure instance profile
aws iam create-instance-profile --instance-profile-name cs627-profile
aws iam add-role-to-instance-profile \
  --instance-profile-name cs627-profile \
  --role-name cs627-ec2-role
```

## 2. S3 Bucket Setup

```bash
# Create bucket with unique name
export S3_BUCKET="cs627-training-$(whoami)-$(date +%Y%m%d)"
aws s3 mb s3://$S3_BUCKET --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket $S3_BUCKET \
  --versioning-configuration Status=Enabled

# Create folder structure
for folder in tokens weights results logs; do
  aws s3api put-object --bucket $S3_BUCKET --key $folder/
done

# Set lifecycle policy (transition old tokens to cheaper storage)
cat > lifecycle.json <<EOF
{
  "Rules": [{
    "Id": "TransitionOldTokens",
    "Status": "Enabled",
    "Filter": {"Prefix": "tokens/"},
    "Transitions": [
      {"Days": 30, "StorageClass": "STANDARD_IA"},
      {"Days": 90, "StorageClass": "GLACIER_IR"}
    ]
  }]
}
EOF
aws s3api put-bucket-lifecycle-configuration \
  --bucket $S3_BUCKET \
  --lifecycle-configuration file://lifecycle.json
```

## 3. Security Group

```bash
# Create security group
aws ec2 create-security-group \
  --group-name cs627-training \
  --description "CS627 training security group"

# Allow SSH
aws ec2 authorize-security-group-ingress \
  --group-name cs627-training \
  --protocol tcp --port 22 --cidr 0.0.0.0/0

# Allow Jupyter (optional)
aws ec2 authorize-security-group-ingress \
  --group-name cs627-training \
  --protocol tcp --port 8888 --cidr 0.0.0.0/0
```

## 4. Find Latest AMI

```bash
# Find latest PyTorch GPU AMI
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=*Deep Learning AMI GPU PyTorch*" \
  --query 'sort_by(Images, &CreationDate)[-1].[ImageId,Name]' \
  --output text
```

## 5. Launch Instance

```bash
# Launch g5.4xlarge instance
aws ec2 run-instances \
  --image-id ami-04f3e35dc85e9423b \
  --instance-type g5.4xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxx \
  --iam-instance-profile Name=cs627-profile \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cs627-training}]'
```

## 6. Connect and Train

```bash
# Get instance IP
INSTANCE_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=cs627-training" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# SSH to instance
ssh -i your-key.pem ubuntu@$INSTANCE_IP

# On instance: setup and train
git clone https://github.com/WatsonWBlair/cs627.git && cd cs627
python src/Training/pregenerate_tokens.py
python src/Training/train_adapters.py --mode encoder
python src/Training/train_adapters.py --mode decoder

# Upload results to S3
aws s3 cp OptimalWeights/ s3://$S3_BUCKET/weights/ --recursive
```

## 7. Data Transfer

```bash
# Upload tokens to S3
aws s3 cp data/pregenerated_tokens/ s3://$S3_BUCKET/tokens/ --recursive

# Download tokens on instance
aws s3 cp s3://$S3_BUCKET/tokens/ data/pregenerated_tokens/ --recursive

# Sync results back
aws s3 sync OptimalWeights/ s3://$S3_BUCKET/weights/
aws s3 sync Results/ s3://$S3_BUCKET/results/
```

## 8. Cost Optimization

### Spot Instances (70% savings)
```bash
aws ec2 request-spot-instances \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-spec.json
```

### Auto-termination
```bash
# Terminate after training completes
echo "sudo shutdown -h +120" | at now
```

## 9. Cleanup

```bash
# Terminate instance
aws ec2 terminate-instances --instance-ids $INSTANCE_ID

# Delete security group
aws ec2 delete-security-group --group-name cs627-training

# Delete S3 bucket (careful!)
aws s3 rb s3://$S3_BUCKET --force

# Delete IAM resources
aws iam remove-role-from-instance-profile \
  --instance-profile-name cs627-profile --role-name cs627-ec2-role
aws iam delete-instance-profile --instance-profile-name cs627-profile
aws iam detach-role-policy --role-name cs627-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam delete-role --role-name cs627-ec2-role
```

## Automated Scripts

```bash
# Setup all resources
./scripts/setup/setup_aws_resources.sh --region us-east-1
source .env.aws

# Launch instance
./scripts/cloud/launch_training_instance.sh --instance-type g5.4xlarge

# Cleanup
./scripts/setup/cleanup_aws_resources.sh
```

## Memory Configuration

| Batch Size | RAM | Swap | GPU VRAM | Instance |
|------------|-----|------|----------|----------|
| 8 | 16GB | 8GB | 12GB | g5.xlarge |
| 16 | 32GB | 16GB | 16GB | g5.2xlarge |
| 32 | 64GB | 32GB | 24GB | g5.4xlarge |

### Add Swap Space
```bash
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Environment Variables

See [docs/ENVIRONMENT_VARIABLES.md](../ENVIRONMENT_VARIABLES.md) for all configuration options.
