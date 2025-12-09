# AWS Troubleshooting

Common AWS issues and solutions for CS627 training.

## Connection Issues

### "Connection timeout"
```bash
# Check security group allows SSH
aws ec2 describe-security-groups --group-names cs627-training

# Check instance is running
aws ec2 describe-instance-status --instance-ids $INSTANCE_ID

# Verify public IP assigned
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress'
```

### "Permission denied (publickey)"
```bash
# Ensure correct key file
chmod 400 your-key.pem

# Verify username (ubuntu for Deep Learning AMI)
ssh -i your-key.pem ubuntu@<ip>  # Not ec2-user
```

## GPU Issues

### "No CUDA devices"
```bash
# Check GPU
nvidia-smi

# If not found, reinstall drivers
sudo apt install nvidia-driver-525
sudo reboot
```

### "CUDA out of memory"
```bash
# Reduce batch size
BATCH_SIZE=16 python src/Training/pregenerate_tokens.py
BATCH_SIZE=128 python src/Training/train_adapters.py
```

## IAM/S3 Issues

### "Access Denied"
1. Verify IAM role is attached to instance:
   - EC2 → Select instance → Actions → Security → Modify IAM role

2. Check S3 bucket policy allows the role

3. Test with:
```bash
aws sts get-caller-identity  # Shows assumed role
aws s3 ls s3://your-bucket/  # Test S3 access
```

### "Unable to locate credentials"
```bash
# Instance role should provide credentials automatically
# If not, configure region:
aws configure set region us-east-1

# Verify instance has IAM role attached
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
```

## Storage Issues

### "No space left on device"
```bash
# Check disk usage
df -h

# Clear Docker cache
docker system prune -a

# Remove old logs
rm -rf Results/logs/*

# Expand EBS volume (in AWS Console, then resize on instance)
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
```

### Token upload/download slow
```bash
# Use compression
tar -czf tokens.tar.gz data/pregenerated_tokens/
aws s3 cp tokens.tar.gz s3://bucket/

# Enable S3 Transfer Acceleration (in bucket settings)
aws s3 cp tokens.tar.gz s3://bucket/ --endpoint-url https://bucket.s3-accelerate.amazonaws.com
```

## Instance Issues

### Instance won't start
1. Check quota limits: EC2 → Limits
2. Try different availability zone
3. Check for capacity issues with instance type

### Instance terminated unexpectedly
1. Check instance state reason:
```bash
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].StateReason'
```
2. Common causes: Spot interruption, billing issue, shutdown command

## Cost Issues

### Unexpected charges
1. Check for running instances: EC2 → Instances
2. Check EBS volumes: EC2 → Volumes (delete unattached)
3. Check S3 storage: S3 → Metrics
4. Set up billing alerts: Billing → Budgets

### Reduce costs
- Use spot instances (70% savings)
- Auto-terminate after training
- Delete unused EBS snapshots
- Use S3 lifecycle policies

## Monitoring

### View training logs
```bash
tail -f Results/adapter_training/*.log
```

### Monitor resources
```bash
nvidia-smi -l 1  # GPU usage
htop             # CPU/memory
watch df -h      # Disk usage
```

### CloudWatch metrics
```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=InstanceId,Value=$INSTANCE_ID \
  --start-time $(date -d '1 hour ago' -Iseconds) \
  --end-time $(date -Iseconds) \
  --period 300 --statistics Average
```

## Getting Help

- [General troubleshooting](../TROUBLESHOOTING.md)
- [Training issues](../TRAINING_GUIDE.md)
- [AWS documentation](https://docs.aws.amazon.com/ec2/)
