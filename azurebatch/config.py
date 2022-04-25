# Global constant variables (Azure Storage account/Batch details)

# import "config.py" in "python_quickstart_client.py "
# Please note that storing the batch and storage account keys in Azure Key Vault
# is a better practice for Production usage.

"""
Configure Batch and Storage Account credentials
"""

BATCH_ACCOUNT_NAME = ''  # Your batch account name
BATCH_ACCOUNT_KEY = ''  # Your batch account key
BATCH_ACCOUNT_URL = ''  # Your batch account URL
STORAGE_ACCOUNT_NAME = ''
STORAGE_ACCOUNT_KEY = ''
STORAGE_ACCOUNT_DOMAIN = 'blob.core.windows.net' # Your storage account blob service domain

POOL_ID = 'PythonQuickstartPool'  # Your Pool ID
POOL_NODE_COUNT = 2  # Pool node count
POOL_VM_SIZE = 'STANDARD_DS1_V2'  # VM Type/Size
JOB_ID = 'PythonQuickstartJob'  # Job ID
STANDARD_OUT_FILE_NAME = 'stdout.txt'  # Standard Output file