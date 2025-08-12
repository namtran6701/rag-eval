import os

def get_azure_key_vault_secret(secret_name):
    """
    Retrieve a secret value from Azure Key Vault.

    Args:
        secret_name (str): The name of the secret to retrieve.

    Returns:
        str: The value of the secret.

    Raises:
        Exception: If the secret cannot be retrieved.
    """
    from azure.keyvault.secrets import SecretClient
    from azure.identity import DefaultAzureCredential

    try:
        keyVaultName = os.getenv("AZURE_KEY_VAULT_NAME")
        if not keyVaultName:
            raise ValueError("Environment variable 'AZURE_KEY_VAULT_NAME' is not set.")

        KVUri = f"https://{keyVaultName}.vault.azure.net"
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=KVUri, credential=credential)
        print(
            f"[webbackend] retrieving {secret_name} secret from {keyVaultName}."
        )
        retrieved_secret = client.get_secret(secret_name)
        return retrieved_secret.value
    except Exception as e:
        print(f"Failed to retrieve secret '{secret_name}': {e}")
        raise