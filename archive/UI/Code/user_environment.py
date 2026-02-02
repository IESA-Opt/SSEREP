"""
User-specific environment configuration utility.
Automatically detects the current user and loads their environment settings.
"""
import os
import sys
import json
import getpass
from pathlib import Path


def safe_print(text):
    """
    Print text safely, handling encoding issues in batch files.
    Replaces Unicode characters that can't be encoded in the console.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic Unicode characters with ASCII equivalents
        safe_text = text.replace('✓', '[OK]').replace('⚠', '[WARNING]').replace('✗', '[ERROR]')
        print(safe_text)


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_user_config():
    """Load user configuration from user_config.json."""
    config_path = get_project_root() / "user_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"User configuration file not found: {config_path}\n"
            "Please create a user_config.json file in the project root."
        )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def get_current_user():
    """Get the current Windows username."""
    return getpass.getuser()


def get_user_environment_settings(username=None, silent=False):
    """
    Get environment settings for the specified user.
    If username is not provided, uses the current user.
    Falls back to 'default' if user not found in config.
    
    Args:
        username: Optional username to look up
        silent: If True, suppress status messages
    
    Returns:
        dict: User environment settings including:
            - conda_path: Path to conda executable
            - activate_script: Path to activate script
            - environment_name: Name of conda environment
            - python_version: Python version
            - environment_path: Path to environment
    """
    if username is None:
        username = get_current_user()
    
    config = load_user_config()
    users = config.get('users', {})
    
    # Try to get user-specific settings, fall back to default
    if username in users:
        user_settings = users[username]
        if not silent:
            safe_print(f"✓ Loaded environment settings for user: {username}")
    elif 'default' in users:
        user_settings = users['default']
        if not silent:
            safe_print(f"⚠ User '{username}' not found in config. Using default settings.")
    else:
        raise ValueError(
            f"No configuration found for user '{username}' and no default configuration exists.\n"
            "Please add user configuration to user_config.json"
        )
    
    return user_settings


def get_conda_command(script_name=None):
    """
    Get the full conda run command for the current user.
    
    Args:
        script_name: Optional script name to append to the command
    
    Returns:
        str: Full conda run command
    """
    settings = get_user_environment_settings()
    conda_path = settings['conda_path']
    env_name = settings['environment_name']
    
    cmd = f'{conda_path} run -n {env_name}'
    
    if script_name:
        cmd += f' python {script_name}'
    
    return cmd


def get_activation_command(silent=False):
    """
    Get the activation command for the current user's environment.
    Returns empty string if no conda environment is configured.
    
    Args:
        silent: If True, suppress status messages
    
    Returns:
        str: Activation command (empty if no conda)
    """
    settings = get_user_environment_settings(silent=silent)
    
    # Check if this is a conda environment
    if settings.get("activate_script") and settings.get("environment_name"):
        activate_script = settings["activate_script"]
        env_name = settings["environment_name"]
        
        # Return the appropriate activation command
        if activate_script:
            return f'"{activate_script}" {env_name}'
        else:
            # No activation script means standalone Python
            return ""
    else:
        # No conda environment configured
        return ""


def print_environment_info():
    """Print current user's environment information."""
    username = get_current_user()
    settings = get_user_environment_settings()
    
    safe_print("\n" + "="*60)
    safe_print("ENVIRONMENT CONFIGURATION")
    safe_print("="*60)
    safe_print(f"Current User:      {username}")
    safe_print(f"Conda Path:        {settings['conda_path']}")
    safe_print(f"Environment Name:  {settings['environment_name']}")
    safe_print(f"Python Version:    {settings['python_version']}")
    safe_print(f"Environment Path:  {settings['environment_path']}")
    safe_print(f"\nActivation Command:")
    safe_print(f"  {get_activation_command()}")
    safe_print(f"\nConda Run Command:")
    safe_print(f"  {get_conda_command()}")
    safe_print("="*60 + "\n")


if __name__ == "__main__":
    # Test the configuration
    print_environment_info()
