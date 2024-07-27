import logging
from Classes.ProjectStrings import ProjectStrings
import os

# Set up logging to both file and console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("./logs/main.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

LOCK_FILE = '.setup_complete'


def setup_project():
    """Set up the project by initializing necessary components."""
    try:
        strings = ProjectStrings()
        logger.info("Project directories have been set up successfully.")
        return strings
    except Exception as e:
        logger.error(f"Failed to set up project strings: {str(e)}")
        raise


def main():
    """Run the main function of the project."""
    if os.path.exists(LOCK_FILE):
        logger.info("Setup has already been completed. Skipping.")
        return

    try:
        setup_project()
        logger.info("Main function executed successfully.")

        # Create the lock file to indicate the script has run
        with open(LOCK_FILE, 'w') as f:
            f.write('Setup complete')

    except Exception as e:
        logger.error(f"An error occurred in the main function: {str(e)}")


if __name__ == "__main__":
    main()
