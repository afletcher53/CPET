import logging
from Classes.ProjectStrings import ProjectStrings


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_project():
    """Set up the project by initializing necessary components."""
    try:
        strings = ProjectStrings()
        logger.info("Project strings set up successfully. Please add your SUM files to the cpet raw directory, and your CPETdb.xlsx file to the data directory.")
        return strings
    except Exception as e:
        logger.error(f"Failed to set up project strings: {str(e)}")
        raise


def main():
    """Run the main function of the project."""
    try:
        project_strings = setup_project()
        # Add your main logic here
        logger.info("Main function executed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main function: {str(e)}")


if __name__ == "__main__":
    main()
