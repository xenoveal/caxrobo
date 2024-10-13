from controllers.learning import LearningController
from modules.logger import setup_logger

logger = setup_logger('MainLogger', 'logs/main.log')

if __name__ == "__main__":
    #TODO: Clean main.py -> separate to Learning & Running controllers
    LearningController().main()
