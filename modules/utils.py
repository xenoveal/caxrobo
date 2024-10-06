from modules.logger import setup_logger

logger = setup_logger('UtilsLogger', 'logs/utils.log')

def get_env (var_name: str) -> str | None | int | float:
    from dotenv import load_dotenv
    import os

    # Load .env file
    load_dotenv()

    # Access environment variables
    try:
        value = os.getenv(var_name)
        
        if (value == "") | (value == None):
            raise Exception("Variable not exist")

        logger.info(f"[GET-ENV] Success to get {var_name} {value}")
        return value
    
    except Exception as e:
        logger.error(f"[GET-ENV] Failed to get {var_name}\nError: {e}")
        return None
    