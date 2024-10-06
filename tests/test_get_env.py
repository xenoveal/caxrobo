import unittest
from unittest.mock import patch
from modules.utils import get_env

class TestGetEnv(unittest.TestCase):

    @patch('os.getenv')
    def test_get_env_variable_exists(self, mock_getenv):
        # Mock the return value of os.getenv
        mock_getenv.return_value = "test123"
        
        # Call the get_env function
        result = get_env("ENV_TEST")
        
        # Assert the returned value matches the expected value
        self.assertEqual(result, "test123")
        mock_getenv.assert_called_once_with("ENV_TEST")

    @patch('os.getenv')
    def test_get_env_variable_does_not_exist(self, mock_getenv):
        # Mock os.getenv to return None
        mock_getenv.return_value = None
        
        # Call the get_env function
        result = get_env("NON_EXISTENT_VAR")
        
        # Assert the returned value is None
        self.assertIsNone(result)
        mock_getenv.assert_called_once_with("NON_EXISTENT_VAR")

    @patch('os.getenv')
    def test_get_env_variable_error(self, mock_getenv):
        # Simulate an error in os.getenv by raising an exception
        mock_getenv.side_effect = Exception("[TEST-GET-ENV] Mocked exception")
        
        # Call the get_env function
        result = get_env("ENV_TEST")
        
        # Assert the returned value is None
        self.assertIsNone(result)
        mock_getenv.assert_called_once_with("ENV_TEST")

if __name__ == '__main__':
    unittest.main()
