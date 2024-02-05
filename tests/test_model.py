import pytest, joblib, logging, warnings
warnings.filterwarnings('ignore')
import pandas as pd

pipeline = joblib.load('pipeline.pkl')

# Configure the logger
logging.basicConfig(filename="tests/model_testing.log", level=logging.INFO)

# Create a logger
logger = logging.getLogger("tests/model_testing")
logger.setLevel(logging.DEBUG) # Set the default log level for the logger

# Create a handler for INFO-level messages
info_handler = logging.FileHandler("tests/model_performance_info.log")
info_handler.setLevel(logging.INFO) 

# Create a handler for ERROR-level messages
error_handler = logging.FileHandler("tests/error.log")
error_handler.setLevel(logging.ERROR)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Set the formatter for the handlers
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(info_handler)
logger.addHandler(error_handler)

test_cases = [
    # Test Case 1
    {
        "input": pd.DataFrame([[3962.4078,69.23,1066,0.6,9,3,11,10284.516,14,4]],columns=['Profit','Discount Percentage','Postal Code','Discount','Order Day','Order Month','Quantity','Operating Expenses','Ship Day','Order Weekday']),
        "expected_output": 6489.28
    },
    # Test Case 2
    {
        "input": pd.DataFrame([[1022.67,52.95,5938,0.2,21,6,10,7491.308,18,5]],columns=['Profit','Discount Percentage','Postal Code','Discount','Order Day','Order Month','Quantity','Operating Expenses','Ship Day','Order Weekday']),
        "expected_output": 6261.03
    },
    # Test Case 3
    {
        "input": pd.DataFrame([[1245.73,16.88,4500,0.5,27,5,8,8216.935,4,0]],columns=['Profit','Discount Percentage','Postal Code','Discount','Order Day','Order Month','Quantity','Operating Expenses','Ship Day','Order Weekday']),
        "expected_output": 6338.52
    },
    # Test Case 4
    {
        "input": pd.DataFrame([[41.9032,0,42396,0,9,11,2,220.019,12,2]],columns=['Profit','Discount Percentage','Postal Code','Discount','Order Day','Order Month','Quantity','Operating Expenses','Ship Day','Order Weekday']),
        "expected_output": 262.83
    },
    # Test Case 5
    {
        "input": pd.DataFrame([[3.312,21.68,43291,0.2,9,10,5,11.408,13,6]],columns=['Profit','Discount Percentage','Postal Code','Discount','Order Day','Order Month','Quantity','Operating Expenses','Ship Day','Order Weekday']),
        "expected_output": 8.03
    }
]

@pytest.mark.parametrize("test_input, expected_output", [(tc["input"], tc["expected_output"]) for tc in test_cases])
def test_prediction_with_custom_input(test_input,expected_output):
    # Make a prediction on the test input
    prediction = pipeline.predict(test_input)[0]
    logger.info("Difference between predicted and expected sales: %.2f",round(abs(prediction - expected_output),2))

logging.shutdown()



