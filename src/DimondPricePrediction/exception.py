import sys

class customException(Exception):
    def __init__(self, error_message, error_details:sys) -> None:
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()

        self.line_no = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self) -> str:
        return f"Error Occured in Python Script: {self.file_name}, at line number {self.line_no}, ERROR MESSAGE: {self.error_message}"
    

if __name__ == "__main__":
    try:
        a = 1/0     # Intentionally raised error

    except Exception as ex:
        raise customException(ex, sys)
