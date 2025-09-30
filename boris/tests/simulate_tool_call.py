import os
from boris.boriscore.terminal.terminal_interface import TerminalExecutor


# This script simulates terminal tool calls for debugging output, specifically for package installation commands.
# It captures the output and error messages, ensuring that they are cropped to a manageable size for display.
# Parameters for cropping limits can be adjusted as needed.

max_output_tokens = 200
max_error_tokens = 3000
command = "ls"


def main():
    # Instantiate TerminalExecutor with the current directory as the base path
    executor = TerminalExecutor(base_path=os.getcwd())
    print("\n\nRUN COMMAND\n\n")
    try:
        # Execute the command to install packages
        output = executor.run_terminal_tool(shell="powershell", command=command)

        # Print the output message with cropped logs
        print(output)
    except Exception as e:
        # Print error message in a consistent format
        print(f"Error occurred: {str(e)}")

    print("RUN SHELL")
    try:
        # Execute the command to install packages
        output = executor.run_shell(shell="powershell", command=command)

        # Print the output message with cropped logs

        print(output.stdout)
    except Exception as e:
        # Print error message in a consistent format
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
