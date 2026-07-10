import serial
import time
import logging

class VibrationController:
    def __init__(self, port, baudrate=115200, timeout=2):
        """
        Initialize the ArduinoController.

        :param port: Serial port (e.g., 'COM3', '/dev/ttyUSB0')
        :param baudrate: Baud rate for serial communication (default 115200)
        :param timeout: Timeout for serial communication (default 2 seconds)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connection = None

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG)
        self.logger.info("ArduinoController initialized with port: %s, baudrate: %d", port, baudrate)

    def connect(self):
        """Establish a serial connection with the Arduino."""
        try:
            self.connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(2)  # Wait for the Arduino to initialize
            self.logger.info("Connected to Arduino on %s", self.port)
        except serial.SerialException as e:
            self.logger.error("Failed to connect to Arduino: %s", e)
            self.connection = None

    def disconnect(self):
        """Close the serial connection."""
        if self.connection and self.connection.is_open:
            self.connection.close()
            self.logger.info("Disconnected from Arduino.")

    def ensure_connection(self):
        """
        Ensure the serial connection is active. Reconnect if disconnected.
        """
        if not self.connection or not self.connection.is_open:
            self.logger.warning("Connection lost. Attempting to reconnect...")
            self.connect()

    def on(self, time_in_seconds):
        """
        Send the ON command to the Arduino with the specified time duration.

        :param time_in_seconds: Duration to turn the pin HIGH (in seconds)
        """
        self.logger.debug("Preparing to send ON command with duration: %f seconds", time_in_seconds)
        self.ensure_connection()

        if self.connection and self.connection.is_open:
            try:
                time_in_ms = int(time_in_seconds * 1000)  # Convert to milliseconds
                command = f"ON {time_in_ms}\n"  # Format the command
                self.logger.debug("Sending command: %s", command.strip())
                self.connection.write(command.encode('utf-8'))  # Send the command
                response = self.connection.readline().decode('utf-8').strip()  # Read Arduino response
                self.logger.info("Arduino response: %s", response)
            except serial.SerialException as e:
                self.logger.error("Error during communication: %s", e)
        else:
            self.logger.error("Failed to send command: No active connection.")

# Example usage:
# controller = ArduinoController(port='COM3')
# controller.on(2)
