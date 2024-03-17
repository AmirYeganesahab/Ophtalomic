import unittest
from unittest.mock import MagicMock
from led import led_module, serial

class TestLedModule(unittest.TestCase):

    def setUp(self):
        # Create an instance of led_module for testing
        self.ledCard = led_module()

        # Mock the serial port to avoid actual serial communication during tests
        self.ledCard.ser = MagicMock()

    def test_open(self):
        # Test if the open function returns a serial.Serial instance
        result = self.ledCard.open()
        self.assertIsInstance(result, serial.Serial)

    def test_calculate_checksum(self):
        # Test the calculate_checksum function with sample data
        data_bytes = [0xaa, 0x01, 0x00, 0x00, 0x63, 0x0e, 0x01]
        result = self.ledCard.calculate_checksum(data_bytes)
        expected_checksum = int(hex(199),16)  # Replace this with the correct checksum value
        self.assertEqual(result, expected_checksum)

    def test_illumCmd(self):
        # Test if illumination_command is correctly generated
        self.ledCard.illumCmd()
        self.assertIsInstance(self.ledCard.illumination_command, bytes)

    def test_flushCmd(self):
        # Test if flush_command is correctly generated
        self.ledCard.flushCmd(trigger_state=1)
        self.assertIsInstance(self.ledCard.flush_command, bytes)

    def test_trgCmd(self):
        # Test if trigger_command is correctly generated
        self.ledCard.trgCmd()
        self.assertIsInstance(self.ledCard.trigger_command, bytes)

    def test_illuminate(self):
        # Test the illuminate function
        self.ledCard.illuminate()
        self.ledCard.ser.write.assert_called_once_with(self.ledCard.illumination_command)

    def test_flush(self):
        # Test the flush function
        self.ledCard.flush()
        self.ledCard.ser.write.assert_called_once_with(self.ledCard.flush_command)

    def test_trigger(self):
        # Test the trigger function
        self.ledCard.trigger()
        self.ledCard.ser.write.assert_called_once_with(self.ledCard.trigger_command)

    def test_single_trigger(self):
        # Test the single_trigger function
        self.ledCard.single_trigger(led_number=1)
        # Add assertions based on the expected behavior of the function

if __name__ == '__main__':
    unittest.main()
