feat: Implement random data generator and high-precision server time sync utility

Key Features:
- Generates data using hardware random numbers (RDSEED/RDRAND).
- Creates various types of random data, including passwords, lottery numbers, coordinates, and NMS portal addresses.
- Supports bulk data generation using multithreading with a progress display.
- Includes a ladder game and generators for random integers/floats within a specified range.

Server Time Synchronization Module (`time.rs`):
- Acquires time by parsing the HTTP Date header from a remote server via TCP or curl.
- Implements a precision time correction logic that compensates for network latency by measuring RTT (Round-Trip Time).
- Automatically triggers a system action (mouse click or F5 key press) when a user-specified time is reached.
- Supports action execution on multiple operating systems, including Windows, macOS, and Linux.
