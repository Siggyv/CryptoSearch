# Ethereum Transaction Tracker

This script searches the Ethereum blockchain for all transactions involving a specific address, both as sender and receiver.

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your API keys:
   ```
   cp .env.example .env
   ```
4. Edit the `.env` file with your actual API keys

## Usage

Basic usage:
```
python eth_transaction_search.py
```

With date filtering:
```
python eth_transaction_search.py --start-date 2023-01-01 --end-date 2023-12-31
```

With a different address:
```
python eth_transaction_search.py --address 0x123456789abcdef --start-date 2023-01-01
```

With PyTorch parallel processing options:
```
python eth_transaction_search.py --workers 16 --batch-size 100 --use-gpu
```

### Command Line Arguments

- `--address`: Ethereum address to search for (defaults to the address in the script)
- `--start-date`: Start date for filtering transactions (format: YYYY-MM-DD)
- `--end-date`: End date for filtering transactions (format: YYYY-MM-DD)
- `--workers`: Number of parallel workers (defaults to CPU core count)
- `--batch-size`: Number of blocks to process in each batch (default: 50)
- `--use-gpu`: Enable GPU acceleration if available (requires CUDA-compatible GPU)

## How It Works

The script will:
1. First try to use the Etherscan API (recommended, much faster)
2. Fall back to direct blockchain scanning if Etherscan API is not available
3. Save all found transactions to a JSON file
4. Display a summary of the first 10 transactions

When filtering by date:
- For the Etherscan API, the script filters transactions by their timestamps
- For direct blockchain scanning, the script estimates block numbers from dates
- The JSON output filename will include the date range

## Parallel Processing with PyTorch

For direct blockchain scanning, the script uses PyTorch's multiprocessing capabilities to significantly speed up the process:

- Each block is processed in parallel using a pool of workers
- The number of workers defaults to your CPU core count but can be customized
- GPU acceleration is available for compatible systems
- Blocks are processed in batches to optimize memory usage

## Modifying the Target Address

You can either:
1. Use the `--address` command line argument
2. Edit the `TARGET_ADDRESS` variable in `eth_transaction_search.py`

## API Key Information

- **Infura API Key**: Required. Get a free key at [infura.io](https://infura.io)
- **Etherscan API Key**: Optional but recommended for better performance. Get a free key at [etherscan.io](https://etherscan.io)

## Notes

- Direct blockchain scanning is much faster with PyTorch parallelism but still slower than using the Etherscan API
- The script is limited to checking 1000 blocks in direct scanning mode for demonstration purposes
- Block estimation from dates is approximate (assumes 13-second block times)
- When using GPU acceleration, make sure you have PyTorch with CUDA support installed 