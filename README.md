# Ethereum Transaction Tracker

This script searches the Ethereum blockchain for all transactions involving a specific address, both as sender and receiver.

## Requirements

- **Python**: Python 3.8 or higher is required
- **PyTorch**: Version 2.2.0 or higher (the script will automatically install the correct version)
- **GPU (Optional)**: CUDA-compatible GPU for acceleration
- **API Keys**: Infura API key is required; Etherscan API key is recommended

### Version Compatibility Notes

- If you encounter PyTorch installation issues, you may need to install it separately following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/)
- For GPU acceleration, make sure to install the CUDA-compatible version of PyTorch
- Web3.py compatibility: This script works with web3.py version 6.x

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   If this fails for PyTorch, try installing it separately:
   ```
   pip install torch
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

Skip Etherscan API (if experiencing API errors):
```
python eth_transaction_search.py --skip-etherscan
```

### Command Line Arguments

- `--address`: Ethereum address to search for (defaults to the address in the script)
- `--start-date`: Start date for filtering transactions (format: YYYY-MM-DD)
- `--end-date`: End date for filtering transactions (format: YYYY-MM-DD)
- `--workers`: Number of parallel workers (defaults to CPU core count)
- `--batch-size`: Number of blocks to process in each batch (default: 50)
- `--use-gpu`: Enable GPU acceleration if available (requires CUDA-compatible GPU)
- `--skip-etherscan`: Skip Etherscan API and use only direct blockchain scanning

## How It Works

The script will:
1. First try to use the Etherscan API (recommended, much faster) unless `--skip-etherscan` is specified
2. Fall back to direct blockchain scanning if Etherscan API is not available or fails
3. Save all found transactions to a JSON file
4. Display a summary of the first 10 transactions

### Transaction Types Tracked

The script now tracks three types of transactions:
- **Normal**: Standard ETH transfers
- **Internal**: Contract-initiated transfers
- **Token**: ERC-20 token transfers (with proper token symbol and decimals)

### Date Filtering

When filtering by date:
- For the Etherscan API, the script filters transactions by their timestamps
- For direct blockchain scanning, the script estimates block numbers from dates
- The JSON output filename includes the date range

## Parallel Processing with PyTorch

For direct blockchain scanning, the script uses PyTorch's multiprocessing capabilities to significantly speed up the process:

- Each block is processed in parallel using a pool of workers
- The number of workers defaults to your CPU core count but can be customized
- GPU acceleration is available for compatible systems
- Blocks are processed in batches to optimize memory usage

## API Error Handling

The script now includes improved error handling for API requests:
- Detailed error messages for API failures
- Automatic retry with pause on rate limiting
- Proper parameter formatting for all API requests
- Option to skip Etherscan API completely with `--skip-etherscan`

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
- If you experience 400 errors with the Etherscan API, try using the `--skip-etherscan` flag 