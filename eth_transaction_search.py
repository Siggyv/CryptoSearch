#!/usr/bin/env python3
import json
import os
import datetime
import torch
import torch.multiprocessing as mp
from web3 import Web3
from dotenv import load_dotenv
import argparse
import time
import requests

# Load environment variables from .env file
load_dotenv()

# The address we want to track
TARGET_ADDRESS = "0x7FF9cFad3877F21d41Da833E2F775dB0569eE3D9"

# Initialize connection to Ethereum node
# Try to get Infura API key from environment variable
INFURA_API_KEY = os.getenv("INFURA_API_KEY")
if not INFURA_API_KEY:
    print("Warning: No INFURA_API_KEY found in environment variables.")
    print("Please get a free API key from https://infura.io and add it to a .env file:")
    print('INFURA_API_KEY="your_api_key_here"')
    INFURA_API_KEY = input("Or enter your Infura API key now: ")

# --- Debugging line added ---
print(f"Attempting to connect with Infura API Key: {INFURA_API_KEY}")
# --- End debugging line ---

# Connect to Ethereum mainnet with proper error handling
try:
    w3 = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{INFURA_API_KEY}"))
    
    # Test connection with a simple call
    current_block = w3.eth.block_number
    print(f"Connected to Ethereum network. Current block number: {current_block}")
except Exception as e:
    print(f"Error connecting to Infura API: {e}")
    print("Please check your API key and internet connection.")
    print("The exact error might indicate if your API key is invalid or if there's a rate limit issue.")
    exit(1)

print(f"Connected to Ethereum network. Current block number: {w3.eth.block_number}")

def estimate_block_from_date(date):
    """
    Estimate the Ethereum block number from a date
    Uses an approximation based on 13-second block times
    
    Parameters:
    date (datetime): Date to estimate block for
    
    Returns:
    int: Estimated block number
    """
    # Ethereum genesis block date
    genesis_date = datetime.datetime(2015, 7, 30, 3, 26, 13, tzinfo=datetime.timezone.utc)
    
    # Calculate seconds since genesis
    seconds_since_genesis = (date - genesis_date).total_seconds()
    
    # Average block time in Ethereum is around 13 seconds
    estimated_block = int(seconds_since_genesis / 13)
    
    # Ensure we don't go above current block
    current_block = w3.eth.block_number
    return min(estimated_block, current_block)

def validate_date(date_str):
    """
    Validate and parse a date string in YYYY-MM-DD format
    
    Parameters:
    date_str (str): Date string in YYYY-MM-DD format
    
    Returns:
    datetime: Parsed datetime object with UTC timezone
    """
    try:
        # Parse the input date string
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        # Set timezone to UTC
        date = date.replace(tzinfo=datetime.timezone.utc)
        return date
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Please use YYYY-MM-DD format.")

def process_block(args):
    """
    Process a single block to find transactions related to an address
    Includes basic rate limiting (retry on 429 error)
    
    Parameters:
    args (tuple): Tuple containing (block_number, address, infura_key)
    
    Returns:
    list: Transactions in this block involving the address
    """
    block_number, address, infura_key = args
    max_retries = 3
    retry_delay = 1 # seconds
    
    # Each process needs its own Web3 connection
    local_w3 = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{infura_key}"))
    
    for attempt in range(max_retries):
        try:
            block = local_w3.eth.get_block(block_number, full_transactions=True)
            result = []
            
            for tx in block.transactions:
                # Check if the transaction involves our address
                if tx['from'].lower() == address or (tx.get('to') and tx['to'].lower() == address):
                    result.append(tx)
                    
            print(f"Processed block {block_number} - Found {len(result)} transactions")
            
            # Convert Web3 objects to dictionaries to make them serializable
            serializable_txs = []
            for tx in result:
                tx_dict = dict(tx)
                # Convert any non-serializable objects
                for k, v in tx_dict.items():
                    if isinstance(v, (bytes, bytearray)):
                        tx_dict[k] = v.hex()
                serializable_txs.append(tx_dict)
                
            return serializable_txs # Success, exit retry loop
            
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                print(f"Rate limit (429) hit processing block {block_number}. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2 # Basic exponential backoff
            else:
                # Handle other HTTP errors
                print(f"HTTP Error processing block {block_number}: {http_err}")
                return [] # Non-recoverable HTTP error for this block
        except Exception as e:
            # Handle other unexpected errors
            print(f"Unexpected Error processing block {block_number}: {e}")
            return [] # Non-recoverable error for this block
            
    # If all retries fail
    print(f"Failed to process block {block_number} after {max_retries} retries due to rate limiting.")
    return []

def get_transaction_history(address, start_block=0, end_block=None, num_workers=None, batch_size=50, use_gpu=False):
    """
    Get all transactions for a specific address using PyTorch's parallel processing
    
    Parameters:
    address (str): The Ethereum address to search for
    start_block (int): The block number to start searching from
    end_block (int): The block number to end searching at (defaults to latest block)
    num_workers (int): Number of parallel workers (defaults to number of CPU cores)
    batch_size (int): Size of batches to process at once
    use_gpu (bool): Whether to use GPU acceleration if available
    
    Returns:
    list: A list of transaction dictionaries
    """
    if end_block is None:
        end_block = w3.eth.block_number
    
    address = address.lower()
    
    # If num_workers is not specified, use the number of CPU cores
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    print(f"Searching from block {start_block} to {end_block} using PyTorch parallel processing...")
    print(f"Using {num_workers} workers")
    
    # Use the full block range derived from dates (or defaults)
    if start_block > end_block:
        print(f"Warning: Start block {start_block} is after end block {end_block}. No blocks to scan.")
        block_range = []
    else:
        # Range goes from end_block down to start_block (inclusive)
        block_range = list(range(end_block, start_block - 1, -1))
    
    block_range = [b for b in block_range if b >= start_block] # Keep this safety check
    
    # Split blocks into batches
    batches = [block_range[i:i + batch_size] for i in range(0, len(block_range), batch_size)]
    
    # Create a pool of workers
    mp.set_start_method('spawn', force=True)
    
    all_transactions = []
    
    # If PyTorch detected a GPU and use_gpu is True, try to use it
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    if use_gpu and torch.cuda.is_available():
        print(f"Using GPU acceleration: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for processing")
    
    # Process each batch with PyTorch multiprocessing
    for batch_idx, batch in enumerate(batches):
        print(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch)} blocks...")
        
        # Create arguments for each block
        args = [(block_num, address, INFURA_API_KEY) for block_num in batch]
        
        # Use PyTorch's multiprocessing pool
        with mp.Pool(processes=num_workers) as pool:
            batch_results = pool.map(process_block, args)
            
        # Flatten the list of lists
        for result in batch_results:
            all_transactions.extend(result)
            
        print(f"Batch {batch_idx+1} complete. Found {len(all_transactions)} transactions so far.")
    
    return all_transactions

def get_transactions_etherscan(address, start_date=None, end_date=None):
    """
    Alternative method using Etherscan API
    For production use, this is more practical than scanning blocks directly
    
    Parameters:
    address (str): The Ethereum address to search for
    start_date (datetime): Optional start date for filtering transactions
    end_date (datetime): Optional end date for filtering transactions
    
    Returns:
    list: A list of transaction dictionaries
    
    Note: You'll need to get an Etherscan API key and add it to your .env file
    """
    try:
        etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
        if not etherscan_api_key:
            print("No Etherscan API key found. Using direct blockchain scanning instead.")
            return None
        
        # Ensure the address is in the correct format (with 0x prefix and checksum format)
        try:
            # Attempt to checksum the address
            address = Web3.to_checksum_address(address)
        except ValueError:
            # If checksumming fails, just ensure it has 0x prefix
            if not address.startswith('0x'):
                address = '0x' + address
        
        # Convert dates to timestamps if provided
        start_timestamp = int(start_date.timestamp()) if start_date else 0
        end_timestamp = int(end_date.timestamp()) if end_date else int(datetime.datetime.now(datetime.timezone.utc).timestamp())
        
        print(f"Searching for transactions from {start_date} to {end_date if end_date else 'now'}")
        
        # Prepare parameters for outgoing transactions query
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': '0',
            'endblock': '99999999',
            'sort': 'desc',
            'apikey': etherscan_api_key
        }
        
        # Add optional timestamp parameters if dates are provided
        if start_date:
            params['starttime'] = str(int(start_timestamp))
        if end_date:
            params['endtime'] = str(int(end_timestamp))
            
        # Get transactions where address is the sender with error handling
        outgoing_txs = []
        try:
            url = "https://api.etherscan.io/api"
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"Error from Etherscan API (status {response.status_code}): {response.text}")
            else:
                result = response.json()
                if result.get('status') == '1':
                    outgoing_txs = result.get('result', [])
                else:
                    print(f"Etherscan API error: {result.get('message', 'Unknown error')}")
                    # If we hit rate limits, pause briefly
                    if 'rate limit' in result.get('message', '').lower():
                        print("Rate limit hit, pausing for 5 seconds...")
                        time.sleep(5)
        except Exception as e:
            print(f"Error fetching outgoing transactions: {e}")
        
        # Get transactions where address is the receiver
        # For internal transactions (contract calls)
        params['action'] = 'txlistinternal'
        
        # Reset the timestamps as they might not be supported for internal transactions
        if 'starttime' in params:
            del params['starttime']
        if 'endtime' in params:
            del params['endtime']
            
        incoming_txs = []
        try:
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"Error from Etherscan API for internal txs (status {response.status_code}): {response.text}")
            else:
                result = response.json()
                if result.get('status') == '1':
                    internal_txs = result.get('result', [])
                    
                    # Filter by date if needed
                    if start_date or end_date:
                        incoming_txs = [
                            tx for tx in internal_txs 
                            if int(tx.get('timeStamp', 0)) >= start_timestamp and 
                               int(tx.get('timeStamp', 0)) <= end_timestamp
                        ]
                    else:
                        incoming_txs = internal_txs
                else:
                    print(f"Etherscan API error for internal txs: {result.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"Error fetching internal transactions: {e}")
        
        # Also get ERC-20 token transfers if we didn't hit any rate limits
        token_txs = []
        try:
            params['action'] = 'tokentx'
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == '1':
                    all_token_txs = result.get('result', [])
                    
                    # Filter by date if needed
                    if start_date or end_date:
                        token_txs = [
                            tx for tx in all_token_txs 
                            if int(tx.get('timeStamp', 0)) >= start_timestamp and 
                               int(tx.get('timeStamp', 0)) <= end_timestamp
                        ]
                    else:
                        token_txs = all_token_txs
        except Exception as e:
            print(f"Error fetching token transactions (non-critical): {e}")
        
        # Combine all transaction types
        all_txs = outgoing_txs + incoming_txs + token_txs
        
        # If we got results, return them
        if all_txs:
            # Mark the transaction type
            for tx in outgoing_txs:
                tx['tx_type'] = 'normal'
            for tx in incoming_txs:
                tx['tx_type'] = 'internal'
            for tx in token_txs:
                tx['tx_type'] = 'token'
                
            return all_txs
        else:
            print("No transactions found via Etherscan or API error occurred.")
            return None
            
    except ImportError:
        print("Requests library not installed. Using direct blockchain scanning instead.")
        return None
    except Exception as e:
        print(f"Error fetching data from Etherscan: {e}")
        return None

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Search Ethereum transactions for a specific address')
    parser.add_argument('--address', type=str, default=TARGET_ADDRESS,
                        help='Ethereum address to search for')
    parser.add_argument('--start-date', type=str, 
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str,
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (defaults to CPU count)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of blocks to process in each batch')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration if available')
    parser.add_argument('--skip-etherscan', action='store_true',
                        help='Skip Etherscan API and use direct blockchain scanning')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set target address
    address = args.address
    print(f"Searching for transactions related to address: {address}")
    
    # Parse dates if provided
    start_date = validate_date(args.start_date) if args.start_date else None
    end_date = validate_date(args.end_date) if args.end_date else None
    
    start_block = estimate_block_from_date(start_date) if start_date else 0
    end_block = estimate_block_from_date(end_date) if end_date else None
    
    if start_date:
        print(f"Start date: {start_date} (estimated block: {start_block})")
    if end_date:
        print(f"End date: {end_date} (estimated block: {end_block})")
    
    # Try Etherscan API first (more efficient) unless skipped
    etherscan_txs = None
    if not args.skip_etherscan:
        etherscan_txs = get_transactions_etherscan(address, start_date, end_date)
    
    output_filename = f"{address.lower().replace('0x', '')}_transactions"
    if start_date:
        output_filename += f"_from_{args.start_date}"
    if end_date:
        output_filename += f"_to_{args.end_date}"
    output_filename += ".json"
    
    if etherscan_txs:
        print(f"Found {len(etherscan_txs)} transactions using Etherscan API")
        
        # Save transactions to a JSON file
        with open(output_filename, 'w') as f:
            json.dump(etherscan_txs, f, indent=2)
        
        # Print summary
        for i, tx in enumerate(etherscan_txs[:10]):  # Show only first 10
            tx_time = datetime.datetime.fromtimestamp(int(tx.get('timeStamp', 0)), tz=datetime.timezone.utc)
            print(f"Transaction {i+1}:")
            print(f"  Hash: {tx.get('hash', tx.get('blockHash', 'N/A'))}")
            print(f"  Date: {tx_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"  Type: {tx.get('tx_type', 'normal')}")
            print(f"  From: {tx.get('from', 'N/A')}")
            print(f"  To: {tx.get('to', 'N/A')}")
            
            # Handle value display differently for token transactions
            if tx.get('tx_type') == 'token':
                token_value = float(tx.get('value', 0)) / (10 ** int(tx.get('tokenDecimal', 18)))
                print(f"  Value: {token_value} {tx.get('tokenSymbol', 'tokens')}")
            else:
                print(f"  Value: {float(tx.get('value', 0)) / 10**18} ETH")
                
            print(f"  Block: {tx.get('blockNumber', 'N/A')}")
            print("")
        
        if len(etherscan_txs) > 10:
            print(f"... and {len(etherscan_txs) - 10} more transactions")
    else:
        # Fallback to direct blockchain scanning with PyTorch parallel processing
        print("Falling back to PyTorch-powered parallel blockchain scanning...")
        
        # PyTorch parallel processing
        transactions = get_transaction_history(
            address, 
            start_block, 
            end_block, 
            num_workers=args.workers,
            batch_size=args.batch_size,
            use_gpu=args.use_gpu
        )
        
        print(f"Found {len(transactions)} transactions")
        
        # Save transactions to a JSON file
        with open(output_filename, 'w') as f:
            json.dump(transactions, f, indent=2)
        
        # Print summary
        for i, tx in enumerate(transactions[:10]):  # Show only first 10
            # For PyTorch processed transactions, they are already serialized
            # So we format them directly
            from_addr = tx.get('from')
            to_addr = tx.get('to', 'Contract Creation')
            value = float(int(tx.get('value', 0), 16) if isinstance(tx.get('value'), str) else tx.get('value', 0)) / 10**18
            
            # Try to get timestamp from block if available
            if tx.get('blockNumber'):
                try:
                    block_num = int(tx['blockNumber'], 16) if isinstance(tx['blockNumber'], str) else tx['blockNumber']
                    block = w3.eth.get_block(block_num)
                    tx_time = datetime.datetime.fromtimestamp(block.timestamp, tz=datetime.timezone.utc)
                    date_str = tx_time.strftime('%Y-%m-%d %H:%M:%S UTC')
                except:
                    date_str = "N/A"
            else:
                date_str = "N/A"
            
            print(f"Transaction {i+1}:")
            print(f"  Hash: {tx.get('hash')}")
            print(f"  Date: {date_str}")
            print(f"  From: {from_addr}")
            print(f"  To: {to_addr}")
            print(f"  Value: {value} ETH")
            print(f"  Block: {tx.get('blockNumber')}")
            print("")
        
        if len(transactions) > 10:
            print(f"... and {len(transactions) - 10} more transactions")
    
    print(f"All transactions saved to {output_filename}")

if __name__ == "__main__":
    main() 