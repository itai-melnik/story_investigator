"""CLI Entry point for story-investigator."""

import argparse
import sys


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='Story Investigator - RAG implementation for story analysis')
    parser.add_argument('--strategy', choices=['naive', 'graphrag', 'neo4j'], 
                       default='naive', help='RAG strategy to use')
    parser.add_argument('--query', type=str, help='Query to process')
    parser.add_argument('--file', type=str, help='Input file path')
    
    args = parser.parse_args()
    
    # TODO: Implement CLI logic
    print(f"Using strategy: {args.strategy}")
    if args.query:
        print(f"Query: {args.query}")
    if args.file:
        print(f"File: {args.file}")


if __name__ == '__main__':
    main()

