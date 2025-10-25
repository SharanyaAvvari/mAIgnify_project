"""
mAIstro: Multi-Agentic System for Medical AI
Main entry point for the autonomous multi-agent system
"""

import os
import argparse
import yaml
from pathlib import Path
from backend.agents.master_agent import MasterAgent
from typing import Optional

class MAIstroSystem:
    """Main orchestrator for the mAIstro multi-agentic system"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the mAIstro system with configuration"""
        self.config = self.load_config(config_path)
        self.master_agent = None
        self.setup_system()
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def setup_system(self):
        """Initialize the master agent and all sub-agents"""
        print("=" * 60)
        print("Initializing mAIstro Multi-Agentic System")
        print("=" * 60)
        
        # Initialize master agent with LLM configuration
        llm_config = self.config.get('llm', {})
        self.master_agent = MasterAgent(
            llm_model=llm_config.get('model', 'gpt-4o'),
            llm_api_key=llm_config.get('api_key'),
            temperature=llm_config.get('temperature', 0.1)
        )
        
        print(f"✓ Master Agent initialized with {llm_config.get('model')}")
        print(f"✓ All 8 specialized agents loaded")
        print("=" * 60)
    
    def execute_query(self, query: str, verbose: bool = True) -> dict:
        """
        Execute a natural language query through the master agent
        
        Args:
            query: Natural language instruction
            verbose: Print detailed execution logs
            
        Returns:
            Dictionary containing execution results and outputs
        """
        if verbose:
            print("\n" + "=" * 60)
            print("QUERY RECEIVED")
            print("=" * 60)
            print(f"{query}\n")
        
        try:
            # Execute through master agent
            result = self.master_agent.execute(query, verbose=verbose)
            
            if verbose:
                print("\n" + "=" * 60)
                print("EXECUTION COMPLETED")
                print("=" * 60)
                print(f"Status: {result.get('status', 'Unknown')}")
                if 'output_paths' in result:
                    print("\nOutput files saved to:")
                    for path in result['output_paths']:
                        print(f"  - {path}")
            
            return result
            
        except Exception as e:
            print(f"\n❌ Error during execution: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def interactive_mode(self):
        """Run mAIstro in interactive mode for multiple queries"""
        print("\n" + "=" * 60)
        print("mAIstro Interactive Mode")
        print("=" * 60)
        print("Enter your queries in natural language.")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            try:
                query = input("mAIstro> ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("\nExiting mAIstro. Goodbye!")
                    break
                
                if not query:
                    continue
                
                self.execute_query(query)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue


def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="mAIstro: Multi-Agentic System for Medical AI Development"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Single query to execute (non-interactive mode)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Initialize mAIstro system
    system = MAIstroSystem(config_path=args.config)
    
    if args.query:
        # Execute single query
        system.execute_query(args.query, verbose=args.verbose)
    elif args.interactive:
        # Interactive mode
        system.interactive_mode()
    else:
        # Default: interactive mode
        system.interactive_mode()


if __name__ == "__main__":
    main()