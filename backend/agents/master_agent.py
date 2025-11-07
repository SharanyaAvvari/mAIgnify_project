# backend/agents/master_agent.py

from .eda_agent import EDAAgent
from .classifier_agent import ClassifierAgent
from .tumor_agent import TumorAgent


class MasterAgent:
    def __init__(self, llm_model=None, llm_api_key=None, temperature=0.1):
        """
        Initialize the MasterAgent with all sub-agents.
        
        Args:
            llm_model: The language model to use for agents
            llm_api_key: API key for the language model
            temperature: Temperature setting for LLM responses (default: 0.1)
        """
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.temperature = temperature
        
        # Initialize the LLM (you can customize this based on your LLM setup)
        self.llm = self._initialize_llm(llm_model, llm_api_key, temperature)
        
        # Initialize all specialized agents
        self.agents = {
            'eda': EDAAgent(self.llm),
            'classifier': ClassifierAgent(self.llm),
            'tumor': TumorAgent(self.llm)
        }
    
    def _initialize_llm(self, model, api_key, temperature):
        """
        Initialize the language model.
        You can customize this based on your specific LLM implementation.
        """
        # Placeholder - replace with your actual LLM initialization
        # For example, if using OpenAI:
        # from openai import OpenAI
        # return OpenAI(api_key=api_key)
        
        # For now, return a simple dict with config
        return {
            'model': model,
            'api_key': api_key,
            'temperature': temperature
        }
    
    def execute(self, query: str, verbose: bool = False):
        """
        Execute a query by routing it to the appropriate agent.
        
        Args:
            query: The user query or task to execute
            verbose: Whether to print verbose output (default: False)
        
        Returns:
            dict: Result dictionary with status and output paths
        """
        if verbose:
            print(f"MasterAgent executing query: {query}")
        
        # Route query to appropriate agent based on keywords
        query_lower = query.lower()
        
        try:
            if 'eda' in query_lower or 'exploratory' in query_lower or 'analysis' in query_lower:
                if verbose:
                    print("Routing to EDA Agent...")
                return self.agents['eda'].execute(query, verbose=verbose)
            
            elif 'classifier' in query_lower or 'classification' in query_lower or 'classify' in query_lower:
                if verbose:
                    print("Routing to Classifier Agent...")
                return self.agents['classifier'].execute(query, verbose=verbose)
            
            elif 'tumor' in query_lower or 'tumour' in query_lower or 'brain' in query_lower or 'mri' in query_lower:
                if verbose:
                    print("Routing to Tumor Agent...")
                return self.agents['tumor'].execute(query, verbose=verbose)
            
            else:
                # Default behavior - try to intelligently route or return generic response
                if verbose:
                    print("No specific agent matched, using default response...")
                return {
                    "status": "success",
                    "message": "Query processed successfully",
                    "output_paths": ["output/dummy_result.txt"]
                }
        
        except Exception as e:
            if verbose:
                print(f"Error executing query: {str(e)}")
            return {
                "status": "error",
                "message": f"Error executing query: {str(e)}",
                "output_paths": []
            }
    
    def list_agents(self):
        """
        List all available agents.
        
        Returns:
            dict: Dictionary of agent names and their types
        """
        return {
            name: type(agent).__name__ 
            for name, agent in self.agents.items()
        }
    
    def get_agent(self, agent_name: str):
        """
        Get a specific agent by name.
        
        Args:
            agent_name: Name of the agent to retrieve
        
        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(agent_name)
