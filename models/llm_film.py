"""LLM-based FiLM parameter generator.

Uses an LLM to understand the original instruction and VLA actions,
then generates gamma and beta parameters to transform actions based on a new instruction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import json
import re


class LLMFiLMGenerator(nn.Module):
    """
    LLM-based FiLM generator that uses language models to generate
    gamma and beta parameters for action modulation.
    
    The LLM receives:
    - Original instruction (e.g., "push object to the goal")
    - Current action values from VLA (without FiLM)
    - New instruction (e.g., "push object to the right")
    
    And outputs gamma and beta to transform the action.
    """
    
    def __init__(
        self, 
        action_dim: int = 4,
        llm_provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        gamma_shift: float = 1.0,
        cache_responses: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.gamma_shift = gamma_shift
        self.cache_responses = cache_responses
        self._cache = {}
        
    def _get_llm_client(self):
        """Get the appropriate LLM client based on provider."""
        if self.llm_provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI()
            except ImportError:
                print("[LLMFiLM] OpenAI not installed.")
                return None
        else:
            return None
    
    def _build_prompt(self, original_instruction: str, action: np.ndarray, new_instruction: str) -> str:
        """Build the prompt for the LLM to generate gamma and beta."""
        
        action_str = ", ".join([f"{a:.4f}" for a in action])
        
        prompt = f"""You are a robot action modifier. Given an original instruction, the robot's planned action, and a new instruction, you need to compute transformation parameters (gamma and beta) to modify the action.

The action is a {self.action_dim}-dimensional vector representing:
- action[0]: x-direction movement (positive = right, negative = left)
- action[1]: y-direction movement (positive = forward, negative = backward)  
- action[2]: z-direction movement (positive = up, negative = down)
- action[3]: gripper command (positive = close, negative = open)

The transformation formula is: new_action = gamma * original_action + beta

Think step by step:
1. What does the original instruction want the robot to do?
2. What does the new instruction want instead?
3. How should each action dimension be scaled (gamma) and shifted (beta)?

Original Instruction: "{original_instruction}"
Current Action: [{action_str}]
New Instruction: "{new_instruction}"

Respond with ONLY a JSON object containing gamma and beta arrays:
{{"gamma": [g0, g1, g2, g3], "beta": [b0, b1, b2, b3]}}

Guidelines:
- gamma=1.0, beta=0.0 means no change for that dimension
- gamma=0.0 means ignore original action for that dimension
- gamma=-1.0 means reverse direction
- beta adds/subtracts fixed offset

JSON response:"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Tuple[np.ndarray, np.ndarray]:
        """Parse the LLM response to extract gamma and beta."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                gamma = np.array(data["gamma"], dtype=np.float32)
                beta = np.array(data["beta"], dtype=np.float32)
                return gamma, beta
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[LLMFiLM] Failed to parse response: {e}")
        
        # Fallback: return gamma=1, beta=0
        return np.ones(self.action_dim, dtype=np.float32), np.zeros(self.action_dim, dtype=np.float32)
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        client = self._get_llm_client()
        if client is None:
            return ""
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        return response.choices[0].message.content
    
    def generate_film_params_llm(self, original_instruction: str, action: torch.Tensor, new_instruction: str,) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate gamma and beta using LLM.
        
        Args:
            original_instruction: The instruction VLA was conditioned on
            action: (B, action_dim) actions from VLA (without FiLM)
            new_instruction: The new instruction to adapt to
            
        Returns:
            gamma: (B, action_dim) multiplicative factors
            beta: (B, action_dim) additive offsets
        """
        device = action.device
        B = action.size(0)
        
        # For now, process batch element by element (could batch LLM calls)
        gammas = []
        betas = []
        
        for i in range(B):
            action_np = action[i].detach().cpu().numpy()
            
            # Check cache
            # If cache_key is exists in cache, use cached response

            cache_key = (original_instruction, tuple(action_np.round(3)), new_instruction)

            if self.cache_responses and cache_key in self._cache:
                gamma, beta = self._cache[cache_key]
            else:
                # Build prompt and call LLM
                prompt = self._build_prompt(original_instruction, action_np, new_instruction)
                
                if self.llm_provider == "openai":
                    response = self._call_openai(prompt)
                else:
                    response = ""
                
                gamma, beta = self._parse_llm_response(response)
                
                if self.cache_responses:
                    self._cache[cache_key] = (gamma, beta)
            
            gammas.append(gamma)
            betas.append(beta)
        
        gamma_t = torch.tensor(np.stack(gammas), dtype=torch.float32, device=device)
        beta_t = torch.tensor(np.stack(betas), dtype=torch.float32, device=device)

        print(f"[Gamma shape]: {gamma_t.shape}, [Beta shape]: {beta_t.shape}")
        
        return gamma_t, beta_t
    
    def forward(self, action: torch.Tensor, original_instruction: str, new_instruction: str,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate FiLM parameters and apply to action.
        """
        gamma, beta = self.generate_film_params_llm( original_instruction, action, new_instruction)
        
        # Apply FiLM transformation
        new_action = gamma * action + beta
        
        return new_action, gamma, beta
    
    def clear_cache(self):
        """Clear the response cache."""
        self._cache = {}


class LLMFiLMWrapper:
    """
    High-level wrapper for using LLM-FiLM with VLA models.
    
    Example usage:
        wrapper = LLMFiLMWrapper(vla_model, llm_provider="openai")
        
        # Get base action from VLA
        action = vla_model.act(image, text_ids, state)
        
        # Modify action with new instruction via LLM
        new_action = wrapper.modify_action(
            action=action,
            original_instruction="push object to the goal",
            new_instruction="push object to the right"
        )
    """
    
    def __init__(
        self,
        vla_model: nn.Module,
        action_dim: int = 4,
        llm_provider: str = "openai",
        model_name: str = "gpt-4o-mini",
    ):
        self.vla = vla_model
        self.llm_film = LLMFiLMGenerator(
            action_dim=action_dim,
            llm_provider=llm_provider,
            model_name=model_name,
        )
    
    def get_base_action(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Get action from VLA without FiLM modification."""
        with torch.no_grad():
            return self.vla.act(image, text_tokens, state)
    
    def modify_action(
        self,
        action: torch.Tensor,
        original_instruction: str,
        new_instruction: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modify action using LLM-generated FiLM parameters.
        
        Returns:
            new_action, gamma, beta
        """
        return self.llm_film(
            action=action,
            original_instruction=original_instruction,
            new_instruction=new_instruction,
            use_llm=True,
        )
    
    def act_with_new_instruction(
        self,
        image: torch.Tensor,
        original_text_tokens: torch.Tensor,
        state: torch.Tensor,
        original_instruction: str,
        new_instruction: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full pipeline: get VLA action, then modify with LLM-FiLM.
        
        Returns:
            new_action, gamma, beta
        """
        base_action = self.get_base_action(image, original_text_tokens, state)
        return self.modify_action(base_action, original_instruction, new_instruction)
